"""Structured transition model for full-state Markov training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from polymarket_quant.state.transition_targets import DEFAULT_PRIMITIVE_TARGET_COLUMNS
from polymarket_quant.utils.logger import get_logger
from polymarket_quant.utils.math import sigmoid

logger = get_logger(__name__)


@dataclass(frozen=True)
class TransitionModelConfig:
    """Configuration for the first structured full-state transition model."""

    event_duration_seconds: float = 300.0
    feature_columns: tuple[str, ...] | None = None
    primitive_target_columns: tuple[str, ...] | None = None
    drift_max_iter: int = 200
    drift_max_depth: int = 6
    drift_learning_rate: float = 0.05
    diffusion_max_iter: int = 150
    diffusion_max_depth: int = 5
    diffusion_learning_rate: float = 0.05
    diffusion_oof_folds: int = 5
    diffusion_variance_floor: float = 5e-4
    diffusion_variance_cap: float = 5e-2
    numerical_variance_epsilon: float = 1e-18
    random_state: int = 42
    min_training_rows: int = 32


@dataclass
class TransitionModelFitResult:
    """Artifacts produced by transition-model fitting."""

    predictions: pd.DataFrame
    summary: dict[str, Any]
    bundle: "TransitionModelBundle"


@dataclass
class TransitionModelBundle:
    """Serializable bundle of structured transition and latent kernel models."""

    config: TransitionModelConfig
    feature_columns: tuple[str, ...]
    primitive_target_columns: tuple[str, ...]
    drift_models: dict[str, HistGradientBoostingRegressor]
    diffusion_models: dict[str, HistGradientBoostingRegressor]
    spot_mu_model: HistGradientBoostingRegressor | None
    spot_sigma_model: HistGradientBoostingRegressor | None
    default_step_seconds: float

    def predict(self, transition_targets: pd.DataFrame) -> pd.DataFrame:
        """Predict structured transition quantities for each row."""

        if transition_targets.empty:
            return transition_targets.copy()

        predictions = transition_targets.copy()
        X = _feature_frame(predictions, self.feature_columns)

        for target_column in self.primitive_target_columns:
            current_column = f"current_{target_column}"
            drift_output = f"drift_hat_{target_column}"
            diffusion_output = f"diffusion_hat_{target_column}"
            future_output = f"future_hat_{target_column}"

            drift_model = self.drift_models.get(target_column)
            diffusion_model = self.diffusion_models.get(target_column)
            if drift_model is None or diffusion_model is None:
                continue

            drift_hat = pd.Series(drift_model.predict(X), index=predictions.index, dtype=float)
            log_diffusion_hat = pd.Series(diffusion_model.predict(X), index=predictions.index, dtype=float)
            diffusion_hat = _clip_diffusion_variance(np.exp(log_diffusion_hat).astype(float), self.config)

            predictions[drift_output] = drift_hat
            predictions[diffusion_output] = diffusion_hat

            if current_column in predictions.columns:
                current_values = pd.to_numeric(predictions[current_column], errors="coerce")
                predictions[future_output] = current_values + drift_hat

        self._postprocess_future_state(predictions)
        return predictions

    def predict_spot_kernel(self, transition_targets: pd.DataFrame) -> pd.DataFrame:
        """Predict a state-conditioned log-spot jump-diffusion kernel."""

        if transition_targets.empty:
            return transition_targets.copy()

        mu_model = getattr(self, "spot_mu_model", None)
        sigma_model = getattr(self, "spot_sigma_model", None)
        if mu_model is None or sigma_model is None:
            raise ValueError("Transition bundle does not contain parametric spot kernel models")

        predictions = transition_targets.copy()
        X = _feature_frame(predictions, self.feature_columns)

        mu_hat = pd.Series(mu_model.predict(X), index=predictions.index, dtype=float)
        log_sigma_sq_hat = pd.Series(sigma_model.predict(X), index=predictions.index, dtype=float)
        sigma_sq_hat = _positive_variance_from_log_predictions(log_sigma_sq_hat.to_numpy(dtype=float), self.config)
        sigma_hat = np.sqrt(sigma_sq_hat)

        predictions["mu_hat_log_spot_ratio"] = mu_hat
        predictions["sigma_hat_log_spot_ratio"] = sigma_hat
        return predictions

    def predict_spot_kernel_from_event_state(self, event_state_row: dict[str, Any]) -> dict[str, float]:
        """Project a serialized event-state row into spot-kernel parameters."""

        feature_row = _event_state_to_transition_feature_row(
            event_state_row,
            feature_columns=self.feature_columns,
            target_horizon_seconds=self.default_step_seconds,
        )
        predictions = self.predict_spot_kernel(pd.DataFrame([feature_row]))
        return predictions.iloc[0].to_dict()

    def _postprocess_future_state(self, predictions: pd.DataFrame) -> None:
        regime_columns = [
            "regime_normal_posterior",
            "regime_shock_posterior",
            "regime_convergence_posterior",
        ]
        regime_future_columns = [f"future_hat_{column}" for column in regime_columns if f"future_hat_{column}" in predictions.columns]
        if regime_future_columns:
            regime_values = predictions[regime_future_columns].apply(pd.to_numeric, errors="coerce").clip(lower=0.0)
            row_sums = regime_values.sum(axis=1)
            valid = row_sums > 1e-12
            regime_values.loc[valid] = regime_values.loc[valid].div(row_sums.loc[valid], axis=0)
            predictions.loc[:, regime_future_columns] = regime_values

        if "future_hat_latent_logit_probability" in predictions.columns:
            latent_logit = pd.to_numeric(predictions["future_hat_latent_logit_probability"], errors="coerce")
            latent_logit_values = latent_logit.to_numpy(dtype=float, na_value=np.nan)
            latent_probability = np.asarray(sigmoid(latent_logit_values), dtype=float)
            latent_probability[~np.isfinite(latent_logit_values)] = np.nan
            predictions["future_hat_latent_up_probability"] = latent_probability

        if "current_normalized_time_to_end" in predictions.columns and "target_horizon_seconds" in predictions.columns:
            decrement = pd.to_numeric(predictions["target_horizon_seconds"], errors="coerce") / max(
                float(self.config.event_duration_seconds),
                1e-12,
            )
            current_normalized_time = pd.to_numeric(predictions["current_normalized_time_to_end"], errors="coerce")
            predictions["future_hat_normalized_time_to_end"] = (current_normalized_time - decrement).clip(lower=0.0, upper=1.0)


def fit_transition_model(
    transition_targets: pd.DataFrame,
    config: TransitionModelConfig | None = None,
) -> TransitionModelFitResult:
    """Fit a structured transition model on matched future state pairs."""

    config = config or TransitionModelConfig()
    if transition_targets.empty:
        raise ValueError("Transition-target dataset is empty")

    training_rows = transition_targets.copy()
    if "has_future_target" in training_rows.columns:
        training_rows = training_rows[training_rows["has_future_target"].fillna(False)].copy()
    if training_rows.empty:
        raise ValueError("Transition-target dataset contains no matched future targets")
    if len(training_rows) < config.min_training_rows:
        raise ValueError(
            f"Need at least {config.min_training_rows} matched rows to fit a transition model; got {len(training_rows)}"
        )

    feature_columns = _resolve_feature_columns(training_rows, config)
    primitive_target_columns = _resolve_primitive_target_columns(training_rows, config)
    X = _feature_frame(training_rows, feature_columns)

    drift_models: dict[str, HistGradientBoostingRegressor] = {}
    diffusion_models: dict[str, HistGradientBoostingRegressor] = {}
    target_summaries: dict[str, dict[str, float | int]] = {}
    spot_mu_model: HistGradientBoostingRegressor | None = None
    spot_sigma_model: HistGradientBoostingRegressor | None = None

    for target_column in primitive_target_columns:
        delta_column = f"target_delta_{target_column}"
        if delta_column not in training_rows.columns:
            continue

        y = pd.to_numeric(training_rows[delta_column], errors="coerce")
        valid = y.notna()
        if valid.sum() < config.min_training_rows:
            logger.warning("Skipping %s because only %s non-null targets are available", target_column, int(valid.sum()))
            continue

        drift_model = HistGradientBoostingRegressor(
            max_iter=config.drift_max_iter,
            max_depth=config.drift_max_depth,
            learning_rate=config.drift_learning_rate,
            random_state=config.random_state,
        )
        drift_model.fit(X.loc[valid], y.loc[valid])
        drift_pred = pd.Series(drift_model.predict(X.loc[valid]), index=y.loc[valid].index, dtype=float)
        oof_drift_pred = _out_of_fold_drift_predictions(
            drift_model=drift_model,
            X=X.loc[valid],
            y=y.loc[valid],
            config=config,
        )
        residual = y.loc[valid] - oof_drift_pred

        diffusion_target = np.log(np.clip(np.square(residual), config.diffusion_variance_floor, config.diffusion_variance_cap))
        diffusion_model = HistGradientBoostingRegressor(
            max_iter=config.diffusion_max_iter,
            max_depth=config.diffusion_max_depth,
            learning_rate=config.diffusion_learning_rate,
            random_state=config.random_state,
        )
        diffusion_model.fit(X.loc[valid], diffusion_target)

        drift_models[target_column] = drift_model
        diffusion_models[target_column] = diffusion_model
        target_summaries[target_column] = {
            "rows": int(valid.sum()),
            "mae": float(mean_absolute_error(y.loc[valid], drift_pred)),
            "rmse": float(mean_squared_error(y.loc[valid], drift_pred) ** 0.5),
            "r2": float(r2_score(y.loc[valid], drift_pred)),
            "residual_std": float(residual.std(ddof=0)),
        }

    if not drift_models:
        raise ValueError("No primitive transition targets had enough non-null rows to fit")

    spot_kernel = _build_spot_kernel_training_targets(training_rows)
    spot_mu_model: HistGradientBoostingRegressor | None
    spot_sigma_model: HistGradientBoostingRegressor | None
    spot_rate = spot_kernel["delta"] / spot_kernel["dt"]
    if int(spot_kernel["valid"].sum()) >= int(config.min_training_rows):
        spot_mu_model = HistGradientBoostingRegressor(
            max_iter=config.drift_max_iter,
            max_depth=config.drift_max_depth,
            learning_rate=config.drift_learning_rate,
            random_state=config.random_state,
        )
        spot_mu_model.fit(X.loc[spot_kernel["valid"]], spot_rate)
        spot_mu_oof = _out_of_fold_drift_predictions(
            drift_model=spot_mu_model,
            X=X.loc[spot_kernel["valid"]],
            y=spot_rate,
            config=config,
        )
        spot_residual = spot_kernel["delta"] - (spot_mu_oof * spot_kernel["dt"])
        spot_sigma_sq = np.maximum(
            np.square(spot_residual) / spot_kernel["dt"],
            float(config.numerical_variance_epsilon),
        )
        spot_sigma_model = HistGradientBoostingRegressor(
            max_iter=config.diffusion_max_iter,
            max_depth=config.diffusion_max_depth,
            learning_rate=config.diffusion_learning_rate,
            random_state=config.random_state,
        )
        spot_sigma_model.fit(X.loc[spot_kernel["valid"]], np.log(spot_sigma_sq))
    else:
        spot_mu_model = None
        spot_sigma_model = None

    bundle = TransitionModelBundle(
        config=config,
        feature_columns=feature_columns,
        primitive_target_columns=tuple(drift_models.keys()),
        drift_models=drift_models,
        diffusion_models=diffusion_models,
        spot_mu_model=spot_mu_model,
        spot_sigma_model=spot_sigma_model,
        default_step_seconds=_default_step_seconds(training_rows),
    )
    predictions = bundle.predict(transition_targets)

    if spot_mu_model is not None and spot_sigma_model is not None:
        spot_kernel_predictions = bundle.predict_spot_kernel(transition_targets)
        spot_kernel_columns = (
            "mu_hat_log_spot_ratio",
            "sigma_hat_log_spot_ratio",
        )
        for column in spot_kernel_columns:
            if column in spot_kernel_predictions.columns:
                predictions[column] = spot_kernel_predictions[column]

    summary = {
        "training_rows": int(len(training_rows)),
        "default_step_seconds": float(bundle.default_step_seconds),
        "feature_columns": list(feature_columns),
        "primitive_target_columns": list(bundle.primitive_target_columns),
        "parametric_spot_kernel": {
            "target": "log_spot_ratio",
            "kernel_columns": [
                "mu_hat_log_spot_ratio",
                "sigma_hat_log_spot_ratio",
            ],
            "step_seconds_median": float(spot_kernel["dt"].median()) if len(spot_kernel["dt"]) else float("nan"),
            "rows": int(spot_kernel["valid"].sum()),
            "has_models": bool(spot_mu_model is not None and spot_sigma_model is not None),
        },
        "target_metrics": target_summaries,
    }
    return TransitionModelFitResult(predictions=predictions, summary=summary, bundle=bundle)


def _resolve_feature_columns(
    transition_targets: pd.DataFrame,
    config: TransitionModelConfig,
) -> tuple[str, ...]:
    if config.feature_columns is not None:
        feature_columns = tuple(column for column in config.feature_columns if column in transition_targets.columns)
        if not feature_columns:
            raise ValueError("Configured feature columns were not found in transition_targets")
        return feature_columns

    excluded = {"has_future_target"}
    feature_columns = tuple(
        column
        for column in transition_targets.columns
        if (
            column.startswith("current_")
            or column == "target_horizon_seconds"
        )
        and column not in excluded
    )
    if not feature_columns:
        raise ValueError("No default feature columns were found in transition_targets")
    return feature_columns


def _resolve_primitive_target_columns(
    transition_targets: pd.DataFrame,
    config: TransitionModelConfig,
) -> tuple[str, ...]:
    configured = config.primitive_target_columns or DEFAULT_PRIMITIVE_TARGET_COLUMNS
    primitive_columns = tuple(
        column for column in configured if f"target_delta_{column}" in transition_targets.columns
    )
    if not primitive_columns:
        raise ValueError("No primitive target delta columns were found in transition_targets")
    return primitive_columns


def _feature_frame(rows: pd.DataFrame, feature_columns: tuple[str, ...]) -> pd.DataFrame:
    feature_frame = rows.reindex(columns=list(feature_columns)).copy()
    for column in feature_frame.columns:
        feature_frame[column] = pd.to_numeric(feature_frame[column], errors="coerce")
    return feature_frame


def _out_of_fold_drift_predictions(
    *,
    drift_model: HistGradientBoostingRegressor,
    X: pd.DataFrame,
    y: pd.Series,
    config: TransitionModelConfig,
) -> pd.Series:
    if len(X) < 2:
        return pd.Series(drift_model.predict(X), index=y.index, dtype=float)

    n_splits = min(max(int(config.diffusion_oof_folds), 2), len(X))
    if n_splits < 2:
        return pd.Series(drift_model.predict(X), index=y.index, dtype=float)

    kfold = KFold(n_splits=n_splits, shuffle=False)
    oof_pred = pd.Series(np.nan, index=y.index, dtype=float)
    for train_idx, test_idx in kfold.split(X):
        fold_model = clone(drift_model)
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        if X_train.empty or X_test.empty:
            continue
        fold_model.fit(X_train, y_train)
        oof_pred.iloc[test_idx] = fold_model.predict(X_test)

    missing = oof_pred.isna()
    if missing.any():
        oof_pred.loc[missing] = drift_model.predict(X.loc[missing])
    return oof_pred


def _training_step_seconds(rows: pd.DataFrame) -> pd.Series:
    step_seconds = _frame_numeric_series(rows, "realized_horizon_seconds")
    if step_seconds.notna().any():
        return step_seconds
    return _frame_numeric_series(rows, "target_horizon_seconds")


def _frame_numeric_series(rows: pd.DataFrame, column: str) -> pd.Series:
    if column in rows.columns:
        return pd.to_numeric(rows[column], errors="coerce")
    return pd.Series(np.nan, index=rows.index, dtype=float)


def _clip_diffusion_variance(values: Any, config: TransitionModelConfig) -> np.ndarray:
    variance = np.asarray(values, dtype=float)
    return np.clip(
        variance,
        float(config.diffusion_variance_floor),
        float(config.diffusion_variance_cap),
    )


def _positive_variance_from_log_predictions(log_values: Any, config: TransitionModelConfig) -> np.ndarray:
    variance = np.exp(np.asarray(log_values, dtype=float))
    return np.maximum(variance, float(config.numerical_variance_epsilon))


def _default_step_seconds(rows: pd.DataFrame) -> float:
    realized = pd.to_numeric(rows.get("realized_horizon_seconds"), errors="coerce")
    realized = realized[realized.notna() & (realized > 0.0)]
    if len(realized):
        return float(realized.median())

    target_horizon = pd.to_numeric(rows.get("target_horizon_seconds"), errors="coerce")
    target_horizon = target_horizon[target_horizon.notna() & (target_horizon > 0.0)]
    if len(target_horizon):
        return float(target_horizon.median())
    return 1.0


def _build_spot_kernel_training_targets(rows: pd.DataFrame) -> dict[str, pd.Series]:
    current_return = pd.to_numeric(rows.get("current_spot_return_since_reference"), errors="coerce")
    future_return = pd.to_numeric(rows.get("future_spot_return_since_reference"), errors="coerce")
    dt = _training_step_seconds(rows)

    valid = (
        current_return.notna()
        & future_return.notna()
        & (current_return > -1.0)
        & (future_return > -1.0)
        & dt.notna()
        & (dt > 0.0)
    )
    current_log_ratio = pd.Series(np.nan, index=rows.index, dtype=float)
    future_log_ratio = pd.Series(np.nan, index=rows.index, dtype=float)
    current_log_ratio.loc[valid] = np.log1p(current_return.loc[valid])
    future_log_ratio.loc[valid] = np.log1p(future_return.loc[valid])
    delta = future_log_ratio - current_log_ratio
    return {
        "valid": valid,
        "current": current_log_ratio.loc[valid],
        "future": future_log_ratio.loc[valid],
        "delta": delta.loc[valid],
        "dt": dt.loc[valid],
    }


def _event_state_to_transition_feature_row(
    event_state_row: dict[str, Any],
    *,
    feature_columns: tuple[str, ...],
    target_horizon_seconds: float,
) -> dict[str, Any]:
    feature_row: dict[str, Any] = {}
    for column in feature_columns:
        if column == "target_horizon_seconds":
            feature_row[column] = target_horizon_seconds
            continue
        if column.startswith("current_"):
            source_column = column.removeprefix("current_")
            feature_row[column] = event_state_row.get(source_column)
            continue
        feature_row[column] = event_state_row.get(column)
    return feature_row
