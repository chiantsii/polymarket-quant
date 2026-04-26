"""Structured transition model for full-state Markov training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from polymarket_quant.state.transition_targets import DEFAULT_PRIMITIVE_TARGET_COLUMNS
from polymarket_quant.utils.logger import get_logger
from polymarket_quant.utils.math import sigmoid

logger = get_logger(__name__)


# Rollout should only depend on state components that are actually refreshed
# after each simulated next-state step. This avoids repeatedly extrapolating
# stale orderbook/regime observations far beyond the horizon they describe.
ROLLOUT_SAFE_FEATURE_COLUMNS: tuple[str, ...] = (
    "current_latent_up_probability",
    "current_latent_logit_probability",
    "current_seconds_to_end",
    "current_normalized_time_to_end",
    "current_dist_to_boundary",
    "current_boundary_leverage_ratio",
    "current_up_book_velocity",
    "current_down_book_velocity",
    "target_horizon_seconds",
)


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
    jump_max_iter: int = 200
    jump_max_depth: int = 5
    jump_learning_rate: float = 0.05
    jump_std_multiplier: float = 2.5
    jump_abs_latent_logit_threshold: float = 0.12
    diffusion_oof_folds: int = 5
    diffusion_variance_floor: float = 5e-4
    diffusion_variance_cap: float = 5e-2
    parametric_min_jump_rows: int = 12
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
    jump_model: HistGradientBoostingClassifier | DummyClassifier
    rollout_feature_columns: tuple[str, ...]
    rollout_drift_model: HistGradientBoostingRegressor | None
    rollout_diffusion_model: HistGradientBoostingRegressor | None
    rollout_jump_model: HistGradientBoostingClassifier | DummyClassifier | None
    latent_mu_model: HistGradientBoostingRegressor | None
    latent_sigma_model: HistGradientBoostingRegressor | None
    latent_jump_mean_model: HistGradientBoostingRegressor | None
    latent_jump_std_model: HistGradientBoostingRegressor | None
    default_step_seconds: float
    jump_label_column: str = "jump_event"

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

        if hasattr(self.jump_model, "predict_proba"):
            jump_probs = self.jump_model.predict_proba(X)
            if jump_probs.ndim == 2 and jump_probs.shape[1] >= 2:
                predictions["jump_intensity_hat"] = jump_probs[:, 1]
            else:
                predictions["jump_intensity_hat"] = jump_probs[:, 0]
        else:
            predictions["jump_intensity_hat"] = self.jump_model.predict(X)

        predictions["jump_intensity_hat"] = pd.to_numeric(predictions["jump_intensity_hat"], errors="coerce").clip(0.0, 1.0)
        return predictions

    def predict_latent_kernel(self, transition_targets: pd.DataFrame) -> pd.DataFrame:
        """Predict the parametric latent jump-diffusion kernel in logit space."""

        if transition_targets.empty:
            return transition_targets.copy()

        predictions = transition_targets.copy()
        feature_columns = getattr(self, "rollout_feature_columns", ()) or self.feature_columns
        X = _feature_frame(predictions, feature_columns)

        mu_model = getattr(self, "latent_mu_model", None)
        sigma_model = getattr(self, "latent_sigma_model", None)
        if mu_model is None or sigma_model is None:
            raise ValueError("Transition bundle does not contain parametric latent kernel models")

        dt = _prediction_step_seconds(predictions)
        mu_hat = pd.Series(mu_model.predict(X), index=predictions.index, dtype=float)
        log_sigma_sq_hat = pd.Series(sigma_model.predict(X), index=predictions.index, dtype=float)
        sigma_sq_hat = _clip_diffusion_variance(np.exp(log_sigma_sq_hat).astype(float), self.config)
        sigma_hat = np.sqrt(sigma_sq_hat)

        jump_model = getattr(self, "rollout_jump_model", None) or self.jump_model
        if hasattr(jump_model, "predict_proba"):
            jump_probs = jump_model.predict_proba(X)
            if jump_probs.ndim == 2 and jump_probs.shape[1] >= 2:
                jump_probability = np.asarray(jump_probs[:, 1], dtype=float)
            else:
                jump_probability = np.asarray(jump_probs[:, 0], dtype=float)
        else:
            jump_probability = np.asarray(jump_model.predict(X), dtype=float)
        jump_probability = np.clip(jump_probability, 0.0, 1.0)
        lambda_hat = _poisson_intensity_from_step_probability(jump_probability, dt.to_numpy(dtype=float))

        jump_mean_model = getattr(self, "latent_jump_mean_model", None)
        jump_std_model = getattr(self, "latent_jump_std_model", None)
        if jump_mean_model is not None:
            jump_mean_hat = pd.Series(jump_mean_model.predict(X), index=predictions.index, dtype=float)
        else:
            jump_mean_hat = pd.Series(0.0, index=predictions.index, dtype=float)

        if jump_std_model is not None:
            log_jump_var_hat = pd.Series(jump_std_model.predict(X), index=predictions.index, dtype=float)
            jump_var_hat = _clip_diffusion_variance(np.exp(log_jump_var_hat).astype(float), self.config)
            jump_std_hat = pd.Series(np.sqrt(jump_var_hat), index=predictions.index, dtype=float)
        else:
            jump_std_hat = pd.Series(0.0, index=predictions.index, dtype=float)

        predictions["mu_hat_latent_logit_probability"] = mu_hat
        predictions["sigma_hat_latent_logit_probability"] = sigma_hat
        predictions["lambda_hat_latent_logit_probability"] = lambda_hat
        predictions["jump_mean_hat_latent_logit_probability"] = jump_mean_hat
        predictions["jump_std_hat_latent_logit_probability"] = jump_std_hat
        predictions["jump_probability_hat_latent_logit_probability"] = jump_probability
        return predictions

    def predict_latent_step(self, transition_targets: pd.DataFrame) -> pd.DataFrame:
        """Predict one-step latent quantities with backward-compatible aliases."""

        mu_model = getattr(self, "latent_mu_model", None)
        sigma_model = getattr(self, "latent_sigma_model", None)
        if mu_model is None or sigma_model is None:
            predictions = self.predict(transition_targets)
            if "future_hat_latent_up_probability" not in predictions.columns and "future_hat_latent_logit_probability" in predictions.columns:
                latent_logit = pd.to_numeric(predictions["future_hat_latent_logit_probability"], errors="coerce")
                latent_logit_values = latent_logit.to_numpy(dtype=float, na_value=np.nan)
                latent_probability = np.asarray(sigmoid(latent_logit_values), dtype=float)
                latent_probability[~np.isfinite(latent_logit_values)] = np.nan
                predictions["future_hat_latent_up_probability"] = latent_probability
            return predictions

        predictions = self.predict_latent_kernel(transition_targets)
        dt = _prediction_step_seconds(predictions)

        drift_hat = predictions["mu_hat_latent_logit_probability"] * dt
        diffusion_hat = np.square(predictions["sigma_hat_latent_logit_probability"]) * dt

        predictions["drift_hat_latent_logit_probability"] = drift_hat
        predictions["diffusion_hat_latent_logit_probability"] = diffusion_hat

        current_column = "current_latent_logit_probability"
        if current_column in predictions.columns:
            current_values = pd.to_numeric(predictions[current_column], errors="coerce")
            future_latent_logit = current_values + drift_hat
            predictions["future_hat_latent_logit_probability"] = future_latent_logit
            latent_logit_values = future_latent_logit.to_numpy(dtype=float, na_value=np.nan)
            latent_probability = np.asarray(sigmoid(latent_logit_values), dtype=float)
            latent_probability[~np.isfinite(latent_logit_values)] = np.nan
            predictions["future_hat_latent_up_probability"] = latent_probability

        predictions["jump_intensity_hat"] = predictions["jump_probability_hat_latent_logit_probability"]
        return predictions

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
    rollout_feature_columns = _resolve_rollout_feature_columns(feature_columns)
    primitive_target_columns = _resolve_primitive_target_columns(training_rows, config)
    X = _feature_frame(training_rows, feature_columns)
    X_rollout = _feature_frame(training_rows, rollout_feature_columns) if rollout_feature_columns else X

    drift_models: dict[str, HistGradientBoostingRegressor] = {}
    diffusion_models: dict[str, HistGradientBoostingRegressor] = {}
    target_summaries: dict[str, dict[str, float | int]] = {}
    rollout_drift_model: HistGradientBoostingRegressor | None = None
    rollout_diffusion_model: HistGradientBoostingRegressor | None = None
    latent_mu_model: HistGradientBoostingRegressor | None = None
    latent_sigma_model: HistGradientBoostingRegressor | None = None
    latent_jump_mean_model: HistGradientBoostingRegressor | None = None
    latent_jump_std_model: HistGradientBoostingRegressor | None = None

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
        if target_column == "latent_logit_probability" and rollout_feature_columns:
            rollout_drift_model = HistGradientBoostingRegressor(
                max_iter=config.drift_max_iter,
                max_depth=config.drift_max_depth,
                learning_rate=config.drift_learning_rate,
                random_state=config.random_state,
            )
            rollout_drift_model.fit(X_rollout.loc[valid], y.loc[valid])
            rollout_oof_drift_pred = _out_of_fold_drift_predictions(
                drift_model=rollout_drift_model,
                X=X_rollout.loc[valid],
                y=y.loc[valid],
                config=config,
            )
            rollout_residual = y.loc[valid] - rollout_oof_drift_pred
            rollout_diffusion_target = np.log(
                np.clip(np.square(rollout_residual), config.diffusion_variance_floor, config.diffusion_variance_cap)
            )
            rollout_diffusion_model = HistGradientBoostingRegressor(
                max_iter=config.diffusion_max_iter,
                max_depth=config.diffusion_max_depth,
                learning_rate=config.diffusion_learning_rate,
                random_state=config.random_state,
            )
            rollout_diffusion_model.fit(X_rollout.loc[valid], rollout_diffusion_target)
        target_summaries[target_column] = {
            "rows": int(valid.sum()),
            "mae": float(mean_absolute_error(y.loc[valid], drift_pred)),
            "rmse": float(mean_squared_error(y.loc[valid], drift_pred) ** 0.5),
            "r2": float(r2_score(y.loc[valid], drift_pred)),
            "residual_std": float(residual.std(ddof=0)),
        }

    if not drift_models:
        raise ValueError("No primitive transition targets had enough non-null rows to fit")

    jump_labels = _build_jump_event_labels(training_rows, config)
    jump_valid = jump_labels.notna()
    jump_y = jump_labels.loc[jump_valid].astype(int)
    rollout_jump_model: HistGradientBoostingClassifier | DummyClassifier | None
    if jump_y.nunique() <= 1:
        jump_model: HistGradientBoostingClassifier | DummyClassifier = DummyClassifier(strategy="most_frequent")
        jump_model.fit(X.loc[jump_valid], jump_y)
        rollout_jump_model = DummyClassifier(strategy="most_frequent")
        rollout_jump_model.fit(X_rollout.loc[jump_valid], jump_y)
    else:
        jump_model = HistGradientBoostingClassifier(
            max_iter=config.jump_max_iter,
            max_depth=config.jump_max_depth,
            learning_rate=config.jump_learning_rate,
            random_state=config.random_state,
        )
        jump_model.fit(X.loc[jump_valid], jump_y)
        rollout_jump_model = HistGradientBoostingClassifier(
            max_iter=config.jump_max_iter,
            max_depth=config.jump_max_depth,
            learning_rate=config.jump_learning_rate,
            random_state=config.random_state,
        )
        rollout_jump_model.fit(X_rollout.loc[jump_valid], jump_y)

    latent_target_column = "latent_logit_probability"
    latent_delta_column = f"target_delta_{latent_target_column}"
    latent_step_seconds = _training_step_seconds(training_rows)
    latent_valid = (
        pd.to_numeric(training_rows.get(latent_delta_column), errors="coerce").notna()
        & latent_step_seconds.notna()
        & (latent_step_seconds > 0.0)
    )
    if latent_valid.sum() < config.min_training_rows:
        raise ValueError("Need enough latent_logit_probability rows to fit parametric latent kernel")

    latent_delta = pd.to_numeric(training_rows.loc[latent_valid, latent_delta_column], errors="coerce")
    latent_dt = latent_step_seconds.loc[latent_valid]
    latent_rate = latent_delta / latent_dt

    latent_mu_model = HistGradientBoostingRegressor(
        max_iter=config.drift_max_iter,
        max_depth=config.drift_max_depth,
        learning_rate=config.drift_learning_rate,
        random_state=config.random_state,
    )
    latent_mu_model.fit(X_rollout.loc[latent_valid], latent_rate)
    latent_mu_oof = _out_of_fold_drift_predictions(
        drift_model=latent_mu_model,
        X=X_rollout.loc[latent_valid],
        y=latent_rate,
        config=config,
    )
    latent_residual = latent_delta - (latent_mu_oof * latent_dt)
    latent_sigma_sq = np.clip(
        np.square(latent_residual) / latent_dt,
        config.diffusion_variance_floor,
        config.diffusion_variance_cap,
    )
    latent_sigma_model = HistGradientBoostingRegressor(
        max_iter=config.diffusion_max_iter,
        max_depth=config.diffusion_max_depth,
        learning_rate=config.diffusion_learning_rate,
        random_state=config.random_state,
    )
    latent_sigma_model.fit(X_rollout.loc[latent_valid], np.log(latent_sigma_sq))
    latent_sigma_hat = np.sqrt(
        _clip_diffusion_variance(
            np.exp(latent_sigma_model.predict(X_rollout.loc[latent_valid])),
            config,
        )
    )

    latent_jump_valid = latent_valid & jump_labels.fillna(0).astype(bool)
    if int(latent_jump_valid.sum()) >= int(config.parametric_min_jump_rows):
        latent_jump_delta = pd.to_numeric(training_rows.loc[latent_jump_valid, latent_delta_column], errors="coerce")
        latent_jump_dt = latent_step_seconds.loc[latent_jump_valid]
        latent_jump_mu = pd.Series(
            latent_mu_model.predict(X_rollout.loc[latent_jump_valid]),
            index=latent_jump_delta.index,
            dtype=float,
        )
        latent_jump_excess = latent_jump_delta - (latent_jump_mu * latent_jump_dt)

        latent_jump_mean_model = HistGradientBoostingRegressor(
            max_iter=config.drift_max_iter,
            max_depth=config.drift_max_depth,
            learning_rate=config.drift_learning_rate,
            random_state=config.random_state,
        )
        latent_jump_mean_model.fit(X_rollout.loc[latent_jump_valid], latent_jump_excess)
        latent_jump_mean_oof = _out_of_fold_drift_predictions(
            drift_model=latent_jump_mean_model,
            X=X_rollout.loc[latent_jump_valid],
            y=latent_jump_excess,
            config=config,
        )
        latent_jump_residual = latent_jump_excess - latent_jump_mean_oof
        latent_jump_var = np.clip(
            np.square(latent_jump_residual),
            config.diffusion_variance_floor,
            config.diffusion_variance_cap,
        )
        latent_jump_std_model = HistGradientBoostingRegressor(
            max_iter=config.diffusion_max_iter,
            max_depth=config.diffusion_max_depth,
            learning_rate=config.diffusion_learning_rate,
            random_state=config.random_state,
        )
        latent_jump_std_model.fit(X_rollout.loc[latent_jump_valid], np.log(latent_jump_var))
    else:
        latent_jump_mean_model = None
        latent_jump_std_model = None

    if hasattr(rollout_jump_model, "predict_proba"):
        jump_probs = rollout_jump_model.predict_proba(X_rollout.loc[latent_valid])
        if jump_probs.ndim == 2 and jump_probs.shape[1] >= 2:
            latent_jump_probability = np.asarray(jump_probs[:, 1], dtype=float)
        else:
            latent_jump_probability = np.asarray(jump_probs[:, 0], dtype=float)
    else:
        latent_jump_probability = np.asarray(rollout_jump_model.predict(X_rollout.loc[latent_valid]), dtype=float)
    latent_jump_probability = np.clip(latent_jump_probability, 0.0, 1.0)
    latent_lambda_hat = _poisson_intensity_from_step_probability(
        latent_jump_probability,
        latent_dt.to_numpy(dtype=float),
    )

    bundle = TransitionModelBundle(
        config=config,
        feature_columns=feature_columns,
        primitive_target_columns=tuple(drift_models.keys()),
        drift_models=drift_models,
        diffusion_models=diffusion_models,
        jump_model=jump_model,
        rollout_feature_columns=rollout_feature_columns,
        rollout_drift_model=rollout_drift_model,
        rollout_diffusion_model=rollout_diffusion_model,
        rollout_jump_model=rollout_jump_model,
        latent_mu_model=latent_mu_model,
        latent_sigma_model=latent_sigma_model,
        latent_jump_mean_model=latent_jump_mean_model,
        latent_jump_std_model=latent_jump_std_model,
        default_step_seconds=_default_step_seconds(training_rows),
    )
    predictions = bundle.predict(transition_targets)
    latent_kernel_predictions = bundle.predict_latent_step(transition_targets)
    latent_kernel_columns = (
        "mu_hat_latent_logit_probability",
        "sigma_hat_latent_logit_probability",
        "lambda_hat_latent_logit_probability",
        "jump_mean_hat_latent_logit_probability",
        "jump_std_hat_latent_logit_probability",
        "jump_probability_hat_latent_logit_probability",
        "drift_hat_latent_logit_probability",
        "diffusion_hat_latent_logit_probability",
        "future_hat_latent_logit_probability",
        "future_hat_latent_up_probability",
        "jump_intensity_hat",
    )
    for column in latent_kernel_columns:
        if column in latent_kernel_predictions.columns:
            predictions[column] = latent_kernel_predictions[column]

    summary = {
        "training_rows": int(len(training_rows)),
        "default_step_seconds": float(bundle.default_step_seconds),
        "feature_columns": list(feature_columns),
        "rollout_feature_columns": list(rollout_feature_columns),
        "primitive_target_columns": list(bundle.primitive_target_columns),
        "jump_event_rate": float(jump_y.mean()) if len(jump_y) else 0.0,
        "parametric_latent_kernel": {
            "target": latent_target_column,
            "kernel_columns": [
                "mu_hat_latent_logit_probability",
                "sigma_hat_latent_logit_probability",
                "lambda_hat_latent_logit_probability",
                "jump_mean_hat_latent_logit_probability",
                "jump_std_hat_latent_logit_probability",
                "jump_probability_hat_latent_logit_probability",
            ],
            "step_seconds_median": float(latent_dt.median()),
            "min_jump_rows": int(config.parametric_min_jump_rows),
            "jump_rows": int(latent_jump_valid.sum()),
            "has_jump_size_models": bool(latent_jump_mean_model is not None and latent_jump_std_model is not None),
            "mu_rate_mae": float(mean_absolute_error(latent_rate, latent_mu_oof)),
            "mu_rate_rmse": float(mean_squared_error(latent_rate, latent_mu_oof) ** 0.5),
            "sigma_mean": float(np.mean(latent_sigma_hat)),
            "lambda_mean": float(np.mean(latent_lambda_hat)),
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


def _resolve_rollout_feature_columns(feature_columns: tuple[str, ...]) -> tuple[str, ...]:
    rollout_columns = tuple(column for column in feature_columns if column in ROLLOUT_SAFE_FEATURE_COLUMNS)
    if "target_horizon_seconds" not in rollout_columns and "target_horizon_seconds" in feature_columns:
        rollout_columns = (*rollout_columns, "target_horizon_seconds")
    if not rollout_columns:
        return feature_columns
    return rollout_columns


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


def _prediction_step_seconds(rows: pd.DataFrame) -> pd.Series:
    dt = _frame_numeric_series(rows, "realized_horizon_seconds")
    if not dt.notna().any():
        dt = _frame_numeric_series(rows, "target_horizon_seconds")
    dt = dt.where(dt > 0.0, np.nan)
    return dt.fillna(1.0)


def _frame_numeric_series(rows: pd.DataFrame, column: str) -> pd.Series:
    if column in rows.columns:
        return pd.to_numeric(rows[column], errors="coerce")
    return pd.Series(np.nan, index=rows.index, dtype=float)


def _poisson_intensity_from_step_probability(step_probability: np.ndarray, dt: np.ndarray) -> np.ndarray:
    clipped_probability = np.clip(np.asarray(step_probability, dtype=float), 0.0, 1.0 - 1e-9)
    safe_dt = np.maximum(np.asarray(dt, dtype=float), 1e-9)
    return -np.log1p(-clipped_probability) / safe_dt


def _clip_diffusion_variance(values: Any, config: TransitionModelConfig) -> np.ndarray:
    variance = np.asarray(values, dtype=float)
    return np.clip(
        variance,
        float(config.diffusion_variance_floor),
        float(config.diffusion_variance_cap),
    )


def _build_jump_event_labels(rows: pd.DataFrame, config: TransitionModelConfig) -> pd.Series:
    latent_delta = pd.to_numeric(rows.get("target_delta_latent_logit_probability"), errors="coerce")
    current_observation_variance = pd.to_numeric(rows.get("current_state_observation_variance"), errors="coerce").clip(lower=0.0)
    expected_std = np.sqrt(current_observation_variance)
    jump_threshold = np.maximum(config.jump_abs_latent_logit_threshold, config.jump_std_multiplier * expected_std)
    latent_jump = latent_delta.abs() > jump_threshold

    regime_columns = [
        "regime_normal_posterior",
        "regime_shock_posterior",
        "regime_convergence_posterior",
    ]
    regime_change = pd.Series(False, index=rows.index)
    current_regime_columns = [f"current_{column}" for column in regime_columns]
    future_regime_columns = [f"future_{column}" for column in regime_columns]
    if all(column in rows.columns for column in current_regime_columns + future_regime_columns):
        current_regime = rows[current_regime_columns].apply(pd.to_numeric, errors="coerce")
        future_regime = rows[future_regime_columns].apply(pd.to_numeric, errors="coerce")
        current_argmax = current_regime.to_numpy(dtype=float).argmax(axis=1)
        future_argmax = future_regime.to_numpy(dtype=float).argmax(axis=1)
        regime_change = pd.Series(current_argmax != future_argmax, index=rows.index)

    jump_event = (latent_jump.fillna(False) | regime_change.fillna(False)).astype(int)
    return jump_event


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
