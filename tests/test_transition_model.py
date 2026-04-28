from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from scripts.fit_transition_model import fit_transition_model_artifacts
from polymarket_quant.state import TransitionModelConfig, fit_transition_model


def _synthetic_transition_targets(n_rows: int = 96) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    start = datetime(2026, 4, 11, 10, 0, 0, tzinfo=timezone.utc)
    rows: list[dict[str, object]] = []
    for idx in range(n_rows):
        imbalance = float(rng.uniform(-0.3, 0.3))
        basis = float(rng.uniform(-0.04, 0.04))
        normalized_time = float(rng.uniform(0.15, 0.95))
        latent_logit = float(rng.normal(0.0, 0.25))
        horizon_seconds = 15.0
        delta_latent_logit = 0.65 * imbalance - 1.2 * basis + 0.2 * (1.0 - normalized_time) + float(
            rng.normal(0.0, 0.02)
        )
        future_latent_logit = latent_logit + delta_latent_logit

        current_regimes = np.array(
            [
                max(0.05, 0.75 - 0.8 * abs(basis)),
                max(0.05, 0.15 + 2.0 * abs(basis)),
                max(0.05, 0.10 + 0.5 * (1.0 - normalized_time)),
            ],
            dtype=float,
        )
        current_regimes = current_regimes / current_regimes.sum()

        future_regimes = np.array(
            [
                max(0.05, current_regimes[0] - 0.20 * abs(delta_latent_logit)),
                max(0.05, current_regimes[1] + 0.20 * abs(delta_latent_logit)),
                max(0.05, current_regimes[2] + 0.05 * (1.0 - normalized_time)),
            ],
            dtype=float,
        )
        future_regimes = future_regimes / future_regimes.sum()

        up_micro = 0.50 + 0.15 * imbalance + 0.5 * basis
        down_micro = 1.0 - up_micro - basis
        future_up_micro = up_micro + 0.3 * delta_latent_logit
        future_down_micro = down_micro - 0.3 * delta_latent_logit

        current_time = start + timedelta(seconds=idx)
        future_time = current_time + timedelta(seconds=int(horizon_seconds))

        row = {
            "event_slug": f"btc-updown-5m-1775578800-{idx // 32}",
            "asset": "BTC",
            "current_collected_at": current_time.isoformat(),
            "future_collected_at": future_time.isoformat(),
            "target_horizon_seconds": horizon_seconds,
            "realized_horizon_seconds": horizon_seconds,
            "horizon_error_seconds": 0.0,
            "has_future_target": True,
            "target_status": "matched",
            "current_normalized_time_to_end": normalized_time,
            "current_spot_price": 100.0 + idx * 0.01,
            "current_spot_bid": 99.9 + idx * 0.01,
            "current_spot_ask": 100.1 + idx * 0.01,
            "current_spot_return_since_reference": basis,
            "current_spot_vol_multiplier": 1.0 + 0.5 * abs(basis),
            "current_external_spot_drift": basis,
            "current_up_best_bid": up_micro - 0.01,
            "current_up_best_ask": up_micro + 0.01,
            "current_down_best_bid": down_micro - 0.01,
            "current_down_best_ask": down_micro + 0.01,
            "current_up_mid_price": up_micro,
            "current_down_mid_price": down_micro,
            "current_up_micro_price": up_micro,
            "current_down_micro_price": down_micro,
            "current_up_spread": 0.02,
            "current_down_spread": 0.02,
            "current_up_bid_depth_top_5": 120.0 + 20.0 * max(imbalance, 0.0),
            "current_up_ask_depth_top_5": 110.0 + 20.0 * max(-imbalance, 0.0),
            "current_down_bid_depth_top_5": 118.0,
            "current_down_ask_depth_top_5": 112.0,
            "current_up_orderbook_imbalance": imbalance,
            "current_down_orderbook_imbalance": -imbalance,
            "current_up_weighted_imbalance": imbalance,
            "current_down_weighted_imbalance": -imbalance,
            "current_up_depth_slope": 10.0,
            "current_down_depth_slope": 10.0,
            "current_up_tick_density": 0.8,
            "current_down_tick_density": 0.8,
            "current_up_book_velocity": abs(delta_latent_logit),
            "current_down_book_velocity": abs(delta_latent_logit) * 0.9,
            "current_cross_book_basis": basis,
            "current_cross_book_bid_basis": basis - 0.01,
            "current_cross_book_ask_basis": basis + 0.01,
            "current_spread_divergence": 0.0,
            "current_dist_to_boundary": min(up_micro, 1.0 - up_micro),
            "current_boundary_leverage_ratio": 1.0 / max(1.0 - up_micro, 1e-6),
            "current_asymmetric_depth_ratio": 1.05,
            "current_book_age_max": 0.5,
            "current_has_full_book_pair": 1.0,
            "current_market_implied_up_probability": 0.5 + 0.5 * basis,
            "current_fundamental_up_probability": 0.5,
            "current_latent_up_probability": 1.0 / (1.0 + np.exp(-latent_logit)),
            "current_latent_logit_probability": latent_logit,
            "current_market_fundamental_basis": basis,
            "current_latent_market_basis": (1.0 / (1.0 + np.exp(-latent_logit))) - (0.5 + 0.5 * basis),
            "current_latent_fundamental_basis": (1.0 / (1.0 + np.exp(-latent_logit))) - 0.5,
            "current_abs_market_fundamental_basis": abs(basis),
            "current_abs_latent_market_basis": abs((1.0 / (1.0 + np.exp(-latent_logit))) - (0.5 + 0.5 * basis)),
            "current_abs_latent_fundamental_basis": abs((1.0 / (1.0 + np.exp(-latent_logit))) - 0.5),
            "current_volatility_per_sqrt_second": 0.001 + 0.0005 * abs(basis),
            "current_state_observation_variance": 0.01 + 0.01 * abs(basis),
            "current_regime_normal_posterior": current_regimes[0],
            "current_regime_shock_posterior": current_regimes[1],
            "current_regime_convergence_posterior": current_regimes[2],
            "future_market_implied_up_probability": 0.5 + 0.5 * (basis + 0.25 * delta_latent_logit),
            "future_spot_return_since_reference": basis + 0.15 * delta_latent_logit,
            "future_up_micro_price": future_up_micro,
            "future_down_micro_price": future_down_micro,
            "future_up_bid_depth_top_5": 120.0 + 12.0 * max(imbalance + 0.1 * delta_latent_logit, 0.0),
            "future_up_ask_depth_top_5": 110.0 + 12.0 * max(-(imbalance + 0.1 * delta_latent_logit), 0.0),
            "future_down_bid_depth_top_5": 118.0 + 4.0 * abs(delta_latent_logit),
            "future_down_ask_depth_top_5": 112.0 + 4.0 * abs(delta_latent_logit),
            "future_cross_book_basis": basis + 0.1 * delta_latent_logit,
            "future_regime_normal_posterior": future_regimes[0],
            "future_regime_shock_posterior": future_regimes[1],
            "future_regime_convergence_posterior": future_regimes[2],
            "future_latent_logit_probability": future_latent_logit,
            "future_seconds_to_end": max(300.0 - idx - horizon_seconds, 0.0),
            "future_normalized_time_to_end": max(normalized_time - horizon_seconds / 300.0, 0.0),
            "target_delta_market_implied_up_probability": 0.25 * delta_latent_logit,
            "target_delta_up_micro_price": future_up_micro - up_micro,
            "target_delta_down_micro_price": future_down_micro - down_micro,
            "target_delta_up_weighted_imbalance": 0.05 * delta_latent_logit,
            "target_delta_down_weighted_imbalance": -0.05 * delta_latent_logit,
            "target_delta_up_bid_depth_top_5": (120.0 + 12.0 * max(imbalance + 0.1 * delta_latent_logit, 0.0))
            - (120.0 + 20.0 * max(imbalance, 0.0)),
            "target_delta_up_ask_depth_top_5": (110.0 + 12.0 * max(-(imbalance + 0.1 * delta_latent_logit), 0.0))
            - (110.0 + 20.0 * max(-imbalance, 0.0)),
            "target_delta_down_bid_depth_top_5": 4.0 * abs(delta_latent_logit),
            "target_delta_down_ask_depth_top_5": 4.0 * abs(delta_latent_logit),
            "target_delta_cross_book_basis": 0.1 * delta_latent_logit,
            "target_delta_latent_logit_probability": delta_latent_logit,
            "target_delta_regime_normal_posterior": future_regimes[0] - current_regimes[0],
            "target_delta_regime_shock_posterior": future_regimes[1] - current_regimes[1],
            "target_delta_regime_convergence_posterior": future_regimes[2] - current_regimes[2],
        }
        rows.append(row)

    return pd.DataFrame(rows)


def test_fit_transition_model_returns_structured_predictions() -> None:
    transition_targets = _synthetic_transition_targets()
    fit_result = fit_transition_model(
        transition_targets,
        config=TransitionModelConfig(min_training_rows=32, random_state=7),
    )

    predictions = fit_result.predictions
    assert len(predictions) == len(transition_targets)
    assert "drift_hat_latent_logit_probability" in predictions.columns
    assert "diffusion_hat_latent_logit_probability" in predictions.columns
    assert "future_hat_latent_logit_probability" in predictions.columns
    assert "future_hat_latent_up_probability" in predictions.columns
    assert "mu_hat_latent_logit_probability" in predictions.columns
    assert "sigma_hat_latent_logit_probability" in predictions.columns
    assert "lambda_hat_latent_logit_probability" in predictions.columns
    assert "mu_hat_log_spot_ratio" in predictions.columns
    assert "sigma_hat_log_spot_ratio" in predictions.columns
    assert "lambda_hat_log_spot_ratio" in predictions.columns
    assert "jump_probability_hat_latent_logit_probability" in predictions.columns
    assert "future_hat_regime_normal_posterior" in predictions.columns
    assert "jump_intensity_hat" in predictions.columns
    assert predictions["jump_intensity_hat"].between(0.0, 1.0).all()
    assert predictions["jump_probability_hat_latent_logit_probability"].between(0.0, 1.0).all()
    assert (predictions["sigma_hat_latent_logit_probability"] > 0.0).all()
    assert (predictions["lambda_hat_latent_logit_probability"] >= 0.0).all()
    assert predictions["future_hat_latent_up_probability"].between(0.0, 1.0).all()
    assert fit_result.summary["training_rows"] == len(transition_targets)
    assert fit_result.summary["default_step_seconds"] == pytest.approx(15.0)
    assert "rollout_feature_columns" in fit_result.summary
    assert "current_latent_logit_probability" in fit_result.summary["rollout_feature_columns"]
    assert "current_market_implied_up_probability" not in fit_result.summary["rollout_feature_columns"]
    assert "parametric_latent_kernel" in fit_result.summary
    assert "parametric_spot_kernel" in fit_result.summary
    assert fit_result.summary["parametric_latent_kernel"]["target"] == "latent_logit_probability"
    assert fit_result.summary["parametric_spot_kernel"]["target"] == "log_spot_ratio"
    assert "mu_hat_latent_logit_probability" in fit_result.summary["parametric_latent_kernel"]["kernel_columns"]
    assert "mu_hat_log_spot_ratio" in fit_result.summary["parametric_spot_kernel"]["kernel_columns"]
    assert fit_result.summary["parametric_latent_kernel"]["step_seconds_median"] == pytest.approx(15.0)
    assert "latent_logit_probability" in fit_result.summary["target_metrics"]


def test_fit_transition_model_artifacts_writes_outputs(tmp_path) -> None:
    transition_targets = _synthetic_transition_targets()
    input_path = tmp_path / "crypto_5m_transition_targets_latest.parquet"
    transition_targets.to_parquet(input_path, index=False)

    result = fit_transition_model_artifacts(
        transition_target_glob=str(input_path),
        output_dir=str(tmp_path / "artifacts"),
        prediction_dir=str(tmp_path / "processed"),
        include_latest=True,
        min_training_rows=32,
        random_state=11,
    )

    assert (tmp_path / "artifacts" / "transition_model_latest.joblib").exists()
    assert (tmp_path / "artifacts" / "transition_model_summary_latest.json").exists()
    assert (tmp_path / "processed" / "crypto_5m_transition_predictions_latest.parquet").exists()
    assert result["training_rows"] == len(transition_targets)
