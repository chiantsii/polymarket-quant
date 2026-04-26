import asyncio

import numpy as np
import pandas as pd
import pytest

from polymarket_quant.pricing import (
    MarkovSimulationEngine,
    MarkovSimulationParams,
    SimulationMarketState,
)


class _StubTransitionBundle:
    rollout_feature_columns = (
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
    feature_columns = (
        "current_market_implied_up_probability",
        "current_fundamental_up_probability",
        "current_latent_up_probability",
        "current_latent_logit_probability",
        "current_up_micro_price",
        "current_down_micro_price",
        "current_up_bid_depth_top_5",
        "current_up_ask_depth_top_5",
        "current_down_bid_depth_top_5",
        "current_down_ask_depth_top_5",
        "current_up_weighted_imbalance",
        "current_down_weighted_imbalance",
        "current_cross_book_basis",
        "current_regime_normal_posterior",
        "current_regime_shock_posterior",
        "current_regime_convergence_posterior",
        "current_normalized_time_to_end",
        "target_horizon_seconds",
    )
    primitive_target_columns = (
        "latent_logit_probability",
        "regime_normal_posterior",
        "regime_shock_posterior",
        "regime_convergence_posterior",
        "market_implied_up_probability",
        "up_micro_price",
        "down_micro_price",
        "up_weighted_imbalance",
        "down_weighted_imbalance",
        "up_bid_depth_top_5",
        "up_ask_depth_top_5",
        "down_bid_depth_top_5",
        "down_ask_depth_top_5",
        "cross_book_basis",
    )

    def predict_latent_step(self, rows: pd.DataFrame) -> pd.DataFrame:
        predictions = rows.copy()
        current_latent = pd.to_numeric(predictions["current_latent_logit_probability"], errors="coerce").fillna(0.0)
        predictions["drift_hat_latent_logit_probability"] = 0.02
        predictions["diffusion_hat_latent_logit_probability"] = 1e-4
        predictions["future_hat_latent_logit_probability"] = current_latent + 0.02
        predictions["future_hat_latent_up_probability"] = 1.0 / (
            1.0 + np.exp(-predictions["future_hat_latent_logit_probability"])
        )
        predictions["jump_intensity_hat"] = 0.02
        return predictions


class _ParametricStubTransitionBundle:
    rollout_feature_columns = _StubTransitionBundle.rollout_feature_columns
    feature_columns = _StubTransitionBundle.feature_columns
    primitive_target_columns = _StubTransitionBundle.primitive_target_columns
    latent_mu_model = object()
    latent_sigma_model = object()

    def predict_latent_kernel(self, rows: pd.DataFrame) -> pd.DataFrame:
        predictions = rows.copy()
        predictions["mu_hat_latent_logit_probability"] = 0.02
        predictions["sigma_hat_latent_logit_probability"] = 0.0
        predictions["lambda_hat_latent_logit_probability"] = 0.0
        predictions["jump_mean_hat_latent_logit_probability"] = 0.0
        predictions["jump_std_hat_latent_logit_probability"] = 0.0
        predictions["jump_probability_hat_latent_logit_probability"] = 0.0
        return predictions

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        predictions = rows.copy()
        current_latent = pd.to_numeric(predictions["current_latent_logit_probability"], errors="coerce").fillna(0.0)
        current_market = pd.to_numeric(predictions["current_market_implied_up_probability"], errors="coerce").fillna(0.5)
        current_up_micro = pd.to_numeric(predictions["current_up_micro_price"], errors="coerce").fillna(current_market)
        current_down_micro = pd.to_numeric(predictions["current_down_micro_price"], errors="coerce").fillna(1.0 - current_market)
        current_up_bid_depth = pd.to_numeric(predictions["current_up_bid_depth_top_5"], errors="coerce").fillna(100.0)
        current_up_ask_depth = pd.to_numeric(predictions["current_up_ask_depth_top_5"], errors="coerce").fillna(100.0)
        current_down_bid_depth = pd.to_numeric(predictions["current_down_bid_depth_top_5"], errors="coerce").fillna(100.0)
        current_down_ask_depth = pd.to_numeric(predictions["current_down_ask_depth_top_5"], errors="coerce").fillna(100.0)
        current_up_imbalance = pd.to_numeric(predictions["current_up_weighted_imbalance"], errors="coerce").fillna(0.0)
        current_down_imbalance = pd.to_numeric(predictions["current_down_weighted_imbalance"], errors="coerce").fillna(0.0)

        predictions["future_hat_latent_logit_probability"] = current_latent + 0.02
        predictions["diffusion_hat_latent_logit_probability"] = 1e-4
        predictions["future_hat_market_implied_up_probability"] = np.clip(current_market + 0.01, 0.0, 1.0)
        predictions["diffusion_hat_market_implied_up_probability"] = 1e-5
        predictions["future_hat_up_micro_price"] = np.clip(current_up_micro + 0.01, 0.0, 1.0)
        predictions["future_hat_down_micro_price"] = np.clip(current_down_micro - 0.01, 0.0, 1.0)
        predictions["diffusion_hat_up_micro_price"] = 1e-5
        predictions["diffusion_hat_down_micro_price"] = 1e-5
        predictions["future_hat_up_weighted_imbalance"] = np.clip(current_up_imbalance * 0.5, -1.0, 1.0)
        predictions["future_hat_down_weighted_imbalance"] = np.clip(current_down_imbalance * 0.5, -1.0, 1.0)
        predictions["diffusion_hat_up_weighted_imbalance"] = 1e-4
        predictions["diffusion_hat_down_weighted_imbalance"] = 1e-4
        predictions["future_hat_up_bid_depth_top_5"] = current_up_bid_depth
        predictions["future_hat_up_ask_depth_top_5"] = current_up_ask_depth
        predictions["future_hat_down_bid_depth_top_5"] = current_down_bid_depth
        predictions["future_hat_down_ask_depth_top_5"] = current_down_ask_depth
        predictions["diffusion_hat_up_bid_depth_top_5"] = 1e-4
        predictions["diffusion_hat_up_ask_depth_top_5"] = 1e-4
        predictions["diffusion_hat_down_bid_depth_top_5"] = 1e-4
        predictions["diffusion_hat_down_ask_depth_top_5"] = 1e-4
        predictions["future_hat_cross_book_basis"] = predictions["future_hat_up_micro_price"] + predictions["future_hat_down_micro_price"] - 1.0
        predictions["diffusion_hat_cross_book_basis"] = 1e-5
        predictions["future_hat_regime_normal_posterior"] = 0.85
        predictions["future_hat_regime_shock_posterior"] = 0.10
        predictions["future_hat_regime_convergence_posterior"] = 0.05
        predictions["diffusion_hat_regime_normal_posterior"] = 1e-6
        predictions["diffusion_hat_regime_shock_posterior"] = 1e-6
        predictions["diffusion_hat_regime_convergence_posterior"] = 1e-6
        predictions["jump_intensity_hat"] = 0.02
        return predictions


class _StaticParametricStubTransitionBundle:
    rollout_feature_columns = _StubTransitionBundle.rollout_feature_columns
    feature_columns = _StubTransitionBundle.feature_columns
    primitive_target_columns = _StubTransitionBundle.primitive_target_columns
    latent_mu_model = object()
    latent_sigma_model = object()

    def predict_latent_kernel(self, rows: pd.DataFrame) -> pd.DataFrame:
        predictions = rows.copy()
        predictions["mu_hat_latent_logit_probability"] = 0.0
        predictions["sigma_hat_latent_logit_probability"] = 0.0
        predictions["lambda_hat_latent_logit_probability"] = 0.0
        predictions["jump_mean_hat_latent_logit_probability"] = 0.0
        predictions["jump_std_hat_latent_logit_probability"] = 0.0
        predictions["jump_probability_hat_latent_logit_probability"] = 0.0
        return predictions


def test_markov_simulation_engine_is_biased_by_imbalance() -> None:
    params = MarkovSimulationParams(
        drift=0.0,
        diffusion_vol=0.0,
        jump_intensity=0.0,
        n_paths=512,
        dt_seconds=1.0,
        imbalance_drift_scale=0.50,
    )
    engine = MarkovSimulationEngine(params=params)

    bullish = engine.simulate(
        initial_probability=0.50,
        horizon_seconds=10.0,
        market_state=SimulationMarketState(imbalance_signal=0.20),
        seed=42,
    )
    bearish = engine.simulate(
        initial_probability=0.50,
        horizon_seconds=10.0,
        market_state=SimulationMarketState(imbalance_signal=-0.20),
        seed=42,
    )

    assert bullish.expected_terminal_probability > 0.50
    assert bearish.expected_terminal_probability < 0.50
    assert bullish.expected_terminal_probability > bearish.expected_terminal_probability


def test_markov_simulation_engine_supports_asyncio_calls() -> None:
    engine = MarkovSimulationEngine(
        params=MarkovSimulationParams(
            drift=0.0,
            diffusion_vol=0.0,
            jump_intensity=0.0,
            n_paths=128,
            dt_seconds=1.0,
        )
    )

    result = asyncio.run(
        engine.simulate_async(
            initial_probability=0.60,
            horizon_seconds=10.0,
            market_state=SimulationMarketState(),
            seed=7,
        )
    )

    assert result.expected_terminal_probability == pytest.approx(0.60)
    assert result.terminal_probability_std == pytest.approx(0.0)
    up_distribution = result.aggregate()
    down_distribution = result.aggregate(invert_probability=True)
    assert up_distribution.expected_fair_price == pytest.approx(0.60)
    assert down_distribution.expected_fair_price == pytest.approx(0.40)


def test_markov_simulation_engine_supports_repeated_next_state_rollout() -> None:
    engine = MarkovSimulationEngine(
        params=MarkovSimulationParams(
            drift=0.0,
            diffusion_vol=0.0,
            jump_intensity=0.0,
            n_paths=128,
            dt_seconds=1.0,
            rollout_horizon_seconds=10.0,
        ),
        transition_bundle=_StubTransitionBundle(),
        event_duration_seconds=300.0,
    )

    initial_event_state = {
        "market_implied_up_probability": 0.52,
        "fundamental_up_probability": 0.50,
        "latent_up_probability": 0.52,
        "latent_logit_probability": float(np.log(0.52 / 0.48)),
        "up_micro_price": 0.53,
        "down_micro_price": 0.47,
        "up_bid_depth_top_5": 100.0,
        "up_ask_depth_top_5": 100.0,
        "down_bid_depth_top_5": 100.0,
        "down_ask_depth_top_5": 100.0,
        "up_weighted_imbalance": 0.10,
        "down_weighted_imbalance": -0.10,
        "cross_book_basis": 0.0,
        "regime_normal_posterior": 0.8,
        "regime_shock_posterior": 0.1,
        "regime_convergence_posterior": 0.1,
        "normalized_time_to_end": 0.8,
    }

    result = engine.simulate(
        initial_probability=0.52,
        horizon_seconds=30.0,
        market_state=SimulationMarketState(),
        initial_event_state=initial_event_state,
        seed=42,
    )

    assert result.diagnostics["simulation_mode"] == "repeated_next_state_rollout"
    assert result.diagnostics["n_steps"] == 3
    assert 0.50 < result.expected_terminal_probability < 0.60


def test_markov_simulation_engine_supports_parametric_kernel_rollout() -> None:
    engine = MarkovSimulationEngine(
        params=MarkovSimulationParams(
            drift=0.0,
            diffusion_vol=0.0,
            jump_intensity=0.0,
            n_paths=64,
            dt_seconds=1.0,
            rollout_horizon_seconds=5.0,
        ),
        transition_bundle=_ParametricStubTransitionBundle(),
        event_duration_seconds=300.0,
    )

    initial_event_state = {
        "latent_up_probability": 0.50,
        "latent_logit_probability": 0.0,
        "normalized_time_to_end": 0.5,
        "seconds_to_end": 15.0,
        "dist_to_boundary": 0.5,
        "boundary_leverage_ratio": 2.0,
        "up_book_velocity": 0.0,
        "down_book_velocity": 0.0,
    }

    result = engine.simulate(
        initial_probability=0.50,
        horizon_seconds=15.0,
        market_state=SimulationMarketState(),
        initial_event_state=initial_event_state,
        seed=7,
    )

    assert result.expected_terminal_probability > 0.50
    assert result.terminal_probability_std == pytest.approx(0.0)
    assert result.diagnostics["simulation_mode"] == "repeated_next_state_rollout"
    assert result.diagnostics["rollout_kernel"] == "parametric_latent_jump_diffusion"


def test_markov_simulation_engine_repeats_single_state_for_all_paths() -> None:
    engine = MarkovSimulationEngine(
        params=MarkovSimulationParams(
            drift=0.0,
            diffusion_vol=0.0,
            jump_intensity=0.0,
            n_paths=32,
            dt_seconds=1.0,
            rollout_horizon_seconds=5.0,
        ),
        transition_bundle=_StaticParametricStubTransitionBundle(),
        event_duration_seconds=300.0,
    )

    initial_probability = 4.005560367936432e-4
    initial_event_state = {
        "latent_up_probability": initial_probability,
        "latent_logit_probability": float(np.log(initial_probability / (1.0 - initial_probability))),
        "normalized_time_to_end": 0.01,
        "seconds_to_end": 0.9492,
        "dist_to_boundary": initial_probability,
        "boundary_leverage_ratio": 1.0 / initial_probability,
        "up_book_velocity": 0.01,
        "down_book_velocity": 0.01,
    }

    result = engine.simulate(
        initial_probability=initial_probability,
        horizon_seconds=0.9492,
        market_state=SimulationMarketState(),
        initial_event_state=initial_event_state,
        seed=7,
    )

    assert np.allclose(result.terminal_probabilities, initial_probability)
    assert result.expected_terminal_probability == pytest.approx(initial_probability)


def test_markov_simulation_batch_falls_back_to_each_rows_initial_probability() -> None:
    engine = MarkovSimulationEngine(
        params=MarkovSimulationParams(
            drift=0.0,
            diffusion_vol=0.0,
            jump_intensity=0.0,
            n_paths=8,
            dt_seconds=1.0,
            rollout_horizon_seconds=10.0,
        ),
        transition_bundle=_StaticParametricStubTransitionBundle(),
        event_duration_seconds=300.0,
    )

    def _return_invalid_latent_state(
        *,
        path_states: pd.DataFrame,
        predictions: pd.DataFrame,
        step_seconds: float | np.ndarray,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        invalid_states = path_states.copy()
        invalid_states["latent_up_probability"] = np.nan
        invalid_states["latent_logit_probability"] = np.nan
        return invalid_states

    engine._sample_next_rollout_latent_state = _return_invalid_latent_state  # type: ignore[method-assign]

    initial_event_states = [
        {
            "latent_up_probability": 0.10,
            "latent_logit_probability": float(np.log(0.10 / 0.90)),
            "normalized_time_to_end": 0.4,
            "seconds_to_end": 3.0,
            "dist_to_boundary": 0.10,
            "boundary_leverage_ratio": 10.0,
            "up_book_velocity": 0.0,
            "down_book_velocity": 0.0,
        },
        {
            "latent_up_probability": 0.90,
            "latent_logit_probability": float(np.log(0.90 / 0.10)),
            "normalized_time_to_end": 0.4,
            "seconds_to_end": 3.0,
            "dist_to_boundary": 0.10,
            "boundary_leverage_ratio": 10.0,
            "up_book_velocity": 0.0,
            "down_book_velocity": 0.0,
        },
    ]

    results = engine.simulate_event_state_batch(
        initial_event_states=initial_event_states,
        seed=7,
    )

    assert len(results) == 2
    assert np.allclose(results[0].terminal_probabilities, 0.10)
    assert np.allclose(results[1].terminal_probabilities, 0.90)
    assert results[0].expected_terminal_probability == pytest.approx(0.10)
    assert results[1].expected_terminal_probability == pytest.approx(0.90)
