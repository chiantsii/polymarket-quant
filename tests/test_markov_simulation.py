import asyncio

import numpy as np
import pytest

from polymarket_quant.pricing import (
    MarkovSimulationEngine,
    MarkovSimulationParams,
    SimulationMarketState,
)


def _state(
    *,
    spot_price: float,
    reference_spot_price: float,
    spot_volatility_per_sqrt_second: float,
) -> SimulationMarketState:
    return SimulationMarketState(
        spot_price=spot_price,
        reference_spot_price=reference_spot_price,
        spot_volatility_per_sqrt_second=spot_volatility_per_sqrt_second,
    )


def test_markov_simulation_prices_deterministic_up_payoff() -> None:
    engine = MarkovSimulationEngine(
        params=MarkovSimulationParams(
            spot_log_drift_per_second=0.0,
            base_spot_volatility_per_sqrt_second=0.0005,
            spot_jump_intensity_per_second=0.0,
            spot_jump_log_return_std=0.0,
            n_paths=128,
            simulation_dt_seconds=1.0,
        )
    )

    result = engine.simulate(
        horizon_seconds=30.0,
        market_state=_state(
            spot_price=101.0,
            reference_spot_price=100.0,
            spot_volatility_per_sqrt_second=0.0,
        ),
        seed=7,
    )

    assert np.allclose(result.terminal_spot_values, 101.0)
    assert np.allclose(result.terminal_payoffs, 1.0)
    assert result.expected_terminal_probability == pytest.approx(1.0)
    assert result.terminal_probability_std == pytest.approx(0.0)
    up_distribution = result.aggregate()
    down_distribution = result.aggregate(invert_probability=True)
    assert up_distribution.expected_fair_price == pytest.approx(1.0)
    assert down_distribution.expected_fair_price == pytest.approx(0.0)


def test_markov_simulation_prices_deterministic_down_payoff() -> None:
    engine = MarkovSimulationEngine(
        params=MarkovSimulationParams(
            spot_log_drift_per_second=0.0,
            base_spot_volatility_per_sqrt_second=0.0005,
            spot_jump_intensity_per_second=0.0,
            spot_jump_log_return_std=0.0,
            n_paths=128,
            simulation_dt_seconds=1.0,
        )
    )

    result = engine.simulate(
        horizon_seconds=30.0,
        market_state=_state(
            spot_price=99.0,
            reference_spot_price=100.0,
            spot_volatility_per_sqrt_second=0.0,
        ),
        seed=7,
    )

    assert np.allclose(result.terminal_payoffs, 0.0)
    assert result.expected_terminal_probability == pytest.approx(0.0)
    assert result.terminal_probability_std == pytest.approx(0.0)


def test_markov_simulation_treats_reference_tie_as_up_payoff() -> None:
    engine = MarkovSimulationEngine(
        params=MarkovSimulationParams(
            spot_log_drift_per_second=0.0,
            base_spot_volatility_per_sqrt_second=0.0,
            spot_jump_intensity_per_second=0.0,
            spot_jump_log_return_std=0.0,
            n_paths=32,
            simulation_dt_seconds=1.0,
        )
    )

    result = engine.simulate(
        horizon_seconds=30.0,
        market_state=_state(
            spot_price=100.0,
            reference_spot_price=100.0,
            spot_volatility_per_sqrt_second=0.0,
        ),
        seed=11,
    )

    assert np.allclose(result.terminal_payoffs, 1.0)
    assert result.expected_terminal_probability == pytest.approx(1.0)


def test_markov_simulation_probability_increases_with_more_favorable_spot() -> None:
    engine = MarkovSimulationEngine(
        params=MarkovSimulationParams(
            spot_log_drift_per_second=0.0,
            base_spot_volatility_per_sqrt_second=0.0005,
            spot_jump_intensity_per_second=0.0,
            spot_jump_log_return_std=0.0,
            n_paths=4_000,
            simulation_dt_seconds=1.0,
        )
    )

    bullish = engine.simulate(
        horizon_seconds=120.0,
        market_state=_state(
            spot_price=101.0,
            reference_spot_price=100.0,
            spot_volatility_per_sqrt_second=0.0005,
        ),
        seed=42,
    )
    bearish = engine.simulate(
        horizon_seconds=120.0,
        market_state=_state(
            spot_price=99.0,
            reference_spot_price=100.0,
            spot_volatility_per_sqrt_second=0.0005,
        ),
        seed=42,
    )

    assert bullish.expected_terminal_probability > 0.5
    assert bearish.expected_terminal_probability < 0.5
    assert bullish.expected_terminal_probability > bearish.expected_terminal_probability


def test_markov_simulation_engine_supports_asyncio_calls() -> None:
    engine = MarkovSimulationEngine(
        params=MarkovSimulationParams(
            spot_log_drift_per_second=0.0,
            base_spot_volatility_per_sqrt_second=0.0005,
            spot_jump_intensity_per_second=0.0,
            spot_jump_log_return_std=0.0,
            n_paths=256,
            simulation_dt_seconds=1.0,
        )
    )

    result = asyncio.run(
        engine.simulate_async(
            horizon_seconds=60.0,
            market_state=_state(
                spot_price=101.0,
                reference_spot_price=100.0,
                spot_volatility_per_sqrt_second=0.0005,
            ),
            seed=7,
        )
    )

    assert 0.0 <= result.expected_terminal_probability <= 1.0
    assert result.diagnostics["simulation_mode"] == "spot_terminal_binary_payoff_rollout"


def test_markov_simulation_batch_prices_each_rows_own_threshold() -> None:
    engine = MarkovSimulationEngine(
        params=MarkovSimulationParams(
            spot_log_drift_per_second=0.0,
            base_spot_volatility_per_sqrt_second=0.0005,
            spot_jump_intensity_per_second=0.0,
            spot_jump_log_return_std=0.0,
            n_paths=2_000,
            simulation_dt_seconds=1.0,
        )
    )

    results = engine.simulate_event_state_batch(
        initial_event_states=[
            {
                "spot_price": 101.0,
                "reference_spot_price": 100.0,
                "volatility_per_sqrt_second": 0.0005,
                "spot_vol_multiplier": 1.0,
                "seconds_to_end": 120.0,
            },
            {
                "spot_price": 99.0,
                "reference_spot_price": 100.0,
                "volatility_per_sqrt_second": 0.0005,
                "spot_vol_multiplier": 1.0,
                "seconds_to_end": 120.0,
            },
        ],
        seed=7,
    )

    assert len(results) == 2
    assert results[0].expected_terminal_probability > results[1].expected_terminal_probability


def test_markov_simulation_prefers_learned_sigma_over_event_state_volatility() -> None:
    engine = MarkovSimulationEngine(
        params=MarkovSimulationParams(
            spot_log_drift_per_second=0.0,
            base_spot_volatility_per_sqrt_second=0.0005,
            spot_jump_intensity_per_second=0.0,
            spot_jump_log_return_std=0.0,
            n_paths=128,
            simulation_dt_seconds=1.0,
        )
    )

    result = engine.simulate(
        horizon_seconds=30.0,
        market_state=SimulationMarketState(
            spot_price=100.0,
            reference_spot_price=100.0,
            spot_volatility_per_sqrt_second=0.123,
            learned_spot_volatility_per_sqrt_second=0.456,
        ),
        initial_event_state={
            "spot_price": 100.0,
            "reference_spot_price": 100.0,
            "volatility_per_sqrt_second": 0.0025,
        },
        seed=11,
    )

    assert result.diagnostics["conditioned_spot_volatility_per_sqrt_second"] == pytest.approx(0.456)


def test_markov_simulation_prefers_spot_kernel_sigma_over_market_state_sigma() -> None:
    engine = MarkovSimulationEngine(
        params=MarkovSimulationParams(
            spot_log_drift_per_second=0.0,
            base_spot_volatility_per_sqrt_second=0.0005,
            spot_jump_intensity_per_second=0.0,
            spot_jump_log_return_std=0.0,
            n_paths=128,
            simulation_dt_seconds=1.0,
        )
    )

    result = engine._conditioned_spot_diffusion(
        market_state=SimulationMarketState(
            spot_price=100.0,
            reference_spot_price=100.0,
            spot_volatility_per_sqrt_second=0.123,
            learned_spot_volatility_per_sqrt_second=0.456,
        ),
        base_volatility=0.0025,
        spot_kernel={"sigma_hat_log_spot_ratio": 0.0789},
    )

    assert result == pytest.approx(0.0789)


def test_markov_simulation_uses_learned_drift_directly() -> None:
    engine = MarkovSimulationEngine(params=MarkovSimulationParams())

    result = engine._conditioned_spot_drift(
        market_state=SimulationMarketState(
            spot_price=100.0,
            reference_spot_price=100.0,
            spot_volatility_per_sqrt_second=0.0005,
            learned_spot_log_drift_per_second=2.0e-4,
        ),
        spot_kernel=None,
    )

    assert result == pytest.approx(2.0e-4)


def test_markov_simulation_applies_volatility_floor_to_tiny_positive_sigma() -> None:
    engine = MarkovSimulationEngine(
        params=MarkovSimulationParams(
            base_spot_volatility_per_sqrt_second=0.0005,
            min_effective_spot_volatility_per_sqrt_second=2.0e-5,
        )
    )

    result = engine._conditioned_spot_diffusion(
        market_state=SimulationMarketState(
            spot_price=100.0,
            reference_spot_price=100.0,
            spot_volatility_per_sqrt_second=0.123,
            learned_spot_volatility_per_sqrt_second=2.0e-6,
        ),
        base_volatility=0.0025,
        spot_kernel=None,
    )

    assert result == pytest.approx(2.0e-5)


def test_markov_simulation_keeps_zero_sigma_deterministic() -> None:
    engine = MarkovSimulationEngine(
        params=MarkovSimulationParams(
            base_spot_volatility_per_sqrt_second=0.0005,
            min_effective_spot_volatility_per_sqrt_second=2.0e-5,
        )
    )

    result = engine._conditioned_spot_diffusion(
        market_state=SimulationMarketState(
            spot_price=100.0,
            reference_spot_price=100.0,
            spot_volatility_per_sqrt_second=0.0,
            learned_spot_volatility_per_sqrt_second=0.0,
        ),
        base_volatility=0.0,
        spot_kernel=None,
    )

    assert result == pytest.approx(0.0)


def test_markov_simulation_applies_drift_decay_to_effective_rollout_drift() -> None:
    no_decay_engine = MarkovSimulationEngine(
        params=MarkovSimulationParams(
            spot_log_drift_per_second=1e-4,
            spot_drift_decay_kappa_per_second=0.0,
            base_spot_volatility_per_sqrt_second=0.0,
            spot_jump_intensity_per_second=0.0,
            spot_jump_log_return_std=0.0,
            n_paths=16,
            simulation_dt_seconds=1.0,
        )
    )
    decay_engine = MarkovSimulationEngine(
        params=MarkovSimulationParams(
            spot_log_drift_per_second=1e-4,
            spot_drift_decay_kappa_per_second=np.log(2.0) / 60.0,
            base_spot_volatility_per_sqrt_second=0.0,
            spot_jump_intensity_per_second=0.0,
            spot_jump_log_return_std=0.0,
            n_paths=16,
            simulation_dt_seconds=1.0,
        )
    )

    no_decay = no_decay_engine.simulate(
        horizon_seconds=300.0,
        market_state=_state(
            spot_price=100.0,
            reference_spot_price=100.0,
            spot_volatility_per_sqrt_second=0.0,
        ),
        seed=1,
    )
    decay = decay_engine.simulate(
        horizon_seconds=300.0,
        market_state=_state(
            spot_price=100.0,
            reference_spot_price=100.0,
            spot_volatility_per_sqrt_second=0.0,
        ),
        seed=1,
    )

    assert decay.diagnostics["effective_accumulated_spot_drift_log_return"] < no_decay.diagnostics["effective_accumulated_spot_drift_log_return"]
    assert float(np.mean(decay.terminal_spot_values)) < float(np.mean(no_decay.terminal_spot_values))


def test_markov_simulation_reports_effective_accumulated_drift_without_cap() -> None:
    engine = MarkovSimulationEngine(
        params=MarkovSimulationParams(
            spot_log_drift_per_second=1.0e-4,
            spot_drift_decay_kappa_per_second=0.0,
            base_spot_volatility_per_sqrt_second=0.0,
            spot_jump_intensity_per_second=0.0,
            spot_jump_log_return_std=0.0,
            n_paths=8,
            simulation_dt_seconds=1.0,
        )
    )

    result = engine.simulate(
        horizon_seconds=300.0,
        market_state=_state(
            spot_price=100.0,
            reference_spot_price=100.0,
            spot_volatility_per_sqrt_second=0.0,
        ),
        seed=1,
    )

    assert result.diagnostics["effective_accumulated_spot_drift_log_return"] == pytest.approx(0.03)


def test_markov_simulation_uses_manual_jump_parameters_by_default() -> None:
    engine = MarkovSimulationEngine(
        params=MarkovSimulationParams(
            spot_log_drift_per_second=0.0,
            base_spot_volatility_per_sqrt_second=0.0,
            spot_jump_intensity_per_second=0.123,
            spot_jump_log_return_mean=0.0,
            spot_jump_log_return_std=0.004,
            spot_jump_std_multiplier_on_local_sigma=0.0,
            n_paths=8,
            simulation_dt_seconds=1.0,
        )
    )

    result = engine.simulate(
        horizon_seconds=30.0,
        market_state=SimulationMarketState(
            spot_price=100.0,
            reference_spot_price=100.0,
            spot_volatility_per_sqrt_second=0.0,
        ),
        seed=3,
    )

    assert result.diagnostics["conditioned_spot_jump_intensity_per_second"] == pytest.approx(0.123)
    assert result.diagnostics["conditioned_spot_jump_log_return_std"] == pytest.approx(0.004)


def test_markov_simulation_can_scale_manual_jump_std_from_local_sigma() -> None:
    engine = MarkovSimulationEngine(
        params=MarkovSimulationParams(
            spot_log_drift_per_second=0.0,
            base_spot_volatility_per_sqrt_second=0.0,
            spot_jump_intensity_per_second=0.123,
            spot_jump_log_return_mean=0.0,
            spot_jump_log_return_std=0.999,
            spot_jump_std_multiplier_on_local_sigma=20.0,
            n_paths=8,
            simulation_dt_seconds=1.0,
        )
    )

    result = engine.simulate(
        horizon_seconds=30.0,
        market_state=SimulationMarketState(
            spot_price=100.0,
            reference_spot_price=100.0,
            spot_volatility_per_sqrt_second=0.0005,
        ),
        seed=3,
    )

    assert result.diagnostics["conditioned_spot_jump_intensity_per_second"] == pytest.approx(0.123)
    assert result.diagnostics["conditioned_spot_jump_log_return_std"] == pytest.approx(0.01)


def test_markov_simulation_scales_jump_std_from_effective_sigma_floor() -> None:
    engine = MarkovSimulationEngine(
        params=MarkovSimulationParams(
            spot_log_drift_per_second=0.0,
            base_spot_volatility_per_sqrt_second=0.0,
            min_effective_spot_volatility_per_sqrt_second=2.0e-5,
            spot_jump_intensity_per_second=0.123,
            spot_jump_log_return_mean=0.0,
            spot_jump_log_return_std=0.999,
            spot_jump_std_multiplier_on_local_sigma=20.0,
            n_paths=8,
            simulation_dt_seconds=1.0,
        )
    )

    result = engine.simulate(
        horizon_seconds=30.0,
        market_state=SimulationMarketState(
            spot_price=100.0,
            reference_spot_price=100.0,
            spot_volatility_per_sqrt_second=2.0e-6,
            learned_spot_volatility_per_sqrt_second=2.0e-6,
        ),
        seed=3,
    )

    assert result.diagnostics["conditioned_spot_volatility_per_sqrt_second"] == pytest.approx(2.0e-5)
    assert result.diagnostics["conditioned_spot_jump_log_return_std"] == pytest.approx(4.0e-4)


def test_markov_simulation_reports_priced_horizon_and_step_semantics() -> None:
    engine = MarkovSimulationEngine(
        params=MarkovSimulationParams(
            spot_log_drift_per_second=0.0,
            base_spot_volatility_per_sqrt_second=0.0005,
            spot_jump_intensity_per_second=0.0,
            spot_jump_log_return_std=0.0,
            n_paths=16,
            simulation_dt_seconds=1.0,
            rollout_horizon_seconds=15.0,
        )
    )

    result = engine.simulate(
        horizon_seconds=120.0,
        market_state=_state(
            spot_price=101.0,
            reference_spot_price=100.0,
            spot_volatility_per_sqrt_second=0.0005,
        ),
        seed=5,
    )

    assert result.diagnostics["rollout_horizon_seconds"] == pytest.approx(120.0)
    assert result.diagnostics["simulation_step_seconds"] == pytest.approx(15.0)
    assert result.diagnostics["dt_seconds"] == pytest.approx(15.0)
