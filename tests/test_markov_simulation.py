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
    spot_vol_multiplier: float = 1.0,
) -> SimulationMarketState:
    return SimulationMarketState(
        spot_price=spot_price,
        reference_spot_price=reference_spot_price,
        spot_volatility_per_sqrt_second=spot_volatility_per_sqrt_second,
        spot_vol_multiplier=spot_vol_multiplier,
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


def test_markov_simulation_uses_event_state_volatility_directly() -> None:
    engine = MarkovSimulationEngine(
        params=MarkovSimulationParams(
            spot_log_drift_per_second=0.0,
            base_spot_volatility_per_sqrt_second=0.0005,
            spot_jump_intensity_per_second=0.0,
            spot_jump_log_return_std=0.0,
            n_paths=128,
            simulation_dt_seconds=1.0,
            liquidity_volatility_scale=10.0,
            velocity_volatility_scale=10.0,
        )
    )

    result = engine.simulate(
        horizon_seconds=30.0,
        market_state=SimulationMarketState(
            spot_price=100.0,
            reference_spot_price=100.0,
            spot_volatility_per_sqrt_second=0.123,
            liquidity_depth=0.01,
            book_velocity=99.0,
            spot_vol_multiplier=50.0,
            learned_spot_volatility_per_sqrt_second=0.456,
        ),
        initial_event_state={
            "spot_price": 100.0,
            "reference_spot_price": 100.0,
            "volatility_per_sqrt_second": 0.0025,
            "spot_vol_multiplier": 99.0,
            "up_bid_depth_top_5": 1.0,
            "up_ask_depth_top_5": 1.0,
            "down_bid_depth_top_5": 1.0,
            "down_ask_depth_top_5": 1.0,
            "up_book_velocity": 999.0,
            "down_book_velocity": 999.0,
        },
        seed=11,
    )

    assert result.diagnostics["conditioned_spot_volatility_per_sqrt_second"] == pytest.approx(0.0025)


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


def test_markov_simulation_can_force_manual_jump_parameters() -> None:
    engine = MarkovSimulationEngine(
        params=MarkovSimulationParams(
            spot_log_drift_per_second=0.0,
            base_spot_volatility_per_sqrt_second=0.0,
            spot_jump_intensity_per_second=0.123,
            spot_jump_log_return_mean=0.0,
            spot_jump_log_return_std=0.004,
            spot_jump_std_multiplier_on_local_sigma=0.0,
            force_manual_jump_parameters=True,
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
            learned_spot_jump_intensity_per_second=0.0,
            learned_spot_jump_log_return_mean=0.0,
            learned_spot_jump_log_return_std=0.0,
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
            force_manual_jump_parameters=True,
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
