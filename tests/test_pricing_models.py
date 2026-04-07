import numpy as np
import pytest

from polymarket_quant.evaluation.metrics import calculate_brier_score
from polymarket_quant.pricing import (
    BinaryMarketABM,
    ParticleFilter,
    estimate_importance_sampled_probability,
    estimate_monte_carlo_probability,
    estimate_stratified_probability,
)


def test_monte_carlo_probability_is_bounded_and_seeded() -> None:
    result_a = estimate_monte_carlo_probability(
        initial_value=100.0,
        threshold=101.0,
        volatility=0.02,
        n_samples=2_000,
        seed=42,
    )
    result_b = estimate_monte_carlo_probability(
        initial_value=100.0,
        threshold=101.0,
        volatility=0.02,
        n_samples=2_000,
        seed=42,
    )

    assert 0.0 <= result_a.probability <= 1.0
    assert result_a.fair_price == result_a.probability
    assert result_a.probability == result_b.probability


def test_importance_sampling_returns_valid_diagnostics() -> None:
    result = estimate_importance_sampled_probability(
        initial_value=100.0,
        threshold=104.0,
        volatility=0.02,
        n_samples=2_000,
        proposal_shift=1.0,
        seed=42,
    )

    assert 0.0 <= result.probability <= 1.0
    assert result.standard_error >= 0.0
    assert result.diagnostics["effective_sample_size"] > 0.0


def test_stratified_monte_carlo_probability_is_bounded() -> None:
    result = estimate_stratified_probability(
        initial_value=100.0,
        threshold=100.0,
        volatility=0.02,
        n_strata=5,
        samples_per_stratum=200,
        seed=42,
    )

    assert 0.0 <= result.probability <= 1.0
    assert len(result.diagnostics["stratum_estimates"]) == 5


def test_particle_filter_outputs_probability_path() -> None:
    observations = np.array([0.50, 0.52, 0.55, 0.53, 0.56])
    particle_filter = ParticleFilter(n_particles=500, seed=42)

    result = particle_filter.filter(observations)

    assert len(result.probabilities) == len(observations)
    assert np.all((result.probabilities >= 0.0) & (result.probabilities <= 1.0))
    assert np.isclose(np.sum(result.final_weights), 1.0)
    assert np.all(result.effective_sample_sizes > 0.0)


def test_abm_generates_bounded_price_path() -> None:
    model = BinaryMarketABM(n_agents=50, seed=42)

    result = model.simulate(n_steps=25, initial_price=0.5, fundamental_probability=0.55)

    assert len(result.prices) == 25
    assert len(result.net_order_flow) == 25
    assert np.all((result.prices > 0.0) & (result.prices < 1.0))


def test_brier_score_matches_manual_calculation() -> None:
    probs = np.array([0.2, 0.8])
    outcomes = np.array([0, 1])

    score = calculate_brier_score(probs, outcomes)

    assert score == pytest.approx(((0.2 - 0.0) ** 2 + (0.8 - 1.0) ** 2) / 2.0)
