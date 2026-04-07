"""Importance sampling for rare binary event pricing."""

import numpy as np

from polymarket_quant.pricing.common import (
    PricingResult,
    make_rng,
    simulate_gbm_terminal_values,
    validate_binary_pricing_inputs,
)


def estimate_importance_sampled_probability(
    initial_value: float,
    threshold: float,
    drift: float = 0.0,
    volatility: float = 0.01,
    horizon: float = 1.0,
    n_samples: int = 10_000,
    proposal_shift: float = 1.0,
    seed: int | None = None,
) -> PricingResult:
    """Estimate ``P(S_T >= threshold)`` with a shifted-normal proposal.

    ``proposal_shift`` moves the simulation distribution toward the event of
    interest. Positive values emphasize upside events; negative values
    emphasize downside events.
    """
    validate_binary_pricing_inputs(
        initial_value=initial_value,
        threshold=threshold,
        volatility=volatility,
        horizon=horizon,
        n_samples=n_samples,
    )

    rng = make_rng(seed)
    proposal_shocks = rng.normal(loc=proposal_shift, scale=1.0, size=n_samples)
    terminal_values = simulate_gbm_terminal_values(
        initial_value=initial_value,
        drift=drift,
        volatility=volatility,
        horizon=horizon,
        shocks=proposal_shocks,
    )

    # Likelihood ratio: phi(z) / phi(z - proposal_shift).
    log_weights = -0.5 * proposal_shocks**2 + 0.5 * (proposal_shocks - proposal_shift) ** 2
    weights = np.exp(log_weights)
    weighted_indicators = weights * (terminal_values >= threshold)
    probability = float(np.mean(weighted_indicators))
    standard_error = float(np.std(weighted_indicators, ddof=1) / np.sqrt(n_samples))
    effective_sample_size = float(np.sum(weights) ** 2 / np.sum(weights**2))

    return PricingResult(
        probability=float(np.clip(probability, 0.0, 1.0)),
        standard_error=standard_error,
        n_samples=n_samples,
        diagnostics={
            "effective_sample_size": effective_sample_size,
            "mean_weight": float(np.mean(weights)),
            "proposal_shift": proposal_shift,
        },
    )
