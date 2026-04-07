"""Monte Carlo pricing for binary event contracts."""

import numpy as np

from polymarket_quant.pricing.common import (
    PricingResult,
    bernoulli_standard_error,
    make_rng,
    simulate_gbm_terminal_values,
    validate_binary_pricing_inputs,
)


def estimate_monte_carlo_probability(
    initial_value: float,
    threshold: float,
    drift: float = 0.0,
    volatility: float = 0.01,
    horizon: float = 1.0,
    n_samples: int = 10_000,
    seed: int | None = None,
) -> PricingResult:
    """Estimate ``P(S_T >= threshold)`` using GBM Monte Carlo simulation.

    For BTC/ETH 5-minute Up/Down markets, ``threshold`` is typically the
    Chainlink start price and ``initial_value`` is the latest observed price.
    """
    validate_binary_pricing_inputs(
        initial_value=initial_value,
        threshold=threshold,
        volatility=volatility,
        horizon=horizon,
        n_samples=n_samples,
    )

    rng = make_rng(seed)
    shocks = rng.standard_normal(n_samples)
    terminal_values = simulate_gbm_terminal_values(
        initial_value=initial_value,
        drift=drift,
        volatility=volatility,
        horizon=horizon,
        shocks=shocks,
    )
    indicators = terminal_values >= threshold
    probability = float(np.mean(indicators))

    return PricingResult(
        probability=probability,
        standard_error=bernoulli_standard_error(indicators.astype(float)),
        n_samples=n_samples,
        diagnostics={
            "terminal_mean": float(np.mean(terminal_values)),
            "terminal_std": float(np.std(terminal_values)),
        },
    )
