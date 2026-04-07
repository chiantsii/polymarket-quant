"""Stratified Monte Carlo pricing for binary event contracts."""

import numpy as np
from scipy.stats import norm

from polymarket_quant.pricing.common import (
    PricingResult,
    make_rng,
    simulate_gbm_terminal_values,
    validate_binary_pricing_inputs,
)


def estimate_stratified_probability(
    initial_value: float,
    threshold: float,
    drift: float = 0.0,
    volatility: float = 0.01,
    horizon: float = 1.0,
    n_strata: int = 10,
    samples_per_stratum: int = 1_000,
    seed: int | None = None,
) -> PricingResult:
    """Estimate ``P(S_T >= threshold)`` with equal-probability strata."""
    n_samples = n_strata * samples_per_stratum
    validate_binary_pricing_inputs(
        initial_value=initial_value,
        threshold=threshold,
        volatility=volatility,
        horizon=horizon,
        n_samples=n_samples,
    )
    if n_strata <= 0:
        raise ValueError("n_strata must be positive")
    if samples_per_stratum <= 0:
        raise ValueError("samples_per_stratum must be positive")

    rng = make_rng(seed)
    stratum_estimates = []
    for stratum_idx in range(n_strata):
        lower = stratum_idx / n_strata
        upper = (stratum_idx + 1) / n_strata
        uniforms = rng.uniform(lower, upper, size=samples_per_stratum)
        shocks = norm.ppf(uniforms)
        terminal_values = simulate_gbm_terminal_values(
            initial_value=initial_value,
            drift=drift,
            volatility=volatility,
            horizon=horizon,
            shocks=shocks,
        )
        stratum_estimates.append(float(np.mean(terminal_values >= threshold)))

    probability = float(np.mean(stratum_estimates))
    standard_error = float(np.std(stratum_estimates, ddof=1) / np.sqrt(n_strata))

    return PricingResult(
        probability=probability,
        standard_error=standard_error,
        n_samples=n_samples,
        diagnostics={"stratum_estimates": stratum_estimates},
    )
