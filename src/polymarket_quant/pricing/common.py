"""Shared utilities for the active binary-payoff pricing engine."""

import numpy as np


def bernoulli_standard_error(indicators: np.ndarray) -> float:
    """Estimate standard error for a Bernoulli Monte Carlo estimator."""
    if indicators.size <= 1:
        return 0.0
    return float(np.std(indicators, ddof=1) / np.sqrt(indicators.size))
