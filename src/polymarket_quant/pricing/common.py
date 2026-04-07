"""Shared utilities for pricing and probability-estimation models."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class PricingResult:
    """Standard output for binary-event pricing estimators."""

    probability: float
    standard_error: float
    n_samples: int
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def fair_price(self) -> float:
        """Binary contract fair price before fees and execution costs."""
        return self.probability


def make_rng(seed: Optional[int] = None) -> np.random.Generator:
    """Create a NumPy random number generator."""
    return np.random.default_rng(seed)


def validate_binary_pricing_inputs(
    initial_value: float,
    threshold: float,
    volatility: float,
    horizon: float,
    n_samples: int,
) -> None:
    """Validate common GBM binary-pricing inputs."""
    if initial_value <= 0:
        raise ValueError("initial_value must be positive")
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    if volatility < 0:
        raise ValueError("volatility must be non-negative")
    if horizon < 0:
        raise ValueError("horizon must be non-negative")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")


def simulate_gbm_terminal_values(
    initial_value: float,
    drift: float,
    volatility: float,
    horizon: float,
    shocks: np.ndarray,
) -> np.ndarray:
    """Simulate terminal values under a geometric Brownian motion model."""
    if horizon == 0 or volatility == 0:
        return np.full_like(shocks, initial_value * np.exp(drift * horizon), dtype=float)

    diffusion = volatility * np.sqrt(horizon) * shocks
    log_return = (drift - 0.5 * volatility**2) * horizon + diffusion
    return initial_value * np.exp(log_return)


def bernoulli_standard_error(indicators: np.ndarray) -> float:
    """Estimate standard error for a Bernoulli Monte Carlo estimator."""
    if indicators.size <= 1:
        return 0.0
    return float(np.std(indicators, ddof=1) / np.sqrt(indicators.size))
