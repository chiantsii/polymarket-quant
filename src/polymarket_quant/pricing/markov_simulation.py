"""Monte Carlo pricing for short-horizon binary prediction markets.

This pricing layer values the *binary contract payoff* directly:

    payoff = 1{spot_T >= reference_spot_price}

The fair token price is therefore the Monte Carlo estimate of
P(spot_T >= reference_spot_price | current state).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np

from polymarket_quant.pricing.common import bernoulli_standard_error


PricingModel = Callable[[np.ndarray], np.ndarray | float]
ProgressCallback = Callable[[int], None]


@dataclass(frozen=True)
class SimulationMarketState:
    """Current observable inputs used to price the binary payoff."""

    spot_price: float | None = None
    reference_spot_price: float | None = None
    spot_volatility_per_sqrt_second: float = 0.0005
    liquidity_depth: float = 0.0
    book_velocity: float = 0.0
    spot_vol_multiplier: float = 1.0


@dataclass(frozen=True)
class MarkovSimulationParams:
    """Parameters for terminal-spot Monte Carlo pricing."""

    spot_log_drift_per_second: float = 0.0
    base_spot_volatility_per_sqrt_second: float = 0.0005
    spot_jump_intensity_per_second: float = 0.0
    spot_jump_log_return_mean: float = 0.0
    spot_jump_log_return_std: float = 0.0
    simulation_dt_seconds: float = 1.0
    n_paths: int = 1_000
    liquidity_volatility_scale: float = 0.30
    velocity_volatility_scale: float = 0.50
    rollout_horizon_seconds: float = 0.0


@dataclass(frozen=True)
class PriceDistributionSummary:
    """Aggregated binary-token price summary."""

    expected_fair_price: float
    risk_score: float
    n_paths: int
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MarkovSimulationResult:
    """Terminal-spot and binary-payoff distribution from the pricing engine."""

    terminal_spot_values: np.ndarray
    terminal_payoffs: np.ndarray
    expected_terminal_probability: float
    terminal_probability_std: float
    n_paths: int
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def aggregate(
        self,
        pricing_model: Optional[PricingModel] = None,
        *,
        invert_probability: bool = False,
    ) -> PriceDistributionSummary:
        payoffs = 1.0 - self.terminal_payoffs if invert_probability else self.terminal_payoffs
        priced = _apply_pricing_model(pricing_model, payoffs)
        return PriceDistributionSummary(
            expected_fair_price=float(np.mean(priced)),
            risk_score=float(bernoulli_standard_error(payoffs.astype(float))),
            n_paths=self.n_paths,
            diagnostics={
                "pricing_payoff_mean": float(np.mean(payoffs)),
                "pricing_payoff_std": float(np.std(payoffs)),
            },
        )


class MarkovSimulationEngine:
    """Simulate terminal spot paths and value the binary payoff directly."""

    def __init__(
        self,
        params: Optional[MarkovSimulationParams] = None,
        pricing_model: Optional[PricingModel] = None,
    ) -> None:
        self.params = params or MarkovSimulationParams()
        self.pricing_model = pricing_model

    def simulate(
        self,
        *,
        horizon_seconds: float,
        market_state: SimulationMarketState,
        initial_event_state: Optional[Dict[str, Any]] = None,
        seed: int | None = None,
    ) -> MarkovSimulationResult:
        if horizon_seconds < 0:
            raise ValueError("horizon_seconds must be non-negative")
        if self.params.simulation_dt_seconds <= 0:
            raise ValueError("simulation_dt_seconds must be positive")
        if self.params.n_paths <= 0:
            raise ValueError("n_paths must be positive")

        pricing_inputs = self._resolve_pricing_inputs(
            market_state=market_state,
            initial_event_state=initial_event_state,
        )
        initial_spot = pricing_inputs["spot_price"]
        reference_price = pricing_inputs["reference_spot_price"]
        base_volatility = pricing_inputs["volatility_per_sqrt_second"]

        step_seconds = (
            float(self.params.rollout_horizon_seconds)
            if float(self.params.rollout_horizon_seconds) > 0.0
            else float(self.params.simulation_dt_seconds)
        )
        step_seconds = max(step_seconds, 1e-9)

        if horizon_seconds == 0.0:
            terminal_values = np.full(self.params.n_paths, initial_spot, dtype=float)
            terminal_payoffs = (terminal_values >= reference_price).astype(float)
            return MarkovSimulationResult(
                terminal_spot_values=terminal_values,
                terminal_payoffs=terminal_payoffs,
                expected_terminal_probability=float(np.mean(terminal_payoffs)),
                terminal_probability_std=float(np.std(terminal_payoffs)),
                n_paths=self.params.n_paths,
                diagnostics={
                    "n_steps": 0,
                    "dt_seconds": 0.0,
                    "simulation_mode": "spot_terminal_binary_payoff_rollout",
                    "rollout_kernel": "spot_jump_diffusion",
                    "initial_spot_price": initial_spot,
                    "reference_spot_price": reference_price,
                },
            )

        n_steps = max(1, int(np.ceil(horizon_seconds / step_seconds)))
        dt = float(horizon_seconds / n_steps)
        rng = np.random.default_rng(seed)

        conditioned_spot_log_drift = self._conditioned_spot_drift(market_state)
        conditioned_spot_volatility = self._conditioned_spot_diffusion(
            market_state=market_state,
            base_volatility=base_volatility,
        )
        conditioned_spot_jump_intensity = self._conditioned_jump_intensity(market_state)

        diffusion_shocks = rng.standard_normal((self.params.n_paths, n_steps))
        if conditioned_spot_jump_intensity > 0.0 and self.params.spot_jump_log_return_std > 0.0:
            jump_counts = rng.poisson(conditioned_spot_jump_intensity * dt, size=(self.params.n_paths, n_steps))
            jump_sizes = np.zeros((self.params.n_paths, n_steps), dtype=float)
            jump_mask = jump_counts > 0
            if np.any(jump_mask):
                jump_sizes[jump_mask] = rng.normal(
                    loc=self.params.spot_jump_log_return_mean * jump_counts[jump_mask],
                    scale=self.params.spot_jump_log_return_std * np.sqrt(jump_counts[jump_mask]),
                )
        else:
            jump_sizes = np.zeros((self.params.n_paths, n_steps), dtype=float)

        log_initial_spot = float(np.log(initial_spot))
        log_increments = (
            (conditioned_spot_log_drift - 0.5 * conditioned_spot_volatility**2) * dt
            + conditioned_spot_volatility * np.sqrt(dt) * diffusion_shocks
            + jump_sizes
        )
        terminal_log_spot = log_initial_spot + np.sum(log_increments, axis=1)
        terminal_values = np.exp(terminal_log_spot)
        terminal_payoffs = (terminal_values >= reference_price).astype(float)

        return MarkovSimulationResult(
            terminal_spot_values=terminal_values,
            terminal_payoffs=terminal_payoffs,
            expected_terminal_probability=float(np.mean(terminal_payoffs)),
            terminal_probability_std=float(np.std(terminal_payoffs)),
            n_paths=self.params.n_paths,
            diagnostics={
                "n_steps": n_steps,
                "dt_seconds": dt,
                "simulation_mode": "spot_terminal_binary_payoff_rollout",
                "rollout_kernel": "spot_jump_diffusion",
                "initial_spot_price": initial_spot,
                "reference_spot_price": reference_price,
                "conditioned_spot_log_drift_per_second": conditioned_spot_log_drift,
                "conditioned_spot_volatility_per_sqrt_second": conditioned_spot_volatility,
                "conditioned_spot_jump_intensity_per_second": conditioned_spot_jump_intensity,
                "rollout_horizon_seconds": step_seconds,
                "terminal_spot_mean": float(np.mean(terminal_values)),
                "terminal_spot_std": float(np.std(terminal_values)),
            },
        )

    async def simulate_async(
        self,
        *,
        horizon_seconds: float,
        market_state: SimulationMarketState,
        initial_event_state: Optional[Dict[str, Any]] = None,
        seed: int | None = None,
    ) -> MarkovSimulationResult:
        return await asyncio.to_thread(
            self.simulate,
            horizon_seconds=horizon_seconds,
            market_state=market_state,
            initial_event_state=initial_event_state,
            seed=seed,
        )

    def simulate_event_state_batch(
        self,
        *,
        initial_event_states: list[Dict[str, Any]],
        seed: int | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> list[MarkovSimulationResult]:
        if not initial_event_states:
            return []

        rng = np.random.default_rng(seed)
        results: list[MarkovSimulationResult] = []
        for event_state_row in initial_event_states:
            row_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            market_state = self._market_state_from_event_state(event_state_row)
            results.append(
                self.simulate(
                    horizon_seconds=float(max(_coerce_float(event_state_row.get("seconds_to_end"), default=0.0), 0.0)),
                    market_state=market_state,
                    initial_event_state=event_state_row,
                    seed=row_seed,
                )
            )
            if progress_callback is not None:
                progress_callback(1)
        return results

    def _resolve_pricing_inputs(
        self,
        *,
        market_state: SimulationMarketState,
        initial_event_state: Optional[Dict[str, Any]],
    ) -> dict[str, float]:
        row = initial_event_state or {}
        spot_price = _coerce_float(row.get("spot_price"), default=market_state.spot_price)
        reference_spot_price = _coerce_float(
            row.get("reference_spot_price"),
            default=market_state.reference_spot_price,
        )
        volatility_per_sqrt_second = _coerce_float(
            row.get("volatility_per_sqrt_second"),
            default=market_state.spot_volatility_per_sqrt_second,
        )

        if spot_price is None or not np.isfinite(spot_price) or spot_price <= 0.0:
            raise ValueError("spot_price must be present and positive for binary payoff pricing")
        if reference_spot_price is None or not np.isfinite(reference_spot_price) or reference_spot_price <= 0.0:
            raise ValueError("reference_spot_price must be present and positive for binary payoff pricing")
        if volatility_per_sqrt_second is None or not np.isfinite(volatility_per_sqrt_second) or volatility_per_sqrt_second < 0.0:
            volatility_per_sqrt_second = max(self.params.base_spot_volatility_per_sqrt_second, 0.0)

        return {
            "spot_price": float(spot_price),
            "reference_spot_price": float(reference_spot_price),
            "volatility_per_sqrt_second": float(volatility_per_sqrt_second),
        }

    def _market_state_from_event_state(self, event_state_row: Dict[str, Any]) -> SimulationMarketState:
        return SimulationMarketState(
            spot_price=_coerce_float(event_state_row.get("spot_price")),
            reference_spot_price=_coerce_float(event_state_row.get("reference_spot_price")),
            spot_volatility_per_sqrt_second=_coerce_float(
                event_state_row.get("volatility_per_sqrt_second"),
                default=max(self.params.base_spot_volatility_per_sqrt_second, 0.0),
            )
            or max(self.params.base_spot_volatility_per_sqrt_second, 0.0),
            liquidity_depth=_nanmin(
                [
                    _coerce_float(event_state_row.get("up_bid_depth_top_5")),
                    _coerce_float(event_state_row.get("up_ask_depth_top_5")),
                    _coerce_float(event_state_row.get("down_bid_depth_top_5")),
                    _coerce_float(event_state_row.get("down_ask_depth_top_5")),
                ],
                default=0.0,
            ),
            book_velocity=_nanmean(
                [
                    abs(_coerce_float(event_state_row.get("up_book_velocity"), default=0.0) or 0.0),
                    abs(_coerce_float(event_state_row.get("down_book_velocity"), default=0.0) or 0.0),
                ],
                default=0.0,
            ),
            spot_vol_multiplier=_coerce_float(event_state_row.get("spot_vol_multiplier"), default=1.0) or 1.0,
        )

    def _conditioned_spot_drift(self, market_state: SimulationMarketState) -> float:
        del market_state  # Pricing uses current spot level and volatility directly.
        return float(self.params.spot_log_drift_per_second)

    def _conditioned_spot_diffusion(
        self,
        *,
        market_state: SimulationMarketState,
        base_volatility: float,
    ) -> float:
        vol_multiplier = max(float(market_state.spot_vol_multiplier), 0.25)
        liquidity_depth = max(float(market_state.liquidity_depth), 0.0)
        liquidity_penalty = 1.0 + (
            self.params.liquidity_volatility_scale
            / max(np.log1p(liquidity_depth), 1.0)
        )
        velocity_pressure = 1.0 + self.params.velocity_volatility_scale * abs(float(market_state.book_velocity))
        variance_scale = (
            vol_multiplier
            * liquidity_penalty
            * velocity_pressure
        )
        return float(max(base_volatility, 0.0) * np.sqrt(max(variance_scale, 1e-9)))

    def _conditioned_jump_intensity(self, market_state: SimulationMarketState) -> float:
        del market_state
        return float(max(self.params.spot_jump_intensity_per_second, 0.0))


def _apply_pricing_model(
    pricing_model: Optional[PricingModel],
    probabilities: np.ndarray,
) -> np.ndarray:
    if pricing_model is None:
        return np.asarray(probabilities, dtype=float)

    try:
        priced = pricing_model(probabilities)
        priced_array = np.asarray(priced, dtype=float)
    except Exception:
        priced_array = np.vectorize(pricing_model, otypes=[float])(probabilities)

    if priced_array.shape == ():
        priced_array = np.full(probabilities.shape, float(priced_array), dtype=float)
    if priced_array.shape != probabilities.shape:
        raise ValueError("pricing_model must return a scalar or an array matching the input probability shape")
    return priced_array


def _coerce_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(numeric):
        return default
    return numeric


def _nanmean(values: list[float | None], *, default: float) -> float:
    filtered = [value for value in values if value is not None and np.isfinite(value)]
    if not filtered:
        return float(default)
    return float(np.mean(filtered))


def _nanmin(values: list[float | None], *, default: float) -> float:
    filtered = [value for value in values if value is not None and np.isfinite(value)]
    if not filtered:
        return float(default)
    return float(np.min(filtered))
