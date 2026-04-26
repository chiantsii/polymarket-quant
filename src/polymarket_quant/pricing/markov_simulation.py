"""Path-integrated MCMC pricing for binary prediction markets."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd

from polymarket_quant.utils.math import logit, sigmoid


PricingModel = Callable[[np.ndarray], np.ndarray | float]
ProgressCallback = Callable[[int], None]


@dataclass(frozen=True)
class SimulationMarketState:
    """Market-observation state used to condition Markov path simulation."""

    imbalance_signal: float = 0.0
    liquidity_depth: float = 0.0
    book_velocity: float = 0.0
    spot_vol_multiplier: float = 1.0
    cross_book_basis: float = 0.0
    boundary_distance: float = 0.5


@dataclass(frozen=True)
class MarkovSimulationParams:
    """Parameters for logit-state path simulation conditioned on market state."""

    drift: float = 0.0
    diffusion_vol: float = 0.05
    jump_intensity: float = 0.01
    jump_mean: float = 0.0
    jump_std: float = 0.20
    dt_seconds: float = 1.0
    n_paths: int = 1_000
    imbalance_drift_scale: float = 0.35
    volatility_variance_scale: float = 0.35
    liquidity_variance_scale: float = 0.30
    velocity_variance_scale: float = 0.50
    boundary_variance_scale: float = 0.25
    jump_intensity_scale: float = 0.50
    boundary_clip: float = 1e-6
    rollout_horizon_seconds: float = 0.0
    rollout_velocity_decay: float = 0.50


@dataclass(frozen=True)
class PriceDistributionSummary:
    """Aggregated price distribution under a token-specific pricing model."""

    expected_fair_price: float
    risk_score: float
    n_paths: int
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MarkovSimulationResult:
    """Simulated terminal-state distribution from the MCMC engine."""

    terminal_probabilities: np.ndarray
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
        probabilities = 1.0 - self.terminal_probabilities if invert_probability else self.terminal_probabilities
        priced = _apply_pricing_model(pricing_model, probabilities)
        return PriceDistributionSummary(
            expected_fair_price=float(np.mean(priced)),
            risk_score=float(np.std(priced)),
            n_paths=self.n_paths,
            diagnostics={
                "pricing_probability_mean": float(np.mean(probabilities)),
                "pricing_probability_std": float(np.std(probabilities)),
            },
        )


class MarkovSimulationEngine:
    """Simulate latent logit-probability paths under market-conditioned dynamics."""

    def __init__(
        self,
        params: Optional[MarkovSimulationParams] = None,
        pricing_model: Optional[PricingModel] = None,
        transition_bundle: Any | None = None,
        event_duration_seconds: float = 300.0,
    ) -> None:
        self.params = params or MarkovSimulationParams()
        self.pricing_model = pricing_model
        self.transition_bundle = transition_bundle
        self.event_duration_seconds = float(event_duration_seconds)

    def simulate(
        self,
        *,
        initial_probability: float,
        horizon_seconds: float,
        market_state: SimulationMarketState,
        initial_event_state: Optional[Dict[str, Any]] = None,
        seed: int | None = None,
    ) -> MarkovSimulationResult:
        if not 0.0 < initial_probability < 1.0:
            raise ValueError("initial_probability must be strictly between 0 and 1")
        if horizon_seconds < 0:
            raise ValueError("horizon_seconds must be non-negative")
        if self.params.dt_seconds <= 0:
            raise ValueError("dt_seconds must be positive")
        if self.params.n_paths <= 0:
            raise ValueError("n_paths must be positive")

        if self.transition_bundle is not None and initial_event_state:
            return self._simulate_repeated_next_state_rollout(
                initial_event_state=initial_event_state,
                initial_probability=initial_probability,
                horizon_seconds=horizon_seconds,
                seed=seed,
            )

        if horizon_seconds == 0:
            terminal_probabilities = np.full(self.params.n_paths, initial_probability, dtype=float)
            return MarkovSimulationResult(
                terminal_probabilities=terminal_probabilities,
                expected_terminal_probability=float(initial_probability),
                terminal_probability_std=0.0,
                n_paths=self.params.n_paths,
                diagnostics={"n_steps": 0, "dt_seconds": 0.0},
            )

        n_steps = max(1, int(np.ceil(horizon_seconds / self.params.dt_seconds)))
        dt = float(horizon_seconds / n_steps)
        rng = np.random.default_rng(seed)

        conditioned_drift = self._conditioned_drift(market_state)
        conditioned_diffusion = self._conditioned_diffusion(market_state)
        conditioned_jump_intensity = self._conditioned_jump_intensity(market_state)

        diffusion_shocks = rng.standard_normal((self.params.n_paths, n_steps))
        jump_counts = rng.poisson(conditioned_jump_intensity * dt, size=(self.params.n_paths, n_steps))
        jump_sizes = np.zeros((self.params.n_paths, n_steps), dtype=float)
        jump_mask = jump_counts > 0
        if np.any(jump_mask):
            jump_sizes[jump_mask] = rng.normal(
                loc=self.params.jump_mean * jump_counts[jump_mask],
                scale=self.params.jump_std * np.sqrt(jump_counts[jump_mask]),
            )

        increments = (
            conditioned_drift * dt
            + conditioned_diffusion * np.sqrt(dt) * diffusion_shocks
            + jump_sizes
        )
        initial_logit = float(logit(initial_probability, epsilon=self.params.boundary_clip))
        latent_logit_paths = initial_logit + np.cumsum(increments, axis=1)
        logit_floor = float(logit(self.params.boundary_clip, epsilon=self.params.boundary_clip))
        logit_ceiling = float(logit(1.0 - self.params.boundary_clip, epsilon=self.params.boundary_clip))
        latent_logit_paths = np.clip(latent_logit_paths, logit_floor, logit_ceiling)
        terminal_probabilities = np.asarray(sigmoid(latent_logit_paths[:, -1]), dtype=float)

        return MarkovSimulationResult(
            terminal_probabilities=terminal_probabilities,
            expected_terminal_probability=float(np.mean(terminal_probabilities)),
            terminal_probability_std=float(np.std(terminal_probabilities)),
            n_paths=self.params.n_paths,
            diagnostics={
                "n_steps": n_steps,
                "dt_seconds": dt,
                "simulation_mode": "single_kernel_terminal_rollout",
                "conditioned_drift": conditioned_drift,
                "conditioned_diffusion": conditioned_diffusion,
                "conditioned_jump_intensity": conditioned_jump_intensity,
                "imbalance_signal": market_state.imbalance_signal,
                "liquidity_depth": market_state.liquidity_depth,
                "book_velocity": market_state.book_velocity,
                "spot_vol_multiplier": market_state.spot_vol_multiplier,
                "cross_book_basis": market_state.cross_book_basis,
                "boundary_distance": market_state.boundary_distance,
            },
        )

    async def simulate_async(
        self,
        *,
        initial_probability: float,
        horizon_seconds: float,
        market_state: SimulationMarketState,
        initial_event_state: Optional[Dict[str, Any]] = None,
        seed: int | None = None,
    ) -> MarkovSimulationResult:
        return await asyncio.to_thread(
            self.simulate,
            initial_probability=initial_probability,
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

        if self.transition_bundle is None:
            results: list[MarkovSimulationResult] = []
            for row in initial_event_states:
                initial_probability = float(np.clip(row.get("latent_up_probability", np.nan), self.params.boundary_clip, 1.0 - self.params.boundary_clip))
                horizon_seconds = float(max(row.get("seconds_to_end", 0.0), 0.0))
                market_state = SimulationMarketState()
                results.append(
                    self.simulate(
                        initial_probability=initial_probability,
                        horizon_seconds=horizon_seconds,
                        market_state=market_state,
                        initial_event_state=row,
                        seed=seed,
                    )
                )
            return results

        return self._simulate_repeated_next_state_rollout_batch(
            initial_event_states=initial_event_states,
            seed=seed,
            progress_callback=progress_callback,
        )

    def _simulate_repeated_next_state_rollout(
        self,
        *,
        initial_event_state: Dict[str, Any],
        initial_probability: float,
        horizon_seconds: float,
        seed: int | None,
    ) -> MarkovSimulationResult:
        if horizon_seconds == 0:
            terminal_probabilities = np.full(self.params.n_paths, initial_probability, dtype=float)
            return MarkovSimulationResult(
                terminal_probabilities=terminal_probabilities,
                expected_terminal_probability=float(initial_probability),
                terminal_probability_std=0.0,
                n_paths=self.params.n_paths,
                diagnostics={"n_steps": 0, "dt_seconds": 0.0, "simulation_mode": "repeated_next_state_rollout"},
            )

        bundle_default_step = max(float(getattr(self.transition_bundle, "default_step_seconds", 1.0)), 1e-9)
        rollout_step_seconds = (
            max(float(self.params.rollout_horizon_seconds), 1e-9)
            if float(self.params.rollout_horizon_seconds) > 0.0
            else bundle_default_step
        )
        rng = np.random.default_rng(seed)

        feature_state_columns = self._rollout_state_columns()
        if not feature_state_columns:
            raise ValueError("transition_bundle must expose feature_columns")

        path_states = self._initial_rollout_state_frame(
            initial_event_state=initial_event_state,
            feature_state_columns=feature_state_columns,
            initial_probability=initial_probability,
        )

        remaining_seconds = float(horizon_seconds)
        n_steps = 0
        jump_intensity_means: list[float] = []
        step_horizons: list[float] = []

        while remaining_seconds > 1e-9:
            step_seconds = min(rollout_step_seconds, remaining_seconds)
            predictions = self._predict_next_latent_state(
                path_states=path_states,
                feature_state_columns=feature_state_columns,
                target_horizon_seconds=step_seconds,
            )
            path_states = self._sample_next_rollout_latent_state(
                path_states=path_states,
                predictions=predictions,
                step_seconds=step_seconds,
                rng=rng,
            )
            remaining_seconds -= step_seconds
            n_steps += 1
            step_horizons.append(step_seconds)
            jump_intensity_means.append(
                float(self._prediction_jump_intensity(predictions, step_seconds).mean())
            )

        terminal_probabilities = pd.to_numeric(path_states["latent_up_probability"], errors="coerce").fillna(initial_probability).to_numpy(dtype=float)
        terminal_probabilities = np.clip(terminal_probabilities, self.params.boundary_clip, 1.0 - self.params.boundary_clip)

        return MarkovSimulationResult(
            terminal_probabilities=terminal_probabilities,
            expected_terminal_probability=float(np.mean(terminal_probabilities)),
            terminal_probability_std=float(np.std(terminal_probabilities)),
            n_paths=self.params.n_paths,
            diagnostics={
                "n_steps": n_steps,
                "dt_seconds": float(np.mean(step_horizons)) if step_horizons else 0.0,
                "simulation_mode": "repeated_next_state_rollout",
                "rollout_kernel": self._prediction_kernel_name(predictions),
                "rollout_horizon_seconds": rollout_step_seconds,
                "mean_jump_intensity_hat": float(np.mean(jump_intensity_means)) if jump_intensity_means else 0.0,
                "terminal_latent_logit_mean": float(pd.to_numeric(path_states["latent_logit_probability"], errors="coerce").mean()),
            },
        )

    def _simulate_repeated_next_state_rollout_batch(
        self,
        *,
        initial_event_states: list[Dict[str, Any]],
        seed: int | None,
        progress_callback: ProgressCallback | None,
    ) -> list[MarkovSimulationResult]:
        bundle_default_step = max(float(getattr(self.transition_bundle, "default_step_seconds", 1.0)), 1e-9)
        rollout_step_seconds = (
            max(float(self.params.rollout_horizon_seconds), 1e-9)
            if float(self.params.rollout_horizon_seconds) > 0.0
            else bundle_default_step
        )
        rng = np.random.default_rng(seed)

        feature_state_columns = self._rollout_state_columns()
        if not feature_state_columns:
            raise ValueError("transition_bundle must expose feature_columns")

        base_states: list[dict[str, Any]] = []
        horizons: list[float] = []
        for row_id, initial_event_state in enumerate(initial_event_states):
            initial_probability = float(np.clip(initial_event_state.get("latent_up_probability", np.nan), self.params.boundary_clip, 1.0 - self.params.boundary_clip))
            initial_latent_logit = float(logit(initial_probability, epsilon=self.params.boundary_clip))
            state_dict: dict[str, Any] = {"_simulation_row_id": row_id}
            for column in feature_state_columns:
                state_dict[column] = initial_event_state.get(column, np.nan)
            # Preserve the row-specific starting belief so any numerical failure
            # during rollout falls back to the path's own initial state, not to
            # an artificial neutral probability like 0.5.
            state_dict["_initial_probability"] = initial_probability
            state_dict["latent_up_probability"] = initial_probability
            state_dict["latent_logit_probability"] = initial_latent_logit
            base_states.append(state_dict)
            horizons.append(float(max(initial_event_state.get("seconds_to_end", 0.0), 0.0)))

        base_frame = pd.DataFrame(base_states)
        path_states = base_frame.loc[base_frame.index.repeat(self.params.n_paths)].reset_index(drop=True)
        remaining_seconds = np.repeat(np.asarray(horizons, dtype=float), self.params.n_paths)

        row_count = len(initial_event_states)
        step_counts = np.zeros(row_count, dtype=int)
        step_horizon_sums = np.zeros(row_count, dtype=float)
        jump_intensity_sums = np.zeros(row_count, dtype=float)

        while np.any(remaining_seconds > 1e-9):
            active_mask = remaining_seconds > 1e-9
            active_states = path_states.loc[active_mask].copy()
            active_remaining = remaining_seconds[active_mask]
            active_step_seconds = np.minimum(rollout_step_seconds, active_remaining)

            predictions = self._predict_next_latent_state(
                path_states=active_states,
                feature_state_columns=feature_state_columns,
                target_horizon_seconds=active_step_seconds,
            )
            active_states = self._sample_next_rollout_latent_state(
                path_states=active_states,
                predictions=predictions,
                step_seconds=active_step_seconds,
                rng=rng,
            )
            path_states.loc[active_mask, :] = active_states
            remaining_seconds[active_mask] = active_remaining - active_step_seconds

            active_row_ids = active_states["_simulation_row_id"].to_numpy(dtype=int)
            active_frame = pd.DataFrame(
                {
                    "row_id": active_row_ids,
                    "step_seconds": active_step_seconds,
                    "jump_intensity_hat": self._prediction_jump_intensity(predictions, active_step_seconds),
                }
            )
            grouped = active_frame.groupby("row_id", sort=False).agg(
                step_seconds=("step_seconds", "mean"),
                jump_intensity_hat=("jump_intensity_hat", "mean"),
            )
            grouped_ids = grouped.index.to_numpy(dtype=int)
            step_counts[grouped_ids] += 1
            step_horizon_sums[grouped_ids] += grouped["step_seconds"].to_numpy(dtype=float)
            jump_intensity_sums[grouped_ids] += grouped["jump_intensity_hat"].to_numpy(dtype=float)
            if progress_callback is not None:
                progress_callback(int(len(grouped_ids)))

        results: list[MarkovSimulationResult] = []
        latent_probabilities = pd.to_numeric(
            path_states["latent_up_probability"],
            errors="coerce",
        )
        initial_probabilities = pd.to_numeric(
            path_states.get("_initial_probability"),
            errors="coerce",
        ).fillna(0.5)
        latent_probabilities = latent_probabilities.where(latent_probabilities.notna(), initial_probabilities).to_numpy(dtype=float)

        for row_id in range(row_count):
            row_mask = path_states["_simulation_row_id"].to_numpy(dtype=int) == row_id
            terminal_probabilities = np.clip(
                latent_probabilities[row_mask],
                self.params.boundary_clip,
                1.0 - self.params.boundary_clip,
            )
            n_steps = int(step_counts[row_id])
            results.append(
                MarkovSimulationResult(
                    terminal_probabilities=terminal_probabilities,
                    expected_terminal_probability=float(np.mean(terminal_probabilities)),
                    terminal_probability_std=float(np.std(terminal_probabilities)),
                    n_paths=self.params.n_paths,
                    diagnostics={
                        "n_steps": n_steps,
                        "dt_seconds": float(step_horizon_sums[row_id] / max(n_steps, 1)),
                        "simulation_mode": "repeated_next_state_rollout",
                        "rollout_kernel": self._prediction_kernel_name(predictions),
                        "rollout_horizon_seconds": rollout_step_seconds,
                        "mean_jump_intensity_hat": float(jump_intensity_sums[row_id] / max(n_steps, 1)),
                        "terminal_latent_logit_mean": float(
                            pd.to_numeric(
                                path_states.loc[row_mask, "latent_logit_probability"],
                                errors="coerce",
                            ).mean()
                        ),
                    },
                )
            )

        return results

    def _initial_rollout_state_frame(
        self,
        *,
        initial_event_state: Dict[str, Any],
        feature_state_columns: list[str],
        initial_probability: float,
    ) -> pd.DataFrame:
        state_dict: dict[str, Any] = {}
        for column in feature_state_columns:
            state_dict[column] = initial_event_state.get(column, np.nan)

        initial_latent_logit = float(logit(initial_probability, epsilon=self.params.boundary_clip))
        state_dict["latent_up_probability"] = initial_probability
        state_dict["latent_logit_probability"] = initial_latent_logit

        path_states = pd.DataFrame([state_dict] * self.params.n_paths).reset_index(drop=True)
        return path_states

    def _predict_next_latent_state(
        self,
        *,
        path_states: pd.DataFrame,
        feature_state_columns: list[str],
        target_horizon_seconds: float | np.ndarray,
    ) -> pd.DataFrame:
        feature_frame = pd.DataFrame(index=path_states.index)
        for column in feature_state_columns:
            feature_frame[f"current_{column}"] = path_states[column] if column in path_states.columns else np.nan
        if np.isscalar(target_horizon_seconds):
            feature_frame["target_horizon_seconds"] = float(target_horizon_seconds)
        else:
            feature_frame["target_horizon_seconds"] = np.asarray(target_horizon_seconds, dtype=float)
        if (
            hasattr(self.transition_bundle, "predict_latent_kernel")
            and getattr(self.transition_bundle, "latent_mu_model", None) is not None
            and getattr(self.transition_bundle, "latent_sigma_model", None) is not None
        ):
            return self.transition_bundle.predict_latent_kernel(feature_frame)
        if hasattr(self.transition_bundle, "predict_latent_step"):
            return self.transition_bundle.predict_latent_step(feature_frame)
        full_feature_columns = [
            column.removeprefix("current_")
            for column in getattr(self.transition_bundle, "feature_columns", ())
            if column.startswith("current_")
        ]
        for column in full_feature_columns:
            prefixed = f"current_{column}"
            if prefixed not in feature_frame.columns:
                feature_frame[prefixed] = path_states[column] if column in path_states.columns else np.nan
        return self.transition_bundle.predict(feature_frame)

    def _sample_next_rollout_latent_state(
        self,
        *,
        path_states: pd.DataFrame,
        predictions: pd.DataFrame,
        step_seconds: float | np.ndarray,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        next_states = path_states.copy()
        if np.isscalar(step_seconds):
            dt = np.full(len(next_states), float(step_seconds), dtype=float)
        else:
            dt = np.asarray(step_seconds, dtype=float)

        if "mu_hat_latent_logit_probability" in predictions.columns and "sigma_hat_latent_logit_probability" in predictions.columns:
            current_latent = pd.to_numeric(
                path_states.get("latent_logit_probability"),
                errors="coerce",
            ).fillna(0.0).to_numpy(dtype=float)
            mu_hat = pd.to_numeric(
                predictions.get("mu_hat_latent_logit_probability"),
                errors="coerce",
            ).fillna(0.0).to_numpy(dtype=float)
            sigma_hat = pd.to_numeric(
                predictions.get("sigma_hat_latent_logit_probability"),
                errors="coerce",
            ).fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
            sigma_hat = np.sqrt(self._clip_rollout_variance(np.square(sigma_hat)))

            lambda_hat = pd.to_numeric(
                predictions.get("lambda_hat_latent_logit_probability"),
                errors="coerce",
            ).fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
            jump_probability = np.clip(1.0 - np.exp(-lambda_hat * np.maximum(dt, 0.0)), 0.0, 1.0)
            jump_mean_hat = pd.to_numeric(
                predictions.get("jump_mean_hat_latent_logit_probability"),
                errors="coerce",
            ).fillna(0.0).to_numpy(dtype=float)
            jump_std_hat = pd.to_numeric(
                predictions.get("jump_std_hat_latent_logit_probability"),
                errors="coerce",
            ).fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)

            diffusion_shocks = rng.standard_normal(len(next_states))
            jump_draws = rng.uniform(size=len(next_states)) < jump_probability
            jump_sizes = np.zeros(len(next_states), dtype=float)
            if np.any(jump_draws):
                jump_sizes[jump_draws] = rng.normal(
                    loc=jump_mean_hat[jump_draws],
                    scale=jump_std_hat[jump_draws],
                )
            sampled = current_latent + (mu_hat * dt) + (sigma_hat * np.sqrt(dt) * diffusion_shocks) + jump_sizes
        else:
            mean = pd.to_numeric(
                predictions.get("future_hat_latent_logit_probability"),
                errors="coerce",
            ).to_numpy(dtype=float)
            variance = (
                pd.to_numeric(predictions.get("diffusion_hat_latent_logit_probability"), errors="coerce")
                .fillna(0.0)
                .clip(lower=0.0)
                .to_numpy(dtype=float)
                if "diffusion_hat_latent_logit_probability" in predictions.columns
                else np.zeros(len(next_states), dtype=float)
            )
            variance = self._clip_rollout_variance(variance)
            std = np.sqrt(variance)
            shocks = rng.standard_normal(len(next_states))
            sampled = mean + (std * shocks)
        logit_floor = float(logit(self.params.boundary_clip, epsilon=self.params.boundary_clip))
        logit_ceiling = float(logit(1.0 - self.params.boundary_clip, epsilon=self.params.boundary_clip))
        sampled = np.clip(sampled, logit_floor, logit_ceiling)
        next_states["latent_logit_probability"] = sampled

        self._rebuild_latent_rollout_state(next_states, step_seconds=step_seconds)
        return next_states

    def _rebuild_latent_rollout_state(self, path_states: pd.DataFrame, *, step_seconds: float | np.ndarray) -> None:
        if np.isscalar(step_seconds):
            step_series = pd.Series(float(step_seconds), index=path_states.index, dtype=float)
        else:
            step_series = pd.Series(np.asarray(step_seconds, dtype=float), index=path_states.index, dtype=float)

        latent_logit = pd.to_numeric(path_states.get("latent_logit_probability"), errors="coerce")
        if latent_logit is not None:
            latent_logit = latent_logit.clip(
                lower=float(logit(self.params.boundary_clip, epsilon=self.params.boundary_clip)),
                upper=float(logit(1.0 - self.params.boundary_clip, epsilon=self.params.boundary_clip)),
            )
            path_states["latent_logit_probability"] = latent_logit
            latent_logit_values = latent_logit.to_numpy(dtype=float, na_value=np.nan)
            latent_up_values = np.asarray(sigmoid(latent_logit_values), dtype=float)
            latent_up_values[~np.isfinite(latent_logit_values)] = np.nan
            path_states["latent_up_probability"] = latent_up_values

        if "market_implied_up_probability" in path_states.columns and "fundamental_up_probability" in path_states.columns:
            market_probability = pd.to_numeric(path_states["market_implied_up_probability"], errors="coerce")
            fundamental_probability = pd.to_numeric(path_states["fundamental_up_probability"], errors="coerce")
            latent_probability = pd.to_numeric(path_states.get("latent_up_probability"), errors="coerce")
            path_states["market_fundamental_basis"] = market_probability - fundamental_probability
            path_states["latent_market_basis"] = latent_probability - market_probability
            path_states["latent_fundamental_basis"] = latent_probability - fundamental_probability
            path_states["abs_market_fundamental_basis"] = path_states["market_fundamental_basis"].abs()
            path_states["abs_latent_market_basis"] = path_states["latent_market_basis"].abs()
            path_states["abs_latent_fundamental_basis"] = path_states["latent_fundamental_basis"].abs()

        if "seconds_to_end" in path_states.columns:
            seconds_to_end = pd.to_numeric(path_states["seconds_to_end"], errors="coerce").fillna(0.0)
            path_states["seconds_to_end"] = (seconds_to_end - step_series).clip(lower=0.0)

        if "normalized_time_to_end" in path_states.columns and self.event_duration_seconds > 0:
            decrement = step_series / self.event_duration_seconds
            current_normalized_time = pd.to_numeric(path_states["normalized_time_to_end"], errors="coerce").fillna(0.0)
            path_states["normalized_time_to_end"] = (current_normalized_time - decrement).clip(lower=0.0, upper=1.0)

        if "dist_to_boundary" in path_states.columns and "latent_up_probability" in path_states.columns:
            latent_probability = pd.to_numeric(path_states["latent_up_probability"], errors="coerce").fillna(0.5)
            path_states["dist_to_boundary"] = np.minimum(latent_probability, 1.0 - latent_probability)

        if "boundary_leverage_ratio" in path_states.columns and "dist_to_boundary" in path_states.columns:
            distance = pd.to_numeric(path_states["dist_to_boundary"], errors="coerce").fillna(0.5)
            path_states["boundary_leverage_ratio"] = 1.0 / np.maximum(distance, self.params.boundary_clip)

        for side in ("up", "down"):
            velocity_col = f"{side}_book_velocity"
            if velocity_col in path_states.columns:
                path_states[velocity_col] = (
                    pd.to_numeric(path_states[velocity_col], errors="coerce").fillna(0.0) * self.params.rollout_velocity_decay
                )

            book_age_col = f"{side}_book_age_seconds"
            if book_age_col in path_states.columns:
                path_states[book_age_col] = 0.0

        if "book_age_max" in path_states.columns:
            path_states["book_age_max"] = 0.0

    def _rollout_state_columns(self) -> list[str]:
        if hasattr(self.transition_bundle, "predict_latent_step"):
            feature_source = getattr(self.transition_bundle, "rollout_feature_columns", ()) or getattr(
                self.transition_bundle,
                "feature_columns",
                (),
            )
        else:
            feature_source = getattr(self.transition_bundle, "feature_columns", ())
        return sorted(
            {
                column.removeprefix("current_")
                for column in feature_source
                if column.startswith("current_")
            }
        )

    def _clip_rollout_variance(self, variance: np.ndarray) -> np.ndarray:
        floor = float(getattr(getattr(self.transition_bundle, "config", None), "diffusion_variance_floor", 0.0) or 0.0)
        cap = float(getattr(getattr(self.transition_bundle, "config", None), "diffusion_variance_cap", np.inf))
        lower = max(floor, 0.0)
        upper = cap if np.isfinite(cap) else None
        if upper is None:
            return np.clip(variance, lower, None)
        return np.clip(variance, lower, upper)

    def _prediction_jump_intensity(
        self,
        predictions: pd.DataFrame,
        step_seconds: float | np.ndarray,
    ) -> np.ndarray:
        if "jump_intensity_hat" in predictions.columns:
            return pd.to_numeric(
                predictions.get("jump_intensity_hat"),
                errors="coerce",
            ).fillna(0.0).to_numpy(dtype=float)
        if "lambda_hat_latent_logit_probability" in predictions.columns:
            return pd.to_numeric(
                predictions.get("lambda_hat_latent_logit_probability"),
                errors="coerce",
            ).fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
        if "jump_probability_hat_latent_logit_probability" in predictions.columns:
            jump_probability = pd.to_numeric(
                predictions.get("jump_probability_hat_latent_logit_probability"),
                errors="coerce",
            ).fillna(0.0).clip(lower=0.0, upper=1.0).to_numpy(dtype=float)
            if np.isscalar(step_seconds):
                dt = np.full(len(jump_probability), float(step_seconds), dtype=float)
            else:
                dt = np.asarray(step_seconds, dtype=float)
            safe_dt = np.maximum(dt, 1e-9)
            return -np.log1p(-np.clip(jump_probability, 0.0, 1.0 - 1e-9)) / safe_dt
        return np.zeros(len(predictions), dtype=float)

    def _prediction_kernel_name(self, predictions: pd.DataFrame) -> str:
        if "mu_hat_latent_logit_probability" in predictions.columns:
            return "parametric_latent_jump_diffusion"
        return "latent_only_batched"

    def _conditioned_drift(self, market_state: SimulationMarketState) -> float:
        return self.params.drift + (self.params.imbalance_drift_scale * float(market_state.imbalance_signal))

    def _conditioned_diffusion(self, market_state: SimulationMarketState) -> float:
        vol_multiplier = max(float(market_state.spot_vol_multiplier), 0.25)
        liquidity_depth = max(float(market_state.liquidity_depth), 0.0)
        liquidity_penalty = 1.0 + (
            self.params.liquidity_variance_scale
            / max(np.log1p(liquidity_depth), 1.0)
        )
        velocity_pressure = 1.0 + self.params.velocity_variance_scale * abs(float(market_state.book_velocity))
        boundary_pressure = 1.0 + self.params.boundary_variance_scale * max(
            0.0,
            1.0 - (2.0 * float(market_state.boundary_distance)),
        )
        variance_scale = (
            vol_multiplier
            * liquidity_penalty
            * velocity_pressure
            * boundary_pressure
        )
        return float(self.params.diffusion_vol * np.sqrt(max(variance_scale, 1e-9)))

    def _conditioned_jump_intensity(self, market_state: SimulationMarketState) -> float:
        vol_pressure = max(float(market_state.spot_vol_multiplier) - 1.0, 0.0)
        velocity_pressure = abs(float(market_state.book_velocity))
        basis_pressure = abs(float(market_state.cross_book_basis)) / 0.05
        scale = 1.0 + self.params.jump_intensity_scale * (vol_pressure + velocity_pressure + basis_pressure)
        return float(max(self.params.jump_intensity * scale, 0.0))


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
