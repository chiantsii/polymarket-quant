"""MCMC-based fair pricing and edge extraction for BTC/ETH 5-minute binary markets."""

from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
import sys
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from polymarket_quant.pricing import (
    MarkovSimulationEngine,
    MarkovSimulationParams,
    SimulationMarketState,
)
from polymarket_quant.state import build_event_state_dataset

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:  # pragma: no cover - optional dependency
    _tqdm = None


@dataclass(frozen=True)
class MispricingDetectorConfig:
    """Configuration for path-integrated MCMC pricing."""

    pricing_method: str = "markov_mcmc"
    spot_log_drift_per_second: float = 0.0
    fallback_spot_volatility_per_sqrt_second: float = 0.0005
    n_samples: int = 1_000
    simulation_dt_seconds: float = 1.0
    rollout_horizon_seconds: float = 0.0
    spot_jump_intensity_per_second: float = 0.01
    spot_jump_log_return_mean: float = 0.0
    spot_jump_log_return_std: float = 0.20
    liquidity_volatility_scale: float = 0.30
    velocity_volatility_scale: float = 0.50
    edge_threshold: float = 0.0
    max_allowed_risk: float = 0.10
    seed: Optional[int] = 42


class RealTimeMispricingDetector:
    """Estimate fair token prices from terminal binary spot payoffs."""

    def __init__(
        self,
        config: Optional[MispricingDetectorConfig] = None,
        pricing_model=None,
    ) -> None:
        self.config = config or MispricingDetectorConfig()
        if str(self.config.pricing_method).strip().lower() != "markov_mcmc":
            raise ValueError("pricing_method must be 'markov_mcmc'")
        self.pricing_model = pricing_model
        self.engine = MarkovSimulationEngine(
            MarkovSimulationParams(
                spot_log_drift_per_second=self.config.spot_log_drift_per_second,
                base_spot_volatility_per_sqrt_second=self.config.fallback_spot_volatility_per_sqrt_second,
                spot_jump_intensity_per_second=self.config.spot_jump_intensity_per_second,
                spot_jump_log_return_mean=self.config.spot_jump_log_return_mean,
                spot_jump_log_return_std=self.config.spot_jump_log_return_std,
                simulation_dt_seconds=self.config.simulation_dt_seconds,
                n_paths=self.config.n_samples,
                liquidity_volatility_scale=self.config.liquidity_volatility_scale,
                velocity_volatility_scale=self.config.velocity_volatility_scale,
                rollout_horizon_seconds=self.config.rollout_horizon_seconds,
            ),
            pricing_model=self.pricing_model,
        )

    def detect(
        self,
        state_rows: Iterable[Dict[str, Any]],
        *,
        show_progress: bool = False,
        progress_description: str = "Pricing replay",
    ) -> List[Dict[str, Any]]:
        """Return one MCMC valuation row per Up/Down token snapshot-level state."""
        rows = [row for row in state_rows if row]
        if not rows:
            return []

        if self._looks_like_event_state(rows[0]):
            return self._detect_from_event_state(
                rows,
                show_progress=show_progress,
                progress_description=progress_description,
            )

        valuation_rows: List[Dict[str, Any]] = []
        grouped_items = list(self._group_by_snapshot(rows).items())
        for snapshot_key, event_rows in self._iter_with_progress(
            grouped_items,
            show_progress=show_progress,
            description=progress_description,
        ):
            if snapshot_key is None or not event_rows:
                continue

            event_state_row = self._event_state_row_from_token_rows(event_rows)
            if event_state_row is None:
                continue

            if not self._has_pricing_inputs(event_state_row):
                continue

            market_state = self._simulation_market_state(event_rows)
            simulation = self.engine.simulate(
                horizon_seconds=float(max(self._to_float(event_state_row.get("seconds_to_end")) or 0.0, 0.0)),
                market_state=market_state,
                initial_event_state=event_state_row,
                seed=self.config.seed,
            )
            up_distribution = simulation.aggregate(self.pricing_model, invert_probability=False)
            down_distribution = simulation.aggregate(self.pricing_model, invert_probability=True)
            pricing = self._pricing_summary(
                event_state_row=event_state_row,
                simulation=simulation,
                up_distribution=up_distribution,
                down_distribution=down_distribution,
            )

            for row in event_rows:
                valuation_rows.append(
                    self._valuation_row(
                        row=row,
                        pricing=pricing,
                    )
                )

        return valuation_rows

    def _detect_from_event_state(
        self,
        event_state_rows: List[Dict[str, Any]],
        *,
        show_progress: bool = False,
        progress_description: str = "Pricing replay",
    ) -> List[Dict[str, Any]]:
        valuation_rows: List[Dict[str, Any]] = []

        valid_rows: list[dict[str, Any]] = []
        for event_state_row in event_state_rows:
            if self._has_pricing_inputs(event_state_row):
                valid_rows.append(event_state_row)

        if not valid_rows:
            return valuation_rows

        total_progress = self._estimated_batch_rollout_steps(valid_rows)
        with self._progress_manager(
            total=total_progress,
            description=progress_description,
            enabled=show_progress,
        ) as progress:
            simulations = self.engine.simulate_event_state_batch(
                initial_event_states=valid_rows,
                seed=self.config.seed,
                progress_callback=progress.update if progress is not None else None,
            )
        row_simulations = zip(valid_rows, simulations)

        for event_state_row, simulation in row_simulations:
            up_distribution = simulation.aggregate(self.pricing_model, invert_probability=False)
            down_distribution = simulation.aggregate(self.pricing_model, invert_probability=True)
            pricing = self._pricing_summary(
                event_state_row=event_state_row,
                simulation=simulation,
                up_distribution=up_distribution,
                down_distribution=down_distribution,
            )
            valuation_rows.extend(self._event_state_valuation_rows(event_state_row=event_state_row, pricing=pricing))

        return valuation_rows

    def _pricing_summary(
        self,
        *,
        event_state_row: Dict[str, Any],
        simulation,
        up_distribution,
        down_distribution,
    ) -> Dict[str, Any]:
        market_implied_up_probability = self._to_float(event_state_row.get("market_implied_up_probability"))
        fundamental_up_probability = self._to_float(event_state_row.get("fundamental_up_probability"))
        latent_up_probability = self._to_float(event_state_row.get("latent_up_probability"))

        return {
            "pricing_method": "markov_mcmc",
            "pricing_source": "spot_terminal_binary_payoff_pricing",
            "market_implied_up_probability": market_implied_up_probability,
            "fundamental_up_probability": fundamental_up_probability,
            "latent_up_probability": latent_up_probability,
            "reference_spot_price": self._to_float(event_state_row.get("reference_spot_price")),
            # This is the direct binary payoff estimate:
            # across simulated terminal spot paths, how often does spot_T finish
            # above the market's reference opening price?
            "expected_terminal_up_probability": simulation.expected_terminal_probability,
            "terminal_probability_std": simulation.terminal_probability_std,
            "fair_up_probability": simulation.expected_terminal_probability,
            "expected_fair_up_price": up_distribution.expected_fair_price,
            "expected_fair_down_price": down_distribution.expected_fair_price,
            "risk_score_up": up_distribution.risk_score,
            "risk_score_down": down_distribution.risk_score,
            "pricing_standard_error": simulation.terminal_probability_std / np.sqrt(max(simulation.n_paths, 1)),
            "pricing_n_samples": simulation.n_paths,
            "simulation_n_steps": simulation.diagnostics.get("n_steps"),
            "simulation_dt_seconds": simulation.diagnostics.get("dt_seconds"),
            "simulation_mode": simulation.diagnostics.get("simulation_mode"),
            "conditioned_spot_log_drift_per_second": simulation.diagnostics.get("conditioned_spot_log_drift_per_second"),
            "conditioned_spot_volatility_per_sqrt_second": simulation.diagnostics.get("conditioned_spot_volatility_per_sqrt_second"),
            "conditioned_spot_jump_intensity_per_second": simulation.diagnostics.get("conditioned_spot_jump_intensity_per_second"),
            "rollout_horizon_seconds": simulation.diagnostics.get("rollout_horizon_seconds"),
            "rollout_kernel": simulation.diagnostics.get("rollout_kernel"),
            "terminal_spot_mean": simulation.diagnostics.get("terminal_spot_mean"),
            "terminal_spot_std": simulation.diagnostics.get("terminal_spot_std"),
        }

    def _valuation_row(
        self,
        *,
        row: Dict[str, Any],
        pricing: Dict[str, Any],
    ) -> Dict[str, Any]:
        outcome_name = str(row.get("outcome_name", "")).lower()
        fair_token_price = (
            pricing["expected_fair_up_price"]
            if outcome_name == "up"
            else pricing["expected_fair_down_price"]
        )
        risk_score = pricing["risk_score_up"] if outcome_name == "up" else pricing["risk_score_down"]

        best_bid = self._to_float(row.get("best_bid"))
        best_ask = self._to_float(row.get("best_ask"))
        buy_edge = (fair_token_price - best_ask) if best_ask is not None else None
        hold_edge = (fair_token_price - best_bid) if best_bid is not None else None
        sell_edge = (best_bid - fair_token_price) if best_bid is not None else None
        buy_signal = bool(
            buy_edge is not None
            and buy_edge > self.config.edge_threshold
            and risk_score < self.config.max_allowed_risk
        )

        return {
            **row,
            "fair_up_probability": pricing["fair_up_probability"],
            "fair_token_price": fair_token_price,
            "expected_fair_up_price": pricing["expected_fair_up_price"],
            "expected_fair_down_price": pricing["expected_fair_down_price"],
            "buy_edge": buy_edge,
            "hold_edge": hold_edge,
            "sell_edge": sell_edge,
            "risk_score": risk_score,
            "buy_signal": buy_signal,
            "edge_threshold": self.config.edge_threshold,
            "max_allowed_risk": self.config.max_allowed_risk,
            **pricing,
        }

    def _event_state_valuation_rows(
        self,
        *,
        event_state_row: Dict[str, Any],
        pricing: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        shared_columns = [
            column
            for column in event_state_row.keys()
            if not column.startswith("up_") and not column.startswith("down_")
        ]
        base = {column: event_state_row[column] for column in shared_columns}

        rows: list[dict[str, Any]] = []
        for outcome_name in ("Up", "Down"):
            side = outcome_name.lower()
            row = {
                **base,
                "outcome_name": outcome_name,
                "token_id": event_state_row.get(f"{side}_token_id"),
                "book_timestamp": event_state_row.get(f"{side}_book_timestamp"),
                "book_age_seconds": event_state_row.get(f"{side}_book_age_seconds"),
                "best_bid": event_state_row.get(f"{side}_best_bid"),
                "best_ask": event_state_row.get(f"{side}_best_ask"),
                "spread": event_state_row.get(f"{side}_spread"),
                "mid_price": event_state_row.get(f"{side}_mid_price"),
                "bid_depth": event_state_row.get(f"{side}_bid_depth"),
                "ask_depth": event_state_row.get(f"{side}_ask_depth"),
                "bid_depth_top_5": event_state_row.get(f"{side}_bid_depth_top_5"),
                "ask_depth_top_5": event_state_row.get(f"{side}_ask_depth_top_5"),
                "orderbook_imbalance": event_state_row.get(f"{side}_orderbook_imbalance"),
                "weighted_imbalance": event_state_row.get(f"{side}_weighted_imbalance"),
                "book_velocity": event_state_row.get(f"{side}_book_velocity"),
            }
            rows.append(self._valuation_row(row=row, pricing=pricing))
        return rows

    def _simulation_market_state(self, event_rows: List[Dict[str, Any]]) -> SimulationMarketState:
        up_row = next((row for row in event_rows if str(row.get("outcome_name", "")).lower() == "up"), None)
        down_row = next((row for row in event_rows if str(row.get("outcome_name", "")).lower() == "down"), None)

        liquidity_depth = self._nanmin_or_default(
            [
                self._nan_if_none(self._row_market_signal(up_row, "bid_depth_top_5")),
                self._nan_if_none(self._row_market_signal(up_row, "ask_depth_top_5")),
                self._nan_if_none(self._row_market_signal(down_row, "bid_depth_top_5")),
                self._nan_if_none(self._row_market_signal(down_row, "ask_depth_top_5")),
            ],
            default=0.0,
        )

        book_velocity = self._nanmean_or_default(
            [
                abs(self._nan_if_none(self._row_market_signal(up_row, "book_velocity"))),
                abs(self._nan_if_none(self._row_market_signal(down_row, "book_velocity"))),
            ],
            default=0.0,
        )

        spot_vol_multiplier = self._to_float(event_rows[0].get("spot_vol_multiplier")) or 1.0

        return SimulationMarketState(
            spot_price=self._to_float(event_rows[0].get("spot_price")),
            reference_spot_price=self._to_float(event_rows[0].get("reference_spot_price")),
            spot_volatility_per_sqrt_second=self._to_float(event_rows[0].get("volatility_per_sqrt_second"))
            or self.config.fallback_spot_volatility_per_sqrt_second,
            liquidity_depth=liquidity_depth,
            book_velocity=book_velocity,
            spot_vol_multiplier=spot_vol_multiplier,
        )

    def _simulation_market_state_from_event_state(self, event_state_row: Dict[str, Any]) -> SimulationMarketState:
        liquidity_depth = self._nanmin_or_default(
            [
                self._nan_if_none(self._to_float(event_state_row.get("up_bid_depth_top_5"))),
                self._nan_if_none(self._to_float(event_state_row.get("up_ask_depth_top_5"))),
                self._nan_if_none(self._to_float(event_state_row.get("down_bid_depth_top_5"))),
                self._nan_if_none(self._to_float(event_state_row.get("down_ask_depth_top_5"))),
            ],
            default=0.0,
        )
        book_velocity = self._nanmean_or_default(
            [
                abs(self._nan_if_none(self._to_float(event_state_row.get("up_book_velocity")))),
                abs(self._nan_if_none(self._to_float(event_state_row.get("down_book_velocity")))),
            ],
            default=0.0,
        )
        spot_vol_multiplier = self._to_float(event_state_row.get("spot_vol_multiplier")) or 1.0

        return SimulationMarketState(
            spot_price=self._to_float(event_state_row.get("spot_price")),
            reference_spot_price=self._to_float(event_state_row.get("reference_spot_price")),
            spot_volatility_per_sqrt_second=self._to_float(event_state_row.get("volatility_per_sqrt_second"))
            or self.config.fallback_spot_volatility_per_sqrt_second,
            liquidity_depth=liquidity_depth,
            book_velocity=book_velocity,
            spot_vol_multiplier=spot_vol_multiplier,
        )

    def _has_pricing_inputs(self, row: Dict[str, Any]) -> bool:
        seconds_to_end = self._to_float(row.get("seconds_to_end"))
        spot_price = self._to_float(row.get("spot_price"))
        reference_spot_price = self._to_float(row.get("reference_spot_price"))
        return bool(
            seconds_to_end is not None
            and np.isfinite(seconds_to_end)
            and seconds_to_end >= 0
            and spot_price is not None
            and np.isfinite(spot_price)
            and spot_price > 0
            and reference_spot_price is not None
            and np.isfinite(reference_spot_price)
            and reference_spot_price > 0
        )

    def _row_market_signal(self, row: Optional[Dict[str, Any]], *columns: str) -> Optional[float]:
        if row is None:
            return None
        for column in columns:
            value = self._to_float(row.get(column))
            if value is not None:
                return value
        return None

    @staticmethod
    def _nan_if_none(value: Optional[float]) -> float:
        if value is None:
            return np.nan
        try:
            value_float = float(value)
        except (TypeError, ValueError):
            return np.nan
        return value_float if np.isfinite(value_float) else np.nan

    @staticmethod
    def _nanmean_or_default(values: Iterable[float], default: float = 0.0) -> float:
        array = np.asarray(list(values), dtype=float)
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            return float(default)
        return float(np.mean(finite))

    @staticmethod
    def _nanmin_or_default(values: Iterable[float], default: float = 0.0) -> float:
        array = np.asarray(list(values), dtype=float)
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            return float(default)
        return float(np.min(finite))

    def _to_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            value_float = float(value)
        except (TypeError, ValueError):
            return None
        return value_float if np.isfinite(value_float) else None

    @staticmethod
    def _snapshot_key(row: Dict[str, Any]) -> Optional[tuple[str, str]]:
        event_slug = str(row.get("event_slug", "")).strip()
        snapshot_time = str(row.get("collected_at") or row.get("book_timestamp") or "").strip()
        if not event_slug or not snapshot_time:
            return None
        return (event_slug, snapshot_time)

    def _group_by_snapshot(self, rows: Iterable[Dict[str, Any]]) -> Dict[tuple[str, str], List[Dict[str, Any]]]:
        grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            snapshot_key = self._snapshot_key(row)
            if snapshot_key is None:
                continue
            grouped[snapshot_key].append(row)
        return grouped

    @staticmethod
    def _iter_with_progress(
        items: List[Any],
        *,
        show_progress: bool,
        description: str,
    ):
        if not show_progress:
            return items
        if _tqdm is not None:
            return _tqdm(items, total=len(items), desc=description)
        return RealTimeMispricingDetector._simple_progress(items, description=description)

    def _estimated_batch_rollout_steps(self, rows: List[Dict[str, Any]]) -> int:
        if not rows:
            return 0
        if self.config.rollout_horizon_seconds > 0.0:
            step_seconds = float(self.config.rollout_horizon_seconds)
        else:
            step_seconds = float(self.config.simulation_dt_seconds)

        if step_seconds <= 0.0:
            return len(rows)

        seconds_to_end = [
            self._to_float(row.get("seconds_to_end"))
            for row in rows
        ]
        valid_seconds = np.asarray(
            [value for value in seconds_to_end if value is not None and np.isfinite(value) and value > 0.0],
            dtype=float,
        )
        if valid_seconds.size == 0:
            return len(rows)
        return int(np.ceil(valid_seconds / step_seconds).sum())

    @staticmethod
    def _progress_manager(*, total: int, description: str, enabled: bool):
        if not enabled:
            return nullcontext(None)
        if _tqdm is not None:
            return _tqdm(total=total, desc=description)
        return RealTimeMispricingDetector._SimpleProgressBar(total=total, description=description)

    @staticmethod
    def _simple_progress(items: List[Any], *, description: str):
        total = len(items)
        if total == 0:
            return iter(())

        def _generator():
            update_every = max(1, total // 50)
            for index, item in enumerate(items, start=1):
                if index == 1 or index % update_every == 0 or index == total:
                    print(f"{description}: {index}/{total}", file=sys.stderr, flush=True)
                yield item

        return _generator()

    class _SimpleProgressBar:
        def __init__(self, *, total: int, description: str) -> None:
            self.total = max(int(total), 1)
            self.description = description
            self.current = 0
            self.update_every = max(1, self.total // 50)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            if self.current < self.total:
                print(f"{self.description}: {min(self.current, self.total)}/{self.total}", file=sys.stderr, flush=True)

        def update(self, increment: int = 1) -> None:
            self.current = min(self.total, self.current + max(int(increment), 0))
            if self.current == 1 or self.current % self.update_every == 0 or self.current >= self.total:
                print(f"{self.description}: {self.current}/{self.total}", file=sys.stderr, flush=True)

    @staticmethod
    def _looks_like_event_state(row: Dict[str, Any]) -> bool:
        return "up_best_bid" in row or "down_best_bid" in row

    @staticmethod
    def _event_state_row_from_token_rows(event_rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        try:
            event_state = build_event_state_dataset(pd.DataFrame(event_rows))
        except Exception:
            return None
        if event_state.empty:
            return None
        return event_state.iloc[0].to_dict()
