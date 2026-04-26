"""Latent-state Markov helpers for binary-market state construction."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import NormalDist
from typing import Any, Deque, Dict, Iterable, List, Optional

import numpy as np

from polymarket_quant.utils.math import logit, sigmoid


@dataclass(frozen=True)
class LatentMarkovStateConfig:
    """Configuration for current-time latent-state estimation."""

    fallback_volatility_per_sqrt_second: float = 0.0005
    volatility_window_seconds: float = 120.0
    anchor_weight: float = 0.35
    observation_std: float = 0.03
    observation_spread_scale: float = 4.0
    observation_disagreement_scale: float = 2.0
    observation_depth_penalty_scale: float = 1.0
    observation_staleness_penalty_scale: float = 0.25
    event_duration_seconds: float = 300.0


@dataclass(frozen=True)
class LatentMarkovStateEstimate:
    """Current latent-state estimate for a single event snapshot."""

    fundamental_up_probability: float
    market_implied_up_probability: Optional[float]
    latent_up_probability: float
    latent_logit_probability: float
    observation_variance: Optional[float]


class LatentMarkovStateBuilder:
    """Construct observation and latent state rows from aligned market data."""

    def __init__(
        self,
        config: Optional[LatentMarkovStateConfig] = None,
    ) -> None:
        self.config = config or LatentMarkovStateConfig()
        self.reference_spot_prices: Dict[str, float] = {}
        self.reference_sources: Dict[str, str] = {}
        self.spot_history: Dict[str, Deque[tuple[datetime, float]]] = defaultdict(deque)
        self._standard_normal = NormalDist()

    def build(
        self,
        orderbook_summary_rows: Iterable[Dict[str, Any]],
        spot_ticks: Dict[str, Dict[str, Any]],
        reference_prices_by_event: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Return state-enriched rows for each token snapshot."""
        reference_prices_by_event = reference_prices_by_event or {}
        rows = [row for row in orderbook_summary_rows if row]
        for asset, spot_tick in spot_ticks.items():
            self._update_spot_history(asset, spot_tick)

        state_rows: List[Dict[str, Any]] = []
        for event_slug, event_rows in self._group_by(rows, "event_slug").items():
            if not event_slug or not event_rows:
                continue

            # We estimate one shared latent state per event snapshot, then
            # attach that state back to each token-side row in the event.
            asset = str(event_rows[0].get("asset", ""))
            spot_tick = spot_ticks.get(asset, {})
            spot_price = self._to_float(spot_tick.get("price"))
            if spot_price is None or spot_price <= 0:
                continue

            timestamp = self._row_timestamp(event_rows[0], spot_tick)
            if timestamp is None:
                continue
            seconds_to_end = self._seconds_to_end(event_rows[0], timestamp)
            if seconds_to_end < 0:
                continue

            reference_price, reference_source = self._reference_price(
                event_slug=event_slug,
                spot_price=spot_price,
                reference_payload=reference_prices_by_event.get(event_slug),
            )
            # The latent update combines three ingredients:
            # 1. a spot/reference-based fundamental anchor
            # 2. a market-implied observation from the orderbook
            # 3. the previous latent state propagated forward in time
            volatility = self._realized_volatility_per_sqrt_second(asset, timestamp)
            market_implied_up_probability = self._market_implied_up_probability(event_rows)
            estimate = self._estimate_latent_state(
                event_slug=event_slug,
                timestamp=timestamp,
                spot_price=spot_price,
                reference_price=reference_price,
                volatility=volatility,
                seconds_to_end=seconds_to_end,
                market_implied_up_probability=market_implied_up_probability,
                event_rows=event_rows,
            )

            for row in event_rows:
                state_rows.append(
                    {
                        **row,
                        "state_timestamp": timestamp.isoformat(),
                        "book_age_seconds": self._book_age_seconds(row, timestamp),
                        "spot_source": spot_tick.get("source"),
                        "spot_product_id": spot_tick.get("product_id"),
                        "spot_exchange_time": spot_tick.get("exchange_time"),
                        "spot_price": spot_price,
                        "spot_bid": spot_tick.get("bid"),
                        "spot_ask": spot_tick.get("ask"),
                        "reference_spot_price": reference_price,
                        "reference_source": reference_source,
                        "spot_return_since_reference": (spot_price / reference_price) - 1.0,
                        "seconds_to_end": seconds_to_end,
                        "volatility_per_sqrt_second": volatility,
                        "market_implied_up_probability": estimate.market_implied_up_probability,
                        "fundamental_up_probability": estimate.fundamental_up_probability,
                        "latent_up_probability": estimate.latent_up_probability,
                        "latent_logit_probability": estimate.latent_logit_probability,
                        "state_observation_variance": estimate.observation_variance,
                    }
                )

        return state_rows

    def _estimate_latent_state(
        self,
        event_slug: str,
        timestamp: datetime,
        spot_price: float,
        reference_price: float,
        volatility: float,
        seconds_to_end: float,
        market_implied_up_probability: Optional[float],
        event_rows: List[Dict[str, Any]],
    ) -> LatentMarkovStateEstimate:
        # The fundamental probability is the model's external anchor: given
        # spot, reference, volatility, and time-to-end, what is the implied
        # chance that the binary event resolves Up?
        fundamental_probability = self._spot_distance_probability(
            spot_price=spot_price,
            reference_price=reference_price,
            volatility=volatility,
            seconds_to_end=seconds_to_end,
        )
        anchor_logit = float(logit(np.clip(fundamental_probability, 1e-6, 1 - 1e-6)))
        observation_variance = None
        latent_logit = anchor_logit
        if market_implied_up_probability is not None:
            observation_logit = float(logit(np.clip(market_implied_up_probability, 1e-6, 1 - 1e-6)))
            observation_variance = self._observation_variance(event_rows, timestamp)
            anchor_variance = self._anchor_variance(observation_variance)
            # Current latent state is a present-time fusion of the fundamental
            # anchor and the current market observation. Transition dynamics are
            # learned later in the transition kernel, not hard-coded here.
            latent_logit = self._gaussian_update(
                prior_mean=anchor_logit,
                prior_variance=anchor_variance,
                observation_mean=observation_logit,
                observation_variance=observation_variance,
            )

        latent_probability = float(sigmoid(latent_logit))
        return LatentMarkovStateEstimate(
            fundamental_up_probability=fundamental_probability,
            market_implied_up_probability=market_implied_up_probability,
            latent_up_probability=latent_probability,
            latent_logit_probability=latent_logit,
            observation_variance=observation_variance,
        )

    def _market_implied_up_probability(self, event_rows: List[Dict[str, Any]]) -> Optional[float]:
        up_mid = None
        down_mid = None
        for row in event_rows:
            outcome = str(row.get("outcome_name", "")).lower()
            mid = self._row_mid_probability(row)
            if mid is None:
                continue
            if outcome == "up":
                up_mid = mid
            elif outcome == "down":
                down_mid = mid

        # We reconcile the two-sided binary book by averaging the direct Up
        # quote with the Down quote mapped back into Up-probability space.
        candidates = []
        if up_mid is not None:
            candidates.append(up_mid)
        if down_mid is not None:
            candidates.append(1.0 - down_mid)
        if not candidates:
            return None
        return float(np.clip(np.mean(candidates), 1e-6, 1 - 1e-6))

    def _row_mid_probability(self, row: Dict[str, Any]) -> Optional[float]:
        mid = self._to_float(row.get("mid_price"))
        if mid is not None:
            return float(np.clip(mid, 1e-6, 1 - 1e-6))
        best_bid = self._to_float(row.get("best_bid"))
        best_ask = self._to_float(row.get("best_ask"))
        if best_bid is not None and best_ask is not None:
            return float(np.clip((best_bid + best_ask) / 2.0, 1e-6, 1 - 1e-6))
        if best_bid is not None:
            return float(np.clip(best_bid, 1e-6, 1 - 1e-6))
        if best_ask is not None:
            return float(np.clip(best_ask, 1e-6, 1 - 1e-6))
        return None

    def _gaussian_update(
        self,
        prior_mean: float,
        prior_variance: float,
        observation_mean: float,
        observation_variance: float,
    ) -> float:
        prior_precision = 1.0 / max(prior_variance, 1e-9)
        observation_precision = 1.0 / max(observation_variance, 1e-9)
        return float(
            ((prior_mean * prior_precision) + (observation_mean * observation_precision))
            / (prior_precision + observation_precision)
        )

    def _anchor_variance(self, observation_variance: float) -> float:
        anchor_weight = float(np.clip(self.config.anchor_weight, 1e-3, 1.0 - 1e-3))
        return float(max(observation_variance * (1.0 - anchor_weight) / anchor_weight, 1e-6))

    def _observation_variance(self, event_rows: List[Dict[str, Any]], timestamp: datetime) -> float:
        base_variance = max(self.config.observation_std**2, 1e-6)

        logit_half_width_variances: List[float] = []
        observation_logits: List[float] = []
        displayed_depths: List[float] = []
        book_ages: List[float] = []

        for row in event_rows:
            interval = self._row_probability_interval(row)
            if interval is not None:
                lower, upper = interval
                lower_logit = float(logit(lower))
                upper_logit = float(logit(upper))
                observation_logits.append(0.5 * (lower_logit + upper_logit))
                logit_half_width_variances.append((0.5 * abs(upper_logit - lower_logit)) ** 2)

            depth = self._row_displayed_depth(row)
            if depth is not None and depth > 0:
                displayed_depths.append(depth)

            age = self._book_age_seconds(row, timestamp)
            if age is not None and age >= 0:
                book_ages.append(age)

        # Observation noise rises when:
        # - displayed quote intervals are wide
        # - Up and Down observations disagree in logit space
        # - displayed depth is thin
        # - the book is stale relative to the chosen timestamp
        quote_variance = (
            float(np.mean(logit_half_width_variances)) if logit_half_width_variances else base_variance
        )
        disagreement_variance = float(np.var(observation_logits)) if len(observation_logits) > 1 else 0.0

        min_displayed_depth = min(displayed_depths) if displayed_depths else 0.0
        depth_penalty = 1.0 + (
            self.config.observation_depth_penalty_scale
            / max(np.log1p(max(min_displayed_depth, 0.0)), 1.0)
        )

        mean_book_age = float(np.mean(book_ages)) if book_ages else 0.0
        staleness_penalty = 1.0 + (
            self.config.observation_staleness_penalty_scale * max(mean_book_age, 0.0)
        )

        variance = (
            base_variance
            + (self.config.observation_spread_scale * quote_variance)
            + (self.config.observation_disagreement_scale * disagreement_variance)
        ) * depth_penalty * staleness_penalty
        return float(max(variance, 1e-6))

    def _event_spread(self, event_rows: List[Dict[str, Any]]) -> float:
        spreads = []
        for row in event_rows:
            spread = self._to_float(row.get("spread"))
            if spread is None:
                best_bid = self._to_float(row.get("best_bid"))
                best_ask = self._to_float(row.get("best_ask"))
                if best_bid is not None and best_ask is not None:
                    spread = max(0.0, best_ask - best_bid)
            if spread is not None:
                spreads.append(spread)
        return float(np.mean(spreads)) if spreads else 0.0

    def _row_probability_interval(self, row: Dict[str, Any]) -> Optional[tuple[float, float]]:
        outcome = str(row.get("outcome_name", "")).lower()
        best_bid = self._to_float(row.get("best_bid"))
        best_ask = self._to_float(row.get("best_ask"))
        mid = self._row_mid_probability(row)

        lower: Optional[float]
        upper: Optional[float]
        if outcome == "up":
            lower = best_bid if best_bid is not None else mid
            upper = best_ask if best_ask is not None else mid
        elif outcome == "down":
            lower = (1.0 - best_ask) if best_ask is not None else ((1.0 - mid) if mid is not None else None)
            upper = (1.0 - best_bid) if best_bid is not None else ((1.0 - mid) if mid is not None else None)
        else:
            return None

        if lower is None and upper is None:
            return None
        if lower is None:
            lower = upper
        if upper is None:
            upper = lower

        lo = float(np.clip(min(lower, upper), 1e-6, 1.0 - 1e-6))
        hi = float(np.clip(max(lower, upper), 1e-6, 1.0 - 1e-6))
        return lo, hi

    def _row_displayed_depth(self, row: Dict[str, Any]) -> Optional[float]:
        bid_depth = self._to_float(row.get("bid_depth_top_5"))
        ask_depth = self._to_float(row.get("ask_depth_top_5"))
        if bid_depth is not None and ask_depth is not None:
            return max(min(bid_depth, ask_depth), 0.0)
        if bid_depth is not None:
            return max(bid_depth, 0.0)
        if ask_depth is not None:
            return max(ask_depth, 0.0)
        return None

    def _book_age_seconds(self, row: Dict[str, Any], timestamp: datetime) -> Optional[float]:
        book_timestamp = row.get("book_timestamp")
        if book_timestamp is None:
            return None
        try:
            book_dt = datetime.fromisoformat(str(book_timestamp).replace("Z", "+00:00"))
        except ValueError:
            return None
        if book_dt.tzinfo is None:
            book_dt = book_dt.replace(tzinfo=timezone.utc)
        return max(0.0, (timestamp - book_dt).total_seconds())

    def _spot_distance_probability(
        self,
        spot_price: float,
        reference_price: float,
        volatility: float,
        seconds_to_end: float,
    ) -> float:
        # This is the "fundamental" binary probability: under the local
        # diffusion assumptions, how likely is the terminal spot to finish
        # above the event's reference price?
        if reference_price <= 0 or spot_price <= 0:
            return 0.5
        horizon = max(seconds_to_end, 0.0)
        sigma = max(volatility, 0.0)

        # Degenerate paths have a deterministic terminal value, so the binary
        # payoff probability collapses to the terminal threshold check.
        if horizon == 0.0 or sigma == 0.0:
            terminal_value = spot_price
            return 1.0 if terminal_value >= reference_price else 0.0

        sqrt_horizon = np.sqrt(horizon)
        denominator = sigma * sqrt_horizon
        if denominator <= 0:
            return 1.0 if spot_price >= reference_price else 0.0

        d2 = (
            np.log(spot_price / reference_price)
            - 0.5 * (sigma**2) * horizon
        ) / denominator
        return float(np.clip(self._standard_normal.cdf(float(d2)), 1e-6, 1.0 - 1e-6))

    def _reference_price(
        self,
        event_slug: str,
        spot_price: float,
        reference_payload: Optional[Dict[str, Any]],
    ) -> tuple[float, str]:
        payload_price = self._to_float((reference_payload or {}).get("price"))
        payload_source = str((reference_payload or {}).get("source", "external_reference"))
        if payload_price is not None and payload_price > 0:
            self.reference_spot_prices[event_slug] = payload_price
            self.reference_sources[event_slug] = payload_source
            return payload_price, payload_source

        reference_price = self.reference_spot_prices.setdefault(event_slug, spot_price)
        reference_source = self.reference_sources.setdefault(event_slug, "first_observed")
        return reference_price, reference_source

    def _update_spot_history(self, asset: str, spot_tick: Dict[str, Any]) -> None:
        timestamp = self._parse_timestamp(spot_tick.get("collected_at"))
        price = self._to_float(spot_tick.get("price"))
        if timestamp is None or price is None or price <= 0:
            return

        history = self.spot_history[asset]
        history.append((timestamp, price))
        cutoff = timestamp.timestamp() - self.config.volatility_window_seconds
        while history and history[0][0].timestamp() < cutoff:
            history.popleft()

    def _realized_volatility_per_sqrt_second(self, asset: str, timestamp: datetime) -> float:
        history = self.spot_history.get(asset)
        if not history or len(history) < 3:
            return self.config.fallback_volatility_per_sqrt_second

        prices = np.asarray([price for _, price in history], dtype=float)
        times = np.asarray([time.timestamp() for time, _ in history], dtype=float)
        log_returns = np.diff(np.log(prices))
        elapsed = np.diff(times)
        valid = elapsed > 0
        if not np.any(valid):
            return self.config.fallback_volatility_per_sqrt_second

        variance_per_second = np.sum((log_returns[valid] ** 2)) / np.sum(elapsed[valid])
        if not np.isfinite(variance_per_second) or variance_per_second <= 0:
            return self.config.fallback_volatility_per_sqrt_second
        return float(np.sqrt(variance_per_second))

    def _seconds_to_end(self, row: Dict[str, Any], timestamp: datetime) -> float:
        end_time = self._parse_timestamp(row.get("market_end_time"))
        if end_time is None:
            return 0.0
        return max(0.0, (end_time - timestamp).total_seconds())

    def _row_timestamp(self, row: Dict[str, Any], spot_tick: Dict[str, Any]) -> Optional[datetime]:
        return (
            self._parse_timestamp(row.get("collected_at"))
            or self._parse_timestamp(row.get("book_timestamp"))
            or self._parse_timestamp(spot_tick.get("collected_at"))
        )

    def _parse_timestamp(self, value: Any) -> Optional[datetime]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        try:
            text = str(value).replace("Z", "+00:00")
            timestamp = datetime.fromisoformat(text)
        except ValueError:
            return None
        return timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc)

    def _group_by(self, rows: Iterable[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row.get(key))].append(row)
        return grouped

    def _to_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric):
            return None
        return numeric
