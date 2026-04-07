"""Real-time mispricing detection for BTC/ETH 5-minute binary markets."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Deque, Dict, Iterable, List, Optional

import numpy as np

from polymarket_quant.execution.market_maker import BinaryMarketMaker
from polymarket_quant.execution.toxicity import ToxicityMonitor
from polymarket_quant.pricing import (
    ParticleFilter,
    estimate_importance_sampled_probability,
    estimate_monte_carlo_probability,
    estimate_stratified_probability,
)


@dataclass(frozen=True)
class MispricingDetectorConfig:
    """Configuration for live binary-market mispricing detection."""

    min_edge: float = 0.02
    max_toxicity: float = 0.7
    min_depth: float = 0.0
    drift: float = 0.0
    fallback_volatility_per_sqrt_second: float = 0.0005
    volatility_window_seconds: float = 120.0
    event_duration_seconds: float = 300.0
    pricing_method: str = "monte_carlo"
    n_samples: int = 5_000
    n_strata: int = 10
    importance_proposal_shift: float = 1.0
    use_particle_filter: bool = True
    particle_count: int = 500
    seed: Optional[int] = 42


class RealTimeMispricingDetector:
    """Estimate fair token prices and flag executable Polymarket mispricings."""

    def __init__(
        self,
        config: Optional[MispricingDetectorConfig] = None,
        toxicity_monitor: Optional[ToxicityMonitor] = None,
        market_maker: Optional[BinaryMarketMaker] = None,
    ) -> None:
        self.config = config or MispricingDetectorConfig()
        self.toxicity_monitor = toxicity_monitor or ToxicityMonitor()
        self.market_maker = market_maker or BinaryMarketMaker()
        self.reference_spot_prices: Dict[str, float] = {}
        self.reference_sources: Dict[str, str] = {}
        self.spot_history: Dict[str, Deque[tuple[datetime, float]]] = defaultdict(deque)
        self.probability_history: Dict[str, List[float]] = defaultdict(list)
        self.last_spreads: Dict[str, float] = {}

    def detect(
        self,
        orderbook_summary_rows: Iterable[Dict[str, Any]],
        spot_ticks: Dict[str, Dict[str, Any]],
        reference_prices_by_event: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Return one signal row per Up/Down token snapshot."""
        reference_prices_by_event = reference_prices_by_event or {}
        rows = [row for row in orderbook_summary_rows if row]
        for asset, spot_tick in spot_ticks.items():
            self._update_spot_history(asset, spot_tick)

        signals: List[Dict[str, Any]] = []
        for event_slug, event_rows in self._group_by(rows, "event_slug").items():
            if not event_slug or not event_rows:
                continue

            asset = str(event_rows[0].get("asset", ""))
            spot_tick = spot_ticks.get(asset, {})
            spot_price = self._to_float(spot_tick.get("price"))
            if spot_price is None or spot_price <= 0:
                continue

            timestamp = self._row_timestamp(event_rows[0], spot_tick)
            seconds_to_end = self._seconds_to_end(event_rows[0], timestamp)
            if seconds_to_end < 0:
                continue

            reference_price, reference_source = self._reference_price(
                event_slug=event_slug,
                spot_price=spot_price,
                reference_payload=reference_prices_by_event.get(event_slug),
            )
            volatility = self._realized_volatility_per_sqrt_second(asset, timestamp)
            pricing = self._estimate_up_probability(
                event_slug=event_slug,
                spot_price=spot_price,
                reference_price=reference_price,
                volatility=volatility,
                seconds_to_end=seconds_to_end,
            )
            fair_up_probability = pricing["fair_up_probability"]
            toxicity_score = self._toxicity_score(asset, event_rows, timestamp, volatility)

            for row in event_rows:
                signals.append(
                    self._signal_row(
                        row=row,
                        spot_tick=spot_tick,
                        spot_price=spot_price,
                        reference_price=reference_price,
                        reference_source=reference_source,
                        seconds_to_end=seconds_to_end,
                        volatility=volatility,
                        fair_up_probability=fair_up_probability,
                        toxicity_score=toxicity_score,
                        pricing=pricing,
                    )
                )

        return signals

    def _estimate_up_probability(
        self,
        event_slug: str,
        spot_price: float,
        reference_price: float,
        volatility: float,
        seconds_to_end: float,
    ) -> Dict[str, Any]:
        horizon = max(seconds_to_end, 0.0)
        volatility = max(volatility, 0.0)
        method = self.config.pricing_method

        if method == "importance_sampling":
            proposal_shift = self._proposal_shift(spot_price, reference_price)
            result = estimate_importance_sampled_probability(
                initial_value=spot_price,
                threshold=reference_price,
                drift=self.config.drift,
                volatility=volatility,
                horizon=horizon,
                n_samples=self.config.n_samples,
                proposal_shift=proposal_shift,
                seed=self.config.seed,
            )
        elif method == "stratified":
            result = estimate_stratified_probability(
                initial_value=spot_price,
                threshold=reference_price,
                drift=self.config.drift,
                volatility=volatility,
                horizon=horizon,
                n_strata=self.config.n_strata,
                samples_per_stratum=max(1, self.config.n_samples // self.config.n_strata),
                seed=self.config.seed,
            )
        else:
            result = estimate_monte_carlo_probability(
                initial_value=spot_price,
                threshold=reference_price,
                drift=self.config.drift,
                volatility=volatility,
                horizon=horizon,
                n_samples=self.config.n_samples,
                seed=self.config.seed,
            )
            method = "monte_carlo"

        raw_probability = float(np.clip(result.probability, 0.0, 1.0))
        fair_up_probability = raw_probability
        if self.config.use_particle_filter:
            self.probability_history[event_slug].append(raw_probability)
            particle_filter = ParticleFilter(n_particles=self.config.particle_count, seed=self.config.seed)
            filtered = particle_filter.filter(np.asarray(self.probability_history[event_slug]), initial_probability=0.5)
            fair_up_probability = float(filtered.probabilities[-1])

        return {
            "pricing_method": method,
            "raw_up_probability": raw_probability,
            "fair_up_probability": fair_up_probability,
            "pricing_standard_error": result.standard_error,
            "pricing_n_samples": result.n_samples,
        }

    def _signal_row(
        self,
        row: Dict[str, Any],
        spot_tick: Dict[str, Any],
        spot_price: float,
        reference_price: float,
        reference_source: str,
        seconds_to_end: float,
        volatility: float,
        fair_up_probability: float,
        toxicity_score: float,
        pricing: Dict[str, Any],
    ) -> Dict[str, Any]:
        outcome_name = str(row.get("outcome_name", ""))
        fair_token_price = fair_up_probability if outcome_name.lower() == "up" else 1.0 - fair_up_probability
        best_bid = self._to_float(row.get("best_bid"))
        best_ask = self._to_float(row.get("best_ask"))
        bid_depth = self._to_float(row.get("bid_depth")) or 0.0
        ask_depth = self._to_float(row.get("ask_depth")) or 0.0

        buy_edge = (fair_token_price - best_ask) if best_ask is not None else None
        sell_edge = (best_bid - fair_token_price) if best_bid is not None else None
        signal = self._classify_signal(buy_edge, sell_edge, bid_depth, ask_depth, toxicity_score)
        executable_edge = buy_edge if signal == "BUY" else sell_edge if signal == "SELL" else None
        mm_bid, mm_ask = self.market_maker.get_quotes(
            p_fair=fair_token_price,
            inventory=0,
            sigma=volatility * np.sqrt(max(self.config.event_duration_seconds, 1.0)),
            ttr=min(1.0, seconds_to_end / max(self.config.event_duration_seconds, 1.0)),
            toxicity_score=toxicity_score,
        )

        return {
            **row,
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
            "fair_up_probability": fair_up_probability,
            "fair_token_price": fair_token_price,
            "buy_edge": buy_edge,
            "sell_edge": sell_edge,
            "executable_edge": executable_edge,
            "toxicity_score": toxicity_score,
            "signal": signal,
            "mm_bid": float(mm_bid),
            "mm_ask": float(mm_ask),
            **pricing,
        }

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

    def _classify_signal(
        self,
        buy_edge: Optional[float],
        sell_edge: Optional[float],
        bid_depth: float,
        ask_depth: float,
        toxicity_score: float,
    ) -> str:
        if toxicity_score > self.config.max_toxicity:
            return "HOLD"
        if buy_edge is not None and buy_edge >= self.config.min_edge and ask_depth >= self.config.min_depth:
            return "BUY"
        if sell_edge is not None and sell_edge >= self.config.min_edge and bid_depth >= self.config.min_depth:
            return "SELL"
        return "HOLD"

    def _update_spot_history(self, asset: str, spot_tick: Dict[str, Any]) -> None:
        price = self._to_float(spot_tick.get("price"))
        if price is None or price <= 0:
            return

        timestamp = self._parse_datetime(spot_tick.get("collected_at")) or datetime.now(timezone.utc)
        history = self.spot_history[asset]
        history.append((timestamp, price))
        cutoff_seconds = self.config.volatility_window_seconds
        while history and (timestamp - history[0][0]).total_seconds() > cutoff_seconds:
            history.popleft()

    def _realized_volatility_per_sqrt_second(self, asset: str, timestamp: datetime) -> float:
        history = list(self.spot_history.get(asset, []))
        if len(history) < 3:
            return self.config.fallback_volatility_per_sqrt_second

        times = np.asarray([point[0].timestamp() for point in history], dtype=float)
        prices = np.asarray([point[1] for point in history], dtype=float)
        log_returns = np.diff(np.log(prices))
        elapsed = np.diff(times)
        valid = elapsed > 0
        if not np.any(valid):
            return self.config.fallback_volatility_per_sqrt_second

        variance_per_second = float(np.sum(log_returns[valid] ** 2) / np.sum(elapsed[valid]))
        if not np.isfinite(variance_per_second) or variance_per_second <= 0:
            return self.config.fallback_volatility_per_sqrt_second
        return float(np.sqrt(variance_per_second))

    def _toxicity_score(
        self,
        asset: str,
        event_rows: List[Dict[str, Any]],
        timestamp: datetime,
        volatility: float,
    ) -> float:
        jump_z = self._latest_jump_z(asset, volatility)
        spread_widening = self._spread_widening(event_rows)
        fallback = max(self.config.fallback_volatility_per_sqrt_second, 1e-12)
        vol_surge = max(0.0, (volatility / fallback) - 1.0)
        return self.toxicity_monitor.calculate_score(jump_z, spread_widening, vol_surge)

    def _latest_jump_z(self, asset: str, volatility: float) -> float:
        history = list(self.spot_history.get(asset, []))
        if len(history) < 2 or volatility <= 0:
            return 0.0
        prev_time, prev_price = history[-2]
        latest_time, latest_price = history[-1]
        elapsed = max((latest_time - prev_time).total_seconds(), 1e-9)
        log_return = float(np.log(latest_price / prev_price))
        return log_return / (volatility * np.sqrt(elapsed))

    def _spread_widening(self, event_rows: List[Dict[str, Any]]) -> float:
        increases = []
        for row in event_rows:
            token_id = str(row.get("token_id", ""))
            spread = self._to_float(row.get("spread"))
            if not token_id or spread is None:
                continue
            previous = self.last_spreads.get(token_id)
            if previous is not None:
                increases.append(max(0.0, spread - previous))
            self.last_spreads[token_id] = spread
        return float(np.mean(increases)) if increases else 0.0

    def _proposal_shift(self, spot_price: float, reference_price: float) -> float:
        direction = 1.0 if reference_price >= spot_price else -1.0
        return direction * abs(self.config.importance_proposal_shift)

    def _seconds_to_end(self, row: Dict[str, Any], timestamp: datetime) -> float:
        end_time = self._parse_datetime(row.get("market_end_time"))
        if end_time is None:
            return 0.0
        return max(0.0, (end_time - timestamp).total_seconds())

    def _row_timestamp(self, row: Dict[str, Any], spot_tick: Dict[str, Any]) -> datetime:
        return (
            self._parse_datetime(row.get("collected_at"))
            or self._parse_datetime(spot_tick.get("collected_at"))
            or datetime.now(timezone.utc)
        )

    def _parse_datetime(self, value: Any) -> Optional[datetime]:
        if value is None:
            return None
        if isinstance(value, datetime):
            parsed = value
        elif isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
        else:
            return None

        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _to_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _group_by(self, rows: Iterable[Dict[str, Any]], key: str) -> Dict[Any, List[Dict[str, Any]]]:
        grouped: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[row.get(key)].append(row)
        return grouped
