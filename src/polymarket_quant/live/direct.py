from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
import yaml

from polymarket_quant.ingestion.client import PolymarketRESTClient
from polymarket_quant.ingestion.pipeline import IngestionPipeline
from polymarket_quant.ingestion.spot import BinanceSpotPriceClient, BaseSpotPriceClient, SpotFetchError
from polymarket_quant.live.capture import DEFAULT_LIVE_STATE_OUTPUT_DIR
from polymarket_quant.state import LatentMarkovStateBuilder, build_event_state_dataset
from polymarket_quant.state.dataset import build_market_state_rows, finalize_market_state_rows
from polymarket_quant.utils.logger import get_logger

try:
    from windowing import resolve_active_window
except ModuleNotFoundError:
    from scripts.windowing import resolve_active_window

logger = get_logger(__name__)

DEFAULT_DIRECT_LIVE_CONFIG_PATH = "configs/base.yaml"
DEFAULT_DIRECT_EVENT_SLUG_PREFIXES = ("btc-updown-5m", "eth-updown-5m")
DEFAULT_DIRECT_SERIES_SLUGS = ("btc-up-or-down-5m", "eth-up-or-down-5m")


@dataclass(frozen=True)
class DirectLiveSourceConfig:
    config_path: str = DEFAULT_DIRECT_LIVE_CONFIG_PATH
    output_dir: str = DEFAULT_LIVE_STATE_OUTPUT_DIR
    event_duration_seconds: int = 300
    event_slug_prefixes: tuple[str, ...] = DEFAULT_DIRECT_EVENT_SLUG_PREFIXES
    series_slugs: tuple[str, ...] = DEFAULT_DIRECT_SERIES_SLUGS
    spot_tolerance_seconds: float = 2.0


class DirectLiveEventStateSource:
    """Single-process live source: fetch market data, build state, emit new event rows."""

    def __init__(
        self,
        config: DirectLiveSourceConfig | None = None,
        *,
        polymarket_client: PolymarketRESTClient | None = None,
        spot_client: BaseSpotPriceClient | None = None,
    ) -> None:
        self.config = config or DirectLiveSourceConfig()
        with open(self.config.config_path, "r", encoding="utf-8") as handle:
            self._repo_config = yaml.safe_load(handle)

        api_config = self._repo_config.get("api", {})
        data_config = self._repo_config.get("data", {})
        self._spot_products = self._repo_config.get("spot", {}).get("products", {"BTC": "BTCUSDT", "ETH": "ETHUSDT"})

        self.polymarket_client = polymarket_client or PolymarketRESTClient(
            gamma_url=api_config["gamma_url"],
            clob_url=api_config["clob_url"],
        )
        self.spot_client = spot_client or BinanceSpotPriceClient(base_url=api_config.get("binance_url", "https://api.binance.com"))
        self.pipeline = IngestionPipeline(
            client=self.polymarket_client,
            raw_dir=data_config["raw_dir"],
            processed_dir=data_config["processed_dir"],
        )

        self._current_window_start: datetime | None = None
        self._current_window_end: datetime | None = None
        self._current_window_slugs: list[str] = []
        self._event_detail_cache: dict[str, dict[str, Any]] = {}
        self._seen_keys: set[str] = set()
        self._state_builder = LatentMarkovStateBuilder()
        self.last_poll_metrics: dict[str, float] = _empty_poll_metrics()

    def poll_new_rows(self) -> list[dict[str, Any]]:
        window = resolve_active_window(
            event_slug_prefixes=list(self.config.event_slug_prefixes),
            event_duration_seconds=self.config.event_duration_seconds,
            now=datetime.now(timezone.utc),
        )
        if self._current_window_start != window.start:
            self._reset_window(window)

        spot_started = perf_counter()
        batch_spot = self._fetch_spot_batch(window.event_slugs, window.start, window.end)
        spot_fetch_ms = (perf_counter() - spot_started) * 1000.0
        orderbook_started = perf_counter()
        batch_summary, batch_levels = self._fetch_orderbook_batch(window.event_slugs)
        orderbook_fetch_ms = (perf_counter() - orderbook_started) * 1000.0
        self.last_poll_metrics = {
            "spot_fetch_ms": spot_fetch_ms,
            "orderbook_fetch_ms": orderbook_fetch_ms,
            "market_state_ms": 0.0,
            "event_state_ms": 0.0,
        }
        if not batch_summary or not batch_spot:
            return []

        market_state_started = perf_counter()
        reference_prices_by_event = self._reference_prices_for_current_window()
        raw_market_state = build_market_state_rows(
            orderbooks=pd.DataFrame(batch_summary),
            spot=pd.DataFrame(batch_spot),
            orderbook_levels=pd.DataFrame(batch_levels),
            state_builder=self._state_builder,
            spot_tolerance_seconds=self.config.spot_tolerance_seconds,
            event_duration_seconds=float(self.config.event_duration_seconds),
            reference_prices_by_event=reference_prices_by_event,
        )
        market_state = finalize_market_state_rows(
            raw_market_state,
            fallback_volatility_per_sqrt_second=self._state_builder.config.fallback_volatility_per_sqrt_second,
        )
        market_state_ms = (perf_counter() - market_state_started) * 1000.0
        event_state_started = perf_counter()
        event_state = build_event_state_dataset(market_state)
        event_state_ms = (perf_counter() - event_state_started) * 1000.0
        self.last_poll_metrics = {
            "spot_fetch_ms": spot_fetch_ms,
            "orderbook_fetch_ms": orderbook_fetch_ms,
            "market_state_ms": market_state_ms,
            "event_state_ms": event_state_ms,
        }
        self._write_latest_artifacts(market_state, event_state)

        if event_state.empty:
            return []
        event_state = event_state.sort_values(["event_slug", "collected_at"]).reset_index(drop=True)
        keys = event_state["event_slug"].astype(str) + "|" + event_state["collected_at"].astype(str)
        new_mask = ~keys.isin(self._seen_keys)
        if not new_mask.any():
            return []
        self._seen_keys.update(keys.loc[new_mask].tolist())
        return event_state.loc[new_mask].to_dict("records")

    def _reset_window(self, window) -> None:
        self._current_window_start = window.start
        self._current_window_end = window.end
        self._current_window_slugs = list(window.event_slugs)
        self._event_detail_cache = {}
        self._seen_keys = set()
        logger.info(
            "Started new direct-live window %s -> %s for %s",
            window.start.isoformat(),
            window.end.isoformat(),
            window.event_slugs,
        )

    def _fetch_orderbook_batch(self, event_slugs: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        _, level_rows, summary_rows = self.pipeline.collect_crypto_5m_orderbooks_once(
            series_slugs=list(self.config.series_slugs),
            event_limit=len(event_slugs),
            event_slug_prefixes=list(self.config.event_slug_prefixes),
            event_slugs=event_slugs,
            event_details_by_slug=self._event_detail_cache,
        )
        return summary_rows, level_rows

    def _fetch_spot_batch(self, event_slugs: list[str], window_start: datetime, window_end: datetime) -> list[dict[str, Any]]:
        event_slug_map = _event_slug_by_asset(event_slugs, self._spot_products)
        rows: list[dict[str, Any]] = []
        for asset, product_id in self._spot_products.items():
            try:
                tick = self.spot_client.fetch_spot_ticker(asset=asset, product_id=product_id)
            except SpotFetchError as exc:
                logger.warning("Failed to fetch direct-live spot tick for %s: %s", asset, exc)
                continue
            tick["event_slug"] = event_slug_map.get(asset)
            tick["market_window_start"] = window_start.isoformat()
            tick["market_window_end"] = window_end.isoformat()
            rows.append(tick)
        return rows

    def _write_latest_artifacts(self, market_state: pd.DataFrame, event_state: pd.DataFrame) -> None:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        market_state.to_parquet(output_dir / "live_market_state_latest.parquet", index=False)
        event_state.to_parquet(output_dir / "live_event_state_latest.parquet", index=False)

    def _reference_prices_for_current_window(self) -> dict[str, dict[str, Any]]:
        references: dict[str, dict[str, Any]] = {}
        for event_slug in self._current_window_slugs:
            reference_price = self._state_builder.reference_spot_prices.get(event_slug)
            if reference_price is None:
                continue
            references[event_slug] = {
                "price": reference_price,
                "source": self._state_builder.reference_sources.get(event_slug, "first_observed"),
            }
        return references


def _event_slug_by_asset(event_slugs: list[str], spot_products: dict[str, str]) -> dict[str, str]:
    slugs_by_asset: dict[str, str] = {}
    for event_slug in event_slugs:
        for asset in spot_products:
            if event_slug.startswith(str(asset).lower()):
                slugs_by_asset[str(asset).upper()] = event_slug
    return slugs_by_asset


def _empty_poll_metrics() -> dict[str, float]:
    return {
        "spot_fetch_ms": 0.0,
        "orderbook_fetch_ms": 0.0,
        "market_state_ms": 0.0,
        "event_state_ms": 0.0,
    }
