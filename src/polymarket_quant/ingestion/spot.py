"""Spot price ingestion adapters for underlying crypto assets."""

from __future__ import annotations

import abc
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import requests

from polymarket_quant.utils.logger import get_logger

logger = get_logger(__name__)


class SpotFetchError(RuntimeError):
    """Raised when no usable spot snapshot can be fetched for an asset."""


class BaseSpotPriceClient(abc.ABC):
    """Abstract boundary for fetching underlying asset spot prices."""

    @abc.abstractmethod
    def fetch_spot_ticker(self, asset: str, product_id: str) -> Dict[str, Any]:
        """Fetch one spot ticker snapshot for an asset/product pair."""

    def fetch_spot_tickers(self, products_by_asset: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
        """Fetch spot ticker snapshots for multiple assets."""
        return {
            asset: self.fetch_spot_ticker(asset=asset, product_id=product_id)
            for asset, product_id in products_by_asset.items()
        }

    def fetch_reference_price(self, asset: str, product_id: str, reference_time: datetime) -> Dict[str, Any]:
        """Fetch an approximate event reference price for a given timestamp."""
        raise NotImplementedError


class BinanceSpotPriceClient(BaseSpotPriceClient):
    """Public Binance REST adapter for BTC/ETH spot snapshots."""

    def __init__(
        self,
        base_url: str = "https://api.binance.com",
        timeout: float = 10.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def fetch_spot_ticker(self, asset: str, product_id: str) -> Dict[str, Any]:
        """Fetch one Binance best-bid/best-ask snapshot and derive a mid spot price."""
        collected_at = datetime.now(timezone.utc).isoformat()
        url = f"{self.base_url}/api/v3/ticker/bookTicker"
        params = {"symbol": product_id}

        logger.info("Fetching Binance spot ticker for %s (%s)", asset, product_id)
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.RequestException as exc:
            raise SpotFetchError(f"Failed to fetch Binance spot ticker for {product_id}: {exc}") from exc

        if not isinstance(payload, dict):
            raise SpotFetchError(f"Failed to fetch Binance spot ticker for {product_id}: non-dict payload {payload!r}")

        bid = self._to_float(payload.get("bidPrice"))
        ask = self._to_float(payload.get("askPrice"))
        bid_size = self._to_float(payload.get("bidQty"))
        ask_size = self._to_float(payload.get("askQty"))

        if bid is None or ask is None:
            raise SpotFetchError(
                f"Failed to fetch Binance spot ticker for {product_id}: missing numeric bid/ask payload={payload!r}"
            )

        price = (bid + ask) / 2.0

        return {
            "asset": asset,
            "product_id": product_id,
            "source": "binance_book_ticker",
            "collected_at": collected_at,
            "exchange_time": payload.get("time"),
            "price": price,
            "bid": bid,
            "ask": ask,
            "size": bid_size if bid_size is not None else ask_size,
            "volume": None,
            "trade_id": None,
        }

    def fetch_reference_price(self, asset: str, product_id: str, reference_time: datetime) -> Dict[str, Any]:
        """Fetch the 1-minute Binance kline open closest to the event start timestamp."""
        reference_time = reference_time.astimezone(timezone.utc)
        start = reference_time.replace(second=0, microsecond=0)
        end = start + timedelta(seconds=60)
        url = f"{self.base_url}/api/v3/klines"
        params = {
            "symbol": product_id,
            "interval": "1m",
            "startTime": int(start.timestamp() * 1000),
            "endTime": int(end.timestamp() * 1000),
            "limit": 1,
        }

        logger.info("Fetching Binance reference kline for %s (%s)", asset, product_id)
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.RequestException as exc:
            logger.error("Failed to fetch Binance reference kline for %s: %s", product_id, exc)
            return {}

        if not isinstance(payload, list) or not payload:
            return {}

        target_ts_ms = reference_time.timestamp() * 1000.0
        candle = min(payload, key=lambda item: abs(float(item[0]) - target_ts_ms))
        if not isinstance(candle, list) or len(candle) < 5:
            return {}

        candle_start = datetime.fromtimestamp(float(candle[0]) / 1000.0, tz=timezone.utc)
        return {
            "asset": asset,
            "product_id": product_id,
            "source": "binance_1m_kline_open",
            "reference_time": reference_time.isoformat(),
            "candle_start_time": candle_start.isoformat(),
            "price": self._to_float(candle[1]),
            "low": self._to_float(candle[3]),
            "high": self._to_float(candle[2]),
            "close": self._to_float(candle[4]),
            "volume": self._to_float(candle[5]) if len(candle) > 5 else None,
        }

    def _to_float(self, value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
