"""Spot price ingestion adapters for underlying crypto assets."""

from __future__ import annotations

import abc
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import requests

from polymarket_quant.utils.logger import get_logger

logger = get_logger(__name__)


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


class CoinbaseSpotPriceClient(BaseSpotPriceClient):
    """Public Coinbase Exchange REST adapter for BTC/ETH spot tickers."""

    def __init__(self, base_url: str = "https://api.exchange.coinbase.com", timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def fetch_spot_ticker(self, asset: str, product_id: str) -> Dict[str, Any]:
        """Fetch Coinbase product ticker data and normalize numeric fields."""
        url = f"{self.base_url}/products/{product_id}/ticker"
        collected_at = datetime.now(timezone.utc).isoformat()

        logger.info("Fetching Coinbase spot ticker for %s (%s)", asset, product_id)
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.RequestException as exc:
            logger.error("Failed to fetch Coinbase ticker for %s: %s", product_id, exc)
            return {}

        if not isinstance(payload, dict):
            return {}

        return {
            "asset": asset,
            "product_id": product_id,
            "source": "coinbase",
            "collected_at": collected_at,
            "exchange_time": payload.get("time"),
            "price": self._to_float(payload.get("price")),
            "bid": self._to_float(payload.get("bid")),
            "ask": self._to_float(payload.get("ask")),
            "size": self._to_float(payload.get("size")),
            "volume": self._to_float(payload.get("volume")),
            "trade_id": payload.get("trade_id"),
        }

    def fetch_reference_price(self, asset: str, product_id: str, reference_time: datetime) -> Dict[str, Any]:
        """Fetch the 1-minute candle open closest to the event start timestamp."""
        reference_time = reference_time.astimezone(timezone.utc)
        start = reference_time.replace(second=0, microsecond=0)
        end = start + timedelta(seconds=60)
        url = f"{self.base_url}/products/{product_id}/candles"
        params = {
            "start": start.isoformat().replace("+00:00", "Z"),
            "end": end.isoformat().replace("+00:00", "Z"),
            "granularity": 60,
        }

        logger.info("Fetching Coinbase reference candle for %s (%s)", asset, product_id)
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.RequestException as exc:
            logger.error("Failed to fetch Coinbase reference candle for %s: %s", product_id, exc)
            return {}

        if not isinstance(payload, list) or not payload:
            return {}

        target_ts = reference_time.timestamp()
        candle = min(payload, key=lambda item: abs(float(item[0]) - target_ts))
        if not isinstance(candle, list) or len(candle) < 5:
            return {}

        candle_start = datetime.fromtimestamp(float(candle[0]), tz=timezone.utc)
        return {
            "asset": asset,
            "product_id": product_id,
            "source": "coinbase_1m_candle_open",
            "reference_time": reference_time.isoformat(),
            "candle_start_time": candle_start.isoformat(),
            "price": self._to_float(candle[3]),
            "low": self._to_float(candle[1]),
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
