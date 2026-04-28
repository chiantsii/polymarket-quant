from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from polymarket_quant.ingestion.spot import SpotFetchError
from scripts.collect_spot_prices import collect_spot_prices


class _MonotonicClock:
    def __init__(self, values: list[float]):
        self._values = iter(values)
        self._last = values[-1]

    def __call__(self) -> float:
        try:
            self._last = next(self._values)
        except StopIteration:
            pass
        return self._last


def _write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "raw_dir": str(tmp_path / "raw"),
                    "processed_dir": str(tmp_path / "processed"),
                },
                "api": {
                    "binance_url": "https://api.binance.com",
                },
                "spot": {
                    "products": {
                        "BTC": "BTCUSDT",
                        "ETH": "ETHUSDT",
                    }
                },
            }
        )
    )
    return config_path


def test_collect_spot_prices_persists_btc_and_eth_separately(tmp_path, monkeypatch) -> None:
    config_path = _write_config(tmp_path)

    class StubClient:
        def fetch_spot_ticker(self, asset: str, product_id: str):
            return {
                "asset": asset,
                "product_id": product_id,
                "source": "binance_book_ticker",
                "collected_at": "2026-04-28T00:00:00+00:00",
                "exchange_time": None,
                "price": 100000.0 if asset == "BTC" else 3000.0,
                "bid": None,
                "ask": None,
                "size": None,
                "volume": None,
                "trade_id": None,
            }

    monkeypatch.setattr("scripts.collect_spot_prices.BinanceSpotPriceClient", lambda *args, **kwargs: StubClient())
    monkeypatch.setattr("scripts.collect_spot_prices.time.monotonic", _MonotonicClock([0.0, 0.0, 0.0, 0.1, 1.1]))
    monkeypatch.setattr("scripts.collect_spot_prices.time.sleep", lambda _: None)
    monkeypatch.setattr(
        "scripts.collect_spot_prices.resolve_active_window",
        lambda **kwargs: type(
            "FullWindow",
            (),
            {
                "start": datetime(2026, 4, 28, 0, 0, tzinfo=timezone.utc),
                "end": datetime(2026, 4, 28, 0, 5, tzinfo=timezone.utc),
                "event_slugs": ["btc-updown-5m-1777334400", "eth-updown-5m-1777334400"],
            },
        )(),
    )

    result = collect_spot_prices(
        config_path=str(config_path),
        mode="duration",
        interval_seconds=1.0,
        duration_seconds=1.0,
        run_timestamp="20260428_000000",
    )

    assert result["spot_rows"] == 2
    assert result["spot_rows_by_asset"] == {"BTC": 1, "ETH": 1}

    btc_raw = tmp_path / "BTC" / "raw" / "spot" / "binance_spot_ticks_20260428_000000.json"
    eth_raw = tmp_path / "ETH" / "raw" / "spot" / "binance_spot_ticks_20260428_000000.json"
    btc_parquet = tmp_path / "BTC" / "processed" / "spot" / "binance_spot_ticks_20260428_000000.parquet"
    eth_parquet = tmp_path / "ETH" / "processed" / "spot" / "binance_spot_ticks_20260428_000000.parquet"

    assert btc_raw.exists()
    assert eth_raw.exists()
    assert btc_parquet.exists()
    assert eth_parquet.exists()
    assert json.loads(btc_raw.read_text())[0]["asset"] == "BTC"
    assert json.loads(eth_raw.read_text())[0]["asset"] == "ETH"
    assert json.loads(btc_raw.read_text())[0]["event_slug"] == "btc-updown-5m-1777334400"
    assert json.loads(eth_raw.read_text())[0]["event_slug"] == "eth-updown-5m-1777334400"


def test_collect_spot_prices_fails_when_any_required_asset_is_missing(tmp_path, monkeypatch) -> None:
    config_path = _write_config(tmp_path)

    class StubClient:
        def fetch_spot_ticker(self, asset: str, product_id: str):
            if asset == "ETH":
                raise SpotFetchError("dns failure")
            return {
                "asset": asset,
                "product_id": product_id,
                "source": "binance_book_ticker",
                "collected_at": "2026-04-28T00:00:00+00:00",
                "exchange_time": None,
                "price": 100000.0,
                "bid": None,
                "ask": None,
                "size": None,
                "volume": None,
                "trade_id": None,
            }

    monkeypatch.setattr("scripts.collect_spot_prices.BinanceSpotPriceClient", lambda *args, **kwargs: StubClient())
    monkeypatch.setattr("scripts.collect_spot_prices.time.monotonic", _MonotonicClock([0.0, 0.0, 0.0, 0.1, 1.1]))
    monkeypatch.setattr("scripts.collect_spot_prices.time.sleep", lambda _: None)
    monkeypatch.setattr(
        "scripts.collect_spot_prices.resolve_active_window",
        lambda **kwargs: type(
            "FullWindow",
            (),
            {
                "start": datetime(2026, 4, 28, 0, 0, tzinfo=timezone.utc),
                "end": datetime(2026, 4, 28, 0, 5, tzinfo=timezone.utc),
                "event_slugs": ["btc-updown-5m-1777334400", "eth-updown-5m-1777334400"],
            },
        )(),
    )

    with pytest.raises(RuntimeError, match="Spot capture failed for required asset"):
        collect_spot_prices(
            config_path=str(config_path),
            mode="duration",
            interval_seconds=1.0,
            duration_seconds=1.0,
            run_timestamp="20260428_000000",
        )

    btc_parquet = tmp_path / "BTC" / "processed" / "spot" / "binance_spot_ticks_20260428_000000.parquet"
    eth_raw = tmp_path / "ETH" / "raw" / "spot" / "binance_spot_ticks_20260428_000000.json"
    eth_parquet = tmp_path / "ETH" / "processed" / "spot" / "binance_spot_ticks_20260428_000000.parquet"

    assert btc_parquet.exists()
    assert eth_raw.exists()
    assert json.loads(eth_raw.read_text()) == []
    assert not eth_parquet.exists()


def test_collect_spot_prices_rejects_expired_full_window(tmp_path, monkeypatch) -> None:
    config_path = _write_config(tmp_path)
    past_start = datetime.now(timezone.utc) - timedelta(minutes=10)
    past_end = past_start + timedelta(minutes=5)

    class FullWindow:
        def __init__(self, start, end):
            self.start = start
            self.end = end
            self.event_slugs = ["btc-updown-5m-test", "eth-updown-5m-test"]

    monkeypatch.setattr(
        "scripts.collect_spot_prices.resolve_full_window",
        lambda **kwargs: FullWindow(start=past_start, end=past_end),
    )

    with pytest.raises(RuntimeError, match="Refusing to collect an expired full window"):
        collect_spot_prices(
            config_path=str(config_path),
            mode="full-window",
            interval_seconds=1.0,
            duration_seconds=300.0,
            event_duration_seconds=300,
            window_start=str(int(past_start.timestamp())),
            run_timestamp="20260428_000000",
        )
