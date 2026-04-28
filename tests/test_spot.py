from datetime import datetime, timezone

import requests
import pytest

from polymarket_quant.ingestion.spot import BinanceSpotPriceClient, SpotFetchError


def test_binance_spot_client_normalizes_book_ticker_payload() -> None:
    client = BinanceSpotPriceClient()
    payload = {
        "symbol": "BTCUSDT",
        "bidPrice": "68000.00",
        "bidQty": "1.50",
        "askPrice": "68001.00",
        "askQty": "2.00",
        "time": 1775520000000,
    }

    class MockResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return payload

    class MockSession:
        def get(self, url, params, timeout):
            return MockResponse()

    client.session = MockSession()

    ticker = client.fetch_spot_ticker(asset="BTC", product_id="BTCUSDT")

    assert ticker["asset"] == "BTC"
    assert ticker["product_id"] == "BTCUSDT"
    assert ticker["source"] == "binance_book_ticker"
    assert ticker["price"] == 68000.50
    assert ticker["bid"] == 68000.00
    assert ticker["ask"] == 68001.00
    assert ticker["size"] == 1.50


def test_binance_spot_client_raises_on_connection_error() -> None:
    client = BinanceSpotPriceClient()
    class MockSession:
        def get(self, url, params, timeout):
            raise requests.exceptions.ConnectionError("binance unavailable")

    client.session = MockSession()

    with pytest.raises(SpotFetchError, match="Failed to fetch Binance spot ticker"):
        client.fetch_spot_ticker(asset="BTC", product_id="BTCUSDT")


def test_binance_reference_price_uses_kline_open() -> None:
    client = BinanceSpotPriceClient()
    payload = [[1775520000000, "68000.0", "68100.0", "67900.0", "68050.0", "12.5"]]

    class MockResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return payload

    class MockSession:
        def get(self, url, params, timeout):
            return MockResponse()

    client.session = MockSession()

    reference = client.fetch_reference_price(
        asset="BTC",
        product_id="BTCUSDT",
        reference_time=datetime.fromtimestamp(1775520000, tz=timezone.utc),
    )

    assert reference["source"] == "binance_1m_kline_open"
    assert reference["price"] == 68000.0
    assert reference["low"] == 67900.0
    assert reference["high"] == 68100.0
