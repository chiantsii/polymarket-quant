from datetime import datetime, timezone

from polymarket_quant.ingestion.spot import CoinbaseSpotPriceClient


def test_coinbase_spot_client_normalizes_ticker_payload() -> None:
    client = CoinbaseSpotPriceClient()
    payload = {
        "trade_id": 1,
        "price": "68000.50",
        "size": "0.10",
        "time": "2026-04-07T00:00:00Z",
        "bid": "68000.00",
        "ask": "68001.00",
        "volume": "100.25",
    }

    class MockResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return payload

    class MockSession:
        def get(self, url, timeout):
            return MockResponse()

    client.session = MockSession()

    ticker = client.fetch_spot_ticker(asset="BTC", product_id="BTC-USD")

    assert ticker["asset"] == "BTC"
    assert ticker["product_id"] == "BTC-USD"
    assert ticker["price"] == 68000.50
    assert ticker["bid"] == 68000.00
    assert ticker["ask"] == 68001.00


def test_coinbase_reference_price_uses_candle_open() -> None:
    client = CoinbaseSpotPriceClient()
    payload = [[1775520000, 67900.0, 68100.0, 68000.0, 68050.0, 12.5]]

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
        product_id="BTC-USD",
        reference_time=datetime.fromtimestamp(1775520000, tz=timezone.utc),
    )

    assert reference["source"] == "coinbase_1m_candle_open"
    assert reference["price"] == 68000.0
    assert reference["low"] == 67900.0
    assert reference["high"] == 68100.0
