from datetime import datetime, timedelta, timezone
from polymarket_quant.ingestion.client import BasePolymarketClient
from polymarket_quant.ingestion.pipeline import IngestionPipeline


def _utc_iso(dt):
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

class MockPolymarketClient(BasePolymarketClient):
    def fetch_orderbook(self, token_id):
        return {
            "market": "cond_456",
            "asset_id": token_id,
            "timestamp": "1775558597568",
            "hash": "book_hash",
            "bids": [{"price": "0.45", "size": "100.0"}],
            "asks": [{"price": "0.48", "size": "50.0"}],
        }

    def fetch_series(self, slug):
        return [{
            "slug": slug,
            "events": [{
                "id": "event_1",
                "slug": f"{slug}-event-1",
                "title": "Bitcoin Up or Down - Test",
                "startTime": "2026-01-01T00:00:00Z",
                "closed": True,
            }]
        }]

    def fetch_event_by_slug(self, slug):
        return {
            "id": "event_1",
            "slug": slug,
            "title": "Bitcoin Up or Down - Test",
            "startTime": "2026-01-01T00:00:00Z",
            "endDate": "2026-01-01T00:05:00Z",
            "closed": True,
            "markets": [{
                "id": "mkt_123",
                "conditionId": "cond_456",
                "outcomes": '["Up", "Down"]',
                "outcomePrices": '["1", "0"]',
                "endDate": "2026-01-01T00:05:00Z",
                "closed": True,
                "clobTokenIds": '["tok_up", "tok_down"]',
            }],
        }


class OpenMarketMockPolymarketClient(MockPolymarketClient):
    def fetch_series(self, slug):
        now = datetime.now(timezone.utc)
        event_prefix = "btc-updown-5m" if slug.startswith("btc") else "eth-updown-5m"
        return [{
            "slug": slug,
            "events": [
                {
                    "id": "event_wrong_prefix",
                    "slug": "xrp-updown-5m-open-event",
                    "title": "XRP Up or Down - Open Test",
                    "startTime": _utc_iso(now - timedelta(minutes=1)),
                    "endDate": _utc_iso(now + timedelta(minutes=4)),
                    "closed": False,
                },
                {
                    "id": "event_future",
                    "slug": f"{event_prefix}-future-event",
                    "title": "Bitcoin Up or Down - Future Test",
                    "startTime": _utc_iso(now + timedelta(minutes=5)),
                    "endDate": _utc_iso(now + timedelta(minutes=10)),
                    "closed": False,
                },
                {
                    "id": "event_open",
                    "slug": f"{event_prefix}-open-event",
                    "title": "Bitcoin Up or Down - Open Test",
                    "startTime": _utc_iso(now - timedelta(minutes=1)),
                    "endDate": _utc_iso(now + timedelta(minutes=4)),
                    "closed": False,
                },
            ]
        }]

    def fetch_event_by_slug(self, slug):
        now = datetime.now(timezone.utc)
        return {
            "id": "event_open",
            "slug": slug,
            "title": "Bitcoin Up or Down - Open Test",
            "startTime": _utc_iso(now - timedelta(minutes=1)),
            "endDate": _utc_iso(now + timedelta(minutes=4)),
            "closed": False,
            "markets": [{
                "id": "mkt_open",
                "conditionId": "cond_open",
                "outcomes": '["Up", "Down"]',
                "outcomePrices": '["0.51", "0.49"]',
                "eventStartTime": _utc_iso(now - timedelta(minutes=1)),
                "endDate": _utc_iso(now + timedelta(minutes=4)),
                "closed": False,
                "acceptingOrders": True,
                "clobTokenIds": '["tok_up", "tok_down"]',
            }],
        }


class CountingOpenMarketMockPolymarketClient(OpenMarketMockPolymarketClient):
    def __init__(self):
        self.series_calls = 0
        self.event_calls = 0

    def fetch_series(self, slug):
        self.series_calls += 1
        return super().fetch_series(slug)

    def fetch_event_by_slug(self, slug):
        self.event_calls += 1
        return super().fetch_event_by_slug(slug)



def test_live_crypto_5m_orderbook_collection(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"

    client = OpenMarketMockPolymarketClient()
    pipeline = IngestionPipeline(client, str(raw_dir), str(processed_dir))

    raw_snapshots, level_rows, summary_rows = pipeline.collect_crypto_5m_orderbooks_once(
        series_slugs=["btc-up-or-down-5m"],
        event_limit=1,
        event_slug_prefixes=["btc-updown-5m"],
    )

    assert len(raw_snapshots) == 2
    assert len(summary_rows) == 2
    assert len(level_rows) == 4
    assert set(row["side"] for row in level_rows) == {"bid", "ask"}
    assert summary_rows[0]["best_bid"] == 0.45
    assert summary_rows[0]["best_ask"] == 0.48
    assert "orderbook_imbalance" in summary_rows[0]

    pipeline.save_crypto_5m_orderbook_collection(
        raw_snapshots=raw_snapshots,
        level_rows=level_rows,
        summary_rows=summary_rows,
        run_timestamp="test",
    )

    assert (tmp_path / "BTC" / "raw" / "polymarket" / "crypto_5m_orderbooks_raw_test.json").exists()
    assert (tmp_path / "BTC" / "processed" / "polymarket" / "crypto_5m_orderbook_levels_test.parquet").exists()
    assert (tmp_path / "BTC" / "processed" / "polymarket" / "crypto_5m_orderbook_summary_test.parquet").exists()


def test_live_collection_with_explicit_event_slugs_skips_series_fetch(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"

    client = CountingOpenMarketMockPolymarketClient()
    pipeline = IngestionPipeline(client, str(raw_dir), str(processed_dir))

    pipeline.collect_crypto_5m_orderbooks_once(
        series_slugs=["btc-up-or-down-5m", "eth-up-or-down-5m"],
        event_limit=2,
        event_slug_prefixes=["btc-updown-5m", "eth-updown-5m"],
        event_slugs=["btc-updown-5m-open-event", "eth-updown-5m-open-event"],
    )

    assert client.series_calls == 0
    assert client.event_calls == 2


def test_live_collection_reuses_event_detail_cache(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"

    client = CountingOpenMarketMockPolymarketClient()
    pipeline = IngestionPipeline(client, str(raw_dir), str(processed_dir))
    event_cache = {}

    pipeline.collect_crypto_5m_orderbooks_once(
        series_slugs=["btc-up-or-down-5m", "eth-up-or-down-5m"],
        event_limit=2,
        event_slug_prefixes=["btc-updown-5m", "eth-updown-5m"],
        event_slugs=["btc-updown-5m-open-event", "eth-updown-5m-open-event"],
        event_details_by_slug=event_cache,
    )
    pipeline.collect_crypto_5m_orderbooks_once(
        series_slugs=["btc-up-or-down-5m", "eth-up-or-down-5m"],
        event_limit=2,
        event_slug_prefixes=["btc-updown-5m", "eth-updown-5m"],
        event_slugs=["btc-updown-5m-open-event", "eth-updown-5m-open-event"],
        event_details_by_slug=event_cache,
    )

    assert client.series_calls == 0
    assert client.event_calls == 2
    assert set(event_cache) == {"btc-updown-5m-open-event", "eth-updown-5m-open-event"}
