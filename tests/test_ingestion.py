from datetime import datetime, timedelta, timezone
from polymarket_quant.ingestion.client import BasePolymarketClient
from polymarket_quant.ingestion.pipeline import IngestionPipeline
from pathlib import Path
import json


def _utc_iso(dt):
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

class MockPolymarketClient(BasePolymarketClient):
    def fetch_active_markets(self, limit=100):
        return [{
            "id": "mkt_123",
            "condition_id": "cond_456",
            "question": "Will ETH reach $4000 by End of Year?",
            "category": "Crypto",
            "endDate": "2024-12-31T23:59:59Z",
            "active": True,
            "tokens": [
                {"token_id": "tok_y", "outcome": "Yes"},
                {"token_id": "tok_n", "outcome": "No"}
            ]
        }]

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

    def fetch_price_history(self, token_id, interval="max", fidelity=1):
        return {"history": [{"t": 1767225600, "p": 0.51}, {"t": 1767225660, "p": 0.55}]}


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

def test_pipeline_normalization(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    
    client = MockPolymarketClient()
    pipeline = IngestionPipeline(client, str(raw_dir), str(processed_dir))
    
    pipeline.run_market_metadata_ingestion()
    
    # Assert raw file exists
    raw_files = [path for path in raw_dir.glob("markets_raw_*.json") if "latest" not in path.name]
    assert len(raw_files) == 1
    assert (raw_dir / "markets_raw_latest.json").exists()
    
    # Assert processed Parquet exists
    processed_files = [path for path in processed_dir.glob("markets_normalized_*.parquet") if "latest" not in path.name]
    assert len(processed_files) == 1
    assert (processed_dir / "markets_normalized_latest.parquet").exists()
    
    import pandas as pd
    df = pd.read_parquet(processed_files[0])
    
    assert len(df) == 1
    assert df.iloc[0]["market_id"] == "mkt_123"
    assert df.iloc[0]["category"] == "Crypto"
    assert "time_to_resolution_seconds" in df.columns


def test_orderbook_snapshot_pipeline(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"

    client = MockPolymarketClient()
    pipeline = IngestionPipeline(client, str(raw_dir), str(processed_dir))

    pipeline.run_orderbook_snapshot_ingestion()

    raw_files = [path for path in raw_dir.glob("orderbooks_raw_*.json") if "latest" not in path.name]
    assert len(raw_files) == 1
    assert (raw_dir / "orderbooks_raw_latest.json").exists()

    processed_files = [path for path in processed_dir.glob("orderbooks_snapshot_*.parquet") if "latest" not in path.name]
    assert len(processed_files) == 1
    assert (processed_dir / "orderbooks_snapshot_latest.parquet").exists()

    import pandas as pd
    df = pd.read_parquet(processed_files[0])

    assert len(df) == 2
    assert set(df["outcome_name"]) == {"Yes", "No"}
    assert df.iloc[0]["best_bid"] == 0.45
    assert df.iloc[0]["best_ask"] == 0.48


def test_crypto_5m_history_pipeline(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"

    client = MockPolymarketClient()
    pipeline = IngestionPipeline(client, str(raw_dir), str(processed_dir))

    pipeline.run_crypto_5m_history_ingestion(series_slugs=["btc-up-or-down-5m"], event_limit=1)

    raw_files = [path for path in raw_dir.glob("crypto_5m_history_raw_*.json") if "latest" not in path.name]
    assert len(raw_files) == 1
    assert (raw_dir / "crypto_5m_history_raw_latest.json").exists()

    processed_files = [path for path in processed_dir.glob("crypto_5m_price_history_*.parquet") if "latest" not in path.name]
    assert len(processed_files) == 1
    assert (processed_dir / "crypto_5m_price_history_latest.parquet").exists()

    import pandas as pd
    df = pd.read_parquet(processed_files[0])

    assert len(df) == 4
    assert set(df["outcome_name"]) == {"Up", "Down"}
    assert set(df["is_winner"]) == {0, 1}
    assert set(df["asset"]) == {"BTC"}


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

    assert (raw_dir / "crypto_5m_orderbooks_raw_test.json").exists()
    assert (processed_dir / "crypto_5m_orderbook_levels_test.parquet").exists()
    assert (processed_dir / "crypto_5m_orderbook_summary_test.parquet").exists()
