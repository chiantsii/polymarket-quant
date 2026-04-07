from datetime import datetime, timedelta, timezone

import pandas as pd

from scripts.replay_pricing import replay_pricing


def _orderbook_row(timestamp, event_slug: str, outcome_name: str, token_id: str, best_bid: float, best_ask: float):
    event_start = datetime.fromtimestamp(1775578800, tz=timezone.utc)
    event_end = event_start + timedelta(minutes=5)
    return {
        "series_slug": "btc-up-or-down-5m",
        "asset": "BTC",
        "event_id": "event_1",
        "event_slug": event_slug,
        "event_title": "Bitcoin Up or Down - Test",
        "market_id": "mkt_1",
        "condition_id": "cond_1",
        "token_id": token_id,
        "outcome_name": outcome_name,
        "market_start_time": event_start.isoformat(),
        "market_end_time": event_end.isoformat(),
        "closed": False,
        "accepting_orders": True,
        "collected_at": timestamp.isoformat(),
        "book_timestamp": timestamp.isoformat(),
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": best_ask - best_bid,
        "mid_price": (best_bid + best_ask) / 2,
        "bid_depth": 100.0,
        "ask_depth": 100.0,
        "bid_depth_top_5": 100.0,
        "ask_depth_top_5": 100.0,
        "orderbook_imbalance": 0.0,
        "bid_levels": 1,
        "ask_levels": 1,
        "book_hash": "hash",
    }


def _spot_row(timestamp, price: float):
    return {
        "asset": "BTC",
        "product_id": "BTC-USD",
        "source": "coinbase",
        "collected_at": timestamp.isoformat(),
        "exchange_time": timestamp.isoformat(),
        "price": price,
        "bid": price - 0.01,
        "ask": price + 0.01,
        "size": 1.0,
        "volume": 100.0,
        "trade_id": 1,
    }


def test_replay_pricing_generates_labeled_edge_rows(tmp_path):
    event_slug = "btc-updown-5m-1775578800"
    start = datetime.fromtimestamp(1775578800, tz=timezone.utc)
    ts_1 = start + timedelta(seconds=1)
    ts_2 = start + timedelta(seconds=2)

    orderbooks = pd.DataFrame(
        [
            _orderbook_row(ts_1, event_slug, "Up", "tok_up", 0.50, 0.51),
            _orderbook_row(ts_1, event_slug, "Down", "tok_down", 0.48, 0.49),
            _orderbook_row(ts_2, event_slug, "Up", "tok_up", 0.52, 0.53),
            _orderbook_row(ts_2, event_slug, "Down", "tok_down", 0.46, 0.47),
        ]
    )
    spot = pd.DataFrame(
        [
            _spot_row(start, 100.0),
            _spot_row(ts_1, 101.0),
            _spot_row(ts_2, 101.5),
        ]
    )
    resolutions = pd.DataFrame(
        [
            {
                "event_slug": event_slug,
                "token_id": "tok_up",
                "outcome_name": "Up",
                "outcome_price": 1.0,
                "is_winner": 1,
            },
            {
                "event_slug": event_slug,
                "token_id": "tok_down",
                "outcome_name": "Down",
                "outcome_price": 0.0,
                "is_winner": 0,
            },
        ]
    )

    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "processed"
    input_dir.mkdir()
    orderbooks.to_parquet(input_dir / "crypto_5m_orderbook_summary_test.parquet", index=False)
    spot.to_parquet(input_dir / "crypto_spot_ticks_test.parquet", index=False)
    resolutions.to_parquet(input_dir / "crypto_5m_resolutions_test.parquet", index=False)

    result = replay_pricing(
        orderbook_glob=str(input_dir / "crypto_5m_orderbook_summary_*.parquet"),
        spot_glob=str(input_dir / "crypto_spot_ticks_*.parquet"),
        resolution_glob=str(input_dir / "crypto_5m_resolutions_*.parquet"),
        output_dir=str(output_dir),
        n_samples=200,
        spot_tolerance_seconds=2.0,
        use_particle_filter=False,
        run_timestamp="test",
    )

    replay = pd.read_parquet(output_dir / "crypto_5m_pricing_replay_test.parquet")

    assert result["rows"] == 4
    assert result["labeled_rows"] == 4
    assert (output_dir / "crypto_5m_pricing_replay_latest.parquet").exists()
    assert set(replay["outcome_name"]) == {"Up", "Down"}
    assert replay["fair_token_price"].between(0.0, 1.0).all()
    assert replay["brier_component"].notna().all()
