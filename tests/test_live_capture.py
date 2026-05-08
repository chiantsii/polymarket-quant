from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from polymarket_quant.live import (
    DirectLiveEventStateSource,
    DirectLiveSourceConfig,
    LiveCaptureEventStateSource,
    LiveCaptureSourceConfig,
    load_live_capture_state_snapshot,
)


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
        "bid_depth": 120.0,
        "ask_depth": 110.0,
        "bid_depth_top_5": 90.0,
        "ask_depth_top_5": 85.0,
        "orderbook_imbalance": 0.05,
        "bid_levels": 3,
        "ask_levels": 3,
        "book_hash": "hash",
    }


def _orderbook_levels_rows(timestamp, event_slug: str, outcome_name: str, token_id: str):
    rows = []
    bid_prices = [0.50, 0.49, 0.48]
    ask_prices = [0.51, 0.52, 0.54]
    bid_sizes = [60.0, 40.0, 30.0]
    ask_sizes = [55.0, 35.0, 20.0]
    for level, (price, size) in enumerate(zip(bid_prices, bid_sizes), start=1):
        rows.append(
            {
                "event_slug": event_slug,
                "collected_at": timestamp.isoformat(),
                "token_id": token_id,
                "outcome_name": outcome_name,
                "side": "bid",
                "level": level,
                "price": price,
                "size": size,
            }
        )
    for level, (price, size) in enumerate(zip(ask_prices, ask_sizes), start=1):
        rows.append(
            {
                "event_slug": event_slug,
                "collected_at": timestamp.isoformat(),
                "token_id": token_id,
                "outcome_name": outcome_name,
                "side": "ask",
                "level": level,
                "price": price,
                "size": size,
            }
        )
    return rows


def _spot_row_for_event(timestamp, event_slug: str, price: float):
    return {
        "asset": "BTC",
        "product_id": "BTCUSDT",
        "source": "binance_book_ticker",
        "collected_at": timestamp.isoformat(),
        "exchange_time": timestamp.isoformat(),
        "price": price,
        "bid": price - 0.01,
        "ask": price + 0.01,
        "size": 1.0,
        "volume": 100.0,
        "trade_id": 1,
        "event_slug": event_slug,
    }


def _write_live_latest_files(
    root: Path,
    *,
    orderbook_rows: list[dict],
    level_rows: list[dict],
    spot_rows: list[dict],
) -> tuple[Path, Path, Path]:
    summary_path = root / "BTC" / "processed" / "polymarket" / "crypto_5m_orderbook_summary_latest.parquet"
    levels_path = root / "BTC" / "processed" / "polymarket" / "crypto_5m_orderbook_levels_latest.parquet"
    spot_path = root / "BTC" / "processed" / "spot" / "binance_spot_ticks_latest.parquet"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    levels_path.parent.mkdir(parents=True, exist_ok=True)
    spot_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(orderbook_rows).to_parquet(summary_path, index=False)
    pd.DataFrame(level_rows).to_parquet(levels_path, index=False)
    pd.DataFrame(spot_rows).to_parquet(spot_path, index=False)
    return summary_path, levels_path, spot_path


def test_load_live_capture_state_snapshot_builds_event_state_from_latest_files(tmp_path: Path) -> None:
    event_slug = "btc-updown-5m-1775578800"
    start = datetime.fromtimestamp(1775578800, tz=timezone.utc)
    ts_1 = start + timedelta(seconds=1)

    summary_path, levels_path, spot_path = _write_live_latest_files(
        tmp_path,
        orderbook_rows=[
            _orderbook_row(ts_1, event_slug, "Up", "tok_up", 0.50, 0.51),
            _orderbook_row(ts_1, event_slug, "Down", "tok_down", 0.48, 0.49),
        ],
        level_rows=_orderbook_levels_rows(ts_1, event_slug, "Up", "tok_up")
        + _orderbook_levels_rows(ts_1, event_slug, "Down", "tok_down"),
        spot_rows=[
            _spot_row_for_event(start, event_slug, 100.0),
            _spot_row_for_event(ts_1, event_slug, 101.0),
        ],
    )

    snapshot = load_live_capture_state_snapshot(
        LiveCaptureSourceConfig(
            orderbook_summary_glob=str(summary_path),
            orderbook_levels_glob=str(levels_path),
            spot_glob=str(spot_path),
            output_dir=str(tmp_path / "artifacts"),
        )
    )

    assert not snapshot.market_state.empty
    assert not snapshot.event_state.empty
    assert snapshot.event_state.iloc[-1]["event_slug"] == event_slug
    assert 0.0 <= snapshot.event_state.iloc[-1]["latent_up_probability"] <= 1.0


def test_live_capture_event_state_source_only_emits_unseen_rows(tmp_path: Path) -> None:
    event_slug = "btc-updown-5m-1775578800"
    start = datetime.fromtimestamp(1775578800, tz=timezone.utc)
    ts_1 = start + timedelta(seconds=1)
    ts_2 = start + timedelta(seconds=2)

    summary_path, levels_path, spot_path = _write_live_latest_files(
        tmp_path,
        orderbook_rows=[
            _orderbook_row(ts_1, event_slug, "Up", "tok_up", 0.50, 0.51),
            _orderbook_row(ts_1, event_slug, "Down", "tok_down", 0.48, 0.49),
        ],
        level_rows=_orderbook_levels_rows(ts_1, event_slug, "Up", "tok_up")
        + _orderbook_levels_rows(ts_1, event_slug, "Down", "tok_down"),
        spot_rows=[
            _spot_row_for_event(start, event_slug, 100.0),
            _spot_row_for_event(ts_1, event_slug, 101.0),
        ],
    )

    source = LiveCaptureEventStateSource(
        LiveCaptureSourceConfig(
            orderbook_summary_glob=str(summary_path),
            orderbook_levels_glob=str(levels_path),
            spot_glob=str(spot_path),
            output_dir=str(tmp_path / "artifacts"),
        )
    )

    first_rows = source.poll_new_rows()
    assert len(first_rows) == 1
    assert first_rows[0]["collected_at"] == ts_1.isoformat()

    second_rows = source.poll_new_rows()
    assert second_rows == []

    _write_live_latest_files(
        tmp_path,
        orderbook_rows=[
            _orderbook_row(ts_1, event_slug, "Up", "tok_up", 0.50, 0.51),
            _orderbook_row(ts_1, event_slug, "Down", "tok_down", 0.48, 0.49),
            _orderbook_row(ts_2, event_slug, "Up", "tok_up", 0.52, 0.53),
            _orderbook_row(ts_2, event_slug, "Down", "tok_down", 0.46, 0.47),
        ],
        level_rows=_orderbook_levels_rows(ts_1, event_slug, "Up", "tok_up")
        + _orderbook_levels_rows(ts_1, event_slug, "Down", "tok_down")
        + _orderbook_levels_rows(ts_2, event_slug, "Up", "tok_up")
        + _orderbook_levels_rows(ts_2, event_slug, "Down", "tok_down"),
        spot_rows=[
            _spot_row_for_event(start, event_slug, 100.0),
            _spot_row_for_event(ts_1, event_slug, 101.0),
            _spot_row_for_event(ts_2, event_slug, 101.5),
        ],
    )

    third_rows = source.poll_new_rows()
    assert len(third_rows) == 1
    assert third_rows[0]["collected_at"] == ts_2.isoformat()
    assert (tmp_path / "artifacts" / "live_event_state_latest.parquet").exists()


class _StubPolymarketClient:
    def __init__(self, event_slug: str, event_start: datetime) -> None:
        self.event_slug = event_slug
        self.event_start = event_start

    def fetch_series(self, slug: str):
        return []

    def fetch_event_by_slug(self, slug: str):
        assert slug == self.event_slug
        event_end = self.event_start + timedelta(minutes=5)
        return {
            "id": "event_1",
            "slug": self.event_slug,
            "title": "Bitcoin Up or Down - Test",
            "startTime": self.event_start.isoformat(),
            "endDate": event_end.isoformat(),
            "markets": [
                {
                    "id": "mkt_1",
                    "conditionId": "cond_1",
                    "eventStartTime": self.event_start.isoformat(),
                    "endDate": event_end.isoformat(),
                    "closed": False,
                    "acceptingOrders": True,
                    "tokens": [
                        {"token_id": "tok_up", "outcome": "Up"},
                        {"token_id": "tok_down", "outcome": "Down"},
                    ],
                }
            ],
        }

    def fetch_orderbook(self, token_id: str):
        now = datetime.now(timezone.utc)
        return {
            "timestamp": int(now.timestamp()),
            "hash": f"hash_{token_id}",
            "bids": [{"price": "0.50", "size": "60"}, {"price": "0.49", "size": "40"}],
            "asks": [{"price": "0.51", "size": "55"}, {"price": "0.52", "size": "35"}],
        }


class _StubSpotClient:
    def __init__(self, event_start: datetime) -> None:
        self.event_start = event_start

    def fetch_spot_ticker(self, asset: str, product_id: str):
        ts = datetime.now(timezone.utc)
        return {
            "asset": asset,
            "product_id": product_id,
            "source": "binance_book_ticker",
            "collected_at": ts.isoformat(),
            "exchange_time": ts.isoformat(),
            "price": 101.0 if asset == "BTC" else 201.0,
            "bid": 100.99 if asset == "BTC" else 200.99,
            "ask": 101.01 if asset == "BTC" else 201.01,
            "size": 1.0,
            "volume": None,
            "trade_id": None,
        }


def test_direct_live_event_state_source_fetches_and_emits_rows(tmp_path: Path, monkeypatch) -> None:
    now = datetime.now(timezone.utc).replace(microsecond=0)
    event_start = now - timedelta(seconds=30)
    event_slug = f"btc-updown-5m-{int(event_start.timestamp())}"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "api:",
                "  gamma_url: https://gamma-api.polymarket.com",
                "  clob_url: https://clob.polymarket.com",
                "  binance_url: https://api.binance.com",
                "data:",
                f"  raw_dir: {str(tmp_path / 'raw')}",
                f"  processed_dir: {str(tmp_path / 'processed')}",
                "spot:",
                "  products:",
                "    BTC: BTCUSDT",
            ]
        ),
        encoding="utf-8",
    )

    class _Window:
        start = event_start
        end = event_start + timedelta(minutes=5)
        event_slugs = [event_slug]

    monkeypatch.setattr("polymarket_quant.live.direct.resolve_active_window", lambda **_: _Window())

    source = DirectLiveEventStateSource(
        DirectLiveSourceConfig(
            config_path=str(config_path),
            output_dir=str(tmp_path / "artifacts"),
            event_slug_prefixes=("btc-updown-5m",),
            series_slugs=("btc-up-or-down-5m",),
            spot_tolerance_seconds=10.0,
        ),
        polymarket_client=_StubPolymarketClient(event_slug=event_slug, event_start=event_start),
        spot_client=_StubSpotClient(event_start=event_start),
    )

    rows = source.poll_new_rows()
    assert len(rows) == 1
    assert rows[0]["event_slug"] == event_slug
    assert 0.0 <= rows[0]["latent_up_probability"] <= 1.0
    assert (tmp_path / "artifacts" / "live_event_state_latest.parquet").exists()
