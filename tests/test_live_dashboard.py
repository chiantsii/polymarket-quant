import json
from pathlib import Path

import pandas as pd

from polymarket_quant.dashboard.data import build_live_dashboard_snapshot
from polymarket_quant.dashboard.textual_app import (
    _effective_market_poll_seconds,
    _effective_ui_refresh_seconds,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_build_live_dashboard_snapshot_extracts_core_panels(tmp_path: Path) -> None:
    event_state_path = tmp_path / "event_state.parquet"
    live_signal_path = tmp_path / "live_signal.jsonl"
    paper_signal_path = tmp_path / "paper_signal.jsonl"
    paper_order_path = tmp_path / "paper_order.jsonl"
    paper_trade_path = tmp_path / "paper_trade.jsonl"

    pd.DataFrame(
        [
            {
                "event_slug": "btc-updown-5m-test",
                "collected_at": "2026-05-08T09:00:00+00:00",
                "market_implied_up_probability": 0.48,
                "fundamental_up_probability": 0.51,
                "latent_up_probability": 0.53,
            }
        ]
    ).to_parquet(event_state_path, index=False)

    _write_jsonl(
        live_signal_path,
        [
            {
                "event_slug": "btc-updown-5m-test",
                "collected_at": "2026-05-08T09:00:02+00:00",
                "decision": "hold",
                "position_state": "LONG",
                "market_implied_up_probability": 0.48,
                "latent_up_probability": 0.53,
                "fair_up_probability": 0.57,
                "fair_token_price": 0.57,
                "buy_edge": 0.03,
                "hold_edge": 0.05,
                "sell_edge": -0.05,
                "best_bid": 0.52,
                "best_ask": 0.54,
                "seconds_to_end": 180,
                "pending_entry": False,
                "has_position": True,
            }
        ],
    )
    _write_jsonl(
        paper_signal_path,
        [
            {
                "event_slug": "btc-updown-5m-test",
                "collected_at": "2026-05-08T09:00:04+00:00",
                "decision": "exit",
                "position_state": "FLAT",
                "market_implied_up_probability": 0.49,
                "latent_up_probability": 0.54,
                "fair_up_probability": 0.58,
                "fair_token_price": 0.58,
                "buy_edge": 0.04,
                "hold_edge": -0.01,
                "sell_edge": 0.01,
                "best_bid": 0.59,
                "best_ask": 0.60,
                "seconds_to_end": 176,
                "pending_entry": False,
                "has_position": False,
                "closed_trade_exit_reason": "hold_edge_reversal",
            }
        ],
    )
    _write_jsonl(
        paper_order_path,
        [
            {
                "filled_at": "2026-05-08T09:00:04+00:00",
                "side": "SELL",
                "status": "filled",
                "price": 0.59,
                "reason": "hold_edge_reversal",
            }
        ],
    )
    _write_jsonl(
        paper_trade_path,
        [
            {
                "exit_time": "2026-05-08T09:00:04+00:00",
                "exit_reason": "hold_edge_reversal",
                "holding_seconds": 12.0,
                "gross_pnl": 0.03,
                "estimated_fee": 0.01,
                "net_pnl_with_fee": 0.02,
            }
        ],
    )

    snapshot = build_live_dashboard_snapshot(
        event_state_path=event_state_path,
        live_signal_path=live_signal_path,
        paper_signal_path=paper_signal_path,
        paper_order_path=paper_order_path,
        paper_trade_path=paper_trade_path,
        max_points=20,
        max_rows=5,
    )

    assert snapshot["state_panel"]["latent_up_probability"] == 0.54
    assert snapshot["state_panel"]["fair_up_probability"] == 0.58
    assert snapshot["edge_panel"]["best_bid"] == 0.59
    assert snapshot["execution_panel"]["latest_order_state"] == "filled"
    assert snapshot["execution_panel"]["exit_reason"] == "hold_edge_reversal"
    assert snapshot["current_snapshot"]["latest_signal_decision"] == "exit"
    assert snapshot["history_summary"]["filled_orders"] == 1
    assert snapshot["history_summary"]["closed_trades"] == 1
    assert snapshot["pnl_panel"]["trade_count"] == 1
    assert snapshot["pnl_panel"]["realized_pnl"] == 0.02
    assert snapshot["pnl_panel"]["fee_cost"] == 0.01
    assert len(snapshot["recent_signals"]) == 1
    assert len(snapshot["recent_orders"]) == 1
    assert len(snapshot["recent_trades"]) == 1


def test_build_live_dashboard_snapshot_returns_compact_streams(tmp_path: Path) -> None:
    event_state_path = tmp_path / "event_state.parquet"
    live_signal_path = tmp_path / "live_signal.jsonl"
    paper_signal_path = tmp_path / "paper_signal.jsonl"
    paper_order_path = tmp_path / "paper_order.jsonl"
    paper_trade_path = tmp_path / "paper_trade.jsonl"

    pd.DataFrame(
        [
            {
                "event_slug": "evt",
                "collected_at": "2026-05-08T09:00:00+00:00",
                "market_implied_up_probability": 0.48,
                "fundamental_up_probability": 0.51,
                "latent_up_probability": 0.53,
            }
        ]
    ).to_parquet(event_state_path, index=False)
    _write_jsonl(
        live_signal_path,
        [
            {
                "event_slug": "evt",
                "collected_at": "2026-05-08T09:00:02+00:00",
                "decision": "hold",
                "position_state": "LONG",
                "market_implied_up_probability": 0.49,
                "latent_up_probability": 0.54,
                "fair_up_probability": 0.58,
                "fair_token_price": 0.58,
                "buy_edge": 0.04,
                "hold_edge": 0.05,
                "sell_edge": -0.05,
                "best_bid": 0.59,
                "best_ask": 0.60,
                "seconds_to_end": 176,
            }
        ],
    )
    _write_jsonl(paper_signal_path, [])
    _write_jsonl(paper_order_path, [])
    _write_jsonl(paper_trade_path, [])

    snapshot = build_live_dashboard_snapshot(
        event_state_path=event_state_path,
        live_signal_path=live_signal_path,
        paper_signal_path=paper_signal_path,
        paper_order_path=paper_order_path,
        paper_trade_path=paper_trade_path,
    )

    assert snapshot["recent_signals"][0]["Decision"] == "hold"
    assert snapshot["recent_signals"][0]["State"] == "LONG"
    assert snapshot["recent_signals"][0]["Edge"] == "0.0400"
    assert snapshot["recent_orders"] == []
    assert snapshot["recent_trades"] == []


def test_build_live_dashboard_snapshot_builds_asset_snapshots(tmp_path: Path) -> None:
    event_state_path = tmp_path / "event_state.parquet"
    live_signal_path = tmp_path / "live_signal.jsonl"
    paper_signal_path = tmp_path / "paper_signal.jsonl"
    paper_order_path = tmp_path / "paper_order.jsonl"
    paper_trade_path = tmp_path / "paper_trade.jsonl"

    pd.DataFrame(
        [
            {
                "asset": "BTC",
                "event_slug": "btc-updown-5m-test",
                "collected_at": "2026-05-08T09:00:00+00:00",
                "market_implied_up_probability": 0.48,
                "fundamental_up_probability": 0.51,
                "latent_up_probability": 0.53,
            },
            {
                "asset": "ETH",
                "event_slug": "eth-updown-5m-test",
                "collected_at": "2026-05-08T09:00:00+00:00",
                "market_implied_up_probability": 0.61,
                "fundamental_up_probability": 0.58,
                "latent_up_probability": 0.60,
            },
        ]
    ).to_parquet(event_state_path, index=False)
    _write_jsonl(
        live_signal_path,
        [
            {
                "asset": "BTC",
                "event_slug": "btc-updown-5m-test",
                "collected_at": "2026-05-08T09:00:02+00:00",
                "decision": "observe",
                "position_state": "FLAT",
                "market_implied_up_probability": 0.48,
                "latent_up_probability": 0.53,
                "fair_up_probability": 0.57,
                "fair_token_price": 0.57,
                "buy_edge": 0.03,
                "hold_edge": 0.05,
                "sell_edge": -0.05,
                "best_bid": 0.52,
                "best_ask": 0.54,
                "seconds_to_end": 180,
            },
            {
                "asset": "ETH",
                "event_slug": "eth-updown-5m-test",
                "collected_at": "2026-05-08T09:00:02+00:00",
                "decision": "observe",
                "position_state": "FLAT",
                "market_implied_up_probability": 0.61,
                "latent_up_probability": 0.60,
                "fair_up_probability": 0.64,
                "fair_token_price": 0.64,
                "buy_edge": 0.02,
                "hold_edge": 0.01,
                "sell_edge": -0.01,
                "best_bid": 0.63,
                "best_ask": 0.62,
                "seconds_to_end": 180,
            },
        ],
    )
    _write_jsonl(paper_signal_path, [])
    _write_jsonl(paper_order_path, [])
    _write_jsonl(paper_trade_path, [])

    snapshot = build_live_dashboard_snapshot(
        event_state_path=event_state_path,
        live_signal_path=live_signal_path,
        paper_signal_path=paper_signal_path,
        paper_order_path=paper_order_path,
        paper_trade_path=paper_trade_path,
    )

    assert snapshot["asset_order"] == ["BTC", "ETH"]
    assert snapshot["asset_snapshots"]["BTC"]["state_panel"]["event_slug"] == "btc-updown-5m-test"
    assert snapshot["asset_snapshots"]["ETH"]["state_panel"]["event_slug"] == "eth-updown-5m-test"
    assert snapshot["asset_snapshots"]["BTC"]["edge_panel"]["fair_token_price"] == 0.57
    assert snapshot["asset_snapshots"]["ETH"]["edge_panel"]["fair_token_price"] == 0.64
    assert snapshot["probability_series_by_asset"]["BTC"]["market"] == [0.48]
    assert snapshot["probability_series_by_asset"]["ETH"]["market"] == [0.61]
    assert snapshot["probability_series_by_asset"]["BTC"]["fair"] == [0.57]
    assert snapshot["probability_series_by_asset"]["ETH"]["fair"] == [0.64]


def test_textual_dashboard_uses_separate_refresh_and_poll_intervals() -> None:
    assert _effective_ui_refresh_seconds(0.25) == 0.25
    assert _effective_ui_refresh_seconds(0.01) == 0.10
    assert _effective_market_poll_seconds(1.0) == 1.0
    assert _effective_market_poll_seconds(0.01) == 0.25
