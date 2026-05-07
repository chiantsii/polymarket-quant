from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from polymarket_quant.evaluation.edge_strategy import (
    edge_strategy_summary,
    replay_baseline_up_strategy,
    replay_dynamic_buy_strategy,
)


def _row(
    seconds: int,
    signal: str,
    best_bid: float,
    best_ask: float,
    fair_token_price: float,
    buy_edge: float,
    net_buy_edge: float | None = None,
    net_hold_edge: float | None = None,
    toxicity_score: float = 0.1,
    seconds_to_end: float = 200.0,
) -> dict:
    timestamp = datetime(2026, 4, 8, tzinfo=timezone.utc) + timedelta(seconds=seconds)
    hold_edge = fair_token_price - best_bid
    return {
        "asset": "BTC",
        "event_slug": "btc-updown-5m-test",
        "token_id": "tok_up",
        "outcome_name": "Up",
        "collected_at": timestamp.isoformat(),
        "signal": signal,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "fair_token_price": fair_token_price,
        "buy_edge": buy_edge,
        "hold_edge": hold_edge,
        "net_buy_edge": buy_edge if net_buy_edge is None else net_buy_edge,
        "net_hold_edge": hold_edge if net_hold_edge is None else net_hold_edge,
        "entry_fee_penalty": 0.0,
        "exit_fee_penalty": 0.0,
        "adverse_selection_penalty": 0.0,
        "inventory_penalty": 0.0,
        "toxicity_score": toxicity_score,
        "seconds_to_end": seconds_to_end - seconds,
    }


def test_dynamic_buy_strategy_exits_when_edge_is_repaired() -> None:
    rows = pd.DataFrame(
        [
            _row(0, "BUY", best_bid=0.49, best_ask=0.50, fair_token_price=0.56, buy_edge=0.06),
            _row(10, "HOLD", best_bid=0.52, best_ask=0.53, fair_token_price=0.56, buy_edge=0.03),
            _row(20, "HOLD", best_bid=0.557, best_ask=0.56, fair_token_price=0.56, buy_edge=0.00),
        ]
    )

    episodes = replay_dynamic_buy_strategy(
        rows,
        entry_edge=0.03,
        exit_edge=0.005,
        take_profit=0.10,
        stop_loss=0.10,
        max_holding_seconds=60,
    )

    assert len(episodes) == 1
    assert episodes.loc[0, "exit_reason"] == "edge_repaired"
    assert episodes.loc[0, "holding_seconds"] == pytest.approx(20.0)
    assert episodes.loc[0, "pnl"] == pytest.approx(0.057)
    assert episodes.loc[0, "gross_pnl"] == pytest.approx(0.057)


def test_dynamic_buy_strategy_exits_on_stop_loss() -> None:
    rows = pd.DataFrame(
        [
            _row(0, "BUY", best_bid=0.49, best_ask=0.50, fair_token_price=0.56, buy_edge=0.06),
            _row(5, "HOLD", best_bid=0.46, best_ask=0.47, fair_token_price=0.56, buy_edge=0.09),
        ]
    )

    episodes = replay_dynamic_buy_strategy(rows, stop_loss=0.03, take_profit=0.10)

    assert len(episodes) == 1
    assert episodes.loc[0, "exit_reason"] == "stop_loss"
    assert episodes.loc[0, "pnl"] == pytest.approx(-0.04)


def test_dynamic_buy_strategy_does_not_overlap_same_token_entries() -> None:
    rows = pd.DataFrame(
        [
            _row(0, "BUY", best_bid=0.49, best_ask=0.50, fair_token_price=0.56, buy_edge=0.06),
            _row(1, "BUY", best_bid=0.50, best_ask=0.51, fair_token_price=0.57, buy_edge=0.06),
            _row(2, "HOLD", best_bid=0.53, best_ask=0.54, fair_token_price=0.56, buy_edge=0.02),
        ]
    )

    episodes = replay_dynamic_buy_strategy(rows, take_profit=0.02, stop_loss=0.10)

    assert len(episodes) == 1
    assert episodes.loc[0, "entry_price"] == pytest.approx(0.50)
    assert edge_strategy_summary(episodes)["episodes"] == 1


def test_dynamic_buy_strategy_net_pnl_includes_penalties() -> None:
    rows = pd.DataFrame(
        [
            _row(
                0,
                "BUY",
                best_bid=0.49,
                best_ask=0.50,
                fair_token_price=0.56,
                buy_edge=0.06,
                net_buy_edge=0.04,
            ),
            _row(
                10,
                "HOLD",
                best_bid=0.54,
                best_ask=0.55,
                fair_token_price=0.56,
                buy_edge=0.01,
                net_hold_edge=0.0,
            ),
        ]
    )
    rows.loc[0, "entry_fee_penalty"] = 0.01
    rows.loc[1, "exit_fee_penalty"] = 0.005
    rows.loc[0, "adverse_selection_penalty"] = 0.01
    rows.loc[0, "inventory_penalty"] = 0.005

    episodes = replay_dynamic_buy_strategy(
        rows,
        entry_edge=0.03,
        exit_edge=0.001,
        take_profit=0.10,
        stop_loss=0.10,
    )

    assert len(episodes) == 1
    assert episodes.loc[0, "gross_pnl"] == pytest.approx(0.04)
    assert episodes.loc[0, "net_pnl"] == pytest.approx(0.01)
    assert episodes.loc[0, "pnl"] == pytest.approx(0.01)


def _baseline_row(
    seconds: int,
    *,
    outcome_name: str = "Up",
    best_bid: float = 0.50,
    best_ask: float = 0.51,
    fair_token_price: float = 0.55,
    buy_edge: float = 0.04,
    seconds_to_end: float = 300.0,
    event_slug: str = "btc-updown-5m-test",
) -> dict:
    timestamp = datetime(2026, 5, 7, tzinfo=timezone.utc) + timedelta(seconds=seconds)
    return {
        "asset": "BTC",
        "event_slug": event_slug,
        "token_id": "tok_up" if outcome_name == "Up" else "tok_down",
        "outcome_name": outcome_name,
        "collected_at": timestamp.isoformat(),
        "best_bid": best_bid,
        "best_ask": best_ask,
        "fair_token_price": fair_token_price,
        "buy_edge": buy_edge,
        "hold_edge": fair_token_price - best_bid,
        "sell_edge": best_bid - fair_token_price,
        "seconds_to_end": seconds_to_end,
    }


def test_baseline_up_strategy_requires_two_snapshot_confirmation() -> None:
    rows = pd.DataFrame(
        [
            _baseline_row(0, buy_edge=0.031, best_ask=0.50, best_bid=0.49, fair_token_price=0.55, seconds_to_end=240),
            _baseline_row(2, buy_edge=0.033, best_ask=0.51, best_bid=0.50, fair_token_price=0.55, seconds_to_end=238),
            _baseline_row(4, buy_edge=0.020, best_ask=0.52, best_bid=0.51, fair_token_price=0.56, seconds_to_end=236),
            _baseline_row(20, buy_edge=0.010, best_ask=0.52, best_bid=0.54, fair_token_price=0.53, seconds_to_end=220),
        ]
    )

    episodes = replay_baseline_up_strategy(rows)

    assert len(episodes) == 1
    assert episodes.loc[0, "signal_confirmation_time"] == rows.loc[1, "collected_at"]
    assert episodes.loc[0, "entry_time"] == rows.loc[2, "collected_at"]
    assert episodes.loc[0, "entry_price"] == pytest.approx(0.52)
    assert episodes.loc[0, "exit_reason"] == "hold_edge_reversal"
    assert episodes.loc[0, "entry_confirmation_snapshots"] == 2
    assert episodes.loc[0, "entry_execution_delay_seconds"] == pytest.approx(2.0)


def test_baseline_up_strategy_ignores_down_rows_and_oversized_edges() -> None:
    rows = pd.DataFrame(
        [
            _baseline_row(0, outcome_name="Down", buy_edge=0.08, seconds_to_end=240, event_slug="evt-1"),
            _baseline_row(1, outcome_name="Down", buy_edge=0.09, seconds_to_end=239, event_slug="evt-1"),
            _baseline_row(0, outcome_name="Up", buy_edge=0.20, seconds_to_end=240, event_slug="evt-2"),
            _baseline_row(1, outcome_name="Up", buy_edge=0.22, seconds_to_end=239, event_slug="evt-2"),
        ]
    )

    episodes = replay_baseline_up_strategy(rows, max_edge_cap=0.15)

    assert episodes.empty


def test_baseline_up_strategy_exits_on_max_holding_time() -> None:
    rows = pd.DataFrame(
        [
            _baseline_row(0, buy_edge=0.04, fair_token_price=0.56, best_bid=0.49, best_ask=0.50, seconds_to_end=240),
            _baseline_row(2, buy_edge=0.05, fair_token_price=0.56, best_bid=0.49, best_ask=0.50, seconds_to_end=238),
            _baseline_row(4, buy_edge=0.02, fair_token_price=0.56, best_bid=0.49, best_ask=0.51, seconds_to_end=236),
            _baseline_row(65, buy_edge=0.05, fair_token_price=0.60, best_bid=0.55, best_ask=0.56, seconds_to_end=175),
        ]
    )

    episodes = replay_baseline_up_strategy(rows, max_holding_seconds=60.0)

    assert len(episodes) == 1
    assert episodes.loc[0, "exit_reason"] == "max_holding_time"
    assert episodes.loc[0, "holding_seconds"] == pytest.approx(61.0)


def test_baseline_up_strategy_summary_counts_exit_reasons() -> None:
    rows = pd.DataFrame(
        [
            _baseline_row(0, buy_edge=0.04, fair_token_price=0.56, best_bid=0.49, best_ask=0.50, seconds_to_end=240, event_slug="evt-1"),
            _baseline_row(2, buy_edge=0.05, fair_token_price=0.56, best_bid=0.49, best_ask=0.50, seconds_to_end=238, event_slug="evt-1"),
            _baseline_row(4, buy_edge=0.02, fair_token_price=0.56, best_bid=0.50, best_ask=0.51, seconds_to_end=236, event_slug="evt-1"),
            _baseline_row(20, buy_edge=0.01, fair_token_price=0.53, best_bid=0.54, best_ask=0.55, seconds_to_end=220, event_slug="evt-1"),
            _baseline_row(0, buy_edge=0.04, fair_token_price=0.56, best_bid=0.49, best_ask=0.50, seconds_to_end=240, event_slug="evt-2"),
            _baseline_row(2, buy_edge=0.05, fair_token_price=0.56, best_bid=0.49, best_ask=0.50, seconds_to_end=238, event_slug="evt-2"),
            _baseline_row(4, buy_edge=0.02, fair_token_price=0.56, best_bid=0.50, best_ask=0.51, seconds_to_end=236, event_slug="evt-2"),
            _baseline_row(70, buy_edge=0.05, fair_token_price=0.60, best_bid=0.55, best_ask=0.56, seconds_to_end=170, event_slug="evt-2"),
        ]
    )

    episodes = replay_baseline_up_strategy(rows, max_holding_seconds=60.0)
    summary = edge_strategy_summary(episodes)

    assert summary["episodes"] == 2
    assert summary["pnl_basis"] == "net_pnl_with_fee"
    assert "mean_execution_cost" in summary
    assert "mean_estimated_fee" in summary
    assert "hold_edge_reversal" in summary["exit_reasons"]
    assert "max_holding_time" in summary["exit_reasons"]


def test_baseline_up_strategy_adds_pnl_decomposition_columns() -> None:
    rows = pd.DataFrame(
        [
            _baseline_row(0, buy_edge=0.04, fair_token_price=0.56, best_bid=0.49, best_ask=0.50, seconds_to_end=240),
            _baseline_row(2, buy_edge=0.05, fair_token_price=0.56, best_bid=0.49, best_ask=0.50, seconds_to_end=238),
            _baseline_row(4, buy_edge=0.02, fair_token_price=0.56, best_bid=0.50, best_ask=0.51, seconds_to_end=236),
            _baseline_row(20, buy_edge=0.01, fair_token_price=0.59, best_bid=0.57, best_ask=0.58, seconds_to_end=220),
        ]
    )

    episodes = replay_baseline_up_strategy(rows)

    assert len(episodes) == 1
    assert episodes.loc[0, "entry_mid_price"] == pytest.approx(0.505)
    assert episodes.loc[0, "exit_mid_price"] == pytest.approx(0.575)
    assert episodes.loc[0, "mid_pnl"] == pytest.approx(0.07)
    assert episodes.loc[0, "gross_pnl"] == pytest.approx(0.06)
    assert episodes.loc[0, "execution_cost"] == pytest.approx(0.01)
    assert episodes.loc[0, "estimated_fee"] > 0
    assert episodes.loc[0, "net_pnl_with_fee"] < episodes.loc[0, "gross_pnl"]
