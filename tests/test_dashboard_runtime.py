import json
from pathlib import Path

from polymarket_quant.dashboard.runtime import EmbeddedLiveRuntime, EmbeddedLiveRuntimeConfig, _format_poll_timing


class _StubSource:
    def __init__(self, rows: list[list[dict]]) -> None:
        self.rows = rows
        self.index = 0
        self.last_poll_metrics = {
            "spot_fetch_ms": 11.0,
            "orderbook_fetch_ms": 22.0,
            "market_state_ms": 33.0,
            "event_state_ms": 44.0,
        }

    def poll_new_rows(self) -> list[dict]:
        if self.index >= len(self.rows):
            return []
        payload = self.rows[self.index]
        self.index += 1
        return payload


class _StubDetector:
    def detect(self, rows, show_progress=False):
        return rows


def _event_state_row(collected_at: str, *, buy_edge: float, hold_edge: float, best_bid: float, best_ask: float) -> dict:
    return {
        "event_slug": "btc-updown-5m-test",
        "collected_at": collected_at,
        "outcome_name": "Up",
        "token_id": "tok_up",
        "market_implied_up_probability": 0.65,
        "fundamental_up_probability": 0.66,
        "latent_up_probability": 0.68,
        "fair_up_probability": 0.58,
        "fair_token_price": 0.58,
        "buy_edge": buy_edge,
        "hold_edge": hold_edge,
        "sell_edge": best_bid - 0.58,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "seconds_to_end": 200.0,
    }


def test_embedded_live_runtime_writes_signal_and_order_artifacts(tmp_path: Path) -> None:
    runtime = EmbeddedLiveRuntime(
        EmbeddedLiveRuntimeConfig(
            live_state_output_dir=str(tmp_path / "live_state"),
            live_signal_output_dir=str(tmp_path / "live_signal"),
            paper_output_dir=str(tmp_path / "paper"),
            quiet_runtime_logs=True,
        ),
        source=_StubSource(
            [
                [_event_state_row("2026-05-08T09:00:00+00:00", buy_edge=0.04, hold_edge=0.08, best_bid=0.50, best_ask=0.54)],
                [_event_state_row("2026-05-08T09:00:02+00:00", buy_edge=0.05, hold_edge=0.08, best_bid=0.50, best_ask=0.54)],
                [_event_state_row("2026-05-08T09:00:04+00:00", buy_edge=0.01, hold_edge=0.01, best_bid=0.57, best_ask=0.58)],
                [_event_state_row("2026-05-08T09:00:06+00:00", buy_edge=0.00, hold_edge=-0.01, best_bid=0.57, best_ask=0.58)],
            ]
        ),
        detector=_StubDetector(),
    )

    first_result = runtime.poll_once()
    runtime.poll_once()
    runtime.poll_once()
    runtime.poll_once()

    signal_rows = _read_jsonl(Path(runtime.snapshot_paths()["paper_signal_path"]))
    order_rows = _read_jsonl(Path(runtime.snapshot_paths()["paper_order_path"]))
    trade_rows = _read_jsonl(Path(runtime.snapshot_paths()["paper_trade_path"]))

    assert len(signal_rows) == 4
    assert len(order_rows) == 2
    assert order_rows[0]["side"] == "BUY"
    assert order_rows[1]["side"] == "SELL"
    assert len(trade_rows) == 1
    assert trade_rows[0]["exit_reason"] == "hold_edge_reversal"
    assert first_result["spot_fetch_ms"] == 11.0
    assert first_result["orderbook_fetch_ms"] == 22.0
    assert first_result["market_state_ms"] == 33.0
    assert first_result["event_state_ms"] == 44.0
    assert float(first_result["pricing_ms"]) >= 0.0
    assert float(first_result["executor_ms"]) >= 0.0


def test_format_poll_timing_contains_expected_fields() -> None:
    message = _format_poll_timing(
        {
            "spot_fetch_ms": 1.0,
            "orderbook_fetch_ms": 2.0,
            "market_state_ms": 3.0,
            "event_state_ms": 4.0,
            "pricing_ms": 5.0,
            "executor_ms": 6.0,
            "poll_total_ms": 7.0,
            "new_event_rows": 8,
            "signal_events": 9,
            "order_events": 10,
            "closed_trades": 11,
        }
    )

    assert "spot_fetch_ms=1.0" in message
    assert "orderbook_fetch_ms=2.0" in message
    assert "market_state_ms=3.0" in message
    assert "event_state_ms=4.0" in message
    assert "pricing_ms=5.0" in message
    assert "executor_ms=6.0" in message
    assert "poll_total_ms=7.0" in message


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows
