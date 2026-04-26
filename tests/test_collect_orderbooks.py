import json
from datetime import datetime, timezone

import pytest

from polymarket_quant.ingestion.storage import save_json_and_parquet_rows
from scripts.run_window_capture import _resolve_window_capture_plan, run_window_capture
from scripts.windowing import FullWindow
from scripts.windowing import next_window_start, resolve_full_window


def test_next_window_start_uses_next_five_minute_boundary() -> None:
    now = datetime.fromtimestamp(1775575585, tz=timezone.utc)

    start = next_window_start(now, event_duration_seconds=300)

    assert int(start.timestamp()) == 1775575800


def test_resolve_full_window_builds_btc_eth_event_slugs() -> None:
    full_window = resolve_full_window(
        event_slug_prefixes=["btc-updown-5m", "eth-updown-5m"],
        event_duration_seconds=300,
        window_start="1775575800",
    )

    assert int(full_window.start.timestamp()) == 1775575800
    assert int(full_window.end.timestamp()) == 1775576100
    assert full_window.event_slugs == ["btc-updown-5m-1775575800", "eth-updown-5m-1775575800"]


def test_run_window_capture_rejects_zero_windows() -> None:
    with pytest.raises(ValueError):
        run_window_capture(windows=0)


def test_resolve_window_capture_plan_uses_window_start_timestamps() -> None:
    starts = [1775575800, 1775576100, 1775576400]
    plans = [
        _resolve_window_capture_plan(
            event_slug_prefixes=["btc-updown-5m", "eth-updown-5m"],
            event_duration_seconds=300,
            window_start=str(starts[0]),
            window_index=index,
        )
        for index in range(3)
    ]

    assert [item["window_start"] for item in plans] == [str(start) for start in starts]
    assert [item["run_timestamp"] for item in plans] == [
        datetime.fromtimestamp(start, tz=timezone.utc).strftime("%Y%m%d_%H%M%S") for start in starts
    ]


def test_resolve_window_capture_plan_catches_up_when_scheduler_falls_behind(monkeypatch) -> None:
    latest_start = 1775576400

    def fake_resolve_full_window(
        event_slug_prefixes=["btc-updown-5m", "eth-updown-5m"],
        event_duration_seconds=300,
        window_start=None,
    ):
        if window_start is None:
            start_epoch = latest_start
        else:
            start_epoch = int(window_start)
        start_dt = datetime.fromtimestamp(start_epoch, tz=timezone.utc)
        end_dt = datetime.fromtimestamp(start_epoch + event_duration_seconds, tz=timezone.utc)
        return FullWindow(
            start=start_dt,
            end=end_dt,
            event_slugs=[f"{prefix}-{start_epoch}" for prefix in event_slug_prefixes],
        )

    monkeypatch.setattr("scripts.run_window_capture.resolve_full_window", fake_resolve_full_window)

    plan = _resolve_window_capture_plan(
        event_slug_prefixes=["btc-updown-5m", "eth-updown-5m"],
        event_duration_seconds=300,
        window_start=None,
        window_index=2,
        last_planned_window_start_epoch=1775575800,
    )

    assert plan["window_start"] == str(latest_start)
    assert plan["skipped_window_count"] == 1


def test_save_json_and_parquet_rows_writes_empty_raw_json(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"

    save_json_and_parquet_rows(
        rows=[],
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        raw_name="spot_raw_20260408_165000.json",
        latest_raw_name="spot_raw_latest.json",
        parquet_name="spot_20260408_165000.parquet",
        latest_parquet_name="spot_latest.parquet",
    )

    raw_path = raw_dir / "spot_raw_20260408_165000.json"
    latest_raw_path = raw_dir / "spot_raw_latest.json"

    assert raw_path.exists()
    assert latest_raw_path.exists()
    assert json.loads(raw_path.read_text()) == []
    assert json.loads(latest_raw_path.read_text()) == []
    assert not processed_dir.exists()
