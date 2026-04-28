import json
from datetime import datetime, timezone
from datetime import timedelta

import pytest
import yaml

from polymarket_quant.ingestion.storage import save_json_and_parquet_rows
from scripts.collect_orderbooks import collect_orderbooks
from scripts.run_window_capture import run_window_capture
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


def test_run_window_capture_uses_continuous_duration(monkeypatch) -> None:
    observed = {}

    def fake_capture(**kwargs):
        observed.update(kwargs)
        return {"orderbooks": {"polls": 1}, "spot": {"polls": 1}, "run_timestamp": None}

    monkeypatch.setattr("scripts.run_window_capture._run_single_window_capture", fake_capture)

    results = run_window_capture(
        windows=3,
        event_duration_seconds=300,
        interval_seconds=1.0,
        event_slug_prefixes=["btc-updown-5m", "eth-updown-5m"],
        series_slugs=["btc-up-or-down-5m", "eth-up-or-down-5m"],
    )

    assert len(results) == 1
    assert observed["mode"] == "duration"
    assert observed["duration_seconds"] == 900.0
    assert observed["event_duration_seconds"] == 300


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


def test_collect_orderbooks_rejects_expired_full_window(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "raw_dir": str(tmp_path / "raw"),
                    "processed_dir": str(tmp_path / "processed"),
                },
                "api": {
                    "gamma_url": "https://gamma-api.polymarket.com",
                    "clob_url": "https://clob.polymarket.com",
                },
            }
        )
    )
    past_start = datetime.now(timezone.utc) - timedelta(minutes=10)
    past_end = past_start + timedelta(minutes=5)

    monkeypatch.setattr(
        "scripts.collect_orderbooks.resolve_full_window",
        lambda **kwargs: FullWindow(
            start=past_start,
            end=past_end,
            event_slugs=["btc-updown-5m-test", "eth-updown-5m-test"],
        ),
    )

    with pytest.raises(RuntimeError, match="Refusing to collect an expired full window"):
        collect_orderbooks(
            config_path=str(config_path),
            mode="full-window",
            interval_seconds=1.0,
            duration_seconds=300.0,
            event_duration_seconds=300,
            window_start=str(int(past_start.timestamp())),
            run_timestamp="20260428_000000",
            event_limit=2,
            event_slug_prefixes=["btc-updown-5m", "eth-updown-5m"],
            series_slugs=["btc-up-or-down-5m", "eth-up-or-down-5m"],
        )
