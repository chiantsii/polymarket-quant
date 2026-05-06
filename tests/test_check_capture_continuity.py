from datetime import datetime, timedelta, timezone

import pandas as pd

from scripts.check_capture_continuity import (
    _event_start_from_slug,
    _summarize_event_slug_sequence,
    _summarize_event_source,
)


def test_summarize_event_slug_sequence_counts_missing_windows() -> None:
    frame = pd.DataFrame(
        [
            {"asset": "BTC", "event_slug": "btc-updown-5m-1775578800"},
            {"asset": "BTC", "event_slug": "btc-updown-5m-1775579100"},
            {"asset": "BTC", "event_slug": "btc-updown-5m-1775579700"},
        ]
    )

    summary = _summarize_event_slug_sequence(frame, event_duration_seconds=300.0)

    last_row = summary.iloc[-1]
    assert last_row["gap_seconds"] == 600.0
    assert last_row["missing_windows"] == 1


def test_summarize_event_source_reports_internal_gap_and_boundary_offsets() -> None:
    event_slug = "btc-updown-5m-1775578800"
    event_start = _event_start_from_slug(event_slug)
    assert event_start is not None
    rows = pd.DataFrame(
        [
            {"asset": "BTC", "event_slug": event_slug, "_collected_at_dt": event_start + timedelta(seconds=2)},
            {"asset": "BTC", "event_slug": event_slug, "_collected_at_dt": event_start + timedelta(seconds=3)},
            {"asset": "BTC", "event_slug": event_slug, "_collected_at_dt": event_start + timedelta(seconds=8)},
        ]
    )

    summary = _summarize_event_source(rows, source="spot", event_duration_seconds=300.0)
    record = summary.iloc[0]

    assert record["rows"] == 3
    assert record["start_offset_seconds"] == 2.0
    assert record["max_internal_gap_seconds"] == 5.0
    assert record["end_shortfall_seconds"] == 292.0
