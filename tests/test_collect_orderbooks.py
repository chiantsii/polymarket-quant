from datetime import datetime, timezone

from scripts.collect_orderbooks import _next_window_start, _resolve_full_window


def test_next_window_start_uses_next_five_minute_boundary() -> None:
    now = datetime.fromtimestamp(1775575585, tz=timezone.utc)

    start = _next_window_start(now, event_duration_seconds=300)

    assert int(start.timestamp()) == 1775575800


def test_resolve_full_window_builds_btc_eth_event_slugs() -> None:
    slugs, start, end = _resolve_full_window(
        event_slug_prefixes=["btc-updown-5m", "eth-updown-5m"],
        event_duration_seconds=300,
        window_start="1775575800",
    )

    assert int(start.timestamp()) == 1775575800
    assert int(end.timestamp()) == 1775576100
    assert slugs == ["btc-updown-5m-1775575800", "eth-updown-5m-1775575800"]
