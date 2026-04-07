import time
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class FullWindow:
    start: datetime
    end: datetime
    event_slugs: list[str]


def resolve_full_window(
    event_slug_prefixes: list[str],
    event_duration_seconds: int,
    window_start: str | None = None,
) -> FullWindow:
    """Resolve full-window timestamps and event slugs from UTC 5m boundaries."""
    start_time = parse_window_start(window_start) if window_start else next_window_start(
        datetime.now(timezone.utc),
        event_duration_seconds,
    )
    end_time = datetime.fromtimestamp(start_time.timestamp() + event_duration_seconds, tz=timezone.utc)
    start_epoch = int(start_time.timestamp())
    event_slugs = [f"{prefix}-{start_epoch}" for prefix in event_slug_prefixes]
    return FullWindow(start=start_time, end=end_time, event_slugs=event_slugs)


def next_window_start(now: datetime, event_duration_seconds: int) -> datetime:
    now = now.astimezone(timezone.utc)
    now_epoch = int(now.timestamp())
    next_epoch = ((now_epoch // event_duration_seconds) + 1) * event_duration_seconds
    return datetime.fromtimestamp(next_epoch, tz=timezone.utc)


def parse_window_start(value: str) -> datetime:
    if value.isdigit():
        return datetime.fromtimestamp(int(value), tz=timezone.utc)

    parsed = parse_iso_datetime(value)
    if parsed is None:
        raise ValueError("--window-start must be Unix seconds or an ISO timestamp")
    return parsed


def wait_until(start_time: datetime, poll_seconds: float, logger) -> None:
    while True:
        now = datetime.now(timezone.utc)
        seconds_to_start = (start_time - now).total_seconds()
        if seconds_to_start <= 0:
            return
        sleep_seconds = min(seconds_to_start, max(poll_seconds, 1.0))
        logger.info("Waiting %.2f seconds for full window to start at %s", sleep_seconds, start_time.isoformat())
        time.sleep(sleep_seconds)


def parse_iso_datetime(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
