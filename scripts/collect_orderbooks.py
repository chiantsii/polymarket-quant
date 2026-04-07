import argparse
import time
from datetime import datetime, timezone

import yaml

from polymarket_quant.ingestion.client import PolymarketRESTClient
from polymarket_quant.ingestion.pipeline import IngestionPipeline
from polymarket_quant.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect live BTC/ETH 5m orderbook time-series data.")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config file")
    parser.add_argument(
        "--mode",
        choices=["duration", "full-window"],
        default="duration",
        help="Use duration mode or wait for the next complete 5m event window",
    )
    parser.add_argument("--interval-seconds", type=float, default=5.0, help="Seconds between orderbook polls")
    parser.add_argument("--duration-seconds", type=float, default=300.0, help="Total collection duration")
    parser.add_argument("--event-duration-seconds", type=int, default=300, help="Full-window event duration")
    parser.add_argument(
        "--window-start",
        type=str,
        default=None,
        help="Optional full-window start time as Unix seconds or ISO timestamp",
    )
    parser.add_argument("--event-limit", type=int, default=1, help="Current events per BTC/ETH series to scan")
    parser.add_argument(
        "--event-slug-prefixes",
        nargs="+",
        default=["btc-updown-5m", "eth-updown-5m"],
        help="Only collect events whose slugs start with these prefixes",
    )
    parser.add_argument(
        "--series-slugs",
        nargs="+",
        default=["btc-up-or-down-5m", "eth-up-or-down-5m"],
        help="Gamma series slugs to collect",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    client = PolymarketRESTClient(
        gamma_url=config["api"]["gamma_url"],
        clob_url=config["api"]["clob_url"],
    )
    pipeline = IngestionPipeline(
        client=client,
        raw_dir=config["data"]["raw_dir"],
        processed_dir=config["data"]["processed_dir"],
    )

    window_slugs = None
    if args.mode == "full-window":
        window_slugs, window_start, window_end = _resolve_full_window(
            event_slug_prefixes=args.event_slug_prefixes,
            event_duration_seconds=args.event_duration_seconds,
            window_start=args.window_start,
        )
        _wait_until(window_start, args.interval_seconds)
        args.duration_seconds = max(0.0, (window_end - datetime.now(timezone.utc)).total_seconds())
        args.event_limit = len(window_slugs)
        logger.info(
            "Collecting full event window %s -> %s for %s",
            window_start.isoformat(),
            window_end.isoformat(),
            window_slugs,
        )

    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    deadline = time.monotonic() + args.duration_seconds
    raw_snapshots = []
    level_rows = []
    summary_rows = []
    poll_count = 0

    logger.info(
        "Starting live orderbook collector for %s seconds at %s second intervals",
        args.duration_seconds,
        args.interval_seconds,
    )

    try:
        while time.monotonic() < deadline:
            poll_started = time.monotonic()
            batch_raw, batch_levels, batch_summary = pipeline.collect_crypto_5m_orderbooks_once(
                series_slugs=args.series_slugs,
                event_limit=args.event_limit,
                event_slug_prefixes=args.event_slug_prefixes,
                event_slugs=window_slugs,
            )
            raw_snapshots.extend(batch_raw)
            level_rows.extend(batch_levels)
            summary_rows.extend(batch_summary)
            poll_count += 1

            logger.info(
                "Poll %s collected %s token books, %s level rows",
                poll_count,
                len(batch_raw),
                len(batch_levels),
            )

            elapsed = time.monotonic() - poll_started
            time.sleep(max(0.0, args.interval_seconds - elapsed))
    except KeyboardInterrupt:
        logger.info("Collector interrupted; saving partial data.")
    finally:
        pipeline.save_crypto_5m_orderbook_collection(
            raw_snapshots=raw_snapshots,
            level_rows=level_rows,
            summary_rows=summary_rows,
            run_timestamp=run_timestamp,
        )


def _resolve_full_window(
    event_slug_prefixes: list[str],
    event_duration_seconds: int,
    window_start: str | None = None,
) -> tuple[list[str], datetime, datetime]:
    """Resolve the full 5m event slugs from the UTC event-start timestamp."""
    start_time = _parse_window_start(window_start) if window_start else _next_window_start(
        datetime.now(timezone.utc),
        event_duration_seconds,
    )
    end_time = datetime.fromtimestamp(start_time.timestamp() + event_duration_seconds, tz=timezone.utc)
    start_epoch = int(start_time.timestamp())
    slugs = [f"{prefix}-{start_epoch}" for prefix in event_slug_prefixes]
    return slugs, start_time, end_time


def _next_window_start(now: datetime, event_duration_seconds: int) -> datetime:
    now = now.astimezone(timezone.utc)
    now_epoch = int(now.timestamp())
    next_epoch = ((now_epoch // event_duration_seconds) + 1) * event_duration_seconds
    return datetime.fromtimestamp(next_epoch, tz=timezone.utc)


def _parse_window_start(value: str) -> datetime:
    if value.isdigit():
        return datetime.fromtimestamp(int(value), tz=timezone.utc)

    parsed = _parse_iso_datetime(value)
    if parsed is None:
        raise ValueError("--window-start must be Unix seconds or an ISO timestamp")
    return parsed


def _wait_until(start_time: datetime, poll_seconds: float) -> None:
    while True:
        now = datetime.now(timezone.utc)
        seconds_to_start = (start_time - now).total_seconds()
        if seconds_to_start <= 0:
            return
        sleep_seconds = min(seconds_to_start, max(poll_seconds, 1.0))
        logger.info("Waiting %.2f seconds for full window to start at %s", sleep_seconds, start_time.isoformat())
        time.sleep(sleep_seconds)


def _parse_iso_datetime(value) -> datetime | None:
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


if __name__ == "__main__":
    main()
