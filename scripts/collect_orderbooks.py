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
        window_slugs, window_end = _wait_for_next_full_window(
            client=client,
            series_slugs=args.series_slugs,
            event_slug_prefixes=args.event_slug_prefixes,
            poll_seconds=args.interval_seconds,
        )
        args.duration_seconds = max(0.0, (window_end - datetime.now(timezone.utc)).total_seconds())
        args.event_limit = len(window_slugs)
        logger.info("Collecting full event window for %s until %s", window_slugs, window_end.isoformat())

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


def _wait_for_next_full_window(
    client: PolymarketRESTClient,
    series_slugs: list[str],
    event_slug_prefixes: list[str],
    poll_seconds: float,
) -> tuple[list[str], datetime]:
    """Wait until BTC and ETH next-window events become current, then return their slugs/end time."""
    target_slugs: list[str] | None = None
    target_start: datetime | None = None
    target_end: datetime | None = None

    while True:
        now = datetime.now(timezone.utc)
        upcoming = _next_events_by_series(client, series_slugs, event_slug_prefixes, now)
        if len(upcoming) == len(series_slugs):
            starts = [_parse_iso_datetime(event.get("startTime") or event.get("startDate")) for event in upcoming.values()]
            ends = [_parse_iso_datetime(event.get("endDate")) for event in upcoming.values()]
            if all(starts) and all(ends):
                latest_start = max(starts)
                earliest_end = min(ends)
                if target_slugs is None:
                    target_slugs = [event["slug"] for event in upcoming.values()]
                    target_start = latest_start
                    target_end = earliest_end
                    logger.info("Next complete window is %s -> %s for %s", target_start, target_end, target_slugs)

                if target_start <= now < target_end:
                    return target_slugs, target_end

                sleep_seconds = min(max((target_start - now).total_seconds(), 0.0), max(poll_seconds, 1.0))
                logger.info("Waiting %.2f seconds for next complete window to start", sleep_seconds)
                time.sleep(sleep_seconds)
                continue

        logger.info("Waiting for next BTC/ETH 5m window metadata.")
        time.sleep(max(poll_seconds, 1.0))


def _next_events_by_series(
    client: PolymarketRESTClient,
    series_slugs: list[str],
    event_slug_prefixes: list[str],
    now: datetime,
) -> dict[str, dict]:
    events_by_series = {}
    for series_slug in series_slugs:
        series_payloads = client.fetch_series(series_slug)
        if not series_payloads:
            continue
        events = series_payloads[0].get("events", [])
        candidates = [
            event
            for event in events
            if event.get("closed") is not True
            and isinstance(event.get("slug"), str)
            and any(event["slug"].startswith(prefix) for prefix in event_slug_prefixes)
        ]
        candidates = [
            event
            for event in candidates
            if (_parse_iso_datetime(event.get("startTime") or event.get("startDate")) or now) >= now
            and (_parse_iso_datetime(event.get("endDate")) or now) > now
        ]
        candidates = sorted(candidates, key=lambda event: event.get("startTime") or event.get("startDate") or "")
        if candidates:
            events_by_series[series_slug] = candidates[0]
    return events_by_series


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
