import argparse
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone

import yaml

from polymarket_quant.ingestion.client import PolymarketRESTClient
from polymarket_quant.ingestion.pipeline import IngestionPipeline
from polymarket_quant.utils.logger import get_logger

try:
    from windowing import resolve_active_window, resolve_full_window, wait_until
except ModuleNotFoundError:
    from scripts.windowing import resolve_active_window, resolve_full_window, wait_until

logger = get_logger(__name__)

DEFAULT_EVENT_SLUG_PREFIXES = ["btc-updown-5m", "eth-updown-5m"]
DEFAULT_SERIES_SLUGS = ["btc-up-or-down-5m", "eth-up-or-down-5m"]


def _submit_chunk_write(
    writer: ThreadPoolExecutor,
    pending_writes: list[Future[None]],
    pipeline: IngestionPipeline,
    raw_snapshots: list[dict],
    level_rows: list[dict],
    summary_rows: list[dict],
    run_timestamp: str,
) -> None:
    if not raw_snapshots:
        return
    pending_writes.append(
        writer.submit(
            pipeline.save_crypto_5m_orderbook_collection,
            raw_snapshots=raw_snapshots,
            level_rows=level_rows,
            summary_rows=summary_rows,
            run_timestamp=run_timestamp,
        )
    )


def collect_orderbooks(
    config_path: str = "configs/base.yaml",
    mode: str = "duration",
    interval_seconds: float = 5.0,
    duration_seconds: float = 300.0,
    event_duration_seconds: int = 300,
    window_start: str | None = None,
    run_timestamp: str | None = None,
    event_limit: int = 1,
    event_slug_prefixes: list[str] | None = None,
    series_slugs: list[str] | None = None,
) -> dict[str, int | str]:
    """Collect Polymarket BTC/ETH 5m orderbooks and persist raw/processed datasets."""
    event_slug_prefixes = event_slug_prefixes or DEFAULT_EVENT_SLUG_PREFIXES
    series_slugs = series_slugs or DEFAULT_SERIES_SLUGS

    with open(config_path, "r") as f:
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
    active_window = None
    if mode == "full-window":
        full_window = resolve_full_window(
            event_slug_prefixes=event_slug_prefixes,
            event_duration_seconds=event_duration_seconds,
            window_start=window_start,
        )
        now = datetime.now(timezone.utc)
        if now >= full_window.end:
            raise RuntimeError(
                "Refusing to collect an expired full window: "
                f"{full_window.start.isoformat()} -> {full_window.end.isoformat()}"
            )
        wait_until(full_window.start, interval_seconds, logger)
        now = datetime.now(timezone.utc)
        if now >= full_window.end:
            raise RuntimeError(
                "Full window expired before orderbook collection could start: "
                f"{full_window.start.isoformat()} -> {full_window.end.isoformat()}"
            )
        duration_seconds = (full_window.end - now).total_seconds()
        run_timestamp = run_timestamp or full_window.start.strftime("%Y%m%d_%H%M%S")
        window_slugs = full_window.event_slugs
        active_window = full_window
        event_limit = len(window_slugs)
        logger.info(
            "Collecting full event window %s -> %s for %s",
            full_window.start.isoformat(),
            full_window.end.isoformat(),
            window_slugs,
        )

    run_timestamp = run_timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    deadline = time.monotonic() + duration_seconds
    raw_snapshots: list[dict] = []
    level_rows: list[dict] = []
    summary_rows: list[dict] = []
    event_detail_cache: dict[str, dict] = {}
    total_raw_snapshots = 0
    total_level_rows = 0
    total_summary_rows = 0
    poll_count = 0
    writer = ThreadPoolExecutor(max_workers=1)
    pending_writes: list[Future[None]] = []

    logger.info(
        "Starting live orderbook collector for %s seconds at %s second intervals",
        duration_seconds,
        interval_seconds,
    )

    chunk_window = active_window or resolve_active_window(
        event_slug_prefixes=event_slug_prefixes,
        event_duration_seconds=event_duration_seconds,
    )
    if mode == "duration":
        window_slugs = chunk_window.event_slugs
        event_limit = len(window_slugs)

    try:
        while time.monotonic() < deadline:
            current_now = datetime.now(timezone.utc)
            if mode == "duration":
                current_window = resolve_active_window(
                    event_slug_prefixes=event_slug_prefixes,
                    event_duration_seconds=event_duration_seconds,
                    now=current_now,
                )
                if current_window.start != chunk_window.start:
                    _submit_chunk_write(
                        writer=writer,
                        pending_writes=pending_writes,
                        pipeline=pipeline,
                        raw_snapshots=raw_snapshots,
                        level_rows=level_rows,
                        summary_rows=summary_rows,
                        run_timestamp=chunk_window.start.strftime("%Y%m%d_%H%M%S"),
                    )
                    raw_snapshots = []
                    level_rows = []
                    summary_rows = []
                    chunk_window = current_window
                    window_slugs = current_window.event_slugs
                    event_limit = len(window_slugs)
                    event_detail_cache = {}

            poll_started = time.monotonic()
            batch_raw, batch_levels, batch_summary = pipeline.collect_crypto_5m_orderbooks_once(
                series_slugs=series_slugs,
                event_limit=event_limit,
                event_slug_prefixes=event_slug_prefixes,
                event_slugs=window_slugs,
                event_details_by_slug=event_detail_cache,
            )
            raw_snapshots.extend(batch_raw)
            level_rows.extend(batch_levels)
            summary_rows.extend(batch_summary)
            total_raw_snapshots += len(batch_raw)
            total_level_rows += len(batch_levels)
            total_summary_rows += len(batch_summary)
            poll_count += 1

            logger.info(
                "Poll %s collected %s token books, %s level rows",
                poll_count,
                len(batch_raw),
                len(batch_levels),
            )

            elapsed = time.monotonic() - poll_started
            time.sleep(max(0.0, interval_seconds - elapsed))
    except KeyboardInterrupt:
        logger.info("Collector interrupted; saving partial data.")
    finally:
        if raw_snapshots:
            _submit_chunk_write(
                writer=writer,
                pending_writes=pending_writes,
                pipeline=pipeline,
                raw_snapshots=raw_snapshots,
                level_rows=level_rows,
                summary_rows=summary_rows,
                run_timestamp=chunk_window.start.strftime("%Y%m%d_%H%M%S"),
            )
        writer.shutdown(wait=True)
        for pending_write in pending_writes:
            pending_write.result()

    return {
        "raw_snapshots": total_raw_snapshots,
        "level_rows": total_level_rows,
        "summary_rows": total_summary_rows,
        "polls": poll_count,
        "run_timestamp": run_timestamp,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect live BTC/ETH 5m orderbook time-series data.")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config file")
    parser.add_argument(
        "--mode",
        choices=["duration", "full-window"],
        default="duration",
        help="Use continuous duration mode or aligned full-window mode",
    )
    parser.add_argument("--interval-seconds", type=float, default=5.0, help="Seconds between orderbook polls")
    parser.add_argument("--duration-seconds", type=float, default=300.0, help="Total collection duration")
    parser.add_argument("--event-duration-seconds", type=int, default=300, help="Chunk flush interval in seconds")
    parser.add_argument(
        "--window-start",
        type=str,
        default=None,
        help="Optional aligned start time as Unix seconds or ISO timestamp",
    )
    parser.add_argument("--event-limit", type=int, default=1, help="Current events per BTC/ETH series to scan")
    parser.add_argument(
        "--event-slug-prefixes",
        nargs="+",
        default=DEFAULT_EVENT_SLUG_PREFIXES,
        help="Only collect events whose slugs start with these prefixes",
    )
    parser.add_argument(
        "--series-slugs",
        nargs="+",
        default=DEFAULT_SERIES_SLUGS,
        help="Gamma series slugs to collect",
    )
    args = parser.parse_args()

    collect_orderbooks(
        config_path=args.config,
        mode=args.mode,
        interval_seconds=args.interval_seconds,
        duration_seconds=args.duration_seconds,
        event_duration_seconds=args.event_duration_seconds,
        window_start=args.window_start,
        run_timestamp=None,
        event_limit=args.event_limit,
        event_slug_prefixes=args.event_slug_prefixes,
        series_slugs=args.series_slugs,
    )


if __name__ == "__main__":
    main()
