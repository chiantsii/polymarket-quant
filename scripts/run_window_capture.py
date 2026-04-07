import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from polymarket_quant.utils.logger import get_logger

try:
    from backfill_resolutions import backfill_resolutions
    from collect_orderbooks import DEFAULT_EVENT_SLUG_PREFIXES, DEFAULT_SERIES_SLUGS, collect_orderbooks
    from collect_spot_prices import collect_spot_prices
    from windowing import resolve_full_window
except ModuleNotFoundError:
    from scripts.backfill_resolutions import backfill_resolutions
    from scripts.collect_orderbooks import DEFAULT_EVENT_SLUG_PREFIXES, DEFAULT_SERIES_SLUGS, collect_orderbooks
    from scripts.collect_spot_prices import collect_spot_prices
    from scripts.windowing import resolve_full_window

logger = get_logger(__name__)


def run_window_capture(
    config_path: str = "configs/base.yaml",
    interval_seconds: float = 1.0,
    event_duration_seconds: int = 300,
    window_start: str | None = None,
    windows: int = 1,
    backfill: bool = False,
    settlement_wait_seconds: float = 600.0,
    event_slug_prefixes: list[str] | None = None,
    series_slugs: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Collect aligned Polymarket orderbook and spot data for full 5m windows."""
    if windows < 1:
        raise ValueError("--windows must be at least 1")

    event_slug_prefixes = event_slug_prefixes or DEFAULT_EVENT_SLUG_PREFIXES
    series_slugs = series_slugs or DEFAULT_SERIES_SLUGS

    first_window = resolve_full_window(
        event_slug_prefixes=event_slug_prefixes,
        event_duration_seconds=event_duration_seconds,
        window_start=window_start,
    )
    first_start_epoch = int(first_window.start.timestamp())
    results = []

    for window_index in range(windows):
        current_start_epoch = first_start_epoch + window_index * event_duration_seconds
        current_window_start = str(current_start_epoch)
        full_window = resolve_full_window(
            event_slug_prefixes=event_slug_prefixes,
            event_duration_seconds=event_duration_seconds,
            window_start=current_window_start,
        )

        logger.info(
            "Starting window capture %s/%s for %s -> %s (%s)",
            window_index + 1,
            windows,
            full_window.start.isoformat(),
            full_window.end.isoformat(),
            full_window.event_slugs,
        )

        results.append(
            _run_single_window_capture(
                config_path=config_path,
                interval_seconds=interval_seconds,
                event_duration_seconds=event_duration_seconds,
                window_start=current_window_start,
                event_slug_prefixes=event_slug_prefixes,
                series_slugs=series_slugs,
            )
        )

    if backfill:
        logger.info(
            "Waiting %.2f seconds before backfilling resolution labels",
            settlement_wait_seconds,
        )
        time.sleep(max(0.0, settlement_wait_seconds))
        backfill_result = backfill_resolutions(
            config_path=config_path,
            event_duration_seconds=float(event_duration_seconds),
            settlement_delay_seconds=0.0,
            event_slug_prefixes=event_slug_prefixes,
        )
        logger.info("Backfill result: %s", backfill_result)

    return results


def _run_single_window_capture(
    config_path: str,
    interval_seconds: float,
    event_duration_seconds: int,
    window_start: str,
    event_slug_prefixes: list[str],
    series_slugs: list[str],
) -> dict[str, Any]:
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(
                collect_orderbooks,
                config_path=config_path,
                mode="full-window",
                interval_seconds=interval_seconds,
                duration_seconds=float(event_duration_seconds),
                event_duration_seconds=event_duration_seconds,
                window_start=window_start,
                event_limit=len(event_slug_prefixes),
                event_slug_prefixes=event_slug_prefixes,
                series_slugs=series_slugs,
            ): "orderbooks",
            executor.submit(
                collect_spot_prices,
                config_path=config_path,
                mode="full-window",
                interval_seconds=interval_seconds,
                duration_seconds=float(event_duration_seconds),
                event_duration_seconds=event_duration_seconds,
                window_start=window_start,
            ): "spot",
        }

        results: dict[str, Any] = {}
        for future in as_completed(futures):
            collector_name = futures[future]
            results[collector_name] = future.result()
            logger.info("%s collector completed: %s", collector_name, results[collector_name])

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect aligned Polymarket orderbook and BTC/ETH spot data for full 5m windows."
    )
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config file")
    parser.add_argument("--interval-seconds", type=float, default=1.0, help="Seconds between polls")
    parser.add_argument("--event-duration-seconds", type=int, default=300, help="Full-window event duration")
    parser.add_argument(
        "--window-start",
        type=str,
        default=None,
        help="Optional first window start as Unix seconds or ISO timestamp",
    )
    parser.add_argument("--windows", type=int, default=1, help="Number of consecutive full windows to collect")
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Wait after capture and backfill resolution labels from collected event slugs",
    )
    parser.add_argument(
        "--settlement-wait-seconds",
        type=float,
        default=600.0,
        help="Seconds to wait before backfill when --backfill is enabled",
    )
    parser.add_argument(
        "--event-slug-prefixes",
        nargs="+",
        default=DEFAULT_EVENT_SLUG_PREFIXES,
        help="Event slug prefixes to collect",
    )
    parser.add_argument(
        "--series-slugs",
        nargs="+",
        default=DEFAULT_SERIES_SLUGS,
        help="Gamma series slugs matching the event slug prefixes",
    )
    args = parser.parse_args()

    run_window_capture(
        config_path=args.config,
        interval_seconds=args.interval_seconds,
        event_duration_seconds=args.event_duration_seconds,
        window_start=args.window_start,
        windows=args.windows,
        backfill=args.backfill,
        settlement_wait_seconds=args.settlement_wait_seconds,
        event_slug_prefixes=args.event_slug_prefixes,
        series_slugs=args.series_slugs,
    )


if __name__ == "__main__":
    main()
