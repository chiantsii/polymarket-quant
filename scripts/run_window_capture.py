import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from polymarket_quant.utils.logger import get_logger

try:
    from collect_orderbooks import DEFAULT_EVENT_SLUG_PREFIXES, DEFAULT_SERIES_SLUGS, collect_orderbooks
    from collect_spot_prices import collect_spot_prices
    from windowing import parse_window_start, wait_until
except ModuleNotFoundError:
    from scripts.collect_orderbooks import DEFAULT_EVENT_SLUG_PREFIXES, DEFAULT_SERIES_SLUGS, collect_orderbooks
    from scripts.collect_spot_prices import collect_spot_prices
    from scripts.windowing import parse_window_start, wait_until

logger = get_logger(__name__)


def run_window_capture(
    config_path: str = "configs/base.yaml",
    interval_seconds: float = 1.0,
    event_duration_seconds: int = 300,
    window_start: str | None = None,
    windows: int = 1,
    event_slug_prefixes: list[str] | None = None,
    series_slugs: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Collect continuous BTC/ETH market data for windows * event_duration_seconds."""
    if windows < 1:
        raise ValueError("--windows must be at least 1")

    event_slug_prefixes = event_slug_prefixes or DEFAULT_EVENT_SLUG_PREFIXES
    series_slugs = series_slugs or DEFAULT_SERIES_SLUGS
    total_duration_seconds = float(windows * event_duration_seconds)

    logger.info(
        "Starting continuous aligned capture for %.1f seconds (%s x %ss chunks)",
        total_duration_seconds,
        windows,
        event_duration_seconds,
    )

    if window_start is not None:
        start_time = parse_window_start(window_start)
        wait_until(start_time, interval_seconds, logger)

    result = _run_single_window_capture(
        config_path=config_path,
        interval_seconds=interval_seconds,
        event_duration_seconds=event_duration_seconds,
        window_start=window_start,
        run_timestamp=None,
        event_slug_prefixes=event_slug_prefixes,
        series_slugs=series_slugs,
        mode="duration",
        duration_seconds=total_duration_seconds,
    )
    return [result]


def _run_single_window_capture(
    config_path: str,
    interval_seconds: float,
    event_duration_seconds: int,
    window_start: str | None,
    run_timestamp: str | None,
    event_slug_prefixes: list[str],
    series_slugs: list[str],
    mode: str = "duration",
    duration_seconds: float | None = None,
) -> dict[str, Any]:
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(
                collect_orderbooks,
                config_path=config_path,
                mode=mode,
                interval_seconds=interval_seconds,
                duration_seconds=float(duration_seconds if duration_seconds is not None else event_duration_seconds),
                event_duration_seconds=event_duration_seconds,
                window_start=window_start,
                run_timestamp=run_timestamp,
                event_limit=len(event_slug_prefixes),
                event_slug_prefixes=event_slug_prefixes,
                series_slugs=series_slugs,
            ): "orderbooks",
            executor.submit(
                collect_spot_prices,
                config_path=config_path,
                mode=mode,
                interval_seconds=interval_seconds,
                duration_seconds=float(duration_seconds if duration_seconds is not None else event_duration_seconds),
                event_duration_seconds=event_duration_seconds,
                window_start=window_start,
                run_timestamp=run_timestamp,
                event_slug_prefixes=event_slug_prefixes,
            ): "spot",
        }

        results: dict[str, Any] = {}
        for future in as_completed(futures):
            collector_name = futures[future]
            results[collector_name] = future.result()
            logger.info("%s collector completed: %s", collector_name, results[collector_name])

    return {"run_timestamp": run_timestamp, **results}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continuously collect aligned Polymarket orderbook and BTC/ETH spot data."
    )
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config file")
    parser.add_argument("--interval-seconds", type=float, default=1.0, help="Seconds between polls")
    parser.add_argument(
        "--event-duration-seconds",
        type=int,
        default=300,
        help="Write chunk interval in seconds",
    )
    parser.add_argument(
        "--window-start",
        type=str,
        default=None,
        help="Optional aligned capture start as Unix seconds or ISO timestamp",
    )
    parser.add_argument(
        "--windows",
        type=int,
        default=1,
        help="Total capture duration measured in event-sized chunks",
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
        event_slug_prefixes=args.event_slug_prefixes,
        series_slugs=args.series_slugs,
    )


if __name__ == "__main__":
    main()
