import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from typing import Any

from polymarket_quant.utils.logger import get_logger

try:
    from collect_orderbooks import DEFAULT_EVENT_SLUG_PREFIXES, DEFAULT_SERIES_SLUGS, collect_orderbooks
    from collect_spot_prices import collect_spot_prices
    from windowing import resolve_full_window
except ModuleNotFoundError:
    from scripts.collect_orderbooks import DEFAULT_EVENT_SLUG_PREFIXES, DEFAULT_SERIES_SLUGS, collect_orderbooks
    from scripts.collect_spot_prices import collect_spot_prices
    from scripts.windowing import resolve_full_window

logger = get_logger(__name__)


def _resolve_window_capture_plan(
    event_slug_prefixes: list[str],
    event_duration_seconds: int,
    window_start: str | None,
    window_index: int,
    last_planned_window_start_epoch: int | None = None,
) -> dict[str, Any]:
    skipped_window_count = 0

    if window_start is not None:
        first_window = resolve_full_window(
            event_slug_prefixes=event_slug_prefixes,
            event_duration_seconds=event_duration_seconds,
            window_start=window_start,
        )
        target_start_epoch = int(first_window.start.timestamp()) + (window_index * event_duration_seconds)
    else:
        latest_window = resolve_full_window(
            event_slug_prefixes=event_slug_prefixes,
            event_duration_seconds=event_duration_seconds,
        )
        latest_start_epoch = int(latest_window.start.timestamp())
        if last_planned_window_start_epoch is None:
            target_start_epoch = latest_start_epoch
        else:
            desired_start_epoch = last_planned_window_start_epoch + event_duration_seconds
            target_start_epoch = max(desired_start_epoch, latest_start_epoch)
            skipped_window_count = max(0, (target_start_epoch - desired_start_epoch) // event_duration_seconds)

    current_window_start = str(target_start_epoch)
    full_window = resolve_full_window(
        event_slug_prefixes=event_slug_prefixes,
        event_duration_seconds=event_duration_seconds,
        window_start=current_window_start,
    )
    return {
        "window_index": window_index,
        "window_start": current_window_start,
        "run_timestamp": full_window.start.strftime("%Y%m%d_%H%M%S"),
        "full_window": full_window,
        "skipped_window_count": skipped_window_count,
    }


def _log_queued_window_capture(plan: dict[str, Any], windows: int) -> None:
    if plan["skipped_window_count"] > 0:
        logger.warning(
            "Capture scheduler fell behind; skipped %s stale full-window(s) and resumed at %s -> %s (%s)",
            plan["skipped_window_count"],
            plan["full_window"].start.isoformat(),
            plan["full_window"].end.isoformat(),
            plan["full_window"].event_slugs,
        )

    logger.info(
        "Queueing window capture %s/%s for %s -> %s (%s)",
        plan["window_index"] + 1,
        windows,
        plan["full_window"].start.isoformat(),
        plan["full_window"].end.isoformat(),
        plan["full_window"].event_slugs,
    )


def run_window_capture(
    config_path: str = "configs/base.yaml",
    interval_seconds: float = 1.0,
    event_duration_seconds: int = 300,
    window_start: str | None = None,
    windows: int = 1,
    event_slug_prefixes: list[str] | None = None,
    series_slugs: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Collect aligned Polymarket orderbook and spot data for full 5m windows."""
    if windows < 1:
        raise ValueError("--windows must be at least 1")

    event_slug_prefixes = event_slug_prefixes or DEFAULT_EVENT_SLUG_PREFIXES
    series_slugs = series_slugs or DEFAULT_SERIES_SLUGS

    queued_window_count = min(2, windows)
    next_window_index = 0
    results_by_index: dict[int, dict[str, Any]] = {}
    last_planned_window_start_epoch: int | None = None

    with ThreadPoolExecutor(max_workers=queued_window_count) as executor:
        future_to_index = {}

        while next_window_index < queued_window_count:
            plan = _resolve_window_capture_plan(
                event_slug_prefixes=event_slug_prefixes,
                event_duration_seconds=event_duration_seconds,
                window_start=window_start,
                window_index=next_window_index,
                last_planned_window_start_epoch=last_planned_window_start_epoch,
            )
            _log_queued_window_capture(plan, windows)
            future = executor.submit(
                _run_single_window_capture,
                config_path=config_path,
                interval_seconds=interval_seconds,
                event_duration_seconds=event_duration_seconds,
                window_start=plan["window_start"],
                run_timestamp=plan["run_timestamp"],
                event_slug_prefixes=event_slug_prefixes,
                series_slugs=series_slugs,
            )
            future_to_index[future] = next_window_index
            last_planned_window_start_epoch = int(plan["window_start"])
            next_window_index += 1

        while future_to_index:
            completed, _ = wait(future_to_index.keys(), return_when=FIRST_COMPLETED)
            for future in completed:
                finished_index = future_to_index.pop(future)
                results_by_index[finished_index] = future.result()

                if next_window_index < windows:
                    plan = _resolve_window_capture_plan(
                        event_slug_prefixes=event_slug_prefixes,
                        event_duration_seconds=event_duration_seconds,
                        window_start=window_start,
                        window_index=next_window_index,
                        last_planned_window_start_epoch=last_planned_window_start_epoch,
                    )
                    _log_queued_window_capture(plan, windows)
                    queued_future = executor.submit(
                        _run_single_window_capture,
                        config_path=config_path,
                        interval_seconds=interval_seconds,
                        event_duration_seconds=event_duration_seconds,
                        window_start=plan["window_start"],
                        run_timestamp=plan["run_timestamp"],
                        event_slug_prefixes=event_slug_prefixes,
                        series_slugs=series_slugs,
                    )
                    future_to_index[queued_future] = next_window_index
                    last_planned_window_start_epoch = int(plan["window_start"])
                    next_window_index += 1

    return [results_by_index[index] for index in range(windows)]


def _run_single_window_capture(
    config_path: str,
    interval_seconds: float,
    event_duration_seconds: int,
    window_start: str,
    run_timestamp: str,
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
                run_timestamp=run_timestamp,
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
                run_timestamp=run_timestamp,
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
