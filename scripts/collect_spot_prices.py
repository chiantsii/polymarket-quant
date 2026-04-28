import argparse
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml

from polymarket_quant.ingestion.spot import BinanceSpotPriceClient, SpotFetchError
from polymarket_quant.ingestion.storage import save_json_and_parquet_rows
from polymarket_quant.utils.logger import get_logger

try:
    from windowing import resolve_active_window, resolve_full_window, wait_until
except ModuleNotFoundError:
    from scripts.windowing import resolve_active_window, resolve_full_window, wait_until

logger = get_logger(__name__)
DEFAULT_EVENT_SLUG_PREFIXES = ["btc-updown-5m", "eth-updown-5m"]


def _asset_storage_dirs(data_config: dict[str, Any], asset: str) -> tuple[Path, Path]:
    asset_key = str(asset).upper()
    asset_root = Path(data_config["raw_dir"]).parent / asset_key
    raw_dir = asset_root / "raw" / "spot"
    processed_dir = asset_root / "processed" / "spot"
    return raw_dir, processed_dir


def _persist_spot_rows_by_asset(
    rows_by_asset: dict[str, list[dict[str, Any]]],
    data_config: dict[str, Any],
    run_timestamp: str,
) -> None:
    for asset, asset_rows in rows_by_asset.items():
        raw_dir, processed_dir = _asset_storage_dirs(data_config, asset)
        save_json_and_parquet_rows(
            rows=asset_rows,
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            raw_name=f"binance_spot_ticks_{run_timestamp}.json",
            latest_raw_name="binance_spot_ticks_latest.json",
            parquet_name=f"binance_spot_ticks_{run_timestamp}.parquet",
            latest_parquet_name="binance_spot_ticks_latest.parquet",
        )


def _submit_chunk_write(
    writer: ThreadPoolExecutor,
    pending_writes: list[Future[None]],
    rows_by_asset: dict[str, list[dict[str, Any]]],
    data_config: dict[str, Any],
    run_timestamp: str,
) -> None:
    if not any(rows_by_asset.values()):
        return
    pending_writes.append(
        writer.submit(
            _persist_spot_rows_by_asset,
            rows_by_asset=rows_by_asset,
            data_config=data_config,
            run_timestamp=run_timestamp,
        )
    )


def _event_slug_by_asset(active_window, spot_products: dict[str, str]) -> dict[str, str]:
    slugs_by_asset: dict[str, str] = {}
    for event_slug in active_window.event_slugs:
        for asset in spot_products:
            if event_slug.startswith(str(asset).lower()):
                slugs_by_asset[str(asset).upper()] = event_slug
    return slugs_by_asset


def collect_spot_prices(
    config_path: str = "configs/base.yaml",
    mode: str = "duration",
    interval_seconds: float = 2.0,
    duration_seconds: float = 300.0,
    event_duration_seconds: int = 300,
    window_start: str | None = None,
    run_timestamp: str | None = None,
    event_slug_prefixes: list[str] | None = None,
) -> dict[str, Any]:
    """Collect Binance BTC/ETH spot prices and persist per-asset raw/processed datasets."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    api_config = config.get("api", {})
    data_config = config.get("data", {})
    spot_products = config.get("spot", {}).get("products", {"BTC": "BTCUSDT", "ETH": "ETHUSDT"})
    event_slug_prefixes = event_slug_prefixes or DEFAULT_EVENT_SLUG_PREFIXES
    client = BinanceSpotPriceClient(base_url=api_config.get("binance_url", "https://api.binance.com"))
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
                "Full window expired before spot collection could start: "
                f"{full_window.start.isoformat()} -> {full_window.end.isoformat()}"
            )
        duration_seconds = (full_window.end - now).total_seconds()
        run_timestamp = run_timestamp or full_window.start.strftime("%Y%m%d_%H%M%S")
        active_window = full_window
        logger.info(
            "Collecting spot prices for full event window %s -> %s",
            full_window.start.isoformat(),
            full_window.end.isoformat(),
        )

    run_timestamp = run_timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    deadline = time.monotonic() + duration_seconds
    rows_by_asset: dict[str, list[dict[str, Any]]] = {asset: [] for asset in spot_products}
    successes_by_asset: dict[str, int] = {asset: 0 for asset in spot_products}
    failures_by_asset: dict[str, int] = {asset: 0 for asset in spot_products}
    last_error_by_asset: dict[str, str] = {}
    poll_count = 0
    interrupted = False
    writer = ThreadPoolExecutor(max_workers=1)
    pending_writes: list[Future[None]] = []

    logger.info(
        "Starting live spot price collector for %s seconds at %s second intervals",
        duration_seconds,
        interval_seconds,
    )

    chunk_window = active_window or resolve_active_window(
        event_slug_prefixes=event_slug_prefixes,
        event_duration_seconds=event_duration_seconds,
    )

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
                        rows_by_asset=rows_by_asset,
                        data_config=data_config,
                        run_timestamp=chunk_window.start.strftime("%Y%m%d_%H%M%S"),
                    )
                    rows_by_asset = {asset: [] for asset in spot_products}
                    chunk_window = current_window

            poll_started = time.monotonic()
            batch_success_count = 0
            poll_number = poll_count + 1
            event_slug_map = _event_slug_by_asset(chunk_window, spot_products)

            for asset, product_id in spot_products.items():
                try:
                    tick = client.fetch_spot_ticker(asset=asset, product_id=product_id)
                except SpotFetchError as exc:
                    failures_by_asset[asset] += 1
                    last_error_by_asset[asset] = str(exc)
                    logger.error("Poll %s failed to collect %s spot tick: %s", poll_number, asset, exc)
                    continue

                tick["event_slug"] = event_slug_map.get(asset)
                tick["market_window_start"] = chunk_window.start.isoformat()
                tick["market_window_end"] = chunk_window.end.isoformat()
                rows_by_asset[asset].append(tick)
                successes_by_asset[asset] += 1
                batch_success_count += 1

            poll_count += 1

            logger.info(
                "Poll %s collected %s/%s spot ticks",
                poll_count,
                batch_success_count,
                len(spot_products),
            )

            elapsed = time.monotonic() - poll_started
            time.sleep(max(0.0, interval_seconds - elapsed))
    except KeyboardInterrupt:
        interrupted = True
        logger.info("Spot price collector interrupted; saving partial data.")
    finally:
        if any(rows_by_asset.values()):
            _submit_chunk_write(
                writer=writer,
                pending_writes=pending_writes,
                rows_by_asset=rows_by_asset,
                data_config=data_config,
                run_timestamp=chunk_window.start.strftime("%Y%m%d_%H%M%S"),
            )
        writer.shutdown(wait=True)
        for pending_write in pending_writes:
            pending_write.result()

    missing_assets = [asset for asset, success_count in successes_by_asset.items() if success_count == 0]
    if missing_assets and not interrupted:
        last_errors = {asset: last_error_by_asset.get(asset, "no successful polls captured") for asset in missing_assets}
        raise RuntimeError(
            "Spot capture failed for required asset(s): "
            f"{missing_assets}. This window is incomplete and should not feed market-state construction. "
            f"Last errors: {last_errors}"
        )

    return {
        "spot_rows": sum(successes_by_asset.values()),
        "spot_rows_by_asset": dict(successes_by_asset),
        "spot_failures_by_asset": dict(failures_by_asset),
        "polls": poll_count,
        "run_timestamp": run_timestamp,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect live BTC/ETH spot price time-series data.")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config file")
    parser.add_argument(
        "--mode",
        choices=["duration", "full-window"],
        default="duration",
        help="Use continuous duration mode or aligned full-window mode",
    )
    parser.add_argument("--interval-seconds", type=float, default=2.0, help="Seconds between spot polls")
    parser.add_argument("--duration-seconds", type=float, default=300.0, help="Total collection duration")
    parser.add_argument("--event-duration-seconds", type=int, default=300, help="Chunk flush interval in seconds")
    parser.add_argument(
        "--window-start",
        type=str,
        default=None,
        help="Optional aligned start time as Unix seconds or ISO timestamp",
    )
    args = parser.parse_args()

    collect_spot_prices(
        config_path=args.config,
        mode=args.mode,
        interval_seconds=args.interval_seconds,
        duration_seconds=args.duration_seconds,
        event_duration_seconds=args.event_duration_seconds,
        window_start=args.window_start,
        run_timestamp=None,
    )


if __name__ == "__main__":
    main()
