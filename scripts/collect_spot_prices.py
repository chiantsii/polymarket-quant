import argparse
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

import yaml

from polymarket_quant.ingestion.spot import CoinbaseSpotPriceClient
from polymarket_quant.ingestion.storage import save_json_and_parquet_rows
from polymarket_quant.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect live BTC/ETH spot price time-series data.")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config file")
    parser.add_argument(
        "--mode",
        choices=["duration", "full-window"],
        default="duration",
        help="Use duration mode or wait for the next complete 5m event window",
    )
    parser.add_argument("--interval-seconds", type=float, default=2.0, help="Seconds between spot polls")
    parser.add_argument("--duration-seconds", type=float, default=300.0, help="Total collection duration")
    parser.add_argument("--event-duration-seconds", type=int, default=300, help="Full-window event duration")
    parser.add_argument(
        "--window-start",
        type=str,
        default=None,
        help="Optional full-window start time as Unix seconds or ISO timestamp",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    api_config = config.get("api", {})
    data_config = config.get("data", {})
    spot_products = config.get("spot", {}).get("products", {"BTC": "BTC-USD", "ETH": "ETH-USD"})
    client = CoinbaseSpotPriceClient(base_url=api_config.get("coinbase_url", "https://api.exchange.coinbase.com"))

    if args.mode == "full-window":
        from windowing import resolve_full_window, wait_until

        full_window = resolve_full_window(
            event_slug_prefixes=["btc-updown-5m", "eth-updown-5m"],
            event_duration_seconds=args.event_duration_seconds,
            window_start=args.window_start,
        )
        wait_until(full_window.start, args.interval_seconds, logger)
        args.duration_seconds = max(0.0, (full_window.end - datetime.now(timezone.utc)).total_seconds())
        logger.info(
            "Collecting spot prices for full event window %s -> %s",
            full_window.start.isoformat(),
            full_window.end.isoformat(),
        )

    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    deadline = time.monotonic() + args.duration_seconds
    spot_rows: List[Dict[str, Any]] = []
    poll_count = 0

    logger.info(
        "Starting live spot price collector for %s seconds at %s second intervals",
        args.duration_seconds,
        args.interval_seconds,
    )

    try:
        while time.monotonic() < deadline:
            poll_started = time.monotonic()
            spot_ticks = client.fetch_spot_tickers(spot_products)
            batch_rows = [tick for tick in spot_ticks.values() if tick]
            spot_rows.extend(batch_rows)
            poll_count += 1

            logger.info("Poll %s collected %s spot ticks", poll_count, len(batch_rows))

            elapsed = time.monotonic() - poll_started
            time.sleep(max(0.0, args.interval_seconds - elapsed))
    except KeyboardInterrupt:
        logger.info("Spot price collector interrupted; saving partial data.")
    finally:
        save_json_and_parquet_rows(
            rows=spot_rows,
            raw_dir=data_config["raw_dir"],
            processed_dir=data_config["processed_dir"],
            raw_name=f"crypto_spot_ticks_raw_{run_timestamp}.json",
            latest_raw_name="crypto_spot_ticks_raw_latest.json",
            parquet_name=f"crypto_spot_ticks_{run_timestamp}.parquet",
            latest_parquet_name="crypto_spot_ticks_latest.parquet",
        )


if __name__ == "__main__":
    main()
