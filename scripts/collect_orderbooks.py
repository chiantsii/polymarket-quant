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


if __name__ == "__main__":
    main()
