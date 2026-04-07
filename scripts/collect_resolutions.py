import argparse
import time
from datetime import datetime, timezone
from typing import Any, Dict

import yaml

from polymarket_quant.ingestion.client import PolymarketRESTClient
from polymarket_quant.ingestion.pipeline import IngestionPipeline
from polymarket_quant.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect resolved BTC/ETH 5m Up/Down winner labels.")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config file")
    parser.add_argument("--interval-seconds", type=float, default=60.0, help="Seconds between resolution polls")
    parser.add_argument("--duration-seconds", type=float, default=300.0, help="Total collection duration")
    parser.add_argument("--event-limit", type=int, default=20, help="Recently closed events per BTC/ETH series to scan")
    parser.add_argument(
        "--include-unresolved",
        action="store_true",
        help="Include closed events whose token winner cannot be inferred yet",
    )
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
    raw_records_by_event: Dict[str, Dict[str, Any]] = {}
    resolution_rows_by_token: Dict[tuple[str, str], Dict[str, Any]] = {}
    poll_count = 0

    logger.info(
        "Starting resolution collector for %s seconds at %s second intervals",
        args.duration_seconds,
        args.interval_seconds,
    )

    try:
        while time.monotonic() < deadline:
            poll_started = time.monotonic()
            batch_raw, batch_rows = pipeline.collect_crypto_5m_resolutions_once(
                series_slugs=args.series_slugs,
                event_limit=args.event_limit,
                event_slug_prefixes=args.event_slug_prefixes,
                resolved_only=not args.include_unresolved,
            )

            for raw_record in batch_raw:
                event = raw_record.get("event", {})
                event_slug = str(event.get("slug", ""))
                if event_slug:
                    raw_records_by_event[event_slug] = raw_record

            for row in batch_rows:
                key = (str(row.get("event_slug", "")), str(row.get("token_id", "")))
                if all(key):
                    resolution_rows_by_token[key] = row

            poll_count += 1
            logger.info(
                "Poll %s collected %s unique events, %s unique token labels",
                poll_count,
                len(raw_records_by_event),
                len(resolution_rows_by_token),
            )

            elapsed = time.monotonic() - poll_started
            time.sleep(max(0.0, args.interval_seconds - elapsed))
    except KeyboardInterrupt:
        logger.info("Resolution collector interrupted; saving partial data.")
    finally:
        pipeline.save_crypto_5m_resolutions(
            raw_records=list(raw_records_by_event.values()),
            resolution_rows=list(resolution_rows_by_token.values()),
            run_timestamp=run_timestamp,
        )


if __name__ == "__main__":
    main()
