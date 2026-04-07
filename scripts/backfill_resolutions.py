import argparse
import glob
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pandas as pd
import yaml

from polymarket_quant.ingestion.client import PolymarketRESTClient
from polymarket_quant.ingestion.pipeline import IngestionPipeline
from polymarket_quant.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_EVENT_SLUG_PREFIXES = ["btc-updown-5m", "eth-updown-5m"]


def backfill_resolutions(
    config_path: str = "configs/base.yaml",
    input_glob: str = "data/processed/crypto_5m_orderbook_summary_*.parquet",
    event_limit: int | None = None,
    event_duration_seconds: float = 300.0,
    settlement_delay_seconds: float = 60.0,
    include_unresolved: bool = False,
    event_slug_prefixes: List[str] | None = None,
) -> dict[str, int | str]:
    """Backfill BTC/ETH 5m resolution labels for collected event slugs."""
    event_slug_prefixes = event_slug_prefixes or DEFAULT_EVENT_SLUG_PREFIXES

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    event_slugs = _load_event_slugs(
        input_glob=input_glob,
        event_slug_prefixes=event_slug_prefixes,
        event_limit=event_limit,
        event_duration_seconds=event_duration_seconds,
        settlement_delay_seconds=settlement_delay_seconds,
    )
    if not event_slugs:
        logger.warning("No event slugs found from %s", input_glob)
        return {"event_slugs": 0, "raw_records": 0, "resolution_rows": 0, "run_timestamp": ""}

    client = PolymarketRESTClient(
        gamma_url=config["api"]["gamma_url"],
        clob_url=config["api"]["clob_url"],
    )
    pipeline = IngestionPipeline(
        client=client,
        raw_dir=config["data"]["raw_dir"],
        processed_dir=config["data"]["processed_dir"],
    )

    logger.info("Backfilling resolution labels for %s event slugs", len(event_slugs))
    raw_records, resolution_rows = pipeline.collect_crypto_5m_resolutions_for_event_slugs(
        event_slugs=event_slugs,
        event_slug_prefixes=event_slug_prefixes,
        resolved_only=not include_unresolved,
    )
    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    pipeline.save_crypto_5m_resolutions(
        raw_records=raw_records,
        resolution_rows=resolution_rows,
        run_timestamp=run_timestamp,
    )
    return {
        "event_slugs": len(event_slugs),
        "raw_records": len(raw_records),
        "resolution_rows": len(resolution_rows),
        "run_timestamp": run_timestamp,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill BTC/ETH 5m resolution labels for event slugs found in collected orderbook data."
    )
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config file")
    parser.add_argument(
        "--input-glob",
        type=str,
        default="data/processed/crypto_5m_orderbook_summary_*.parquet",
        help="Glob for collected orderbook summary parquet files",
    )
    parser.add_argument("--event-limit", type=int, default=None, help="Optional maximum number of event slugs to query")
    parser.add_argument(
        "--event-duration-seconds",
        type=float,
        default=300.0,
        help="Expected 5m event duration used to skip still-live slugs",
    )
    parser.add_argument(
        "--settlement-delay-seconds",
        type=float,
        default=60.0,
        help="Extra delay after event end before querying resolution labels",
    )
    parser.add_argument(
        "--include-unresolved",
        action="store_true",
        help="Include rows whose winner cannot be inferred yet",
    )
    parser.add_argument(
        "--event-slug-prefixes",
        nargs="+",
        default=DEFAULT_EVENT_SLUG_PREFIXES,
        help="Only backfill events whose slugs start with these prefixes",
    )
    args = parser.parse_args()

    backfill_resolutions(
        config_path=args.config,
        input_glob=args.input_glob,
        event_limit=args.event_limit,
        event_duration_seconds=args.event_duration_seconds,
        settlement_delay_seconds=args.settlement_delay_seconds,
        include_unresolved=args.include_unresolved,
        event_slug_prefixes=args.event_slug_prefixes,
    )


def _load_event_slugs(
    input_glob: str,
    event_slug_prefixes: List[str],
    event_limit: int | None = None,
    event_duration_seconds: float = 300.0,
    settlement_delay_seconds: float = 60.0,
    now: datetime | None = None,
) -> List[str]:
    paths = [Path(path) for path in sorted(glob.glob(input_glob))]
    event_slugs = set()
    now = now or datetime.now(timezone.utc)
    for path in paths:
        try:
            df = pd.read_parquet(path, columns=["event_slug"])
        except Exception as exc:
            logger.warning("Skipping %s: %s", path, exc)
            continue

        event_slugs.update(
            str(slug)
            for slug in df["event_slug"].dropna().unique()
            if any(str(slug).startswith(prefix) for prefix in event_slug_prefixes)
            and _is_ready_for_resolution_backfill(
                str(slug),
                event_duration_seconds=event_duration_seconds,
                settlement_delay_seconds=settlement_delay_seconds,
                now=now,
            )
        )

    sorted_slugs = sorted(event_slugs)
    if event_limit is not None:
        return sorted_slugs[-event_limit:]
    return sorted_slugs


def _is_ready_for_resolution_backfill(
    event_slug: str,
    event_duration_seconds: float,
    settlement_delay_seconds: float,
    now: datetime,
) -> bool:
    event_start = _event_start_from_slug(event_slug)
    if event_start is None:
        return True
    seconds_since_start = (now - event_start).total_seconds()
    return seconds_since_start >= event_duration_seconds + settlement_delay_seconds


def _event_start_from_slug(event_slug: str) -> datetime | None:
    suffix = event_slug.rsplit("-", 1)[-1]
    if not suffix.isdigit():
        return None
    return datetime.fromtimestamp(int(suffix), tz=timezone.utc)


if __name__ == "__main__":
    main()
