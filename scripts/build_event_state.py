import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from polymarket_quant.state import build_event_state_dataset
from polymarket_quant.state.dataset import load_parquet_glob, matching_parquet_paths
from polymarket_quant.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_MARKET_STATE_GLOB = "data/processed/crypto_5m_market_state_latest.parquet"


def build_event_state(
    market_state_glob: str = DEFAULT_MARKET_STATE_GLOB,
    output_dir: str = "data/processed",
    batch_mode: str = "file",
    include_latest: bool = False,
    run_timestamp: str | None = None,
) -> dict[str, str | int]:
    run_timestamp = run_timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    parquet_path = output_path / f"crypto_5m_event_state_{run_timestamp}.parquet"
    latest_path = output_path / "crypto_5m_event_state_latest.parquet"

    if batch_mode == "full":
        market_state = load_parquet_glob(market_state_glob, include_latest=include_latest)
        event_state = build_event_state_dataset(market_state)
        event_state.to_parquet(parquet_path, index=False)
        event_state.to_parquet(latest_path, index=False)
        rows = len(event_state)
    elif batch_mode == "file":
        rows = _build_event_state_file_batched(
            market_state_glob=market_state_glob,
            include_latest=include_latest,
            parquet_path=parquet_path,
            latest_path=latest_path,
        )
    else:
        raise ValueError(f"Unsupported batch_mode: {batch_mode}")

    logger.info("Saved %s event-state rows to %s", rows, parquet_path)
    return {
        "rows": rows,
        "output_path": str(parquet_path),
        "latest_path": str(latest_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build event-level continuous state from token-level market state.")
    parser.add_argument(
        "--market-state-glob",
        default=DEFAULT_MARKET_STATE_GLOB,
        help="Market-state parquet path or glob",
    )
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    parser.add_argument(
        "--batch-mode",
        choices=["file", "full"],
        default="file",
        help="Use file-batched event-state construction to reduce peak memory, or load all inputs at once",
    )
    parser.add_argument("--include-latest", action="store_true", help="Include *_latest.parquet inputs")
    args = parser.parse_args()

    build_event_state(
        market_state_glob=args.market_state_glob,
        output_dir=args.output_dir,
        batch_mode=args.batch_mode,
        include_latest=args.include_latest,
    )


def _build_event_state_file_batched(
    *,
    market_state_glob: str,
    include_latest: bool,
    parquet_path: Path,
    latest_path: Path,
) -> int:
    market_state_paths = matching_parquet_paths(market_state_glob, include_latest=include_latest)
    if not market_state_paths:
        raise FileNotFoundError(f"No parquet files matched {market_state_glob}")

    total_started_at = perf_counter()
    row_count = 0
    batch_count = 0
    writer: pq.ParquetWriter | None = None
    arrow_schema: pa.Schema | None = None

    if parquet_path.exists():
        parquet_path.unlink()
    if latest_path.exists():
        latest_path.unlink()

    try:
        for index, market_state_path in enumerate(market_state_paths, start=1):
            batch_started_at = perf_counter()
            logger.info(
                "Processing event-state market_state file %s/%s: %s",
                index,
                len(market_state_paths),
                market_state_path.name,
            )
            market_state = pd.read_parquet(market_state_path)
            if market_state.empty:
                logger.info(
                    "Skipping empty market_state file %s/%s: %s",
                    index,
                    len(market_state_paths),
                    market_state_path.name,
                )
                continue

            event_state_batch = build_event_state_dataset(market_state)
            if event_state_batch.empty:
                logger.info(
                    "Skipping event-state batch %s/%s after transformation produced no rows: %s",
                    index,
                    len(market_state_paths),
                    market_state_path.name,
                )
                continue

            batch_table = pa.Table.from_pandas(event_state_batch, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(parquet_path, batch_table.schema)
                arrow_schema = batch_table.schema
                logger.info(
                    "Initialized streamed event-state parquet writer at %s with %s columns",
                    parquet_path,
                    len(batch_table.schema.names),
                )
            else:
                batch_table = pa.Table.from_pandas(
                    event_state_batch.reindex(columns=arrow_schema.names),
                    schema=arrow_schema,
                    preserve_index=False,
                )

            writer.write_table(batch_table)
            batch_rows = event_state_batch.shape[0]
            row_count += batch_rows
            batch_count += 1
            logger.info(
                "Finished event-state file %s/%s: wrote %s rows in %.2fs (cumulative rows=%s, streamed batches=%s)",
                index,
                len(market_state_paths),
                batch_rows,
                perf_counter() - batch_started_at,
                row_count,
                batch_count,
            )

        if writer is None or row_count == 0:
            raise ValueError("No event-state rows were generated from file batches.")
    finally:
        if writer is not None:
            writer.close()

    shutil.copyfile(parquet_path, latest_path)
    logger.info(
        "Finished streamed event-state write: %s rows across %s batch(es) in %.2fs",
        row_count,
        batch_count,
        perf_counter() - total_started_at,
    )
    logger.info("Copied latest event-state parquet to %s", latest_path)
    return row_count


if __name__ == "__main__":
    main()
