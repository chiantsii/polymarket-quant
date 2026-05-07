import argparse
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import pandas as pd

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
    if batch_mode == "full":
        market_state = load_parquet_glob(market_state_glob, include_latest=include_latest)
        event_state = build_event_state_dataset(market_state)
    elif batch_mode == "file":
        event_state = _build_event_state_file_batched(
            market_state_glob=market_state_glob,
            include_latest=include_latest,
        )
    else:
        raise ValueError(f"Unsupported batch_mode: {batch_mode}")

    run_timestamp = run_timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    parquet_path = output_path / f"crypto_5m_event_state_{run_timestamp}.parquet"
    latest_path = output_path / "crypto_5m_event_state_latest.parquet"
    event_state.to_parquet(parquet_path, index=False)
    event_state.to_parquet(latest_path, index=False)

    logger.info("Saved %s event-state rows to %s", len(event_state), parquet_path)
    return {
        "rows": len(event_state),
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
):
    market_state_paths = matching_parquet_paths(market_state_glob, include_latest=include_latest)
    if not market_state_paths:
        raise FileNotFoundError(f"No parquet files matched {market_state_glob}")

    with TemporaryDirectory(prefix="event_state_batches_") as tmpdir:
        temp_dir = Path(tmpdir)
        event_state_batch_paths: list[Path] = []

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
            batch_output_path = temp_dir / f"event_state_batch_{index:04d}.parquet"
            event_state_batch.to_parquet(batch_output_path, index=False)
            event_state_batch_paths.append(batch_output_path)
            logger.info(
                "Finished event-state file %s/%s: produced %s rows in %.2fs (spilled to %s)",
                index,
                len(market_state_paths),
                len(event_state_batch),
                perf_counter() - batch_started_at,
                batch_output_path.name,
            )

        if not event_state_batch_paths:
            raise ValueError("No event-state rows were generated from file batches.")

        logger.info("Concatenating %s event-state batch parquet file(s)", len(event_state_batch_paths))
        concat_started_at = perf_counter()
        event_state = pd.concat((pd.read_parquet(path) for path in event_state_batch_paths), ignore_index=True)
        sort_cols = [column for column in ("event_slug", "collected_at") if column in event_state.columns]
        if sort_cols:
            event_state = event_state.sort_values(sort_cols).reset_index(drop=True)
        logger.info(
            "Concatenated %s event-state rows from temp parquet in %.2fs",
            len(event_state),
            perf_counter() - concat_started_at,
        )
        return event_state


if __name__ == "__main__":
    main()
