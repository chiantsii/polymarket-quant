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

DEFAULT_MARKET_STATE_GLOB = "data/*/processed/market_state/shards/*.parquet"
DEFAULT_OUTPUT_ROOT = "data"
EVENT_STATE_PREFIX = "crypto_5m_event_state"


def build_event_state(
    market_state_glob: str = DEFAULT_MARKET_STATE_GLOB,
    output_dir: str = DEFAULT_OUTPUT_ROOT,
    batch_mode: str = "file",
    include_latest: bool = False,
    run_timestamp: str | None = None,
) -> dict[str, object]:
    run_timestamp = run_timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if batch_mode == "full":
        market_state = load_parquet_glob(market_state_glob, include_latest=include_latest)
        event_state = build_event_state_dataset(market_state)
        rows = len(event_state)
        asset_outputs = _write_asset_event_state_outputs(
            rows=event_state,
            output_root=output_root,
            run_timestamp=run_timestamp,
        )
    elif batch_mode == "file":
        rows, asset_outputs = _build_event_state_file_batched(
            market_state_glob=market_state_glob,
            include_latest=include_latest,
            output_root=output_root,
            run_timestamp=run_timestamp,
        )
    else:
        raise ValueError(f"Unsupported batch_mode: {batch_mode}")

    logger.info("Saved %s event-state rows across %s asset output(s)", rows, len(asset_outputs))
    result: dict[str, object] = {
        "rows": rows,
        "output_paths": {asset: payload["output_path"] for asset, payload in asset_outputs.items()},
        "latest_paths": {asset: payload["latest_path"] for asset, payload in asset_outputs.items()},
        "shard_dirs": {asset: payload["shard_dir"] for asset, payload in asset_outputs.items()},
    }
    if len(asset_outputs) == 1:
        asset_payload = next(iter(asset_outputs.values()))
        result["output_path"] = asset_payload["output_path"]
        result["latest_path"] = asset_payload["latest_path"]
        result["shard_dir"] = asset_payload["shard_dir"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build event-level continuous state from token-level market state.")
    parser.add_argument(
        "--market-state-glob",
        default=DEFAULT_MARKET_STATE_GLOB,
        help="Market-state parquet path or glob",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_ROOT, help="Output root directory")
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
    output_root: Path,
    run_timestamp: str,
) -> tuple[int, dict[str, dict[str, str]]]:
    market_state_paths = matching_parquet_paths(market_state_glob, include_latest=include_latest)
    if not market_state_paths:
        raise FileNotFoundError(f"No parquet files matched {market_state_glob}")

    total_started_at = perf_counter()
    row_count = 0
    batch_count = 0
    writers_by_asset: dict[str, pq.ParquetWriter] = {}
    arrow_schema_by_asset: dict[str, pa.Schema] = {}
    asset_outputs: dict[str, dict[str, str]] = {}

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

            logger.info(
                "Loaded market_state file %s/%s: %s rows, %s columns. Building event-state rows...",
                index,
                len(market_state_paths),
                len(market_state),
                len(market_state.columns),
            )
            event_state_batch = build_event_state_dataset(market_state)
            if event_state_batch.empty:
                logger.info(
                    "Skipping event-state batch %s/%s after transformation produced no rows: %s",
                    index,
                    len(market_state_paths),
                    market_state_path.name,
                )
                continue

            shard_updates = 0
            batch_rows = 0
            for asset, asset_batch in event_state_batch.groupby("asset", sort=False):
                asset_name = str(asset)
                payload = asset_outputs.get(asset_name)
                if payload is None:
                    payload = _prepare_asset_event_state_paths(
                        output_root=output_root,
                        asset=asset_name,
                        run_timestamp=run_timestamp,
                    )
                    asset_outputs[asset_name] = payload

                asset_batch = (
                    asset_batch.sort_values(["event_slug", "collected_at"])
                    .drop_duplicates(subset=["event_slug", "collected_at"], keep="last")
                    .reset_index(drop=True)
                )
                batch_table = pa.Table.from_pandas(asset_batch, preserve_index=False)
                writer = writers_by_asset.get(asset_name)
                if writer is None:
                    parquet_path = Path(payload["output_path"])
                    latest_path = Path(payload["latest_path"])
                    if parquet_path.exists():
                        parquet_path.unlink()
                    if latest_path.exists():
                        latest_path.unlink()
                    writer = pq.ParquetWriter(parquet_path, batch_table.schema)
                    writers_by_asset[asset_name] = writer
                    arrow_schema_by_asset[asset_name] = batch_table.schema
                    logger.info(
                        "Initialized streamed event-state parquet writer for %s at %s with %s columns",
                        asset_name,
                        parquet_path,
                        len(batch_table.schema.names),
                    )
                    writer.write_table(batch_table)
                else:
                    batch_table = pa.Table.from_pandas(
                        asset_batch.reindex(columns=arrow_schema_by_asset[asset_name].names),
                        schema=arrow_schema_by_asset[asset_name],
                        preserve_index=False,
                    )
                    writer.write_table(batch_table)

                shard_updates += _upsert_event_state_shards(
                    rows=asset_batch,
                    shards_dir=Path(payload["shard_dir"]),
                )
                batch_rows += asset_batch.shape[0]
                row_count += asset_batch.shape[0]
            batch_count += 1
            logger.info(
                "Finished event-state file %s/%s: wrote %s rows in %.2fs (cumulative rows=%s, streamed batches=%s, updated shards=%s)",
                index,
                len(market_state_paths),
                batch_rows,
                perf_counter() - batch_started_at,
                row_count,
                batch_count,
                shard_updates,
            )

        if not writers_by_asset or row_count == 0:
            raise ValueError("No event-state rows were generated from file batches.")
    finally:
        for writer in writers_by_asset.values():
            writer.close()

    for asset, payload in asset_outputs.items():
        shutil.copyfile(payload["output_path"], payload["latest_path"])
        logger.info("Copied latest event-state parquet for %s to %s", asset, payload["latest_path"])
    logger.info("Finished streamed event-state write: %s rows across %s batch(es) in %.2fs", row_count, batch_count, perf_counter() - total_started_at)
    return row_count, asset_outputs


def _prepare_asset_event_state_paths(
    *,
    output_root: Path,
    asset: str,
    run_timestamp: str,
) -> dict[str, str]:
    dataset_dir = output_root / asset / "processed" / "event_state"
    shards_dir = dataset_dir / "shards"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    shards_dir.mkdir(parents=True, exist_ok=True)
    return {
        "output_path": str(dataset_dir / f"{EVENT_STATE_PREFIX}_{run_timestamp}.parquet"),
        "latest_path": str(dataset_dir / f"{EVENT_STATE_PREFIX}_latest.parquet"),
        "shard_dir": str(shards_dir),
    }


def _write_asset_event_state_outputs(
    *,
    rows: pd.DataFrame,
    output_root: Path,
    run_timestamp: str,
) -> dict[str, dict[str, str]]:
    if rows.empty:
        raise ValueError("No event-state rows were produced.")
    if "asset" not in rows.columns:
        raise ValueError("Event-state rows are missing required asset column.")

    asset_outputs: dict[str, dict[str, str]] = {}
    for asset, asset_rows in rows.groupby("asset", sort=False):
        asset_name = str(asset)
        payload = _prepare_asset_event_state_paths(
            output_root=output_root,
            asset=asset_name,
            run_timestamp=run_timestamp,
        )
        asset_frame = (
            asset_rows.sort_values(["event_slug", "collected_at"])
            .drop_duplicates(subset=["event_slug", "collected_at"], keep="last")
            .reset_index(drop=True)
        )
        asset_frame.to_parquet(payload["output_path"], index=False)
        shutil.copyfile(payload["output_path"], payload["latest_path"])
        shard_updates = _upsert_event_state_shards(
            rows=asset_frame,
            shards_dir=Path(payload["shard_dir"]),
        )
        logger.info(
            "Wrote %s event_state rows for %s to %s and updated %s event shard(s)",
            len(asset_frame),
            asset_name,
            payload["output_path"],
            shard_updates,
        )
        asset_outputs[asset_name] = payload
    return asset_outputs


def _upsert_event_state_shards(
    *,
    rows: pd.DataFrame,
    shards_dir: Path,
) -> int:
    shard_updates = 0
    sort_cols = ["event_slug", "collected_at"]
    for event_slug, event_rows in rows.groupby("event_slug", sort=False):
        shard_path = shards_dir / f"{event_slug}.parquet"
        if shard_path.exists():
            existing = pd.read_parquet(shard_path)
            combined = pd.concat([existing, event_rows], ignore_index=True, sort=False)
        else:
            combined = event_rows.copy()

        combined = (
            combined.sort_values(sort_cols)
            .drop_duplicates(subset=sort_cols, keep="last")
            .reset_index(drop=True)
        )
        combined.to_parquet(shard_path, index=False)
        shard_updates += 1
    return shard_updates


if __name__ == "__main__":
    main()
