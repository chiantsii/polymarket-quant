"""Small persistence helpers for ingestion scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from polymarket_quant.utils.logger import get_logger

logger = get_logger(__name__)


def save_json_and_parquet_rows(
    rows: List[Dict[str, Any]],
    raw_dir: str | Path,
    processed_dir: str | Path,
    raw_name: str,
    latest_raw_name: str,
    parquet_name: str,
    latest_parquet_name: str,
) -> None:
    """Persist normalized rows as raw JSON plus analytics-ready parquet."""
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    with open(raw_path / raw_name, "w") as f:
        json.dump(rows, f)
    with open(raw_path / latest_raw_name, "w") as f:
        json.dump(rows, f)

    if not rows:
        logger.warning("No rows to save for %s; wrote empty raw JSON only", parquet_name)
        return

    processed_path.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_parquet(processed_path / parquet_name, index=False)
    df.to_parquet(processed_path / latest_parquet_name, index=False)
    logger.info("Saved %s rows to %s", len(df), processed_path / parquet_name)
