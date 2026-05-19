from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import glob

import pandas as pd

from polymarket_quant.state import build_event_state_dataset, build_market_state_dataset
from polymarket_quant.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_LIVE_ORDERBOOK_SUMMARY_GLOB = "data/*/processed/polymarket/crypto_5m_orderbook_summary_*.parquet"
DEFAULT_LIVE_ORDERBOOK_LEVELS_GLOB = "data/*/processed/polymarket/crypto_5m_orderbook_levels_*.parquet"
DEFAULT_LIVE_SPOT_GLOB = "data/*/processed/spot/binance_spot_ticks_*.parquet"
DEFAULT_LIVE_STATE_OUTPUT_DIR = "artifacts/live_state"


@dataclass(frozen=True)
class LiveCaptureSourceConfig:
    orderbook_summary_glob: str = DEFAULT_LIVE_ORDERBOOK_SUMMARY_GLOB
    orderbook_levels_glob: str = DEFAULT_LIVE_ORDERBOOK_LEVELS_GLOB
    spot_glob: str = DEFAULT_LIVE_SPOT_GLOB
    output_dir: str = DEFAULT_LIVE_STATE_OUTPUT_DIR
    spot_tolerance_seconds: float = 2.0
    event_duration_seconds: float = 300.0


@dataclass(frozen=True)
class LiveCaptureStateSnapshot:
    market_state: pd.DataFrame
    event_state: pd.DataFrame
    summary_paths: tuple[str, ...]
    level_paths: tuple[str, ...]
    spot_paths: tuple[str, ...]


class LiveCaptureEventStateSource:
    """Build incremental event-state rows from the latest timestamped capture batches."""

    def __init__(self, config: LiveCaptureSourceConfig | None = None) -> None:
        self.config = config or LiveCaptureSourceConfig()
        self._seen_keys: set[str] = set()

    def poll_new_rows(self) -> list[dict[str, Any]]:
        snapshot = load_live_capture_state_snapshot(self.config)
        self._write_latest_artifacts(snapshot)

        event_state = snapshot.event_state
        if event_state.empty:
            return []

        event_state = event_state.sort_values(["event_slug", "collected_at"]).reset_index(drop=True)
        keys = event_state["event_slug"].astype(str) + "|" + event_state["collected_at"].astype(str)
        new_mask = ~keys.isin(self._seen_keys)
        if not new_mask.any():
            self._prune_seen_keys(event_state)
            return []

        new_keys = keys.loc[new_mask].tolist()
        self._seen_keys.update(new_keys)
        self._prune_seen_keys(event_state)
        return event_state.loc[new_mask].to_dict("records")

    def _write_latest_artifacts(self, snapshot: LiveCaptureStateSnapshot) -> None:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        snapshot.market_state.to_parquet(output_dir / "live_market_state_latest.parquet", index=False)
        snapshot.event_state.to_parquet(output_dir / "live_event_state_latest.parquet", index=False)

    def _prune_seen_keys(self, event_state: pd.DataFrame) -> None:
        if event_state.empty:
            self._seen_keys.clear()
            return
        current_slugs = set(event_state["event_slug"].astype(str).unique().tolist())
        self._seen_keys = {
            key for key in self._seen_keys if key.split("|", 1)[0] in current_slugs
        }


def load_live_capture_state_snapshot(
    config: LiveCaptureSourceConfig | None = None,
) -> LiveCaptureStateSnapshot:
    config = config or LiveCaptureSourceConfig()

    summary_paths = _latest_timestamped_paths_by_asset(config.orderbook_summary_glob)
    spot_paths = _latest_timestamped_paths_by_asset(config.spot_glob)
    level_paths = _latest_timestamped_paths_by_asset(config.orderbook_levels_glob)

    if not summary_paths:
        logger.warning("No live orderbook summary parquet files matched %s", config.orderbook_summary_glob)
        return LiveCaptureStateSnapshot(
            market_state=pd.DataFrame(),
            event_state=pd.DataFrame(),
            summary_paths=(),
            level_paths=tuple(str(path) for path in level_paths),
            spot_paths=tuple(str(path) for path in spot_paths),
        )
    if not spot_paths:
        logger.warning("No live spot parquet files matched %s", config.spot_glob)
        return LiveCaptureStateSnapshot(
            market_state=pd.DataFrame(),
            event_state=pd.DataFrame(),
            summary_paths=tuple(str(path) for path in summary_paths),
            level_paths=tuple(str(path) for path in level_paths),
            spot_paths=(),
        )

    orderbooks = _read_parquet_paths(summary_paths)
    spot = _read_parquet_paths(spot_paths)
    orderbook_levels = _read_parquet_paths(level_paths) if level_paths else pd.DataFrame()

    if orderbooks.empty or spot.empty:
        return LiveCaptureStateSnapshot(
            market_state=pd.DataFrame(),
            event_state=pd.DataFrame(),
            summary_paths=tuple(str(path) for path in summary_paths),
            level_paths=tuple(str(path) for path in level_paths),
            spot_paths=tuple(str(path) for path in spot_paths),
        )

    market_state = build_market_state_dataset(
        orderbooks=orderbooks,
        spot=spot,
        orderbook_levels=orderbook_levels,
        spot_tolerance_seconds=config.spot_tolerance_seconds,
        event_duration_seconds=config.event_duration_seconds,
    )
    event_state = build_event_state_dataset(market_state)
    return LiveCaptureStateSnapshot(
        market_state=market_state,
        event_state=event_state,
        summary_paths=tuple(str(path) for path in summary_paths),
        level_paths=tuple(str(path) for path in level_paths),
        spot_paths=tuple(str(path) for path in spot_paths),
    )


def _latest_timestamped_paths_by_asset(pattern: str) -> list[Path]:
    matched_paths = [Path(path) for path in sorted(glob.glob(pattern))]
    timestamped_paths = [path for path in matched_paths if not path.name.endswith("_latest.parquet")]
    if matched_paths and not timestamped_paths:
        logger.warning(
            "Ignoring aggregate *_latest.parquet capture inputs for %s; expected timestamped batch parquet files.",
            pattern,
        )
    latest_by_asset: dict[str, Path] = {}
    for path in timestamped_paths:
        asset = path.parents[2].name
        current = latest_by_asset.get(asset)
        if current is None or path.name > current.name:
            latest_by_asset[asset] = path
    return [latest_by_asset[asset] for asset in sorted(latest_by_asset)]


def _read_parquet_paths(paths: list[Path]) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
