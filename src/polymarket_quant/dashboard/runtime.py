from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from polymarket_quant.evaluation import BaselineUpPaperTradingExecutor, BaselineUpStrategyConfig
from polymarket_quant.live import (
    DEFAULT_LIVE_ORDERBOOK_LEVELS_GLOB,
    DEFAULT_LIVE_ORDERBOOK_SUMMARY_GLOB,
    DEFAULT_LIVE_SPOT_GLOB,
    DEFAULT_LIVE_STATE_OUTPUT_DIR,
    DEFAULT_TRANSITION_MODEL_BTC_PATH,
    DEFAULT_TRANSITION_MODEL_ETH_PATH,
    DEFAULT_TRANSITION_MODEL_PATH,
    DirectLiveEventStateSource,
    DirectLiveSourceConfig,
    LiveCaptureEventStateSource,
    LiveCaptureSourceConfig,
    build_pricing_detector,
)
from polymarket_quant.signals.mispricing import AssetAwareMispricingDetector, RealTimeMispricingDetector
from polymarket_quant.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_EVENT_STATE_PATH = "data/processed/crypto_5m_event_state_latest.parquet"
DEFAULT_LIVE_SIGNAL_OUTPUT_DIR = "artifacts/live_signal_loop"
DEFAULT_PAPER_OUTPUT_DIR = "artifacts/paper_trading"


@dataclass(frozen=True)
class EmbeddedLiveRuntimeConfig:
    runtime_mode: str = "embedded-live"
    source_mode: str = "direct-live"
    transition_model_path: str = DEFAULT_TRANSITION_MODEL_PATH
    transition_model_btc_path: str = DEFAULT_TRANSITION_MODEL_BTC_PATH
    transition_model_eth_path: str = DEFAULT_TRANSITION_MODEL_ETH_PATH
    config_path: str = "configs/base.yaml"
    event_state_path: str = DEFAULT_EVENT_STATE_PATH
    orderbook_summary_glob: str = DEFAULT_LIVE_ORDERBOOK_SUMMARY_GLOB
    orderbook_levels_glob: str = DEFAULT_LIVE_ORDERBOOK_LEVELS_GLOB
    spot_glob: str = DEFAULT_LIVE_SPOT_GLOB
    live_state_output_dir: str = DEFAULT_LIVE_STATE_OUTPUT_DIR
    live_signal_output_dir: str = DEFAULT_LIVE_SIGNAL_OUTPUT_DIR
    paper_output_dir: str = DEFAULT_PAPER_OUTPUT_DIR
    event_duration_seconds: int = 300
    event_slug_prefixes: tuple[str, ...] = ("btc-updown-5m", "eth-updown-5m")
    series_slugs: tuple[str, ...] = ("btc-up-or-down-5m", "eth-up-or-down-5m")
    n_samples: int = 1_000
    simulation_dt_seconds: float = 1.0
    rollout_horizon_seconds: float = 0.0
    edge_threshold: float = 0.0
    open_cooldown_seconds: float = 30.0
    entry_edge_threshold: float = 0.03
    exit_hold_edge_threshold: float = 0.0
    max_holding_seconds: float | None = None
    forced_exit_seconds_to_end: float | None = None
    max_edge_cap: float = 0.12
    market_probability_exclusion_low: float | None = 0.40
    market_probability_exclusion_high: float | None = 0.60
    confident_exit_fair_probability_threshold: float = 0.86
    confident_exit_window_seconds: float = 60.0
    confident_exit_hold_edge_floor: float = -0.02
    quiet_runtime_logs: bool = True
    print_timing_to_terminal: bool = False


class EmbeddedLiveRuntime:
    """Single-process live pipeline: fetch -> state -> fair/edge -> paper execution."""

    def __init__(
        self,
        config: EmbeddedLiveRuntimeConfig,
        *,
        source: Any | None = None,
        detector: RealTimeMispricingDetector | AssetAwareMispricingDetector | None = None,
        executor: BaselineUpPaperTradingExecutor | None = None,
    ) -> None:
        self.config = config
        if self.config.quiet_runtime_logs:
            _quiet_textual_runtime_logging()

        self.source = source or _build_source(config)
        self.detector = detector or _build_detector(config)
        self.executor = executor or BaselineUpPaperTradingExecutor(
            BaselineUpStrategyConfig(
                event_duration_seconds=float(config.event_duration_seconds),
                open_cooldown_seconds=config.open_cooldown_seconds,
                entry_edge_threshold=config.entry_edge_threshold,
                exit_hold_edge_threshold=config.exit_hold_edge_threshold,
                max_holding_seconds=config.max_holding_seconds,
                forced_exit_seconds_to_end=config.forced_exit_seconds_to_end,
                max_edge_cap=config.max_edge_cap,
                market_probability_exclusion_low=config.market_probability_exclusion_low,
                market_probability_exclusion_high=config.market_probability_exclusion_high,
                confident_exit_fair_probability_threshold=config.confident_exit_fair_probability_threshold,
                confident_exit_window_seconds=config.confident_exit_window_seconds,
                confident_exit_hold_edge_floor=config.confident_exit_hold_edge_floor,
            )
        )

        self.live_signal_latest_path = Path(config.live_signal_output_dir) / "live_signal_events_latest.jsonl"
        self.paper_signal_latest_path = Path(config.paper_output_dir) / "paper_signal_events_latest.jsonl"
        self.paper_order_latest_path = Path(config.paper_output_dir) / "paper_order_events_latest.jsonl"
        self.paper_trade_latest_path = Path(config.paper_output_dir) / "paper_trade_ledger_latest.jsonl"
        self.live_event_state_latest_path = Path(config.live_state_output_dir) / "live_event_state_latest.parquet"

        self._prepare_output_files()

    def poll_once(self) -> dict[str, int | float]:
        poll_started = perf_counter()
        new_event_rows = self.source.poll_new_rows()
        timing_metrics = _source_timing_metrics(self.source)
        if not new_event_rows:
            result = {
                "new_event_rows": 0,
                "signal_events": 0,
                "order_events": 0,
                "closed_trades": 0,
                **timing_metrics,
                "pricing_ms": 0.0,
                "executor_ms": 0.0,
                "poll_total_ms": (perf_counter() - poll_started) * 1000.0,
            }
            _log_poll_timing(result, print_to_terminal=self.config.print_timing_to_terminal)
            return result

        pricing_started = perf_counter()
        pricing_rows = self.detector.detect(new_event_rows, show_progress=False)
        pricing_ms = (perf_counter() - pricing_started) * 1000.0
        pricing_rows = sorted(
            pricing_rows,
            key=lambda row: (
                str(row.get("event_slug", "")),
                str(row.get("collected_at", "")),
                str(row.get("outcome_name", "")),
            ),
        )

        signal_events: list[dict[str, Any]] = []
        order_events: list[dict[str, Any]] = []
        trade_events: list[dict[str, Any]] = []

        executor_started = perf_counter()
        for row in pricing_rows:
            result = self.executor.process_row(row)
            if result is None:
                continue
            signal_event = {
                key: value
                for key, value in result.items()
                if key not in {"row", "closed_trade", "order_events", "closed_episode"}
            }
            signal_events.append(signal_event)
            order_events.extend(result.get("order_events", []))
            if result.get("closed_trade") is not None:
                trade_events.append(result["closed_trade"])
        executor_ms = (perf_counter() - executor_started) * 1000.0

        self._append_jsonl(self.live_signal_latest_path, signal_events)
        self._append_jsonl(self.paper_signal_latest_path, signal_events)
        self._append_jsonl(self.paper_order_latest_path, order_events)
        self._append_jsonl(self.paper_trade_latest_path, trade_events)

        result = {
            "new_event_rows": len(new_event_rows),
            "signal_events": len(signal_events),
            "order_events": len(order_events),
            "closed_trades": len(trade_events),
            **timing_metrics,
            "pricing_ms": pricing_ms,
            "executor_ms": executor_ms,
            "poll_total_ms": (perf_counter() - poll_started) * 1000.0,
        }
        _log_poll_timing(result, print_to_terminal=self.config.print_timing_to_terminal)
        return result

    def snapshot_paths(self) -> dict[str, str]:
        return {
            "event_state_path": str(self.live_event_state_latest_path),
            "live_signal_path": str(self.live_signal_latest_path),
            "paper_signal_path": str(self.paper_signal_latest_path),
            "paper_order_path": str(self.paper_order_latest_path),
            "paper_trade_path": str(self.paper_trade_latest_path),
        }

    def _prepare_output_files(self) -> None:
        for path in [
            self.live_signal_latest_path,
            self.paper_signal_latest_path,
            self.paper_order_latest_path,
            self.paper_trade_latest_path,
        ]:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")
        self.live_event_state_latest_path.parent.mkdir(parents=True, exist_ok=True)

    def _append_jsonl(self, path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        with path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(_json_ready(row), sort_keys=True) + "\n")


class _ParquetEventStateSource:
    def __init__(self, event_state_path: str) -> None:
        self.path = Path(event_state_path)
        self._seen_keys: set[str] = set()

    def poll_new_rows(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        frame = pd.read_parquet(self.path)
        if frame.empty:
            return []
        frame = frame.sort_values(["event_slug", "collected_at"]).reset_index(drop=True)
        keys = frame["event_slug"].astype(str) + "|" + frame["collected_at"].astype(str)
        new_mask = ~keys.isin(self._seen_keys)
        if not new_mask.any():
            return []
        self._seen_keys.update(keys.loc[new_mask].tolist())
        return frame.loc[new_mask].to_dict("records")


def _build_source(config: EmbeddedLiveRuntimeConfig) -> Any:
    if config.source_mode == "direct-live":
        return DirectLiveEventStateSource(
            DirectLiveSourceConfig(
                config_path=config.config_path,
                output_dir=config.live_state_output_dir,
                event_duration_seconds=config.event_duration_seconds,
                event_slug_prefixes=config.event_slug_prefixes,
                series_slugs=config.series_slugs,
            )
        )
    if config.source_mode == "live-capture":
        return LiveCaptureEventStateSource(
            LiveCaptureSourceConfig(
                orderbook_summary_glob=config.orderbook_summary_glob,
                orderbook_levels_glob=config.orderbook_levels_glob,
                spot_glob=config.spot_glob,
                output_dir=config.live_state_output_dir,
            )
        )
    return _ParquetEventStateSource(config.event_state_path)


def _build_detector(config: EmbeddedLiveRuntimeConfig) -> RealTimeMispricingDetector | AssetAwareMispricingDetector:
    return build_pricing_detector(
        transition_model_path=config.transition_model_path,
        transition_model_btc_path=config.transition_model_btc_path,
        transition_model_eth_path=config.transition_model_eth_path,
        n_samples=config.n_samples,
        simulation_dt_seconds=config.simulation_dt_seconds,
        rollout_horizon_seconds=config.rollout_horizon_seconds,
        edge_threshold=config.edge_threshold,
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _source_timing_metrics(source: Any) -> dict[str, float]:
    default = {
        "spot_fetch_ms": 0.0,
        "orderbook_fetch_ms": 0.0,
        "market_state_ms": 0.0,
        "event_state_ms": 0.0,
    }
    metrics = getattr(source, "last_poll_metrics", None)
    if not isinstance(metrics, dict):
        return default
    return {
        key: float(metrics.get(key, 0.0) or 0.0)
        for key in default
    }


def _log_poll_timing(metrics: dict[str, int | float], *, print_to_terminal: bool = False) -> None:
    message = _format_poll_timing(metrics)
    if print_to_terminal:
        print(message, file=sys.__stderr__, flush=True)
        return
    logger.info(message)


def _format_poll_timing(metrics: dict[str, int | float]) -> str:
    return (
        "poll timing "
        "spot_fetch_ms=%.1f "
        "orderbook_fetch_ms=%.1f "
        "market_state_ms=%.1f "
        "event_state_ms=%.1f "
        "pricing_ms=%.1f "
        "executor_ms=%.1f "
        "poll_total_ms=%.1f "
        "new_event_rows=%s "
        "signal_events=%s "
        "order_events=%s "
        "closed_trades=%s"
    ) % (
        float(metrics.get("spot_fetch_ms", 0.0) or 0.0),
        float(metrics.get("orderbook_fetch_ms", 0.0) or 0.0),
        float(metrics.get("market_state_ms", 0.0) or 0.0),
        float(metrics.get("event_state_ms", 0.0) or 0.0),
        float(metrics.get("pricing_ms", 0.0) or 0.0),
        float(metrics.get("executor_ms", 0.0) or 0.0),
        float(metrics.get("poll_total_ms", 0.0) or 0.0),
        int(metrics.get("new_event_rows", 0) or 0),
        int(metrics.get("signal_events", 0) or 0),
        int(metrics.get("order_events", 0) or 0),
        int(metrics.get("closed_trades", 0) or 0),
    )


def _quiet_textual_runtime_logging() -> None:
    for logger_name in [
        "polymarket_quant.ingestion.client",
        "polymarket_quant.ingestion.spot",
        "polymarket_quant.state.dataset",
        "polymarket_quant.live.direct",
        "urllib3.connectionpool",
    ]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
