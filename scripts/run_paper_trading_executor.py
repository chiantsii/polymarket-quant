import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

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
from polymarket_quant.evaluation import BaselineUpPaperTradingExecutor, BaselineUpStrategyConfig
from polymarket_quant.signals.mispricing import AssetAwareMispricingDetector, RealTimeMispricingDetector
from polymarket_quant.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_EVENT_STATE_PATH = "data/processed/crypto_5m_event_state_latest.parquet"
def run_paper_trading_executor(
    *,
    source_mode: str = "direct-live",
    event_state_path: str = DEFAULT_EVENT_STATE_PATH,
    transition_model_path: str = DEFAULT_TRANSITION_MODEL_PATH,
    transition_model_btc_path: str = DEFAULT_TRANSITION_MODEL_BTC_PATH,
    transition_model_eth_path: str = DEFAULT_TRANSITION_MODEL_ETH_PATH,
    orderbook_summary_glob: str = DEFAULT_LIVE_ORDERBOOK_SUMMARY_GLOB,
    orderbook_levels_glob: str = DEFAULT_LIVE_ORDERBOOK_LEVELS_GLOB,
    spot_glob: str = DEFAULT_LIVE_SPOT_GLOB,
    live_state_output_dir: str = DEFAULT_LIVE_STATE_OUTPUT_DIR,
    config_path: str = "configs/base.yaml",
    event_duration_seconds: int = 300,
    event_slug_prefixes: tuple[str, ...] = ("btc-updown-5m", "eth-updown-5m"),
    series_slugs: tuple[str, ...] = ("btc-up-or-down-5m", "eth-up-or-down-5m"),
    output_dir: str = "artifacts/paper_trading",
    poll_interval_seconds: float = 2.0,
    max_polls: int = 0,
    n_samples: int = 1_000,
    simulation_dt_seconds: float = 1.0,
    rollout_horizon_seconds: float = 0.0,
    edge_threshold: float = 0.0,
    entry_edge_threshold: float = 0.03,
    exit_hold_edge_threshold: float = 0.0,
    max_holding_seconds: float = 60.0,
    max_edge_cap: float = 0.12,
) -> dict[str, Any]:
    detector = _build_detector(
        transition_model_path=transition_model_path,
        transition_model_btc_path=transition_model_btc_path,
        transition_model_eth_path=transition_model_eth_path,
        n_samples=n_samples,
        simulation_dt_seconds=simulation_dt_seconds,
        rollout_horizon_seconds=rollout_horizon_seconds,
        edge_threshold=edge_threshold,
    )
    executor = BaselineUpPaperTradingExecutor(
        BaselineUpStrategyConfig(
            entry_edge_threshold=entry_edge_threshold,
            exit_hold_edge_threshold=exit_hold_edge_threshold,
            max_holding_seconds=max_holding_seconds,
            max_edge_cap=max_edge_cap,
        )
    )

    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    signal_path = out_dir / f"paper_signal_events_{run_timestamp}.jsonl"
    order_path = out_dir / f"paper_order_events_{run_timestamp}.jsonl"
    trade_path = out_dir / f"paper_trade_ledger_{run_timestamp}.jsonl"
    latest_signal_path = out_dir / "paper_signal_events_latest.jsonl"
    latest_order_path = out_dir / "paper_order_events_latest.jsonl"
    latest_trade_path = out_dir / "paper_trade_ledger_latest.jsonl"

    seen_keys: set[str] = set()
    if source_mode == "direct-live":
        live_source = DirectLiveEventStateSource(
            DirectLiveSourceConfig(
                config_path=config_path,
                output_dir=live_state_output_dir,
                event_duration_seconds=event_duration_seconds,
                event_slug_prefixes=event_slug_prefixes,
                series_slugs=series_slugs,
            )
        )
    elif source_mode == "live-capture":
        live_source = LiveCaptureEventStateSource(
            LiveCaptureSourceConfig(
                orderbook_summary_glob=orderbook_summary_glob,
                orderbook_levels_glob=orderbook_levels_glob,
                spot_glob=spot_glob,
                output_dir=live_state_output_dir,
            )
        )
    else:
        live_source = None
    poll_count = 0
    processed_event_rows = 0
    processed_orders = 0
    closed_trades = 0

    with (
        signal_path.open("a", encoding="utf-8") as signal_file,
        order_path.open("a", encoding="utf-8") as order_file,
        trade_path.open("a", encoding="utf-8") as trade_file,
    ):
        while True:
            poll_count += 1
            new_event_rows = live_source.poll_new_rows() if live_source is not None else _load_new_event_state_rows(Path(event_state_path), seen_keys)
            if new_event_rows:
                pricing_rows = detector.detect(new_event_rows, show_progress=False)
                pricing_rows = sorted(
                    pricing_rows,
                    key=lambda row: (
                        str(row.get("event_slug", "")),
                        str(row.get("collected_at", "")),
                        str(row.get("outcome_name", "")),
                    ),
                )
                for row in pricing_rows:
                    result = executor.process_row(row)
                    if result is None:
                        continue
                    signal_event = {
                        key: value
                        for key, value in result.items()
                        if key not in {"row", "closed_trade", "order_events"}
                    }
                    signal_file.write(json.dumps(_json_ready(signal_event), sort_keys=True) + "\n")
                    signal_file.flush()

                    for order_event in result.get("order_events", []):
                        order_file.write(json.dumps(_json_ready(order_event), sort_keys=True) + "\n")
                        processed_orders += 1
                    order_file.flush()

                    if result.get("closed_trade") is not None:
                        trade_file.write(json.dumps(_json_ready(result["closed_trade"]), sort_keys=True) + "\n")
                        trade_file.flush()
                        closed_trades += 1
                    logger.info(
                        "[paper] %s %s decision=%s state=%s orders=%s closed_trade=%s",
                        signal_event.get("collected_at"),
                        signal_event.get("event_slug"),
                        signal_event.get("decision"),
                        signal_event.get("position_state"),
                        len(result.get("order_events", [])),
                        result.get("closed_trade") is not None,
                    )
                processed_event_rows += len(new_event_rows)

            latest_signal_path.write_text(signal_path.read_text(encoding="utf-8"), encoding="utf-8")
            latest_order_path.write_text(order_path.read_text(encoding="utf-8"), encoding="utf-8")
            latest_trade_path.write_text(trade_path.read_text(encoding="utf-8"), encoding="utf-8")
            if max_polls > 0 and poll_count >= max_polls:
                break
            time.sleep(max(poll_interval_seconds, 0.0))

    return {
        "source_mode": source_mode,
        "signal_path": str(signal_path),
        "order_path": str(order_path),
        "trade_path": str(trade_path),
        "latest_signal_path": str(latest_signal_path),
        "latest_order_path": str(latest_order_path),
        "latest_trade_path": str(latest_trade_path),
        "live_event_state_path": str(Path(live_state_output_dir) / "live_event_state_latest.parquet"),
        "processed_event_rows": processed_event_rows,
        "processed_orders": processed_orders,
        "closed_trades": closed_trades,
        "polls": poll_count,
    }


def _build_detector(
    *,
    transition_model_path: str,
    transition_model_btc_path: str,
    transition_model_eth_path: str,
    n_samples: int,
    simulation_dt_seconds: float,
    rollout_horizon_seconds: float,
    edge_threshold: float,
) -> RealTimeMispricingDetector | AssetAwareMispricingDetector:
    return build_pricing_detector(
        transition_model_path=transition_model_path,
        transition_model_btc_path=transition_model_btc_path,
        transition_model_eth_path=transition_model_eth_path,
        n_samples=n_samples,
        simulation_dt_seconds=simulation_dt_seconds,
        rollout_horizon_seconds=rollout_horizon_seconds,
        edge_threshold=edge_threshold,
    )


def _load_new_event_state_rows(path: Path, seen_keys: set[str]) -> list[dict[str, Any]]:
    if not path.exists():
        logger.warning("Event-state parquet not found yet: %s", path)
        return []

    frame = pd.read_parquet(path)
    if frame.empty:
        return []
    frame = frame.sort_values(["event_slug", "collected_at"]).reset_index(drop=True)
    keys = frame["event_slug"].astype(str) + "|" + frame["collected_at"].astype(str)
    new_mask = ~keys.isin(seen_keys)
    if not new_mask.any():
        return []
    seen_keys.update(keys.loc[new_mask].tolist())
    return frame.loc[new_mask].to_dict("records")


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local paper-trading executor from live capture files or an event_state parquet.")
    parser.add_argument("--source-mode", choices=["direct-live", "live-capture", "event-state"], default="direct-live", help="Fetch market data directly, use collector latest files, or use an event_state parquet")
    parser.add_argument("--event-state-path", default=DEFAULT_EVENT_STATE_PATH, help="Latest event_state parquet path (event-state mode)")
    parser.add_argument("--transition-model-path", default=DEFAULT_TRANSITION_MODEL_PATH, help="Fallback shared transition model artifact")
    parser.add_argument("--transition-model-btc-path", default=DEFAULT_TRANSITION_MODEL_BTC_PATH, help="BTC transition model artifact")
    parser.add_argument("--transition-model-eth-path", default=DEFAULT_TRANSITION_MODEL_ETH_PATH, help="ETH transition model artifact")
    parser.add_argument("--config-path", default="configs/base.yaml", help="Repo config path for direct-live mode")
    parser.add_argument("--orderbook-summary-glob", default=DEFAULT_LIVE_ORDERBOOK_SUMMARY_GLOB, help="Live orderbook summary parquet glob")
    parser.add_argument("--orderbook-levels-glob", default=DEFAULT_LIVE_ORDERBOOK_LEVELS_GLOB, help="Live orderbook levels parquet glob")
    parser.add_argument("--spot-glob", default=DEFAULT_LIVE_SPOT_GLOB, help="Live spot parquet glob")
    parser.add_argument("--live-state-output-dir", default=DEFAULT_LIVE_STATE_OUTPUT_DIR, help="Output dir for rebuilt live market/event state parquet")
    parser.add_argument("--event-duration-seconds", type=int, default=300, help="Active market window size for direct-live mode")
    parser.add_argument("--event-slug-prefixes", nargs="+", default=["btc-updown-5m", "eth-updown-5m"], help="Direct-live event slug prefixes")
    parser.add_argument("--series-slugs", nargs="+", default=["btc-up-or-down-5m", "eth-up-or-down-5m"], help="Direct-live Gamma series slugs")
    parser.add_argument("--output-dir", default="artifacts/paper_trading", help="Output directory for paper JSONL logs")
    parser.add_argument("--poll-interval-seconds", type=float, default=2.0, help="Polling interval for latest parquet")
    parser.add_argument("--max-polls", type=int, default=0, help="Stop after N polls; 0 means run forever")
    parser.add_argument("--n-samples", type=int, default=1000, help="Pricing simulation paths")
    parser.add_argument("--simulation-dt-seconds", type=float, default=1.0, help="Simulation dt in seconds")
    parser.add_argument("--rollout-horizon-seconds", type=float, default=0.0, help="Optional rollout step override")
    parser.add_argument("--edge-threshold", type=float, default=0.0, help="Detector buy-signal threshold")
    parser.add_argument("--entry-edge-threshold", type=float, default=0.03, help="Baseline strategy entry threshold")
    parser.add_argument("--exit-hold-edge-threshold", type=float, default=0.0, help="Baseline strategy exit hold-edge threshold")
    parser.add_argument("--max-holding-seconds", type=float, default=60.0, help="Baseline strategy max holding time")
    parser.add_argument("--max-edge-cap", type=float, default=0.12, help="Ignore oversized entry edges above this cap")
    args = parser.parse_args()

    run_paper_trading_executor(
        source_mode=args.source_mode,
        event_state_path=args.event_state_path,
        transition_model_path=args.transition_model_path,
        transition_model_btc_path=args.transition_model_btc_path,
        transition_model_eth_path=args.transition_model_eth_path,
        orderbook_summary_glob=args.orderbook_summary_glob,
        orderbook_levels_glob=args.orderbook_levels_glob,
        spot_glob=args.spot_glob,
        live_state_output_dir=args.live_state_output_dir,
        config_path=args.config_path,
        event_duration_seconds=args.event_duration_seconds,
        event_slug_prefixes=tuple(args.event_slug_prefixes),
        series_slugs=tuple(args.series_slugs),
        output_dir=args.output_dir,
        poll_interval_seconds=args.poll_interval_seconds,
        max_polls=args.max_polls,
        n_samples=args.n_samples,
        simulation_dt_seconds=args.simulation_dt_seconds,
        rollout_horizon_seconds=args.rollout_horizon_seconds,
        edge_threshold=args.edge_threshold,
        entry_edge_threshold=args.entry_edge_threshold,
        exit_hold_edge_threshold=args.exit_hold_edge_threshold,
        max_holding_seconds=args.max_holding_seconds,
        max_edge_cap=args.max_edge_cap,
    )


if __name__ == "__main__":
    main()
