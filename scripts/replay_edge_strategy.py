import argparse
import glob
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from polymarket_quant.evaluation.edge_strategy import edge_strategy_summary, replay_baseline_up_strategy
from polymarket_quant.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_PRICING_REPLAY_GLOB = "data/processed/crypto_5m_pricing_replay_*.parquet"


def replay_edge_strategy(
    pricing_replay_glob: str = DEFAULT_PRICING_REPLAY_GLOB,
    output_dir: str = "data/processed",
    event_duration_seconds: float = 300.0,
    open_cooldown_seconds: float = 30.0,
    entry_edge_threshold: float = 0.03,
    entry_confirmation_snapshots: int = 2,
    min_entry_seconds_to_end: float = 120.0,
    max_entry_seconds_to_end: float = 300.0,
    no_new_entry_seconds_to_end: float = 60.0,
    max_holding_seconds: float | None = None,
    forced_exit_seconds_to_end: float | None = None,
    max_edge_cap: float = 0.12,
    exit_hold_edge_threshold: float = 0.0,
    market_probability_exclusion_low: float | None = 0.40,
    market_probability_exclusion_high: float | None = 0.60,
    confident_exit_fair_probability_threshold: float = 0.86,
    confident_exit_window_seconds: float = 60.0,
    confident_exit_hold_edge_floor: float = -0.02,
    include_latest: bool = False,
    run_timestamp: str | None = None,
) -> dict:
    """Replay the baseline Up-only taker strategy on pricing-replay rows."""
    replay = _load_replay(pricing_replay_glob, include_latest=include_latest)
    episodes = replay_baseline_up_strategy(
        replay,
        event_duration_seconds=event_duration_seconds,
        open_cooldown_seconds=open_cooldown_seconds,
        entry_edge_threshold=entry_edge_threshold,
        entry_confirmation_snapshots=entry_confirmation_snapshots,
        min_entry_seconds_to_end=min_entry_seconds_to_end,
        max_entry_seconds_to_end=max_entry_seconds_to_end,
        no_new_entry_seconds_to_end=no_new_entry_seconds_to_end,
        max_holding_seconds=max_holding_seconds,
        forced_exit_seconds_to_end=forced_exit_seconds_to_end,
        max_edge_cap=max_edge_cap,
        exit_hold_edge_threshold=exit_hold_edge_threshold,
        market_probability_exclusion_low=market_probability_exclusion_low,
        market_probability_exclusion_high=market_probability_exclusion_high,
        confident_exit_fair_probability_threshold=confident_exit_fair_probability_threshold,
        confident_exit_window_seconds=confident_exit_window_seconds,
        confident_exit_hold_edge_floor=confident_exit_hold_edge_floor,
    )

    run_timestamp = run_timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    parquet_path = output_path / f"crypto_5m_edge_strategy_{run_timestamp}.parquet"
    latest_path = output_path / "crypto_5m_edge_strategy_latest.parquet"
    episodes.to_parquet(parquet_path, index=False)
    episodes.to_parquet(latest_path, index=False)

    summary = edge_strategy_summary(episodes)
    logger.info("Saved %s baseline strategy episodes to %s", len(episodes), parquet_path)
    logger.info("Baseline strategy summary: %s", summary)
    return {
        "episodes": len(episodes),
        "output_path": str(parquet_path),
        "latest_path": str(latest_path),
        "summary": summary,
    }


def _load_replay(pattern: str, include_latest: bool) -> pd.DataFrame:
    paths = [Path(path) for path in sorted(glob.glob(pattern))]
    if not include_latest:
        paths = [path for path in paths if "latest" not in path.name]
    if not paths and not include_latest:
        paths = [Path(path) for path in sorted(glob.glob(pattern))]
    if not paths:
        raise FileNotFoundError(f"No pricing replay parquet files matched {pattern}")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay the baseline Up-only taker strategy.")
    parser.add_argument("--pricing-replay-glob", default=DEFAULT_PRICING_REPLAY_GLOB, help="Pricing replay parquet glob")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    parser.add_argument("--event-duration-seconds", type=float, default=300.0, help="Nominal event duration in seconds")
    parser.add_argument("--open-cooldown-seconds", type=float, default=30.0, help="Skip new entries during the first N seconds after market open")
    parser.add_argument("--entry-edge-threshold", type=float, default=0.03, help="Minimum buy edge required to enter")
    parser.add_argument(
        "--entry-confirmation-snapshots",
        type=int,
        default=2,
        help="Number of consecutive qualifying snapshots required before entry",
    )
    parser.add_argument(
        "--min-entry-seconds-to-end",
        type=float,
        default=120.0,
        help="Minimum seconds to end required for a new entry",
    )
    parser.add_argument(
        "--max-entry-seconds-to-end",
        type=float,
        default=300.0,
        help="Maximum seconds to end allowed for a new entry",
    )
    parser.add_argument(
        "--no-new-entry-seconds-to-end",
        type=float,
        default=60.0,
        help="Block new entries once seconds-to-end falls below this threshold",
    )
    parser.add_argument("--max-holding-seconds", type=float, default=None, help="Optional maximum position holding time; disabled by default")
    parser.add_argument("--forced-exit-seconds-to-end", type=float, default=None, help="Optional forced exit time-to-end threshold; disabled by default")
    parser.add_argument("--max-edge-cap", type=float, default=0.12, help="Ignore oversized entry edges above this cap")
    parser.add_argument(
        "--exit-hold-edge-threshold",
        type=float,
        default=0.0,
        help="Exit when hold_edge falls below this threshold",
    )
    parser.add_argument("--market-probability-exclusion-low", type=float, default=0.40, help="Lower bound of market implied probability chop zone exclusion")
    parser.add_argument("--market-probability-exclusion-high", type=float, default=0.60, help="Upper bound of market implied probability chop zone exclusion")
    parser.add_argument("--confident-exit-fair-probability-threshold", type=float, default=0.86, help="Relax exit threshold when fair probability is above this level")
    parser.add_argument("--confident-exit-window-seconds", type=float, default=60.0, help="Apply dynamic confident exit logic inside this time-to-end window")
    parser.add_argument("--confident-exit-hold-edge-floor", type=float, default=-0.02, help="Most negative hold-edge threshold allowed near settlement for high-confidence states")
    parser.add_argument("--include-latest", action="store_true", help="Include *_latest.parquet inputs")
    args = parser.parse_args()

    replay_edge_strategy(
        pricing_replay_glob=args.pricing_replay_glob,
        output_dir=args.output_dir,
        event_duration_seconds=args.event_duration_seconds,
        open_cooldown_seconds=args.open_cooldown_seconds,
        entry_edge_threshold=args.entry_edge_threshold,
        entry_confirmation_snapshots=args.entry_confirmation_snapshots,
        min_entry_seconds_to_end=args.min_entry_seconds_to_end,
        max_entry_seconds_to_end=args.max_entry_seconds_to_end,
        no_new_entry_seconds_to_end=args.no_new_entry_seconds_to_end,
        max_holding_seconds=args.max_holding_seconds,
        forced_exit_seconds_to_end=args.forced_exit_seconds_to_end,
        max_edge_cap=args.max_edge_cap,
        exit_hold_edge_threshold=args.exit_hold_edge_threshold,
        market_probability_exclusion_low=args.market_probability_exclusion_low,
        market_probability_exclusion_high=args.market_probability_exclusion_high,
        confident_exit_fair_probability_threshold=args.confident_exit_fair_probability_threshold,
        confident_exit_window_seconds=args.confident_exit_window_seconds,
        confident_exit_hold_edge_floor=args.confident_exit_hold_edge_floor,
        include_latest=args.include_latest,
    )


if __name__ == "__main__":
    main()
