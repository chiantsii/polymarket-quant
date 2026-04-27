import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from polymarket_quant.signals.mispricing import MispricingDetectorConfig, RealTimeMispricingDetector
from polymarket_quant.state.dataset import load_parquet_glob
from polymarket_quant.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_EVENT_STATE_GLOB = "data/processed/crypto_5m_event_state_latest.parquet"
def _build_detector_config(
    *,
    pricing_method: str,
    n_samples: int,
    fallback_spot_volatility_per_sqrt_second: float,
    edge_threshold: float,
    max_allowed_risk: float,
    simulation_dt_seconds: float,
    rollout_horizon_seconds: float,
) -> MispricingDetectorConfig:
    """Create the canonical pricing/state assumption bundle for replay."""
    return MispricingDetectorConfig(
        pricing_method=pricing_method,
        n_samples=n_samples,
        fallback_spot_volatility_per_sqrt_second=fallback_spot_volatility_per_sqrt_second,
        edge_threshold=edge_threshold,
        max_allowed_risk=max_allowed_risk,
        simulation_dt_seconds=simulation_dt_seconds,
        rollout_horizon_seconds=rollout_horizon_seconds,
    )


def replay_pricing(
    event_state_glob: str = DEFAULT_EVENT_STATE_GLOB,
    output_dir: str = "data/processed",
    pricing_method: str = "markov_mcmc",
    n_samples: int = 1_000,
    fallback_spot_volatility_per_sqrt_second: float = 0.0005,
    edge_threshold: float = 0.0,
    max_allowed_risk: float = 0.10,
    simulation_dt_seconds: float = 1.0,
    rollout_horizon_seconds: float = 0.0,
    max_rows: int | None = None,
    show_progress: bool = False,
    include_latest: bool = False,
    run_timestamp: str | None = None,
) -> dict[str, Any]:
    """Replay serialized event-state rows through the pricing detector.

    This script is intentionally downstream of state construction:
    `build_market_state.py` and `build_event_state.py` own state creation,
    while `replay_pricing.py` only consumes `event_state` rows and prices them.
    """
    event_state = load_parquet_glob(event_state_glob, include_latest=include_latest)

    detector_config = _build_detector_config(
        pricing_method=pricing_method,
        n_samples=n_samples,
        fallback_spot_volatility_per_sqrt_second=fallback_spot_volatility_per_sqrt_second,
        edge_threshold=edge_threshold,
        max_allowed_risk=max_allowed_risk,
        simulation_dt_seconds=simulation_dt_seconds,
        rollout_horizon_seconds=rollout_horizon_seconds,
    )
    if max_rows is not None:
        event_state = event_state.head(max(int(max_rows), 0)).copy()

    # Log the implied workload before we start. This is especially useful for
    # rollout-based pricing, where runtime scales with rows x steps x paths.
    _log_replay_workload_estimate(
        event_state=event_state,
        n_samples=n_samples,
        pricing_method=pricing_method,
        rollout_horizon_seconds=rollout_horizon_seconds,
        simulation_dt_seconds=simulation_dt_seconds,
    )

    detector = RealTimeMispricingDetector(detector_config)

    replay_rows = detector.detect(
        state_rows=event_state.to_dict("records"),
        show_progress=show_progress,
        progress_description="Pricing replay",
    )

    if not replay_rows:
        raise ValueError("No replay pricing rows were generated. Check the event-state input rows.")

    replay = pd.DataFrame(replay_rows)

    # Persist both a timestamped artifact and a *_latest pointer so downstream
    # inspection scripts can either compare runs or just read the newest output.
    run_timestamp = run_timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    parquet_path = output_path / f"crypto_5m_pricing_replay_{run_timestamp}.parquet"
    latest_path = output_path / "crypto_5m_pricing_replay_latest.parquet"
    replay.to_parquet(parquet_path, index=False)
    replay.to_parquet(latest_path, index=False)

    logger.info("Saved %s pricing replay rows to %s", len(replay), parquet_path)

    return {
        "rows": len(replay),
        "output_path": str(parquet_path),
        "latest_path": str(latest_path),
    }


def _log_replay_workload_estimate(
    *,
    event_state: pd.DataFrame,
    n_samples: int,
    pricing_method: str,
    rollout_horizon_seconds: float,
    simulation_dt_seconds: float,
) -> None:
    """Emit a lightweight runtime estimate before pricing replay starts."""
    if event_state.empty:
        logger.warning("Pricing replay has no event-state rows to process.")
        return

    if pricing_method != "markov_mcmc":
        logger.info("Pricing replay will process %s event-state rows.", len(event_state))
        return

    seconds_to_end = pd.to_numeric(event_state.get("seconds_to_end"), errors="coerce").dropna()
    if seconds_to_end.empty:
        logger.info("Pricing replay will process %s event-state rows.", len(event_state))
        return

    if rollout_horizon_seconds > 0.0:
        step_seconds = float(rollout_horizon_seconds)
    else:
        step_seconds = float(simulation_dt_seconds)

    if step_seconds <= 0.0:
        logger.info(
            "Pricing replay will process %s event-state rows using single-kernel terminal rollout with %s paths.",
            len(event_state),
            n_samples,
        )
        return

    estimated_steps = np.ceil(seconds_to_end / max(step_seconds, 1e-9)).astype(int)
    total_rollout_steps = int(estimated_steps.sum())
    total_path_steps = int(total_rollout_steps * max(int(n_samples), 0))
    logger.info(
        (
            "Pricing replay workload estimate: rows=%s, paths=%s, step_seconds=%.3f, "
            "median_steps_per_row=%s, max_steps_per_row=%s, total_rollout_steps=%s, total_path_steps=%s"
        ),
        len(event_state),
        n_samples,
        step_seconds,
        int(np.median(estimated_steps)),
        int(np.max(estimated_steps)),
        total_rollout_steps,
        total_path_steps,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay serialized event-state rows through pricing models.")
    parser.add_argument("--event-state-glob", default=DEFAULT_EVENT_STATE_GLOB, help="Event-state parquet glob")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory for replay parquet")
    parser.add_argument(
        "--pricing-method",
        choices=[
            "markov_mcmc",
        ],
        default="markov_mcmc",
        help="Pricing estimator to replay",
    )
    parser.add_argument("--n-samples", type=int, default=1_000, help="Simulation paths for pricing methods that require them")
    parser.add_argument("--simulation-dt-seconds", type=float, default=1.0, help="Simulation step size in seconds")
    parser.add_argument(
        "--rollout-horizon-seconds",
        type=float,
        default=0.0,
        help="Optional simulation step override in seconds. Use 0 to follow --simulation-dt-seconds.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on the number of event-state rows to replay. Useful for smoke tests and debugging.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable progress output during pricing replay")
    parser.add_argument("--edge-threshold", type=float, default=0.0, help="Minimum edge required to signal a buy")
    parser.add_argument("--max-allowed-risk", type=float, default=0.10, help="Maximum risk score allowed for a buy signal")
    parser.add_argument(
        "--fallback-spot-volatility-per-sqrt-second",
        type=float,
        default=0.0005,
        help="Fallback spot volatility used when event_state is missing volatility_per_sqrt_second",
    )
    parser.add_argument("--include-latest", action="store_true", help="Include *_latest.parquet inputs")
    args = parser.parse_args()

    result = replay_pricing(
        event_state_glob=args.event_state_glob,
        output_dir=args.output_dir,
        pricing_method=args.pricing_method,
        n_samples=args.n_samples,
        edge_threshold=args.edge_threshold,
        max_allowed_risk=args.max_allowed_risk,
        simulation_dt_seconds=args.simulation_dt_seconds,
        rollout_horizon_seconds=args.rollout_horizon_seconds,
        max_rows=args.max_rows,
        show_progress=not args.no_progress,
        fallback_spot_volatility_per_sqrt_second=args.fallback_spot_volatility_per_sqrt_second,
        include_latest=args.include_latest,
    )
    logger.info("Replay complete: %s", result)


if __name__ == "__main__":
    main()
