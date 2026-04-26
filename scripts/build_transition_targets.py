import argparse
from datetime import datetime, timezone
from pathlib import Path

from polymarket_quant.state.dataset import load_parquet_glob
from polymarket_quant.state.transition_targets import (
    TransitionTargetConfig,
    build_transition_target_dataset,
    build_transition_target_summary,
)
from polymarket_quant.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_EVENT_STATE_GLOB = "data/processed/crypto_5m_event_state_latest.parquet"


def build_transition_targets(
    event_state_glob: str = DEFAULT_EVENT_STATE_GLOB,
    output_dir: str = "data/processed",
    include_latest: bool = False,
    run_timestamp: str | None = None,
    pairing_mode: str = "next",
    horizons_seconds: tuple[float, ...] = (15.0,),
    tolerance_seconds: float = 2.0,
    include_unmatched: bool = False,
) -> dict[str, object]:
    event_state = load_parquet_glob(event_state_glob, include_latest=include_latest)
    config = TransitionTargetConfig(
        pairing_mode=pairing_mode,
        horizons_seconds=tuple(horizons_seconds),
        tolerance_seconds=tolerance_seconds,
        include_unmatched=include_unmatched,
    )
    transition_targets = build_transition_target_dataset(event_state=event_state, config=config)
    summary = build_transition_target_summary(transition_targets)

    run_timestamp = run_timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    parquet_path = output_path / f"crypto_5m_transition_targets_{run_timestamp}.parquet"
    latest_path = output_path / "crypto_5m_transition_targets_latest.parquet"
    transition_targets.to_parquet(parquet_path, index=False)
    transition_targets.to_parquet(latest_path, index=False)

    logger.info(
        "Saved %s transition-target rows to %s for pairing_mode=%s horizons=%s",
        len(transition_targets),
        parquet_path,
        config.validated_pairing_mode(),
        tuple(config.validated_horizons()),
    )
    return {
        **summary,
        "output_path": str(parquet_path),
        "latest_path": str(latest_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build structured full-state transition targets from event-state data.")
    parser.add_argument(
        "--event-state-glob",
        default=DEFAULT_EVENT_STATE_GLOB,
        help="Event-state parquet path or glob",
    )
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    parser.add_argument("--include-latest", action="store_true", help="Include *_latest.parquet inputs")
    parser.add_argument(
        "--pairing-mode",
        choices=["next", "horizon"],
        default="next",
        help="Use next-observation pairing or fixed-horizon pairing",
    )
    parser.add_argument(
        "--horizons-seconds",
        type=float,
        nargs="+",
        default=(15.0,),
        help="One or more future horizons, in seconds. Used only when --pairing-mode horizon",
    )
    parser.add_argument(
        "--tolerance-seconds",
        type=float,
        default=2.0,
        help="Maximum allowable mismatch between requested and matched horizon",
    )
    parser.add_argument(
        "--include-unmatched",
        action="store_true",
        help="Keep rows without a valid future match and label them via target_status",
    )
    args = parser.parse_args()

    build_transition_targets(
        event_state_glob=args.event_state_glob,
        output_dir=args.output_dir,
        include_latest=args.include_latest,
        pairing_mode=args.pairing_mode,
        horizons_seconds=tuple(args.horizons_seconds),
        tolerance_seconds=args.tolerance_seconds,
        include_unmatched=args.include_unmatched,
    )


if __name__ == "__main__":
    main()
