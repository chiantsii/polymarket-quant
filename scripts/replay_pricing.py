import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from polymarket_quant.live.model_loader import (
    DEFAULT_TRANSITION_MODEL_BTC_PATH,
    DEFAULT_TRANSITION_MODEL_ETH_PATH,
)
from polymarket_quant.pricing import DEFAULT_SPOT_DRIFT_DECAY_KAPPA_PER_SECOND
from polymarket_quant.pricing import (
    DEFAULT_MANUAL_SPOT_JUMP_INTENSITY_PER_SECOND,
    DEFAULT_MANUAL_SPOT_JUMP_LOG_RETURN_MEAN,
    DEFAULT_MANUAL_SPOT_JUMP_LOG_RETURN_STD,
    DEFAULT_MANUAL_SPOT_JUMP_STD_MULTIPLIER_ON_LOCAL_SIGMA,
)
from polymarket_quant.signals.mispricing import MispricingDetectorConfig, RealTimeMispricingDetector
from polymarket_quant.state.dataset import matching_parquet_paths
from polymarket_quant.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_EVENT_STATE_GLOB = "data/*/processed/event_state/shards/*.parquet"
DEFAULT_OUTPUT_ROOT = "data"


def _validate_transition_bundle(bundle: Any) -> None:
    required_attributes = (
        "spot_mu_model",
        "spot_sigma_model",
    )
    missing = [attribute for attribute in required_attributes if getattr(bundle, attribute, None) is None]
    if missing:
        raise ValueError(
            "Transition model artifact does not contain the learned spot kernel required for pricing. "
            f"Missing: {missing}. Re-run scripts/fit_transition_model.py to refresh the artifact."
        )


def _build_detector_config(
    *,
    pricing_method: str,
    n_samples: int,
    spot_drift_decay_kappa_per_second: float,
    spot_jump_intensity_per_second: float,
    spot_jump_log_return_mean: float,
    spot_jump_log_return_std: float,
    spot_jump_std_multiplier_on_local_sigma: float,
    fallback_spot_volatility_per_sqrt_second: float,
    edge_threshold: float,
    simulation_dt_seconds: float,
    rollout_horizon_seconds: float,
    transition_bundle: Any,
) -> MispricingDetectorConfig:
    return MispricingDetectorConfig(
        pricing_method=pricing_method,
        n_samples=n_samples,
        spot_drift_decay_kappa_per_second=spot_drift_decay_kappa_per_second,
        spot_jump_intensity_per_second=spot_jump_intensity_per_second,
        spot_jump_log_return_mean=spot_jump_log_return_mean,
        spot_jump_log_return_std=spot_jump_log_return_std,
        spot_jump_std_multiplier_on_local_sigma=spot_jump_std_multiplier_on_local_sigma,
        fallback_spot_volatility_per_sqrt_second=fallback_spot_volatility_per_sqrt_second,
        edge_threshold=edge_threshold,
        simulation_dt_seconds=simulation_dt_seconds,
        rollout_horizon_seconds=rollout_horizon_seconds,
        transition_bundle=transition_bundle,
    )


def replay_pricing(
    event_state_glob: str = DEFAULT_EVENT_STATE_GLOB,
    output_dir: str = DEFAULT_OUTPUT_ROOT,
    pricing_method: str = "markov_mcmc",
    n_samples: int = 1_000,
    spot_drift_decay_kappa_per_second: float = DEFAULT_SPOT_DRIFT_DECAY_KAPPA_PER_SECOND,
    spot_jump_intensity_per_second: float = DEFAULT_MANUAL_SPOT_JUMP_INTENSITY_PER_SECOND,
    spot_jump_log_return_mean: float = DEFAULT_MANUAL_SPOT_JUMP_LOG_RETURN_MEAN,
    spot_jump_log_return_std: float = DEFAULT_MANUAL_SPOT_JUMP_LOG_RETURN_STD,
    spot_jump_std_multiplier_on_local_sigma: float = DEFAULT_MANUAL_SPOT_JUMP_STD_MULTIPLIER_ON_LOCAL_SIGMA,
    fallback_spot_volatility_per_sqrt_second: float = 0.0005,
    edge_threshold: float = 0.0,
    simulation_dt_seconds: float = 1.0,
    rollout_horizon_seconds: float = 0.0,
    transition_model_btc_path: str = DEFAULT_TRANSITION_MODEL_BTC_PATH,
    transition_model_eth_path: str = DEFAULT_TRANSITION_MODEL_ETH_PATH,
    transition_bundles_by_asset: dict[str, Any] | None = None,
    transition_bundle: Any | None = None,
    max_rows: int | None = None,
    show_progress: bool = False,
    include_latest: bool = False,
) -> dict[str, Any]:
    event_state_paths = matching_parquet_paths(event_state_glob, include_latest=include_latest)
    if not event_state_paths:
        raise FileNotFoundError(f"No parquet files matched {event_state_glob}")

    bundles_by_asset = _resolve_transition_bundles_by_asset(
        transition_model_btc_path=transition_model_btc_path,
        transition_model_eth_path=transition_model_eth_path,
        transition_bundles_by_asset=transition_bundles_by_asset,
        transition_bundle=transition_bundle,
    )
    detectors_by_asset = {
        asset: RealTimeMispricingDetector(
            _build_detector_config(
                pricing_method=pricing_method,
                n_samples=n_samples,
                spot_drift_decay_kappa_per_second=spot_drift_decay_kappa_per_second,
                spot_jump_intensity_per_second=spot_jump_intensity_per_second,
                spot_jump_log_return_mean=spot_jump_log_return_mean,
                spot_jump_log_return_std=spot_jump_log_return_std,
                spot_jump_std_multiplier_on_local_sigma=spot_jump_std_multiplier_on_local_sigma,
                fallback_spot_volatility_per_sqrt_second=fallback_spot_volatility_per_sqrt_second,
                edge_threshold=edge_threshold,
                simulation_dt_seconds=simulation_dt_seconds,
                rollout_horizon_seconds=rollout_horizon_seconds,
                transition_bundle=bundle,
            )
        )
        for asset, bundle in bundles_by_asset.items()
    }

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    shard_output_paths: dict[str, str] = {}
    assets_seen: set[str] = set()
    remaining_rows = max_rows if max_rows is None else max(int(max_rows), 0)

    for shard_index, event_state_path in enumerate(event_state_paths, start=1):
        event_state = pd.read_parquet(event_state_path)
        if event_state.empty:
            logger.info(
                "Skipping empty event_state shard %s/%s: %s",
                shard_index,
                len(event_state_paths),
                event_state_path.name,
            )
            continue

        asset, event_slug = _validate_event_state_shard(event_state=event_state, shard_path=event_state_path)
        detector = detectors_by_asset.get(asset)
        if detector is None:
            raise FileNotFoundError(
                f"No transition model configured for asset {asset}. "
                f"Expected artifact at {transition_model_btc_path if asset == 'BTC' else transition_model_eth_path}."
            )

        if remaining_rows is not None:
            if remaining_rows <= 0:
                break
            event_state = event_state.head(remaining_rows).copy()
            if event_state.empty:
                break
            remaining_rows -= len(event_state)

        logger.info(
            "Processing pricing replay shard %s/%s: %s (%s rows)",
            shard_index,
            len(event_state_paths),
            event_state_path.name,
            len(event_state),
        )
        _log_replay_workload_estimate(
            event_state=event_state,
            n_samples=n_samples,
            pricing_method=pricing_method,
            rollout_horizon_seconds=rollout_horizon_seconds,
            simulation_dt_seconds=simulation_dt_seconds,
        )

        replay_rows = detector.detect(
            state_rows=event_state.to_dict("records"),
            show_progress=show_progress,
            progress_description=f"Pricing replay [{asset}:{event_slug}]",
        )
        if not replay_rows:
            raise ValueError(f"No replay pricing rows were generated for event-state shard {event_state_path}")

        replay = pd.DataFrame(replay_rows)
        output_path = _pricing_replay_shard_path(output_root=output_root, asset=asset, event_slug=event_slug)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        replay.to_parquet(output_path, index=False)

        total_rows += len(replay)
        shard_output_paths[event_slug] = str(output_path)
        assets_seen.add(asset)
        logger.info("Saved %s pricing replay rows to %s", len(replay), output_path)

    if not shard_output_paths:
        raise ValueError("No pricing replay shards were generated.")

    return {
        "rows": total_rows,
        "shard_paths": shard_output_paths,
        "assets": sorted(assets_seen),
    }


def _resolve_transition_bundles_by_asset(
    *,
    transition_model_btc_path: str,
    transition_model_eth_path: str,
    transition_bundles_by_asset: dict[str, Any] | None,
    transition_bundle: Any | None,
) -> dict[str, Any]:
    if transition_bundles_by_asset:
        normalized = {str(asset).strip().upper(): bundle for asset, bundle in transition_bundles_by_asset.items()}
        for bundle in normalized.values():
            _validate_transition_bundle(bundle)
        return normalized

    if transition_bundle is not None:
        _validate_transition_bundle(transition_bundle)
        return {"BTC": transition_bundle, "ETH": transition_bundle}

    bundle_paths = {
        "BTC": transition_model_btc_path,
        "ETH": transition_model_eth_path,
    }
    bundles_by_asset: dict[str, Any] = {}
    for asset, model_path in bundle_paths.items():
        bundle_path = Path(model_path)
        if not bundle_path.exists():
            raise FileNotFoundError(
                f"Transition model artifact not found for {asset} at {model_path}. "
                "Run scripts/fit_transition_model.py before replay pricing."
            )
        bundle = joblib.load(bundle_path)
        _validate_transition_bundle(bundle)
        bundles_by_asset[asset] = bundle
    return bundles_by_asset


def _validate_event_state_shard(*, event_state: pd.DataFrame, shard_path: Path) -> tuple[str, str]:
    if "asset" not in event_state.columns:
        raise ValueError(f"Event-state shard is missing asset column: {shard_path}")
    if "event_slug" not in event_state.columns:
        raise ValueError(f"Event-state shard is missing event_slug column: {shard_path}")

    assets = sorted(event_state["asset"].dropna().astype(str).str.upper().unique().tolist())
    event_slugs = sorted(event_state["event_slug"].dropna().astype(str).unique().tolist())
    if len(assets) != 1:
        raise ValueError(f"Expected one asset per event_state shard, found {assets} in {shard_path}")
    if len(event_slugs) != 1:
        raise ValueError(f"Expected one event_slug per event_state shard, found {event_slugs} in {shard_path}")
    return assets[0], event_slugs[0]


def _pricing_replay_shard_path(*, output_root: Path, asset: str, event_slug: str) -> Path:
    return output_root / asset / "processed" / "pricing_replay" / "shards" / f"{event_slug}.parquet"


def _log_replay_workload_estimate(
    *,
    event_state: pd.DataFrame,
    n_samples: int,
    pricing_method: str,
    rollout_horizon_seconds: float,
    simulation_dt_seconds: float,
) -> None:
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
    parser = argparse.ArgumentParser(description="Replay serialized event-state shards through pricing models.")
    parser.add_argument("--event-state-glob", default=DEFAULT_EVENT_STATE_GLOB, help="Event-state shard glob")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_ROOT, help="Output root directory for replay shards")
    parser.add_argument(
        "--pricing-method",
        choices=[
            "markov_mcmc",
        ],
        default="markov_mcmc",
        help="Pricing estimator to replay",
    )
    parser.add_argument("--n-samples", type=int, default=1_000, help="Simulation paths for pricing methods that require them")
    parser.add_argument(
        "--spot-drift-decay-kappa-per-second",
        type=float,
        default=float(DEFAULT_SPOT_DRIFT_DECAY_KAPPA_PER_SECOND),
        help="Exponential decay rate applied to the learned spot drift during rollout. Default is a 5-second half-life.",
    )
    parser.add_argument(
        "--spot-jump-intensity-per-second",
        type=float,
        default=float(DEFAULT_MANUAL_SPOT_JUMP_INTENSITY_PER_SECOND),
        help="Manual jump intensity per second. Default is the selected 50/day zero-mean jump prior.",
    )
    parser.add_argument(
        "--spot-jump-log-return-mean",
        type=float,
        default=float(DEFAULT_MANUAL_SPOT_JUMP_LOG_RETURN_MEAN),
        help="Manual jump log-return mean. Default keeps the jump prior zero-mean.",
    )
    parser.add_argument(
        "--spot-jump-log-return-std",
        type=float,
        default=float(DEFAULT_MANUAL_SPOT_JUMP_LOG_RETURN_STD),
        help="Manual fixed jump log-return std. When the local-sigma multiplier is positive, this fixed std is ignored.",
    )
    parser.add_argument(
        "--spot-jump-std-multiplier-on-local-sigma",
        type=float,
        default=float(DEFAULT_MANUAL_SPOT_JUMP_STD_MULTIPLIER_ON_LOCAL_SIGMA),
        help=(
            "Optional manual jump-std multiplier applied to each row's local spot volatility. "
            "Default is the selected 20x local-sigma jump prior. When positive, replay uses "
            "jump_std = multiplier * volatility_per_sqrt_second instead of the fixed jump std."
        ),
    )
    parser.add_argument(
        "--transition-model-btc-path",
        default=DEFAULT_TRANSITION_MODEL_BTC_PATH,
        help="BTC transition model artifact path",
    )
    parser.add_argument(
        "--transition-model-eth-path",
        default=DEFAULT_TRANSITION_MODEL_ETH_PATH,
        help="ETH transition model artifact path",
    )
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
        help="Optional global cap on the number of event-state rows to replay. Useful for smoke tests and debugging.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable progress output during pricing replay")
    parser.add_argument("--edge-threshold", type=float, default=0.0, help="Minimum edge required to signal a buy")
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
        spot_drift_decay_kappa_per_second=args.spot_drift_decay_kappa_per_second,
        spot_jump_intensity_per_second=args.spot_jump_intensity_per_second,
        spot_jump_log_return_mean=args.spot_jump_log_return_mean,
        spot_jump_log_return_std=args.spot_jump_log_return_std,
        spot_jump_std_multiplier_on_local_sigma=args.spot_jump_std_multiplier_on_local_sigma,
        edge_threshold=args.edge_threshold,
        simulation_dt_seconds=args.simulation_dt_seconds,
        rollout_horizon_seconds=args.rollout_horizon_seconds,
        transition_model_btc_path=args.transition_model_btc_path,
        transition_model_eth_path=args.transition_model_eth_path,
        max_rows=args.max_rows,
        show_progress=not args.no_progress,
        fallback_spot_volatility_per_sqrt_second=args.fallback_spot_volatility_per_sqrt_second,
        include_latest=args.include_latest,
    )
    logger.info("Replay complete: %s", result)


if __name__ == "__main__":
    main()
