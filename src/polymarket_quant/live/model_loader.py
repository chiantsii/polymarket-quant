from __future__ import annotations

from pathlib import Path

import joblib

from polymarket_quant.pricing import (
    DEFAULT_MANUAL_SPOT_JUMP_INTENSITY_PER_SECOND,
    DEFAULT_MANUAL_SPOT_JUMP_LOG_RETURN_MEAN,
    DEFAULT_MANUAL_SPOT_JUMP_LOG_RETURN_STD,
    DEFAULT_MANUAL_SPOT_JUMP_STD_MULTIPLIER_ON_LOCAL_SIGMA,
    DEFAULT_SPOT_DRIFT_DECAY_KAPPA_PER_SECOND,
)
from polymarket_quant.signals.mispricing import (
    AssetAwareMispricingDetector,
    MispricingDetectorConfig,
    RealTimeMispricingDetector,
)
from polymarket_quant.state import TransitionModelBundle

DEFAULT_TRANSITION_MODEL_PATH = "artifacts/transition_model/transition_model_latest.joblib"
DEFAULT_TRANSITION_MODEL_BTC_PATH = "artifacts/transition_model/transition_model_btc_latest.joblib"
DEFAULT_TRANSITION_MODEL_ETH_PATH = "artifacts/transition_model/transition_model_eth_latest.joblib"


def build_pricing_detector(
    *,
    transition_model_path: str = DEFAULT_TRANSITION_MODEL_PATH,
    transition_model_btc_path: str = DEFAULT_TRANSITION_MODEL_BTC_PATH,
    transition_model_eth_path: str = DEFAULT_TRANSITION_MODEL_ETH_PATH,
    n_samples: int,
    simulation_dt_seconds: float,
    rollout_horizon_seconds: float,
    edge_threshold: float,
) -> RealTimeMispricingDetector | AssetAwareMispricingDetector:
    detectors_by_asset: dict[str, RealTimeMispricingDetector] = {}

    for asset, model_path in {
        "BTC": transition_model_btc_path,
        "ETH": transition_model_eth_path,
    }.items():
        bundle = _load_transition_bundle(model_path)
        if bundle is None:
            continue
        detectors_by_asset[asset] = _build_single_detector(
            bundle=bundle,
            n_samples=n_samples,
            simulation_dt_seconds=simulation_dt_seconds,
            rollout_horizon_seconds=rollout_horizon_seconds,
            edge_threshold=edge_threshold,
        )

    fallback_bundle = _load_transition_bundle(transition_model_path)
    fallback_detector = (
        _build_single_detector(
            bundle=fallback_bundle,
            n_samples=n_samples,
            simulation_dt_seconds=simulation_dt_seconds,
            rollout_horizon_seconds=rollout_horizon_seconds,
            edge_threshold=edge_threshold,
        )
        if fallback_bundle is not None
        else None
    )

    if detectors_by_asset:
        return AssetAwareMispricingDetector(
            detectors_by_asset=detectors_by_asset,
            fallback_detector=fallback_detector,
        )
    if fallback_detector is not None:
        return fallback_detector

    raise FileNotFoundError(
        "No transition model artifacts found. Expected either a shared model at "
        f"{transition_model_path} or per-asset models at {transition_model_btc_path} / {transition_model_eth_path}."
    )


def _build_single_detector(
    *,
    bundle: TransitionModelBundle,
    n_samples: int,
    simulation_dt_seconds: float,
    rollout_horizon_seconds: float,
    edge_threshold: float,
) -> RealTimeMispricingDetector:
    config = MispricingDetectorConfig(
        n_samples=n_samples,
        simulation_dt_seconds=simulation_dt_seconds,
        rollout_horizon_seconds=rollout_horizon_seconds,
        edge_threshold=edge_threshold,
        spot_drift_decay_kappa_per_second=DEFAULT_SPOT_DRIFT_DECAY_KAPPA_PER_SECOND,
        spot_jump_intensity_per_second=DEFAULT_MANUAL_SPOT_JUMP_INTENSITY_PER_SECOND,
        spot_jump_log_return_mean=DEFAULT_MANUAL_SPOT_JUMP_LOG_RETURN_MEAN,
        spot_jump_log_return_std=DEFAULT_MANUAL_SPOT_JUMP_LOG_RETURN_STD,
        spot_jump_std_multiplier_on_local_sigma=DEFAULT_MANUAL_SPOT_JUMP_STD_MULTIPLIER_ON_LOCAL_SIGMA,
        transition_bundle=bundle,
    )
    return RealTimeMispricingDetector(config)


def _load_transition_bundle(path: str) -> TransitionModelBundle | None:
    bundle_path = Path(path)
    if not bundle_path.exists():
        return None
    return joblib.load(bundle_path)
