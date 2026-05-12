import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from polymarket_quant.state.dataset import load_parquet_glob, matching_parquet_paths
from polymarket_quant.state.transition_model import TransitionModelConfig, fit_transition_model
from polymarket_quant.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_TRANSITION_TARGET_GLOB = "data/*/processed/transition_targets/shards/*.parquet"


def fit_transition_model_artifacts(
    transition_target_glob: str = DEFAULT_TRANSITION_TARGET_GLOB,
    output_dir: str = "artifacts/transition_model",
    prediction_dir: str = "data/processed",
    include_latest: bool = False,
    run_timestamp: str | None = None,
    min_training_rows: int = 32,
    random_state: int = 42,
    split_by_asset: bool = True,
    write_predictions: bool = False,
) -> dict[str, object]:
    run_timestamp = run_timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    artifact_path = Path(output_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)
    prediction_path = Path(prediction_dir)
    if write_predictions:
        prediction_path.mkdir(parents=True, exist_ok=True)

    if split_by_asset:
        transition_targets_by_asset = _load_transition_target_shards_by_asset(
            transition_target_glob=transition_target_glob,
            include_latest=include_latest,
        )
        return _fit_transition_model_artifacts_by_asset(
            transition_targets_by_asset=transition_targets_by_asset,
            output_dir=artifact_path,
            prediction_dir=prediction_path,
            run_timestamp=run_timestamp,
            min_training_rows=min_training_rows,
            random_state=random_state,
            write_predictions=write_predictions,
        )

    transition_targets = load_parquet_glob(transition_target_glob, include_latest=include_latest)
    config = TransitionModelConfig(
        min_training_rows=min_training_rows,
        random_state=random_state,
    )
    fit_result = fit_transition_model(transition_targets, config=config)

    model_path = artifact_path / f"transition_model_{run_timestamp}.joblib"
    summary_path = artifact_path / f"transition_model_summary_{run_timestamp}.json"
    latest_model_path = artifact_path / "transition_model_latest.joblib"
    latest_summary_path = artifact_path / "transition_model_summary_latest.json"

    joblib.dump(fit_result.bundle, model_path)
    joblib.dump(fit_result.bundle, latest_model_path)
    summary_path.write_text(json.dumps(fit_result.summary, indent=2, sort_keys=True))
    latest_summary_path.write_text(json.dumps(fit_result.summary, indent=2, sort_keys=True))

    result: dict[str, object] = {
        **fit_result.summary,
        "model_path": str(model_path),
        "latest_model_path": str(latest_model_path),
        "summary_path": str(summary_path),
        "latest_summary_path": str(latest_summary_path),
        "write_predictions": write_predictions,
    }

    if write_predictions:
        predictions_path = prediction_path / f"crypto_5m_transition_predictions_{run_timestamp}.parquet"
        latest_predictions_path = prediction_path / "crypto_5m_transition_predictions_latest.parquet"
        fit_result.predictions.to_parquet(predictions_path, index=False)
        fit_result.predictions.to_parquet(latest_predictions_path, index=False)
        logger.info(
            "Saved transition model to %s and predictions to %s",
            model_path,
            predictions_path,
        )
        result["predictions_path"] = str(predictions_path)
        result["latest_predictions_path"] = str(latest_predictions_path)
    else:
        logger.info("Saved transition model to %s without row-level prediction parquet output", model_path)

    return result


def _fit_transition_model_artifacts_by_asset(
    *,
    transition_targets_by_asset: dict[str, pd.DataFrame],
    output_dir: Path,
    prediction_dir: Path,
    run_timestamp: str,
    min_training_rows: int,
    random_state: int,
    write_predictions: bool,
) -> dict[str, object]:
    results_by_asset: dict[str, dict[str, object]] = {}

    for asset in sorted(transition_targets_by_asset):
        asset_rows = transition_targets_by_asset[asset]
        if asset_rows.empty:
            continue
        config = TransitionModelConfig(
            min_training_rows=min_training_rows,
            random_state=random_state,
        )
        fit_result = fit_transition_model(asset_rows, config=config)
        asset_key = asset.lower()

        model_path = output_dir / f"transition_model_{asset_key}_{run_timestamp}.joblib"
        summary_path = output_dir / f"transition_model_summary_{asset_key}_{run_timestamp}.json"
        latest_model_path = output_dir / f"transition_model_{asset_key}_latest.joblib"
        latest_summary_path = output_dir / f"transition_model_summary_{asset_key}_latest.json"

        joblib.dump(fit_result.bundle, model_path)
        joblib.dump(fit_result.bundle, latest_model_path)
        summary_path.write_text(json.dumps(fit_result.summary, indent=2, sort_keys=True))
        latest_summary_path.write_text(json.dumps(fit_result.summary, indent=2, sort_keys=True))

        asset_result: dict[str, object] = {
            **fit_result.summary,
            "model_path": str(model_path),
            "latest_model_path": str(latest_model_path),
            "summary_path": str(summary_path),
            "latest_summary_path": str(latest_summary_path),
            "training_rows": int(fit_result.summary.get("training_rows", len(asset_rows))),
            "write_predictions": write_predictions,
        }

        if write_predictions:
            predictions_path = prediction_dir / f"crypto_5m_transition_predictions_{asset_key}_{run_timestamp}.parquet"
            latest_predictions_path = prediction_dir / f"crypto_5m_transition_predictions_{asset_key}_latest.parquet"
            fit_result.predictions.to_parquet(predictions_path, index=False)
            fit_result.predictions.to_parquet(latest_predictions_path, index=False)
            asset_result["predictions_path"] = str(predictions_path)
            asset_result["latest_predictions_path"] = str(latest_predictions_path)

        results_by_asset[asset] = asset_result

    if not results_by_asset:
        raise ValueError("No per-asset transition models were trained")

    logger.info(
        "Saved per-asset transition models for %s%s",
        ", ".join(sorted(results_by_asset)),
        "" if write_predictions else " without row-level prediction parquet output",
    )
    return {
        "split_by_asset": True,
        "assets": results_by_asset,
        "asset_count": len(results_by_asset),
        "write_predictions": write_predictions,
    }


def _load_transition_target_shards_by_asset(
    *,
    transition_target_glob: str,
    include_latest: bool,
) -> dict[str, pd.DataFrame]:
    shard_paths = matching_parquet_paths(transition_target_glob, include_latest=include_latest)
    if not shard_paths:
        raise FileNotFoundError(f"No parquet files matched {transition_target_glob}")

    paths_by_asset: dict[str, list[Path]] = {}
    for shard_path in shard_paths:
        asset = _asset_from_shard_path(shard_path)
        paths_by_asset.setdefault(asset, []).append(shard_path)

    transition_targets_by_asset: dict[str, pd.DataFrame] = {}
    for asset, asset_paths in sorted(paths_by_asset.items()):
        logger.info("Loading %s transition-target shard(s) for %s", len(asset_paths), asset)
        transition_targets_by_asset[asset] = pd.concat(
            [pd.read_parquet(path) for path in asset_paths],
            ignore_index=True,
            sort=False,
        )
    return transition_targets_by_asset


def _asset_from_shard_path(path: Path) -> str:
    parts = path.parts
    try:
        processed_idx = parts.index("processed")
    except ValueError as exc:
        raise ValueError(f"Transition-target shard path does not include processed segment: {path}") from exc
    if processed_idx == 0:
        raise ValueError(f"Cannot infer asset from transition-target shard path: {path}")
    asset = str(parts[processed_idx - 1]).strip().upper()
    if not asset:
        raise ValueError(f"Cannot infer asset from transition-target shard path: {path}")
    return asset


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the first structured full-state Markov transition model.")
    parser.add_argument(
        "--transition-target-glob",
        default=DEFAULT_TRANSITION_TARGET_GLOB,
        help="Transition-target parquet path or glob",
    )
    parser.add_argument("--output-dir", default="artifacts/transition_model", help="Directory for model artifacts")
    parser.add_argument("--prediction-dir", default="data/processed", help="Directory for optional prediction parquet outputs")
    parser.add_argument("--include-latest", action="store_true", help="Include *_latest.parquet inputs")
    parser.add_argument("--combined-model", action="store_true", help="Fit one shared model instead of separate BTC/ETH models")
    parser.add_argument("--write-predictions", action="store_true", help="Also write row-level prediction parquet outputs")
    parser.add_argument("--min-training-rows", type=int, default=32, help="Minimum matched rows required to fit")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for sklearn models")
    args = parser.parse_args()

    fit_transition_model_artifacts(
        transition_target_glob=args.transition_target_glob,
        output_dir=args.output_dir,
        prediction_dir=args.prediction_dir,
        include_latest=args.include_latest,
        min_training_rows=args.min_training_rows,
        random_state=args.random_state,
        split_by_asset=not args.combined_model,
        write_predictions=args.write_predictions,
    )


if __name__ == "__main__":
    main()
