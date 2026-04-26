import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib

from polymarket_quant.state.dataset import load_parquet_glob
from polymarket_quant.state.transition_model import TransitionModelConfig, fit_transition_model
from polymarket_quant.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_TRANSITION_TARGET_GLOB = "data/processed/crypto_5m_transition_targets_latest.parquet"


def fit_transition_model_artifacts(
    transition_target_glob: str = DEFAULT_TRANSITION_TARGET_GLOB,
    output_dir: str = "artifacts/transition_model",
    prediction_dir: str = "data/processed",
    include_latest: bool = False,
    run_timestamp: str | None = None,
    min_training_rows: int = 32,
    random_state: int = 42,
) -> dict[str, object]:
    transition_targets = load_parquet_glob(transition_target_glob, include_latest=include_latest)
    config = TransitionModelConfig(
        min_training_rows=min_training_rows,
        random_state=random_state,
    )
    fit_result = fit_transition_model(transition_targets, config=config)

    run_timestamp = run_timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    artifact_path = Path(output_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)
    prediction_path = Path(prediction_dir)
    prediction_path.mkdir(parents=True, exist_ok=True)

    model_path = artifact_path / f"transition_model_{run_timestamp}.joblib"
    summary_path = artifact_path / f"transition_model_summary_{run_timestamp}.json"
    latest_model_path = artifact_path / "transition_model_latest.joblib"
    latest_summary_path = artifact_path / "transition_model_summary_latest.json"

    predictions_path = prediction_path / f"crypto_5m_transition_predictions_{run_timestamp}.parquet"
    latest_predictions_path = prediction_path / "crypto_5m_transition_predictions_latest.parquet"

    joblib.dump(fit_result.bundle, model_path)
    joblib.dump(fit_result.bundle, latest_model_path)
    summary_path.write_text(json.dumps(fit_result.summary, indent=2, sort_keys=True))
    latest_summary_path.write_text(json.dumps(fit_result.summary, indent=2, sort_keys=True))
    fit_result.predictions.to_parquet(predictions_path, index=False)
    fit_result.predictions.to_parquet(latest_predictions_path, index=False)

    logger.info(
        "Saved transition model to %s and predictions to %s",
        model_path,
        predictions_path,
    )
    return {
        **fit_result.summary,
        "model_path": str(model_path),
        "latest_model_path": str(latest_model_path),
        "summary_path": str(summary_path),
        "latest_summary_path": str(latest_summary_path),
        "predictions_path": str(predictions_path),
        "latest_predictions_path": str(latest_predictions_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the first structured full-state Markov transition model.")
    parser.add_argument(
        "--transition-target-glob",
        default=DEFAULT_TRANSITION_TARGET_GLOB,
        help="Transition-target parquet path or glob",
    )
    parser.add_argument("--output-dir", default="artifacts/transition_model", help="Directory for model artifacts")
    parser.add_argument("--prediction-dir", default="data/processed", help="Directory for prediction parquet outputs")
    parser.add_argument("--include-latest", action="store_true", help="Include *_latest.parquet inputs")
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
    )


if __name__ == "__main__":
    main()
