import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Sequence

import joblib
import pandas as pd
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

from polymarket_quant.pricing import DEFAULT_SPOT_DRIFT_DECAY_KAPPA_PER_SECOND
from polymarket_quant.signals.mispricing import MispricingDetectorConfig, RealTimeMispricingDetector
from polymarket_quant.state.dataset import load_parquet_glob
from polymarket_quant.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_EVENT_STATE_GLOB = "data/processed/crypto_5m_event_state_latest.parquet"
DEFAULT_TRANSITION_MODEL_PATH = "artifacts/transition_model/transition_model_latest.joblib"
DEFAULT_OUTPUT_DIR = "artifacts/jump_sensitivity"
DEFAULT_JUMP_INTENSITIES_PER_DAY = (5.0, 20.0, 50.0)
DEFAULT_JUMP_STD_MULTIPLIERS = (10.0, 20.0, 30.0)


def study_jump_sensitivity(
    *,
    event_state_glob: str = DEFAULT_EVENT_STATE_GLOB,
    transition_model_path: str = DEFAULT_TRANSITION_MODEL_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    jump_intensities_per_day: Sequence[float] = DEFAULT_JUMP_INTENSITIES_PER_DAY,
    jump_std_multipliers: Sequence[float] = DEFAULT_JUMP_STD_MULTIPLIERS,
    include_zero_jump_baseline: bool = True,
    jump_mean: float = 0.0,
    n_samples: int = 100,
    simulation_dt_seconds: float = 1.0,
    max_rows: int | None = None,
    drift_decay_kappa_per_second: float = DEFAULT_SPOT_DRIFT_DECAY_KAPPA_PER_SECOND,
    include_latest: bool = False,
    show_progress: bool = True,
) -> dict[str, object]:
    event_state = load_parquet_glob(event_state_glob, include_latest=include_latest)
    if max_rows is not None:
        event_state = event_state.head(max(int(max_rows), 0)).copy()
    rows = event_state.to_dict("records")
    if not rows:
        raise ValueError("No event-state rows available for jump sensitivity study")

    bundle = joblib.load(transition_model_path)
    combos = _build_jump_grid(
        jump_intensities_per_day=jump_intensities_per_day,
        jump_std_multipliers=jump_std_multipliers,
        include_zero_jump_baseline=include_zero_jump_baseline,
    )

    summaries: list[dict[str, Any]] = []
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    combo_iterator = _iter_jump_grid_with_progress(
        combos,
        show_progress=show_progress,
    )
    for scenario_index, (jump_intensity_per_day, jump_std_multiplier) in enumerate(combo_iterator, start=1):
        if tqdm is None or not show_progress:
            logger.info(
                "Running jump scenario %s/%s: intensity=%s/day, jump_std=%sx local sigma",
                scenario_index,
                len(combos),
                float(jump_intensity_per_day),
                float(jump_std_multiplier),
            )
        intensity_per_second = float(max(jump_intensity_per_day, 0.0) / 86400.0)
        detector = RealTimeMispricingDetector(
            config=MispricingDetectorConfig(
                n_samples=int(n_samples),
                simulation_dt_seconds=float(simulation_dt_seconds),
                spot_drift_decay_kappa_per_second=float(drift_decay_kappa_per_second),
                force_manual_jump_parameters=True,
                spot_jump_intensity_per_second=float(intensity_per_second),
                spot_jump_log_return_mean=float(jump_mean),
                spot_jump_log_return_std=0.0,
                spot_jump_std_multiplier_on_local_sigma=float(max(jump_std_multiplier, 0.0)),
                transition_bundle=bundle,
            )
        )
        valuation = pd.DataFrame(detector.detect(rows, show_progress=False))
        summary = _summarize_valuation_frame(
            valuation,
            jump_intensity_per_day=float(jump_intensity_per_day),
            jump_intensity_per_second=float(intensity_per_second),
            jump_std_multiplier_on_local_sigma=float(jump_std_multiplier),
            jump_mean=float(jump_mean),
            n_samples=int(n_samples),
            drift_decay_kappa_per_second=float(drift_decay_kappa_per_second),
        )
        summaries.append(summary)

    summary_frame = (
        pd.DataFrame(summaries)
        .sort_values(
            [
                "jump_intensity_per_day",
                "jump_std_multiplier_on_local_sigma",
            ]
        )
        .reset_index(drop=True)
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    summary_path = output_path / f"crypto_5m_jump_sensitivity_{timestamp}.parquet"
    latest_path = output_path / "crypto_5m_jump_sensitivity_latest.parquet"
    metadata_path = output_path / f"crypto_5m_jump_sensitivity_{timestamp}.json"
    latest_metadata_path = output_path / "crypto_5m_jump_sensitivity_latest.json"

    summary_frame.to_parquet(summary_path, index=False)
    summary_frame.to_parquet(latest_path, index=False)
    metadata = {
        "event_state_glob": event_state_glob,
        "transition_model_path": transition_model_path,
        "output_path": str(summary_path),
        "latest_path": str(latest_path),
        "jump_intensities_per_day": [float(value) for value in jump_intensities_per_day],
        "jump_std_multipliers": [float(value) for value in jump_std_multipliers],
        "include_zero_jump_baseline": bool(include_zero_jump_baseline),
        "jump_mean": float(jump_mean),
        "n_samples": int(n_samples),
        "simulation_dt_seconds": float(simulation_dt_seconds),
        "max_rows": None if max_rows is None else int(max_rows),
        "drift_decay_kappa_per_second": float(drift_decay_kappa_per_second),
        "include_latest": bool(include_latest),
        "show_progress": bool(show_progress),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    latest_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    logger.info("Saved %s jump-sensitivity row(s) to %s", len(summary_frame), summary_path)
    logger.info("\n%s", _render_summary_table(summary_frame))

    return {
        "rows": int(len(summary_frame)),
        "output_path": str(summary_path),
        "latest_path": str(latest_path),
        "metadata_path": str(metadata_path),
        "latest_metadata_path": str(latest_metadata_path),
    }


def _build_jump_grid(
    *,
    jump_intensities_per_day: Sequence[float],
    jump_std_multipliers: Sequence[float],
    include_zero_jump_baseline: bool,
) -> list[tuple[float, float]]:
    combos: list[tuple[float, float]] = []
    if include_zero_jump_baseline:
        combos.append((0.0, 0.0))
    for intensity in jump_intensities_per_day:
        for multiplier in jump_std_multipliers:
            combos.append((float(intensity), float(multiplier)))
    seen: set[tuple[float, float]] = set()
    deduped: list[tuple[float, float]] = []
    for combo in combos:
        if combo in seen:
            continue
        seen.add(combo)
        deduped.append(combo)
    return deduped


def _iter_jump_grid_with_progress(
    combos: Sequence[tuple[float, float]],
    *,
    show_progress: bool,
):
    if not show_progress or tqdm is None:
        return iter(combos)
    return tqdm(
        combos,
        desc="Jump scenarios",
        total=len(combos),
        unit="scenario",
        dynamic_ncols=True,
    )


def _summarize_valuation_frame(
    valuation: pd.DataFrame,
    *,
    jump_intensity_per_day: float,
    jump_intensity_per_second: float,
    jump_std_multiplier_on_local_sigma: float,
    jump_mean: float,
    n_samples: int,
    drift_decay_kappa_per_second: float,
) -> dict[str, Any]:
    if valuation.empty:
        raise ValueError("Valuation frame is empty")
    if "outcome_name" not in valuation.columns:
        raise ValueError("Valuation frame must contain outcome_name")

    up = valuation[valuation["outcome_name"].astype(str) == "Up"].copy()
    if up.empty:
        raise ValueError("Valuation frame does not contain Up rows")

    fair = pd.to_numeric(up["fair_up_probability"], errors="coerce")
    market = pd.to_numeric(up["market_implied_up_probability"], errors="coerce")
    seconds_to_end = pd.to_numeric(up.get("seconds_to_end"), errors="coerce")
    aligned = pd.DataFrame(
        {
            "fair": fair,
            "market": market,
            "seconds_to_end": seconds_to_end,
        }
    ).dropna(subset=["fair", "market"])
    if aligned.empty:
        raise ValueError("Up valuation rows are missing fair or market probabilities")

    conditioned_jump_std = pd.to_numeric(
        up.get(
            "conditioned_spot_jump_log_return_std",
            pd.Series(index=up.index, dtype=float),
        ),
        errors="coerce",
    )
    conditioned_jump_intensity = pd.to_numeric(
        up.get(
            "conditioned_spot_jump_intensity_per_second",
            pd.Series(index=up.index, dtype=float),
        ),
        errors="coerce",
    )

    return {
        "jump_intensity_per_day": float(jump_intensity_per_day),
        "jump_intensity_per_second": float(jump_intensity_per_second),
        "jump_std_multiplier_on_local_sigma": float(jump_std_multiplier_on_local_sigma),
        "jump_mean": float(jump_mean),
        "n_samples": int(n_samples),
        "drift_decay_kappa_per_second": float(drift_decay_kappa_per_second),
        "drift_half_life_seconds": float(math.log(2.0) / drift_decay_kappa_per_second)
        if drift_decay_kappa_per_second > 0.0
        else float("inf"),
        "rows_total": int(len(valuation)),
        "rows_up": int(len(aligned)),
        "extreme_share_1pct": float(((aligned["fair"] <= 0.01) | (aligned["fair"] >= 0.99)).mean()),
        "extreme_share_5pct": float(((aligned["fair"] <= 0.05) | (aligned["fair"] >= 0.95)).mean()),
        "mean_abs_gap_vs_market": float((aligned["fair"] - aligned["market"]).abs().mean()),
        "median_abs_gap_vs_market": float((aligned["fair"] - aligned["market"]).abs().median()),
        "corr_with_market": float(aligned["fair"].corr(aligned["market"])),
        "fair_q25": float(aligned["fair"].quantile(0.25)),
        "fair_q50": float(aligned["fair"].quantile(0.50)),
        "fair_q75": float(aligned["fair"].quantile(0.75)),
        "mean_seconds_to_end": float(pd.to_numeric(aligned["seconds_to_end"], errors="coerce").dropna().mean()),
        "mean_conditioned_jump_std": float(conditioned_jump_std.dropna().mean()),
        "mean_conditioned_jump_intensity_per_second": float(conditioned_jump_intensity.dropna().mean()),
    }


def _render_summary_table(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "No jump-sensitivity rows were computed."
    view = summary[
        [
            "jump_intensity_per_day",
            "jump_std_multiplier_on_local_sigma",
            "extreme_share_1pct",
            "extreme_share_5pct",
            "mean_abs_gap_vs_market",
            "median_abs_gap_vs_market",
            "corr_with_market",
            "fair_q25",
            "fair_q50",
            "fair_q75",
        ]
    ].copy()
    return view.to_string(index=False, float_format=lambda value: f"{value:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run replay-pricing sensitivity tests across manual zero-mean jump priors."
    )
    parser.add_argument("--event-state-glob", default=DEFAULT_EVENT_STATE_GLOB, help="Event-state parquet path or glob")
    parser.add_argument(
        "--transition-model-path",
        default=DEFAULT_TRANSITION_MODEL_PATH,
        help="Transition model artifact used for pricing replay",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for sensitivity artifacts")
    parser.add_argument(
        "--jump-intensities-per-day",
        nargs="+",
        type=float,
        default=list(DEFAULT_JUMP_INTENSITIES_PER_DAY),
        help="Manual zero-mean jump intensities, in expected jumps per day, to evaluate",
    )
    parser.add_argument(
        "--jump-std-multipliers",
        nargs="+",
        type=float,
        default=list(DEFAULT_JUMP_STD_MULTIPLIERS),
        help="Multipliers applied to each row's local spot volatility to form manual jump std",
    )
    parser.add_argument(
        "--no-zero-jump-baseline",
        action="store_true",
        help="Skip the no-jump baseline row (0/day, 0x sigma).",
    )
    parser.add_argument("--jump-mean", type=float, default=0.0, help="Manual jump mean. Keep this at 0 for the current study.")
    parser.add_argument("--n-samples", type=int, default=100, help="Monte Carlo paths per jump scenario")
    parser.add_argument("--simulation-dt-seconds", type=float, default=1.0, help="Simulation step size in seconds")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap on event-state rows for quick cloud smoke tests")
    parser.add_argument(
        "--spot-drift-decay-kappa-per-second",
        type=float,
        default=float(DEFAULT_SPOT_DRIFT_DECAY_KAPPA_PER_SECOND),
        help="Drift-decay kappa held fixed while scanning jump priors. Default is the 5-second baseline.",
    )
    parser.add_argument("--include-latest", action="store_true", help="Include *_latest.parquet inputs")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm scenario progress display.")
    args = parser.parse_args()

    result = study_jump_sensitivity(
        event_state_glob=args.event_state_glob,
        transition_model_path=args.transition_model_path,
        output_dir=args.output_dir,
        jump_intensities_per_day=args.jump_intensities_per_day,
        jump_std_multipliers=args.jump_std_multipliers,
        include_zero_jump_baseline=not args.no_zero_jump_baseline,
        jump_mean=args.jump_mean,
        n_samples=args.n_samples,
        simulation_dt_seconds=args.simulation_dt_seconds,
        max_rows=args.max_rows,
        drift_decay_kappa_per_second=args.spot_drift_decay_kappa_per_second,
        include_latest=args.include_latest,
        show_progress=not args.no_progress,
    )
    logger.info("Jump sensitivity complete: %s", result)


if __name__ == "__main__":
    main()
