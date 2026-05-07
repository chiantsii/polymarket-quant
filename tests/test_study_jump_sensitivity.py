import math

import pandas as pd

from scripts.study_jump_sensitivity import _build_jump_grid, _summarize_valuation_frame


def test_build_jump_grid_includes_zero_baseline_once() -> None:
    combos = _build_jump_grid(
        jump_intensities_per_day=[0.0, 5.0],
        jump_std_multipliers=[0.0, 10.0],
        include_zero_jump_baseline=True,
    )

    assert combos[0] == (0.0, 0.0)
    assert combos.count((0.0, 0.0)) == 1
    assert (5.0, 10.0) in combos


def test_summarize_valuation_frame_computes_jump_metrics() -> None:
    valuation = pd.DataFrame(
        [
            {
                "outcome_name": "Up",
                "fair_up_probability": 0.95,
                "market_implied_up_probability": 0.80,
                "seconds_to_end": 20.0,
                "conditioned_spot_jump_log_return_std": 0.01,
                "conditioned_spot_jump_intensity_per_second": 20.0 / 86400.0,
            },
            {
                "outcome_name": "Up",
                "fair_up_probability": 0.10,
                "market_implied_up_probability": 0.20,
                "seconds_to_end": 30.0,
                "conditioned_spot_jump_log_return_std": 0.02,
                "conditioned_spot_jump_intensity_per_second": 20.0 / 86400.0,
            },
            {
                "outcome_name": "Down",
                "fair_up_probability": 0.90,
                "market_implied_up_probability": 0.70,
                "seconds_to_end": 20.0,
                "conditioned_spot_jump_log_return_std": 0.02,
                "conditioned_spot_jump_intensity_per_second": 20.0 / 86400.0,
            },
        ]
    )

    summary = _summarize_valuation_frame(
        valuation,
        jump_intensity_per_day=20.0,
        jump_intensity_per_second=20.0 / 86400.0,
        jump_std_multiplier_on_local_sigma=20.0,
        jump_mean=0.0,
        n_samples=100,
        drift_decay_kappa_per_second=math.log(2.0) / 5.0,
    )

    assert summary["rows_total"] == 3
    assert summary["rows_up"] == 2
    assert math.isclose(summary["extreme_share_1pct"], 0.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(summary["extreme_share_5pct"], 0.5, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(summary["mean_abs_gap_vs_market"], 0.125, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(summary["median_abs_gap_vs_market"], 0.125, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(summary["mean_conditioned_jump_std"], 0.015, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(summary["mean_conditioned_jump_intensity_per_second"], 20.0 / 86400.0, rel_tol=0.0, abs_tol=1e-12)
