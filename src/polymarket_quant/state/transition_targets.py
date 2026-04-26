"""Structured transition-target builders for full-state Markov training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from polymarket_quant.utils.logger import get_logger

logger = get_logger(__name__)


STATE_METADATA_COLUMNS: tuple[str, ...] = (
    "series_slug",
    "asset",
    "event_id",
    "event_title",
    "market_id",
    "condition_id",
)


DEFAULT_STATE_COLUMNS: tuple[str, ...] = (
    # M_t: quote / liquidity / observation block
    "spot_price",
    "spot_bid",
    "spot_ask",
    "spot_return_since_reference",
    "spot_vol_multiplier",
    "external_spot_drift",
    "up_best_bid",
    "up_best_ask",
    "down_best_bid",
    "down_best_ask",
    "up_mid_price",
    "down_mid_price",
    "up_micro_price",
    "down_micro_price",
    "up_spread",
    "down_spread",
    "up_bid_depth_top_5",
    "up_ask_depth_top_5",
    "down_bid_depth_top_5",
    "down_ask_depth_top_5",
    "up_orderbook_imbalance",
    "down_orderbook_imbalance",
    "up_weighted_imbalance",
    "down_weighted_imbalance",
    "up_depth_slope",
    "down_depth_slope",
    "up_tick_density",
    "down_tick_density",
    "up_book_velocity",
    "down_book_velocity",
    "cross_book_basis",
    "cross_book_bid_basis",
    "cross_book_ask_basis",
    "spread_divergence",
    "dist_to_boundary",
    "boundary_leverage_ratio",
    "asymmetric_depth_ratio",
    "book_age_max",
    "has_full_book_pair",
    # L_t: latent / mechanism block
    "market_implied_up_probability",
    "fundamental_up_probability",
    "latent_up_probability",
    "latent_logit_probability",
    "market_fundamental_basis",
    "latent_market_basis",
    "latent_fundamental_basis",
    "abs_market_fundamental_basis",
    "abs_latent_market_basis",
    "abs_latent_fundamental_basis",
    "volatility_per_sqrt_second",
    "state_observation_variance",
    "regime_normal_posterior",
    "regime_shock_posterior",
    "regime_convergence_posterior",
    # T_t
    "normalized_time_to_end",
)


DEFAULT_PRIMITIVE_TARGET_COLUMNS: tuple[str, ...] = (
    "latent_logit_probability",
    "regime_normal_posterior",
    "regime_shock_posterior",
    "regime_convergence_posterior",
    "market_implied_up_probability",
    "up_micro_price",
    "down_micro_price",
    "up_weighted_imbalance",
    "down_weighted_imbalance",
    "up_bid_depth_top_5",
    "up_ask_depth_top_5",
    "down_bid_depth_top_5",
    "down_ask_depth_top_5",
    "cross_book_basis",
)


@dataclass(frozen=True)
class TransitionTargetConfig:
    """Configuration for pairing current states with future transition targets."""

    pairing_mode: str = "next"
    horizons_seconds: tuple[float, ...] = (15.0,)
    tolerance_seconds: float = 2.0
    include_unmatched: bool = False

    def validated_horizons(self) -> tuple[float, ...]:
        if self.validated_pairing_mode() != "horizon":
            return ()
        horizons = tuple(float(horizon) for horizon in self.horizons_seconds)
        if not horizons:
            raise ValueError("At least one horizon is required")
        if any(horizon <= 0.0 for horizon in horizons):
            raise ValueError("All horizons must be strictly positive")
        return tuple(dict.fromkeys(horizons))

    def validated_pairing_mode(self) -> str:
        pairing_mode = str(self.pairing_mode).strip().lower()
        if pairing_mode not in {"next", "horizon"}:
            raise ValueError("pairing_mode must be either 'next' or 'horizon'")
        return pairing_mode


def build_transition_target_dataset(
    event_state: pd.DataFrame,
    config: TransitionTargetConfig | None = None,
    state_columns: Sequence[str] | None = None,
    primitive_target_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Build a structured full-state transition table from event-level state rows.

    The returned table contains:
    - metadata describing the current/future state pairing
    - prefixed `current_*` state columns for the current state S_t
    - prefixed `future_*` state columns for the matched future state S_{t+Δ}
    - `target_delta_*` columns for stochastic primitive targets
    """

    config = config or TransitionTargetConfig()
    pairing_mode = config.validated_pairing_mode()
    horizons = config.validated_horizons()

    required = {"event_slug", "collected_at"}
    missing = required - set(event_state.columns)
    if missing:
        raise ValueError(f"Event-state data is missing columns: {sorted(missing)}")

    prepared = event_state.copy()
    prepared["_state_dt"] = pd.to_datetime(prepared["collected_at"], utc=True, errors="coerce")
    prepared = prepared.dropna(subset=["event_slug", "_state_dt"]).sort_values(["event_slug", "_state_dt"]).reset_index(drop=True)
    if prepared.empty:
        raise ValueError("Event-state data does not contain any valid timestamped rows")

    available_state_columns = [column for column in (state_columns or DEFAULT_STATE_COLUMNS) if column in prepared.columns]
    if not available_state_columns:
        raise ValueError("No configured state columns were found in event_state")

    available_metadata_columns = [column for column in STATE_METADATA_COLUMNS if column in prepared.columns]
    available_primitive_targets = [
        column for column in (primitive_target_columns or DEFAULT_PRIMITIVE_TARGET_COLUMNS) if column in available_state_columns
    ]
    if not available_primitive_targets:
        raise ValueError("No configured primitive target columns were found in event_state")

    current_base = prepared[
        ["event_slug", "collected_at", "_state_dt", *available_metadata_columns]
        + [column for column in ("state_timestamp", "seconds_to_end") if column in prepared.columns]
        + available_state_columns
    ].copy()
    current_base = current_base.rename(
        columns={
            "collected_at": "current_collected_at",
            "_state_dt": "_current_state_dt",
            "state_timestamp": "current_state_timestamp",
            "seconds_to_end": "current_seconds_to_end",
            **{column: f"current_{column}" for column in available_state_columns},
        }
    )

    future_base = prepared[
        ["event_slug", "collected_at", "_state_dt"]
        + [column for column in ("state_timestamp", "seconds_to_end") if column in prepared.columns]
        + available_state_columns
    ].copy()
    future_base = future_base.rename(
        columns={
            "collected_at": "future_collected_at",
            "_state_dt": "_future_state_dt",
            "state_timestamp": "future_state_timestamp",
            "seconds_to_end": "future_seconds_to_end",
            **{column: f"future_{column}" for column in available_state_columns},
        }
    )

    frames: list[pd.DataFrame] = []
    if pairing_mode == "next":
        paired = _pair_current_and_next_state(
            current_base=current_base,
            future_base=future_base,
        )
        paired = _add_transition_deltas(paired, primitive_target_columns=available_primitive_targets)
        if not config.include_unmatched:
            paired = paired[paired["has_future_target"]].copy()
        frames.append(paired)
    else:
        for horizon_seconds in horizons:
            paired = _pair_current_and_future_states(
                current_base=current_base,
                future_base=future_base,
                horizon_seconds=horizon_seconds,
                tolerance_seconds=config.tolerance_seconds,
            )
            paired = _add_transition_deltas(paired, primitive_target_columns=available_primitive_targets)
            if not config.include_unmatched:
                paired = paired[paired["has_future_target"]].copy()
            frames.append(paired)

    transition_targets = pd.concat(frames, ignore_index=True)
    transition_targets = transition_targets.sort_values(
        ["event_slug", "current_collected_at", "target_horizon_seconds"]
    ).reset_index(drop=True)
    transition_targets = transition_targets.drop(columns=["_current_state_dt", "_target_state_dt", "_future_state_dt"])
    return transition_targets


def build_transition_target_summary(transition_targets: pd.DataFrame) -> dict[str, Any]:
    """Return a compact summary for logging and CLI output."""

    if transition_targets.empty:
        return {
            "rows": 0,
            "matched_rows": 0,
            "horizons_seconds": [],
            "status_counts": {},
        }

    status_counts = (
        transition_targets["target_status"].value_counts(dropna=False).sort_index().to_dict()
        if "target_status" in transition_targets.columns
        else {}
    )
    matched_rows = int(pd.to_numeric(transition_targets.get("has_future_target"), errors="coerce").fillna(False).sum())
    return {
        "rows": int(len(transition_targets)),
        "matched_rows": matched_rows,
        "horizons_seconds": sorted(
            float(value) for value in pd.to_numeric(transition_targets["target_horizon_seconds"], errors="coerce").dropna().unique()
        ),
        "status_counts": status_counts,
    }


def _pair_current_and_future_states(
    *,
    current_base: pd.DataFrame,
    future_base: pd.DataFrame,
    horizon_seconds: float,
    tolerance_seconds: float,
) -> pd.DataFrame:
    current = current_base.copy()
    current["target_horizon_seconds"] = float(horizon_seconds)
    current["_target_state_dt"] = current["_current_state_dt"] + pd.to_timedelta(horizon_seconds, unit="s")

    paired = pd.merge_asof(
        current.sort_values(["_target_state_dt", "event_slug"]),
        future_base.sort_values(["_future_state_dt", "event_slug"]),
        by="event_slug",
        left_on="_target_state_dt",
        right_on="_future_state_dt",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=tolerance_seconds),
    ).sort_values(["event_slug", "_current_state_dt"]).reset_index(drop=True)

    has_future = paired["future_collected_at"].notna() & (paired["_future_state_dt"] > paired["_current_state_dt"])
    paired["has_future_target"] = has_future

    realized_horizon_seconds = (
        paired["_future_state_dt"] - paired["_current_state_dt"]
    ).dt.total_seconds()
    paired["realized_horizon_seconds"] = realized_horizon_seconds.where(has_future, np.nan)
    paired["horizon_error_seconds"] = paired["realized_horizon_seconds"] - paired["target_horizon_seconds"]
    paired["target_status"] = _target_status(paired, horizon_seconds)
    return paired


def _pair_current_and_next_state(
    *,
    current_base: pd.DataFrame,
    future_base: pd.DataFrame,
) -> pd.DataFrame:
    key_columns = ["event_slug", "_current_state_dt"]
    current = current_base.sort_values(key_columns).reset_index(drop=True).copy()
    future = future_base.sort_values(["event_slug", "_future_state_dt"]).reset_index(drop=True).copy()

    paired_future = future.groupby("event_slug", sort=False).shift(-1)
    paired = current.copy()

    for column in future.columns:
        if column == "event_slug":
            continue
        paired[column] = paired_future[column]

    has_future = paired["future_collected_at"].notna() & (paired["_future_state_dt"] > paired["_current_state_dt"])
    paired["has_future_target"] = has_future
    realized_horizon_seconds = (paired["_future_state_dt"] - paired["_current_state_dt"]).dt.total_seconds()
    paired["realized_horizon_seconds"] = realized_horizon_seconds.where(has_future, np.nan)
    paired["target_horizon_seconds"] = paired["realized_horizon_seconds"]
    paired["_target_state_dt"] = paired["_current_state_dt"] + pd.to_timedelta(
        paired["target_horizon_seconds"].fillna(0.0),
        unit="s",
    )
    paired["horizon_error_seconds"] = 0.0
    paired.loc[~has_future, "horizon_error_seconds"] = np.nan
    paired["target_status"] = _next_target_status(paired)
    return paired


def _target_status(rows: pd.DataFrame, horizon_seconds: float) -> pd.Series:
    status = pd.Series("missing_future_snapshot", index=rows.index, dtype=object)

    matched = rows["has_future_target"].fillna(False)
    status.loc[matched] = "matched"

    nonfuture_match = rows["future_collected_at"].notna() & ~matched
    status.loc[nonfuture_match] = "nonfuture_match"

    current_seconds_to_end = pd.to_numeric(rows.get("current_seconds_to_end"), errors="coerce")
    expired = current_seconds_to_end.notna() & (current_seconds_to_end < horizon_seconds)
    status.loc[expired & ~matched] = "expired_before_horizon"
    return status


def _next_target_status(rows: pd.DataFrame) -> pd.Series:
    status = pd.Series("missing_future_snapshot", index=rows.index, dtype=object)
    matched = rows["has_future_target"].fillna(False)
    status.loc[matched] = "matched"

    current_seconds_to_end = pd.to_numeric(rows.get("current_seconds_to_end"), errors="coerce")
    expired = current_seconds_to_end.notna() & (current_seconds_to_end <= 0.0)
    status.loc[expired & ~matched] = "expired_before_next_observation"
    return status


def _add_transition_deltas(rows: pd.DataFrame, primitive_target_columns: Sequence[str]) -> pd.DataFrame:
    prepared = rows.copy()
    for column in primitive_target_columns:
        current_column = f"current_{column}"
        future_column = f"future_{column}"
        if current_column not in prepared.columns or future_column not in prepared.columns:
            continue

        current_numeric = pd.to_numeric(prepared[current_column], errors="coerce")
        future_numeric = pd.to_numeric(prepared[future_column], errors="coerce")
        if current_numeric.notna().sum() == 0 and future_numeric.notna().sum() == 0:
            continue

        prepared[f"target_delta_{column}"] = future_numeric - current_numeric
    return prepared
