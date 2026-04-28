"""Dataset builders for observation and latent Markov market state."""

from __future__ import annotations

import glob
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from polymarket_quant.state.latent_markov import LatentMarkovStateBuilder
from polymarket_quant.utils.logger import get_logger

logger = get_logger(__name__)


def filter_complete_event_windows(
    orderbooks: pd.DataFrame,
    spot: pd.DataFrame,
    orderbook_levels: pd.DataFrame | None = None,
    *,
    event_duration_seconds: float = 300.0,
    coverage_tolerance_seconds: float = 2.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Keep only interior event_slug windows from continuous capture runs."""
    prepared_orderbooks = prepare_orderbooks(orderbooks)
    prepared_spot = prepare_spot(spot)
    prepared_levels = None
    if orderbook_levels is not None and not orderbook_levels.empty:
        prepared_levels = prepare_orderbook_levels(orderbook_levels)

    del coverage_tolerance_seconds  # slug-first filtering no longer uses boundary timing thresholds

    complete_event_slugs: list[str] = []
    dropped_events: list[dict[str, Any]] = []
    available_spot_slugs = set(prepared_spot.get("event_slug", pd.Series(dtype=object)).dropna().astype(str))

    orderbook_events = (
        prepared_orderbooks[["asset", "event_slug"]]
        .dropna()
        .drop_duplicates()
        .assign(
            event_start_epoch=lambda frame: frame["event_slug"]
            .astype(str)
            .str.rsplit("-", n=1)
            .str[-1]
        )
    )
    orderbook_events["event_start_epoch"] = pd.to_numeric(orderbook_events["event_start_epoch"], errors="coerce")
    orderbook_events = orderbook_events.dropna(subset=["event_start_epoch"]).copy()
    orderbook_events["event_start_epoch"] = orderbook_events["event_start_epoch"].astype(int)

    for asset, asset_events in orderbook_events.groupby("asset", sort=False):
        asset_events = asset_events.sort_values("event_start_epoch").reset_index(drop=True)
        if asset_events.empty:
            continue

        run_breaks = asset_events["event_start_epoch"].diff().ne(int(event_duration_seconds)).fillna(True)
        asset_events["_run_id"] = run_breaks.cumsum()

        for _, run in asset_events.groupby("_run_id", sort=False):
            run_slugs = run["event_slug"].astype(str).tolist()
            if len(run_slugs) <= 2:
                dropped_events.extend({"event_slug": slug, "reason": "edge_window_in_short_run"} for slug in run_slugs)
                continue

            interior_slugs = run_slugs[1:-1]
            edge_slugs = [run_slugs[0], run_slugs[-1]]
            dropped_events.extend({"event_slug": slug, "reason": "edge_window_in_continuous_run"} for slug in edge_slugs)

            for slug in interior_slugs:
                if slug not in available_spot_slugs:
                    dropped_events.append({"event_slug": slug, "reason": "missing_spot_event_slug"})
                    continue
                complete_event_slugs.append(slug)

    complete_event_slugs = sorted(set(complete_event_slugs))
    filtered_orderbooks = orderbooks[orderbooks["event_slug"].isin(complete_event_slugs)].copy()
    filtered_levels = None
    if orderbook_levels is not None:
        filtered_levels = orderbook_levels[orderbook_levels["event_slug"].isin(complete_event_slugs)].copy()

    if dropped_events:
        preview = dropped_events[:5]
        logger.warning(
            "Dropped %s incomplete event windows before market-state construction. Examples: %s",
            len(dropped_events),
            preview,
        )
    logger.info(
        "Retained %s/%s complete event windows for market-state construction",
        len(complete_event_slugs),
        prepared_orderbooks["event_slug"].nunique(),
    )
    return filtered_orderbooks, spot.copy(), filtered_levels


def build_market_state_dataset(
    orderbooks: pd.DataFrame,
    spot: pd.DataFrame,
    orderbook_levels: pd.DataFrame | None = None,
    state_builder: LatentMarkovStateBuilder | None = None,
    spot_tolerance_seconds: float = 2.0,
    event_duration_seconds: float = 300.0,
) -> pd.DataFrame:
    """Build a continuous market-state dataset from aligned orderbook and spot data."""
    state_builder = state_builder or LatentMarkovStateBuilder()

    prepared_orderbooks = prepare_orderbooks(orderbooks)
    if orderbook_levels is not None and not orderbook_levels.empty:
        prepared_levels = prepare_orderbook_levels(orderbook_levels)
        level_features = build_orderbook_level_features(prepared_levels)
        if not level_features.empty:
            merge_keys = ["event_slug", "collected_at", "token_id", "outcome_name"]
            prepared_orderbooks = prepared_orderbooks.merge(level_features, on=merge_keys, how="left")
    prepared_spot = prepare_spot(spot)
    spot_by_asset = {asset: frame for asset, frame in prepared_spot.groupby("asset", sort=False)}
    reference_prices = build_reference_prices(
        orderbooks=prepared_orderbooks,
        spot_by_asset=spot_by_asset,
        event_duration_seconds=event_duration_seconds,
    )

    state_rows = []
    for timestamp, batch in prepared_orderbooks.groupby("_collected_at_dt", sort=True):
        spot_ticks = spot_ticks_asof(
            spot_by_asset=spot_by_asset,
            assets=batch["asset"].dropna().unique(),
            timestamp=timestamp,
            tolerance_seconds=spot_tolerance_seconds,
        )
        if not spot_ticks:
            continue

        batch_records = batch.drop(columns=["_collected_at_dt"]).to_dict("records")
        batch_references = {
            event_slug: reference_prices[event_slug]
            for event_slug in batch["event_slug"].dropna().unique()
            if event_slug in reference_prices
        }
        state_rows.extend(
            state_builder.build(
                orderbook_summary_rows=batch_records,
                spot_ticks=spot_ticks,
                reference_prices_by_event=batch_references,
            )
        )

    if not state_rows:
        raise ValueError("No market-state rows were generated. Check spot/orderbook timestamp overlap.")

    state = pd.DataFrame(state_rows)
    state = _add_market_observation_v2_features(
        state,
        fallback_volatility_per_sqrt_second=state_builder.config.fallback_volatility_per_sqrt_second,
    )
    state = _add_latent_mechanism_features(state)
    return state


def build_event_state_dataset(market_state: pd.DataFrame) -> pd.DataFrame:
    """Collapse token-level market_state rows into one continuous event-level state per timestamp."""
    required = {"event_slug", "collected_at", "outcome_name"}
    missing = required - set(market_state.columns)
    if missing:
        raise ValueError(f"Market-state data is missing columns: {sorted(missing)}")

    prepared = market_state.copy()
    prepared["_outcome_key"] = prepared["outcome_name"].astype(str).str.strip().str.lower()
    prepared = prepared[prepared["_outcome_key"].isin({"up", "down"})].copy()
    if prepared.empty:
        raise ValueError("Market-state data does not contain Up/Down outcome rows")

    key_cols = ["event_slug", "collected_at"]
    shared_cols = [
        "series_slug",
        "asset",
        "event_id",
        "event_title",
        "market_id",
        "condition_id",
        "state_timestamp",
        "market_start_time",
        "market_end_time",
        "closed",
        "accepting_orders",
        "spot_source",
        "spot_product_id",
        "spot_exchange_time",
        "spot_price",
        "spot_bid",
        "spot_ask",
        "reference_spot_price",
        "reference_source",
        "spot_return_since_reference",
        "seconds_to_end",
        "normalized_time_to_end",
        "volatility_per_sqrt_second",
        "spot_vol_multiplier",
        "external_spot_drift",
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
        "state_observation_variance",
    ]
    side_cols = [
        "token_id",
        "book_timestamp",
        "book_age_seconds",
        "book_hash",
        "best_bid",
        "best_ask",
        "spread",
        "mid_price",
        "bid_depth",
        "ask_depth",
        "bid_depth_top_5",
        "ask_depth_top_5",
        "orderbook_imbalance",
        "bid_levels",
        "ask_levels",
        "best_bid_size",
        "best_ask_size",
        "micro_price",
        "bid_depth_slope",
        "ask_depth_slope",
        "depth_slope",
        "bid_tick_density",
        "ask_tick_density",
        "tick_density",
        "weighted_imbalance",
        "book_velocity",
        "mid_price_velocity",
        "micro_price_velocity",
    ]

    shared_cols = [column for column in shared_cols if column in prepared.columns]
    side_cols = [column for column in side_cols if column in prepared.columns]

    base = (
        prepared.sort_values(key_cols + ["_outcome_key"])
        .drop_duplicates(subset=key_cols, keep="last")[key_cols + shared_cols]
        .reset_index(drop=True)
    )

    up = _prefixed_side_frame(prepared, outcome_key="up", key_cols=key_cols, side_cols=side_cols, prefix="up")
    down = _prefixed_side_frame(prepared, outcome_key="down", key_cols=key_cols, side_cols=side_cols, prefix="down")

    event_state = base.merge(up, on=key_cols, how="left").merge(down, on=key_cols, how="left")
    event_state["has_up_book"] = event_state["up_token_id"].notna() if "up_token_id" in event_state.columns else False
    event_state["has_down_book"] = event_state["down_token_id"].notna() if "down_token_id" in event_state.columns else False
    event_state["has_full_book_pair"] = event_state["has_up_book"] & event_state["has_down_book"]
    event_state = _add_event_market_observation_features(event_state)
    return event_state.sort_values(key_cols).reset_index(drop=True)


def load_parquet_glob(pattern: str, include_latest: bool = False) -> pd.DataFrame:
    paths = matching_parquet_paths(pattern, include_latest=include_latest)
    if not paths:
        raise FileNotFoundError(f"No parquet files matched {pattern}")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def load_optional_parquet_glob(pattern: str, include_latest: bool = False) -> pd.DataFrame:
    paths = matching_parquet_paths(pattern, include_latest=include_latest)
    if not paths:
        logger.warning("No parquet files matched %s", pattern)
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def matching_parquet_paths(pattern: str, include_latest: bool = False) -> list[Path]:
    paths = [Path(path) for path in sorted(glob.glob(pattern))]
    if not include_latest:
        paths = [path for path in paths if "latest" not in path.name]
    if not paths and not include_latest:
        paths = [Path(path) for path in sorted(glob.glob(pattern))]
    return paths


def load_orderbook_raw_glob(pattern: str, include_latest: bool = False) -> pd.DataFrame:
    records = load_json_glob_records(pattern, include_latest=include_latest)
    if not records:
        raise FileNotFoundError(f"No raw JSON files matched {pattern}")

    summary_rows = [_summarize_orderbook_record(record) for record in records]
    summary_rows = [row for row in summary_rows if row is not None]
    if not summary_rows:
        raise ValueError(f"No valid orderbook summary rows could be derived from {pattern}")
    return pd.DataFrame(summary_rows)


def load_spot_raw_glob(pattern: str, include_latest: bool = False) -> pd.DataFrame:
    records = load_json_glob_records(pattern, include_latest=include_latest)
    if not records:
        raise FileNotFoundError(f"No raw JSON files matched {pattern}")
    return pd.DataFrame(records)


def load_json_glob_records(pattern: str, include_latest: bool = False) -> list[dict[str, Any]]:
    paths = matching_json_paths(pattern, include_latest=include_latest)
    records: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text())
        if isinstance(payload, list):
            records.extend(item for item in payload if isinstance(item, dict))
        elif isinstance(payload, dict):
            records.append(payload)
    return records


def matching_json_paths(pattern: str, include_latest: bool = False) -> list[Path]:
    paths = [Path(path) for path in sorted(glob.glob(pattern))]
    if not include_latest:
        paths = [path for path in paths if "latest" not in path.name]
    if not paths and not include_latest:
        paths = [Path(path) for path in sorted(glob.glob(pattern))]
    return paths


def prepare_orderbooks(orderbooks: pd.DataFrame) -> pd.DataFrame:
    required = {"collected_at", "event_slug", "asset", "outcome_name", "token_id"}
    missing = required - set(orderbooks.columns)
    if missing:
        raise ValueError(f"Orderbook data is missing columns: {sorted(missing)}")

    prepared = orderbooks.copy()
    prepared["_collected_at_dt"] = pd.to_datetime(prepared["collected_at"], utc=True)
    prepared = prepared.dropna(subset=["_collected_at_dt", "event_slug", "asset"])
    return prepared.sort_values("_collected_at_dt").reset_index(drop=True)


def prepare_orderbook_levels(orderbook_levels: pd.DataFrame) -> pd.DataFrame:
    required = {"collected_at", "event_slug", "token_id", "outcome_name", "side", "level", "price", "size"}
    missing = required - set(orderbook_levels.columns)
    if missing:
        raise ValueError(f"Orderbook level data is missing columns: {sorted(missing)}")

    prepared = orderbook_levels.copy()
    prepared["_collected_at_dt"] = pd.to_datetime(prepared["collected_at"], utc=True)
    prepared["level"] = pd.to_numeric(prepared["level"], errors="coerce")
    prepared["price"] = pd.to_numeric(prepared["price"], errors="coerce")
    prepared["size"] = pd.to_numeric(prepared["size"], errors="coerce")
    prepared = prepared.dropna(
        subset=["_collected_at_dt", "event_slug", "token_id", "outcome_name", "side", "level", "price", "size"]
    )
    prepared["side"] = prepared["side"].astype(str).str.strip().str.lower()
    prepared = prepared[prepared["side"].isin({"bid", "ask"})].copy()
    return prepared.sort_values(["_collected_at_dt", "event_slug", "token_id", "side", "level"]).reset_index(drop=True)


def prepare_spot(spot: pd.DataFrame) -> pd.DataFrame:
    required = {"collected_at", "asset", "price"}
    missing = required - set(spot.columns)
    if missing:
        raise ValueError(f"Spot data is missing columns: {sorted(missing)}")

    prepared = spot.copy()
    prepared["_collected_at_dt"] = pd.to_datetime(prepared["collected_at"], utc=True)
    prepared["price"] = pd.to_numeric(prepared["price"], errors="coerce")
    prepared = prepared.dropna(subset=["_collected_at_dt", "asset", "price"])
    return prepared.sort_values("_collected_at_dt").reset_index(drop=True)


def build_reference_prices(
    orderbooks: pd.DataFrame,
    spot_by_asset: dict[str, pd.DataFrame],
    event_duration_seconds: float,
) -> dict[str, dict[str, Any]]:
    references = {}
    event_rows = orderbooks[["event_slug", "asset"]].drop_duplicates()
    for row in event_rows.to_dict("records"):
        event_slug = str(row["event_slug"])
        asset = str(row["asset"])
        event_start = event_start_from_slug(event_slug)
        asset_spot = spot_by_asset.get(asset)
        if event_start is None or asset_spot is None or asset_spot.empty:
            continue

        event_end = event_start + timedelta(seconds=event_duration_seconds)
        in_window = asset_spot[
            (asset_spot["_collected_at_dt"] >= event_start)
            & (asset_spot["_collected_at_dt"] <= event_end)
        ]
        if in_window.empty:
            continue

        reference_tick = in_window.iloc[0]
        references[event_slug] = {
            "price": float(reference_tick["price"]),
            "source": "first_observed_spot_in_event_window",
            "collected_at": reference_tick["collected_at"],
        }
    return references


def build_orderbook_level_features(orderbook_levels: pd.DataFrame) -> pd.DataFrame:
    if orderbook_levels.empty:
        return pd.DataFrame()

    key_cols = ["event_slug", "collected_at", "token_id", "outcome_name"]
    feature_rows: list[dict[str, Any]] = []

    for key, group in orderbook_levels.groupby(key_cols, sort=False):
        bids = group[group["side"] == "bid"].sort_values("level")
        asks = group[group["side"] == "ask"].sort_values("level")

        best_bid = _first_value(bids, "price")
        best_ask = _first_value(asks, "price")
        best_bid_size = _first_value(bids, "size")
        best_ask_size = _first_value(asks, "size")
        mid_price = _mid_from_quotes(best_bid, best_ask)
        micro_price = _micro_price(best_bid, best_ask, best_bid_size, best_ask_size)

        weighted_bid_depth = _weighted_depth(bids)
        weighted_ask_depth = _weighted_depth(asks)
        weighted_total = weighted_bid_depth + weighted_ask_depth
        weighted_imbalance = (
            (weighted_bid_depth - weighted_ask_depth) / weighted_total
            if weighted_total > 0
            else np.nan
        )

        bid_depth_slope = _cumulative_depth_slope(bids)
        ask_depth_slope = _cumulative_depth_slope(asks)
        depth_slope = np.nanmean([bid_depth_slope, ask_depth_slope])
        if not np.isfinite(depth_slope):
            depth_slope = np.nan

        bid_tick_density = _price_tick_density(bids)
        ask_tick_density = _price_tick_density(asks)
        tick_density = np.nanmin([bid_tick_density, ask_tick_density])
        if not np.isfinite(tick_density):
            tick_density = np.nan

        feature_rows.append(
            {
                "event_slug": key[0],
                "collected_at": key[1],
                "token_id": key[2],
                "outcome_name": key[3],
                "best_bid_size": best_bid_size,
                "best_ask_size": best_ask_size,
                "micro_price": micro_price,
                "bid_depth_slope": bid_depth_slope,
                "ask_depth_slope": ask_depth_slope,
                "depth_slope": depth_slope,
                "bid_tick_density": bid_tick_density,
                "ask_tick_density": ask_tick_density,
                "tick_density": tick_density,
                "weighted_imbalance": weighted_imbalance,
            }
        )

    return pd.DataFrame(feature_rows)


def spot_ticks_asof(
    spot_by_asset: dict[str, pd.DataFrame],
    assets,
    timestamp: pd.Timestamp,
    tolerance_seconds: float,
) -> dict[str, dict[str, Any]]:
    spot_ticks = {}
    for asset in assets:
        frame = spot_by_asset.get(asset)
        if frame is None or frame.empty:
            continue

        idx = frame["_collected_at_dt"].searchsorted(timestamp, side="right") - 1
        if idx < 0:
            continue

        tick = frame.iloc[int(idx)]
        age_seconds = (timestamp - tick["_collected_at_dt"]).total_seconds()
        if age_seconds > tolerance_seconds:
            continue

        tick_dict = tick.drop(labels=["_collected_at_dt"]).to_dict()
        spot_ticks[str(asset)] = tick_dict
    return spot_ticks


def _add_market_observation_v2_features(
    rows: pd.DataFrame,
    *,
    fallback_volatility_per_sqrt_second: float,
) -> pd.DataFrame:
    if rows.empty:
        return rows

    prepared = rows.copy()
    numeric_cols = [
        "best_bid",
        "best_ask",
        "spread",
        "mid_price",
        "bid_depth_top_5",
        "ask_depth_top_5",
        "orderbook_imbalance",
        "micro_price",
        "best_bid_size",
        "best_ask_size",
        "weighted_imbalance",
        "volatility_per_sqrt_second",
    ]
    for column in numeric_cols:
        if column in prepared.columns:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    sort_cols = [column for column in ("event_slug", "token_id", "collected_at") if column in prepared.columns]
    prepared = prepared.sort_values(sort_cols).reset_index(drop=True)

    group_cols = [column for column in ("event_slug", "token_id") if column in prepared.columns]
    if group_cols:
        prepared["_collected_at_dt"] = pd.to_datetime(prepared["collected_at"], utc=True)
        prepared["_dt_seconds"] = (
            prepared.groupby(group_cols)["_collected_at_dt"].diff().dt.total_seconds().replace(0.0, np.nan)
        )

        prepared["mid_price_velocity"] = _group_velocity(prepared, group_cols, "mid_price")
        prepared["micro_price_velocity"] = _group_velocity(prepared, group_cols, "micro_price")
        normalized_bid_depth_change = _group_relative_change(prepared, group_cols, "bid_depth_top_5")
        normalized_ask_depth_change = _group_relative_change(prepared, group_cols, "ask_depth_top_5")
        imbalance_change = _group_abs_change(prepared, group_cols, "weighted_imbalance", fallback="orderbook_imbalance")
        quote_change = prepared["micro_price_velocity"].abs().where(
            prepared["micro_price_velocity"].notna(),
            prepared["mid_price_velocity"].abs(),
        )
        depth_change = normalized_bid_depth_change.abs().fillna(0.0) + normalized_ask_depth_change.abs().fillna(0.0)
        prepared["book_velocity"] = quote_change.fillna(0.0) + imbalance_change.fillna(0.0) + depth_change

        prepared = prepared.drop(columns=["_collected_at_dt", "_dt_seconds"])
    else:
        prepared["mid_price_velocity"] = np.nan
        prepared["micro_price_velocity"] = np.nan
        prepared["book_velocity"] = np.nan

    if fallback_volatility_per_sqrt_second > 0:
        prepared["spot_vol_multiplier"] = (
            pd.to_numeric(prepared.get("volatility_per_sqrt_second"), errors="coerce")
            / fallback_volatility_per_sqrt_second
        )
    else:
        prepared["spot_vol_multiplier"] = np.nan
    prepared["external_spot_drift"] = _spot_distance(prepared)
    return prepared


def _add_latent_mechanism_features(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows

    prepared = rows.copy()
    market_probability = _numeric_series(prepared, "market_implied_up_probability")
    fundamental_probability = _numeric_series(prepared, "fundamental_up_probability")
    latent_probability = _numeric_series(prepared, "latent_up_probability")

    prepared["market_fundamental_basis"] = market_probability - fundamental_probability
    prepared["latent_market_basis"] = latent_probability - market_probability
    prepared["latent_fundamental_basis"] = latent_probability - fundamental_probability
    prepared["abs_market_fundamental_basis"] = prepared["market_fundamental_basis"].abs()
    prepared["abs_latent_market_basis"] = prepared["latent_market_basis"].abs()
    prepared["abs_latent_fundamental_basis"] = prepared["latent_fundamental_basis"].abs()
    prepared["normalized_time_to_end"] = _normalized_time_to_end(prepared)
    return prepared


def _add_event_market_observation_features(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows

    prepared = _add_latent_mechanism_features(rows.copy())

    up_observed = _numeric_series(prepared, "up_micro_price")
    down_observed = _numeric_series(prepared, "down_micro_price")
    if up_observed.notna().sum() == 0:
        up_observed = _numeric_series(prepared, "up_mid_price")
    else:
        up_observed = up_observed.where(up_observed.notna(), _numeric_series(prepared, "up_mid_price"))
    if down_observed.notna().sum() == 0:
        down_observed = _numeric_series(prepared, "down_mid_price")
    else:
        down_observed = down_observed.where(
            down_observed.notna(),
            _numeric_series(prepared, "down_mid_price"),
        )

    prepared["cross_book_basis"] = up_observed + down_observed - 1.0
    prepared["cross_book_bid_basis"] = (
        _numeric_series(prepared, "up_best_bid")
        + _numeric_series(prepared, "down_best_bid")
        - 1.0
    )
    prepared["cross_book_ask_basis"] = (
        _numeric_series(prepared, "up_best_ask")
        + _numeric_series(prepared, "down_best_ask")
        - 1.0
    )
    prepared["spread_divergence"] = (
        _numeric_series(prepared, "up_spread")
        - _numeric_series(prepared, "down_spread")
    )

    up_boundary_reference = up_observed.where(up_observed.notna(), _numeric_series(prepared, "latent_up_probability"))
    prepared["dist_to_boundary"] = np.minimum(up_boundary_reference, 1.0 - up_boundary_reference)
    prepared["boundary_leverage_ratio"] = 1.0 / np.maximum(1.0 - up_boundary_reference, 1e-6)
    prepared["asymmetric_depth_ratio"] = _safe_divide(
        _numeric_series(prepared, "up_bid_depth_top_5"),
        _numeric_series(prepared, "down_bid_depth_top_5"),
    )
    prepared["book_age_max"] = pd.concat(
        [
            _numeric_series(prepared, "up_book_age_seconds"),
            _numeric_series(prepared, "down_book_age_seconds"),
        ],
        axis=1,
    ).max(axis=1, skipna=True)
    prepared["external_spot_drift"] = _spot_distance(prepared)
    if "spot_vol_multiplier" not in prepared.columns:
        prepared["spot_vol_multiplier"] = np.nan
    prepared = _add_regime_posterior_features(prepared)
    return prepared


def _add_regime_posterior_features(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows

    prepared = rows.copy()

    basis_pressure = pd.concat(
        [
            _numeric_series(prepared, "abs_market_fundamental_basis") / 0.05,
            _numeric_series(prepared, "abs_latent_market_basis") / 0.05,
            _numeric_series(prepared, "abs_latent_fundamental_basis") / 0.05,
        ],
        axis=1,
    ).max(axis=1, skipna=True).fillna(0.0)

    mean_book_velocity = pd.concat(
        [
            _numeric_series(prepared, "up_book_velocity").abs(),
            _numeric_series(prepared, "down_book_velocity").abs(),
        ],
        axis=1,
    ).mean(axis=1, skipna=True).fillna(0.0)

    vol_pressure = (_numeric_series(prepared, "spot_vol_multiplier") - 1.0).clip(lower=0.0).fillna(0.0)
    velocity_pressure = (mean_book_velocity / 0.05).clip(lower=0.0)
    time_pressure = (1.0 - _numeric_series(prepared, "normalized_time_to_end")).clip(lower=0.0, upper=1.0).fillna(0.0)
    boundary_proximity = (1.0 - 2.0 * _numeric_series(prepared, "dist_to_boundary")).clip(lower=0.0, upper=1.0).fillna(0.0)
    cross_alignment = (
        1.0 - (_numeric_series(prepared, "cross_book_basis").abs() / 0.05).clip(lower=0.0, upper=1.0)
    ).fillna(0.0)

    normal_score = 1.0 - 0.5 * basis_pressure - 0.5 * vol_pressure - 0.35 * time_pressure
    shock_score = 1.5 * basis_pressure + 1.0 * vol_pressure + 0.75 * velocity_pressure
    convergence_score = 1.75 * time_pressure + 1.0 * boundary_proximity + 0.75 * cross_alignment - 0.25 * vol_pressure

    logits = np.column_stack(
        [
            normal_score.to_numpy(dtype=float),
            shock_score.to_numpy(dtype=float),
            convergence_score.to_numpy(dtype=float),
        ]
    )
    logits = np.where(np.isfinite(logits), logits, 0.0)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits)
    weights = weights / np.maximum(np.sum(weights, axis=1, keepdims=True), 1e-12)

    prepared["regime_normal_posterior"] = weights[:, 0]
    prepared["regime_shock_posterior"] = weights[:, 1]
    prepared["regime_convergence_posterior"] = weights[:, 2]
    return prepared

def event_start_from_slug(event_slug: str) -> datetime | None:
    suffix = event_slug.rsplit("-", 1)[-1]
    if not suffix.isdigit():
        return None
    return datetime.fromtimestamp(int(suffix), tz=timezone.utc)


def _prefixed_side_frame(
    market_state: pd.DataFrame,
    outcome_key: str,
    key_cols: list[str],
    side_cols: list[str],
    prefix: str,
) -> pd.DataFrame:
    side = market_state[market_state["_outcome_key"] == outcome_key].copy()
    if side.empty:
        return pd.DataFrame(columns=key_cols + [f"{prefix}_{column}" for column in side_cols])

    side = (
        side.sort_values(key_cols)
        .drop_duplicates(subset=key_cols, keep="last")[key_cols + side_cols]
        .rename(columns={column: f"{prefix}_{column}" for column in side_cols})
        .reset_index(drop=True)
    )
    return side


def _first_value(rows: pd.DataFrame, column: str) -> float:
    if rows.empty or column not in rows.columns:
        return np.nan
    value = pd.to_numeric(rows.iloc[0][column], errors="coerce")
    return float(value) if pd.notna(value) else np.nan


def _mid_from_quotes(best_bid: float, best_ask: float) -> float:
    if pd.notna(best_bid) and pd.notna(best_ask):
        return float((best_bid + best_ask) / 2.0)
    if pd.notna(best_bid):
        return float(best_bid)
    if pd.notna(best_ask):
        return float(best_ask)
    return np.nan


def _micro_price(best_bid: float, best_ask: float, best_bid_size: float, best_ask_size: float) -> float:
    if all(pd.notna(value) for value in (best_bid, best_ask, best_bid_size, best_ask_size)):
        denominator = best_bid_size + best_ask_size
        if denominator > 0:
            return float((best_ask * best_bid_size + best_bid * best_ask_size) / denominator)
    return _mid_from_quotes(best_bid, best_ask)


def _weighted_depth(rows: pd.DataFrame, top_n: int = 5) -> float:
    if rows.empty:
        return 0.0
    top = rows.nsmallest(top_n, "level")
    if top.empty:
        return 0.0
    weights = 1.0 / np.maximum(pd.to_numeric(top["level"], errors="coerce").to_numpy(dtype=float), 1.0)
    sizes = pd.to_numeric(top["size"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return float(np.sum(weights * sizes))


def _cumulative_depth_slope(rows: pd.DataFrame, top_n: int = 5) -> float:
    if rows.empty:
        return np.nan
    top = rows.nsmallest(top_n, "level")
    if len(top) < 2:
        return np.nan
    levels = pd.to_numeric(top["level"], errors="coerce").to_numpy(dtype=float)
    cumulative = np.cumsum(pd.to_numeric(top["size"], errors="coerce").fillna(0.0).to_numpy(dtype=float))
    slope = np.polyfit(levels, cumulative, deg=1)[0]
    return float(slope) if np.isfinite(slope) else np.nan


def _price_tick_density(rows: pd.DataFrame, top_n: int = 5) -> float:
    if rows.empty:
        return np.nan
    top = rows.nsmallest(top_n, "level")
    prices = pd.to_numeric(top["price"], errors="coerce").dropna().to_numpy(dtype=float)
    if len(prices) <= 1:
        return 1.0 if len(prices) == 1 else np.nan
    diffs = np.abs(np.diff(np.sort(prices)))
    diffs = diffs[diffs > 1e-12]
    if len(diffs) == 0:
        return 1.0
    min_tick = float(np.min(diffs))
    price_range = float(np.max(prices) - np.min(prices))
    expected_levels = max(int(round(price_range / min_tick)) + 1, 1)
    return float(min(len(prices) / expected_levels, 1.0))


def _group_velocity(rows: pd.DataFrame, group_cols: list[str], column: str) -> pd.Series:
    series = _numeric_series(rows, column)
    change = series.groupby([rows[col] for col in group_cols]).diff().abs()
    dt_seconds = pd.to_numeric(rows.get("_dt_seconds"), errors="coerce")
    return _safe_divide(change, dt_seconds)


def _group_relative_change(rows: pd.DataFrame, group_cols: list[str], column: str) -> pd.Series:
    series = _numeric_series(rows, column)
    prev = series.groupby([rows[col] for col in group_cols]).shift(1)
    delta = (series - prev).abs()
    baseline = prev.abs() + 1e-6
    dt_seconds = pd.to_numeric(rows.get("_dt_seconds"), errors="coerce")
    return _safe_divide(delta / baseline, dt_seconds)


def _group_abs_change(rows: pd.DataFrame, group_cols: list[str], column: str, fallback: str | None = None) -> pd.Series:
    series = _numeric_series(rows, column)
    if series.notna().sum() == 0 and fallback is not None:
        series = _numeric_series(rows, fallback)
    change = series.groupby([rows[col] for col in group_cols]).diff().abs()
    dt_seconds = pd.to_numeric(rows.get("_dt_seconds"), errors="coerce")
    return _safe_divide(change, dt_seconds)


def _numeric_series(rows: pd.DataFrame, column: str) -> pd.Series:
    if column not in rows.columns:
        return pd.Series(np.nan, index=rows.index, dtype=float)
    return pd.to_numeric(rows[column], errors="coerce")


def _spot_distance(rows: pd.DataFrame) -> pd.Series:
    existing = _numeric_series(rows, "spot_return_since_reference")
    if existing.notna().any():
        return existing
    spot = _numeric_series(rows, "spot_price")
    reference = _numeric_series(rows, "reference_spot_price")
    return _safe_divide(spot, reference) - 1.0


def _normalized_time_to_end(rows: pd.DataFrame) -> pd.Series:
    seconds_to_end = _numeric_series(rows, "seconds_to_end")
    duration = pd.Series(np.nan, index=rows.index, dtype=float)

    if "market_start_time" in rows.columns and "market_end_time" in rows.columns:
        start = pd.to_datetime(rows["market_start_time"], utc=True, errors="coerce")
        end = pd.to_datetime(rows["market_end_time"], utc=True, errors="coerce")
        duration_from_bounds = (end - start).dt.total_seconds()
        duration = duration.where(duration.notna(), duration_from_bounds)

    if "event_slug" in rows.columns:
        duration_from_history = seconds_to_end.groupby(rows["event_slug"]).transform("max")
        duration = duration.where(duration.notna() & (duration > 0), duration_from_history)

    return _safe_divide(seconds_to_end, duration).clip(lower=0.0, upper=1.0)


def _safe_divide(numerator: pd.Series | Any, denominator: pd.Series | Any) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    result = num / den
    return result.where(den.abs() > 1e-12)


def _summarize_orderbook_record(record: dict[str, Any]) -> dict[str, Any] | None:
    book = record.get("orderbook")
    if not isinstance(book, dict):
        return None

    bids = book.get("bids", [])
    asks = book.get("asks", [])
    best_bid = _best_price(bids, max)
    best_ask = _best_price(asks, min)

    token_row = {key: value for key, value in record.items() if key != "orderbook"}
    return {
        **token_row,
        "book_timestamp": _parse_clob_timestamp(book.get("timestamp")),
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": (best_ask - best_bid) if best_bid is not None and best_ask is not None else None,
        "mid_price": ((best_bid + best_ask) / 2.0) if best_bid is not None and best_ask is not None else None,
        "bid_depth": _sum_sizes(bids),
        "ask_depth": _sum_sizes(asks),
        "bid_depth_top_5": _sum_sizes(bids[:5]),
        "ask_depth_top_5": _sum_sizes(asks[:5]),
        "orderbook_imbalance": _orderbook_imbalance(bids, asks),
        "bid_levels": len(bids),
        "ask_levels": len(asks),
        "book_hash": book.get("hash"),
    }


def _best_price(orders: list[dict[str, Any]], reducer) -> float | None:
    prices = [float(order["price"]) for order in orders if isinstance(order, dict) and "price" in order]
    return reducer(prices) if prices else None


def _sum_sizes(orders: list[dict[str, Any]]) -> float:
    return sum(float(order.get("size", 0.0)) for order in orders if isinstance(order, dict))


def _orderbook_imbalance(
    bids: list[dict[str, Any]],
    asks: list[dict[str, Any]],
) -> float | None:
    bid_depth = _sum_sizes(bids)
    ask_depth = _sum_sizes(asks)
    total_depth = bid_depth + ask_depth
    if total_depth == 0:
        return None
    return (bid_depth - ask_depth) / total_depth


def _parse_clob_timestamp(timestamp: Any) -> str | None:
    if timestamp is None:
        return None
    ts = int(timestamp)
    if ts > 10_000_000_000:
        ts = ts / 1000
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
