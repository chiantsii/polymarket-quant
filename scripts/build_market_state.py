import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from polymarket_quant.state import (
    LatentMarkovStateBuilder,
    LatentMarkovStateConfig,
    build_market_state_dataset,
)
from polymarket_quant.state.dataset import (
    build_market_state_rows,
    build_reference_prices,
    finalize_market_state_rows,
    filter_complete_event_windows,
    load_optional_parquet_glob,
    load_parquet_glob,
    matching_parquet_paths,
    prepare_orderbooks,
    prepare_spot,
)
from polymarket_quant.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_ORDERBOOK_GLOB = "data/*/processed/polymarket/crypto_5m_orderbook_summary_*.parquet"
DEFAULT_ORDERBOOK_LEVELS_GLOB = "data/*/processed/polymarket/crypto_5m_orderbook_levels_*.parquet"
DEFAULT_SPOT_GLOB = "data/*/processed/spot/binance_spot_ticks_*.parquet"
WINDOW_COVERAGE_TOLERANCE_SECONDS = 10.0
SPOT_CONTEXT_BUFFER_SECONDS = 5.0
SUMMARY_PREFIX = "crypto_5m_orderbook_summary_"
LEVELS_PREFIX = "crypto_5m_orderbook_levels_"
SPOT_PREFIX = "binance_spot_ticks_"


def _build_state_config(
    *,
    event_duration_seconds: float,
    fallback_volatility_per_sqrt_second: float,
    volatility_window_seconds: float,
    latent_anchor_weight: float,
    latent_observation_std: float,
    latent_observation_spread_scale: float,
) -> LatentMarkovStateConfig:
    """Create the canonical latent-state assumptions for market-state construction."""
    return LatentMarkovStateConfig(
        fallback_volatility_per_sqrt_second=fallback_volatility_per_sqrt_second,
        volatility_window_seconds=volatility_window_seconds,
        anchor_weight=latent_anchor_weight,
        observation_std=latent_observation_std,
        observation_spread_scale=latent_observation_spread_scale,
        event_duration_seconds=event_duration_seconds,
    )


def build_market_state(
    orderbook_glob: str = DEFAULT_ORDERBOOK_GLOB,
    orderbook_levels_glob: str = DEFAULT_ORDERBOOK_LEVELS_GLOB,
    spot_glob: str = DEFAULT_SPOT_GLOB,
    output_dir: str = "data/processed",
    spot_tolerance_seconds: float = 2.0,
    event_duration_seconds: float = 300.0,
    fallback_volatility_per_sqrt_second: float = 0.0005,
    volatility_window_seconds: float = 120.0,
    latent_anchor_weight: float = 0.35,
    latent_observation_std: float = 0.03,
    latent_observation_spread_scale: float = 4.0,
    batch_mode: str = "file",
    include_latest: bool = False,
    run_timestamp: str | None = None,
) -> dict[str, str | int]:
    """Build market_state from processed orderbook and spot parquet only."""
    state_config = _build_state_config(
        event_duration_seconds=event_duration_seconds,
        fallback_volatility_per_sqrt_second=fallback_volatility_per_sqrt_second,
        volatility_window_seconds=volatility_window_seconds,
        latent_anchor_weight=latent_anchor_weight,
        latent_observation_std=latent_observation_std,
        latent_observation_spread_scale=latent_observation_spread_scale,
    )
    if batch_mode == "full":
        orderbooks = load_parquet_glob(orderbook_glob, include_latest=include_latest)
        orderbook_levels = load_optional_parquet_glob(orderbook_levels_glob, include_latest=include_latest)
        spot = load_parquet_glob(spot_glob, include_latest=include_latest)
        orderbooks, spot, orderbook_levels = filter_complete_event_windows(
            orderbooks=orderbooks,
            spot=spot,
            orderbook_levels=orderbook_levels,
            event_duration_seconds=event_duration_seconds,
            coverage_tolerance_seconds=WINDOW_COVERAGE_TOLERANCE_SECONDS,
        )
        if orderbooks.empty:
            raise ValueError("No complete event windows remained after event_slug-based window reconstruction.")

        state = build_market_state_dataset(
            orderbooks=orderbooks,
            spot=spot,
            orderbook_levels=orderbook_levels,
            state_builder=LatentMarkovStateBuilder(state_config),
            spot_tolerance_seconds=spot_tolerance_seconds,
            event_duration_seconds=event_duration_seconds,
        )
    elif batch_mode == "file":
        state = _build_market_state_file_batched(
            orderbook_glob=orderbook_glob,
            orderbook_levels_glob=orderbook_levels_glob,
            spot_glob=spot_glob,
            state_config=state_config,
            spot_tolerance_seconds=spot_tolerance_seconds,
            event_duration_seconds=event_duration_seconds,
            include_latest=include_latest,
        )
    else:
        raise ValueError(f"Unsupported batch_mode: {batch_mode}")

    run_timestamp = run_timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    parquet_path = output_path / f"crypto_5m_market_state_{run_timestamp}.parquet"
    latest_path = output_path / "crypto_5m_market_state_latest.parquet"
    state.to_parquet(parquet_path, index=False)
    state.to_parquet(latest_path, index=False)

    logger.info("Saved %s market-state rows to %s", len(state), parquet_path)
    return {
        "rows": len(state),
        "output_path": str(parquet_path),
        "latest_path": str(latest_path),
    }


def _build_market_state_file_batched(
    *,
    orderbook_glob: str,
    orderbook_levels_glob: str,
    spot_glob: str,
    state_config: LatentMarkovStateConfig,
    spot_tolerance_seconds: float,
    event_duration_seconds: float,
    include_latest: bool,
) -> pd.DataFrame:
    summary_paths = matching_parquet_paths(orderbook_glob, include_latest=include_latest)
    if not summary_paths:
        raise FileNotFoundError(f"No parquet files matched {orderbook_glob}")
    levels_paths = matching_parquet_paths(orderbook_levels_glob, include_latest=include_latest)
    spot_paths = matching_parquet_paths(spot_glob, include_latest=include_latest)
    if not spot_paths:
        raise FileNotFoundError(f"No parquet files matched {spot_glob}")

    complete_event_slugs, reference_prices = _prepare_batching_metadata(
        summary_paths=summary_paths,
        spot_paths=spot_paths,
        event_duration_seconds=event_duration_seconds,
    )
    if not complete_event_slugs:
        raise ValueError("No complete event windows remained after event_slug-based window reconstruction.")

    level_path_map = _build_file_path_map(levels_paths, prefix=LEVELS_PREFIX)
    spot_path_map = _build_file_path_map(spot_paths, prefix=SPOT_PREFIX)
    spot_context_by_asset: dict[str, pd.DataFrame] = {}
    raw_state_batches: list[pd.DataFrame] = []
    builder = LatentMarkovStateBuilder(state_config)

    for summary_path in summary_paths:
        asset = _path_asset(summary_path)
        batch_key = (asset, _path_suffix(summary_path, SUMMARY_PREFIX))

        orderbooks = pd.read_parquet(summary_path)
        orderbooks = orderbooks[orderbooks["event_slug"].isin(complete_event_slugs)].copy()
        if orderbooks.empty:
            continue

        spot_batch = _load_optional_batch_parquet(spot_path_map.get(batch_key))
        if not spot_batch.empty:
            spot_batch = spot_batch[spot_batch["event_slug"].isin(complete_event_slugs)].copy()
        spot_batch = _combine_spot_context(
            previous_context=spot_context_by_asset.get(asset),
            current_batch=spot_batch,
            spot_tolerance_seconds=spot_tolerance_seconds,
        )
        if spot_batch.empty:
            logger.warning("Skipping %s because no spot context remained after filtering", summary_path)
            continue

        levels_batch = _load_optional_batch_parquet(level_path_map.get(batch_key))
        if not levels_batch.empty:
            levels_batch = levels_batch[levels_batch["event_slug"].isin(complete_event_slugs)].copy()

        raw_state = build_market_state_rows(
            orderbooks=orderbooks,
            spot=spot_batch,
            orderbook_levels=levels_batch if not levels_batch.empty else None,
            state_builder=builder,
            spot_tolerance_seconds=spot_tolerance_seconds,
            event_duration_seconds=event_duration_seconds,
            reference_prices_by_event=reference_prices,
        )
        raw_state_batches.append(raw_state)
        spot_context_by_asset[asset] = _trim_spot_context(
            spot_batch,
            context_seconds=max(spot_tolerance_seconds, SPOT_CONTEXT_BUFFER_SECONDS),
        )

    if not raw_state_batches:
        raise ValueError("No market-state rows were generated from file batches.")

    raw_state = pd.concat(raw_state_batches, ignore_index=True)
    return finalize_market_state_rows(
        raw_state,
        fallback_volatility_per_sqrt_second=state_config.fallback_volatility_per_sqrt_second,
    )


def _prepare_batching_metadata(
    *,
    summary_paths: list[Path],
    spot_paths: list[Path],
    event_duration_seconds: float,
) -> tuple[set[str], dict[str, dict[str, object]]]:
    minimal_orderbooks = pd.concat(
        [
            pd.read_parquet(
                path,
                columns=["collected_at", "event_slug", "asset", "outcome_name", "token_id"],
            )
            for path in summary_paths
        ],
        ignore_index=True,
    )
    minimal_spot = pd.concat(
        [
            pd.read_parquet(
                path,
                columns=["collected_at", "asset", "price", "event_slug"],
            )
            for path in spot_paths
        ],
        ignore_index=True,
    )
    filtered_orderbooks, filtered_spot, _ = filter_complete_event_windows(
        orderbooks=minimal_orderbooks,
        spot=minimal_spot,
        orderbook_levels=None,
        event_duration_seconds=event_duration_seconds,
        coverage_tolerance_seconds=WINDOW_COVERAGE_TOLERANCE_SECONDS,
    )
    complete_event_slugs = set(filtered_orderbooks["event_slug"].astype(str))
    reference_prices = build_reference_prices(
        orderbooks=prepare_orderbooks(filtered_orderbooks),
        spot_by_asset={
            asset: frame
            for asset, frame in prepare_spot(filtered_spot).groupby("asset", sort=False)
        },
        event_duration_seconds=event_duration_seconds,
    )
    return complete_event_slugs, reference_prices


def _build_file_path_map(paths: list[Path], *, prefix: str) -> dict[tuple[str, str], Path]:
    return {
        (_path_asset(path), _path_suffix(path, prefix)): path
        for path in paths
    }


def _path_asset(path: Path) -> str:
    return path.parents[2].name


def _path_suffix(path: Path, prefix: str) -> str:
    if not path.name.startswith(prefix) or not path.name.endswith(".parquet"):
        raise ValueError(f"Unexpected parquet filename: {path.name}")
    return path.name[len(prefix) : -len(".parquet")]


def _load_optional_batch_parquet(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _combine_spot_context(
    *,
    previous_context: pd.DataFrame | None,
    current_batch: pd.DataFrame,
    spot_tolerance_seconds: float,
) -> pd.DataFrame:
    frames = [frame for frame in (previous_context, current_batch) if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    if "collected_at" not in combined.columns:
        return combined
    combined = combined.sort_values("collected_at").drop_duplicates().reset_index(drop=True)
    return _trim_spot_context(
        combined,
        context_seconds=max(spot_tolerance_seconds, SPOT_CONTEXT_BUFFER_SECONDS),
        preserve_all_current=True,
        current_batch=current_batch,
    )


def _trim_spot_context(
    spot_frame: pd.DataFrame,
    *,
    context_seconds: float,
    preserve_all_current: bool = False,
    current_batch: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if spot_frame.empty or "collected_at" not in spot_frame.columns:
        return spot_frame
    prepared = spot_frame.copy()
    prepared["_collected_at_dt"] = pd.to_datetime(prepared["collected_at"], utc=True)
    prepared = prepared.dropna(subset=["_collected_at_dt"]).sort_values("_collected_at_dt")
    cutoff = prepared["_collected_at_dt"].max() - pd.Timedelta(seconds=context_seconds)
    trimmed = prepared[prepared["_collected_at_dt"] >= cutoff].copy()
    if preserve_all_current and current_batch is not None and not current_batch.empty:
        current = current_batch.copy()
        current["_collected_at_dt"] = pd.to_datetime(current["collected_at"], utc=True)
        trimmed = pd.concat([trimmed, current], ignore_index=True).drop_duplicates().sort_values("_collected_at_dt")
    return trimmed.drop(columns=["_collected_at_dt"], errors="ignore").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build continuous latent-state market_state from processed orderbook and spot parquet.")
    parser.add_argument("--orderbook-glob", default=DEFAULT_ORDERBOOK_GLOB, help="Orderbook summary parquet glob")
    parser.add_argument("--orderbook-levels-glob", default=DEFAULT_ORDERBOOK_LEVELS_GLOB, help="Orderbook levels parquet glob")
    parser.add_argument("--spot-glob", default=DEFAULT_SPOT_GLOB, help="Spot parquet glob")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    parser.add_argument("--spot-tolerance-seconds", type=float, default=2.0, help="Max age for as-of spot joins")
    parser.add_argument("--event-duration-seconds", type=float, default=300.0, help="Expected event duration")
    parser.add_argument(
        "--fallback-volatility-per-sqrt-second",
        type=float,
        default=0.0005,
        help="Fallback realized volatility used before enough spot history exists",
    )
    parser.add_argument(
        "--volatility-window-seconds",
        type=float,
        default=120.0,
        help="Spot-history window used to estimate realized volatility",
    )
    parser.add_argument("--latent-anchor-weight", type=float, default=0.35, help="Weight on the fundamental anchor in latent update")
    parser.add_argument("--latent-observation-std", type=float, default=0.03, help="Base observation std in logit-space update")
    parser.add_argument(
        "--latent-observation-spread-scale",
        type=float,
        default=4.0,
        help="Scale linking displayed quote uncertainty to observation variance",
    )
    parser.add_argument(
        "--batch-mode",
        choices=["file", "full"],
        default="file",
        help="Use file-batched state construction to reduce peak memory, or load all inputs at once",
    )
    parser.add_argument("--include-latest", action="store_true", help="Include *_latest.parquet inputs")
    args = parser.parse_args()

    build_market_state(
        orderbook_glob=args.orderbook_glob,
        orderbook_levels_glob=args.orderbook_levels_glob,
        spot_glob=args.spot_glob,
        output_dir=args.output_dir,
        spot_tolerance_seconds=args.spot_tolerance_seconds,
        event_duration_seconds=args.event_duration_seconds,
        fallback_volatility_per_sqrt_second=args.fallback_volatility_per_sqrt_second,
        volatility_window_seconds=args.volatility_window_seconds,
        latent_anchor_weight=args.latent_anchor_weight,
        latent_observation_std=args.latent_observation_std,
        latent_observation_spread_scale=args.latent_observation_spread_scale,
        batch_mode=args.batch_mode,
        include_latest=args.include_latest,
    )


if __name__ == "__main__":
    main()
