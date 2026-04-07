import argparse
import glob
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from polymarket_quant.evaluation.metrics import calculate_brier_score
from polymarket_quant.signals.mispricing import MispricingDetectorConfig, RealTimeMispricingDetector
from polymarket_quant.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_ORDERBOOK_GLOB = "data/processed/crypto_5m_orderbook_summary_*.parquet"
DEFAULT_SPOT_GLOB = "data/processed/crypto_spot_ticks_*.parquet"
DEFAULT_RESOLUTION_GLOB = "data/processed/crypto_5m_resolutions_*.parquet"


def replay_pricing(
    orderbook_glob: str = DEFAULT_ORDERBOOK_GLOB,
    spot_glob: str = DEFAULT_SPOT_GLOB,
    resolution_glob: str = DEFAULT_RESOLUTION_GLOB,
    output_dir: str = "data/processed",
    pricing_method: str = "monte_carlo",
    n_samples: int = 1_000,
    min_edge: float = 0.02,
    max_toxicity: float = 0.7,
    min_depth: float = 0.0,
    spot_tolerance_seconds: float = 2.0,
    event_duration_seconds: float = 300.0,
    fallback_volatility_per_sqrt_second: float = 0.0005,
    use_particle_filter: bool = False,
    include_latest: bool = False,
    run_timestamp: str | None = None,
) -> dict[str, Any]:
    """Replay historical orderbook/spot snapshots through the pricing detector."""
    orderbooks = _load_parquet_glob(orderbook_glob, include_latest=include_latest)
    spot = _load_parquet_glob(spot_glob, include_latest=include_latest)
    resolutions = _load_optional_parquet_glob(resolution_glob, include_latest=include_latest)

    orderbooks = _prepare_orderbooks(orderbooks)
    spot = _prepare_spot(spot)
    spot_by_asset = {asset: frame for asset, frame in spot.groupby("asset", sort=False)}
    reference_prices = _build_reference_prices(
        orderbooks=orderbooks,
        spot_by_asset=spot_by_asset,
        event_duration_seconds=event_duration_seconds,
    )

    detector = RealTimeMispricingDetector(
        MispricingDetectorConfig(
            min_edge=min_edge,
            max_toxicity=max_toxicity,
            min_depth=min_depth,
            pricing_method=pricing_method,
            n_samples=n_samples,
            fallback_volatility_per_sqrt_second=fallback_volatility_per_sqrt_second,
            event_duration_seconds=event_duration_seconds,
            use_particle_filter=use_particle_filter,
        )
    )

    replay_rows = []
    for timestamp, batch in orderbooks.groupby("_collected_at_dt", sort=True):
        spot_ticks = _spot_ticks_asof(
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
        replay_rows.extend(
            detector.detect(
                orderbook_summary_rows=batch_records,
                spot_ticks=spot_ticks,
                reference_prices_by_event=batch_references,
            )
        )

    if not replay_rows:
        raise ValueError("No replay pricing rows were generated. Check spot/orderbook timestamp overlap.")

    replay = pd.DataFrame(replay_rows)
    replay = _attach_resolution_labels(replay, resolutions)
    replay = _add_brier_component(replay)

    run_timestamp = run_timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    parquet_path = output_path / f"crypto_5m_pricing_replay_{run_timestamp}.parquet"
    latest_path = output_path / "crypto_5m_pricing_replay_latest.parquet"
    replay.to_parquet(parquet_path, index=False)
    replay.to_parquet(latest_path, index=False)

    labeled = replay.dropna(subset=["is_winner"]) if "is_winner" in replay.columns else pd.DataFrame()
    brier = None
    if not labeled.empty:
        brier = calculate_brier_score(
            labeled["fair_token_price"].astype(float).to_numpy(),
            labeled["is_winner"].astype(int).to_numpy(),
        )

    logger.info("Saved %s pricing replay rows to %s", len(replay), parquet_path)
    if brier is not None:
        logger.info("Replay Brier score: %.6f over %s labeled rows", brier, len(labeled))

    return {
        "rows": len(replay),
        "labeled_rows": len(labeled),
        "brier_score": brier,
        "output_path": str(parquet_path),
        "latest_path": str(latest_path),
    }


def _load_parquet_glob(pattern: str, include_latest: bool = False) -> pd.DataFrame:
    paths = _matching_parquet_paths(pattern, include_latest=include_latest)
    if not paths:
        raise FileNotFoundError(f"No parquet files matched {pattern}")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def _load_optional_parquet_glob(pattern: str, include_latest: bool = False) -> pd.DataFrame:
    paths = _matching_parquet_paths(pattern, include_latest=include_latest)
    if not paths:
        logger.warning("No resolution parquet files matched %s; replay will be unlabeled", pattern)
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def _matching_parquet_paths(pattern: str, include_latest: bool = False) -> list[Path]:
    paths = [Path(path) for path in sorted(glob.glob(pattern))]
    if not include_latest:
        paths = [path for path in paths if "latest" not in path.name]
    if not paths and not include_latest:
        paths = [Path(path) for path in sorted(glob.glob(pattern))]
    return paths


def _prepare_orderbooks(orderbooks: pd.DataFrame) -> pd.DataFrame:
    required = {"collected_at", "event_slug", "asset", "outcome_name", "token_id"}
    missing = required - set(orderbooks.columns)
    if missing:
        raise ValueError(f"Orderbook data is missing columns: {sorted(missing)}")

    prepared = orderbooks.copy()
    prepared["_collected_at_dt"] = pd.to_datetime(prepared["collected_at"], utc=True)
    prepared = prepared.dropna(subset=["_collected_at_dt", "event_slug", "asset"])
    return prepared.sort_values("_collected_at_dt").reset_index(drop=True)


def _prepare_spot(spot: pd.DataFrame) -> pd.DataFrame:
    required = {"collected_at", "asset", "price"}
    missing = required - set(spot.columns)
    if missing:
        raise ValueError(f"Spot data is missing columns: {sorted(missing)}")

    prepared = spot.copy()
    prepared["_collected_at_dt"] = pd.to_datetime(prepared["collected_at"], utc=True)
    prepared["price"] = pd.to_numeric(prepared["price"], errors="coerce")
    prepared = prepared.dropna(subset=["_collected_at_dt", "asset", "price"])
    return prepared.sort_values("_collected_at_dt").reset_index(drop=True)


def _build_reference_prices(
    orderbooks: pd.DataFrame,
    spot_by_asset: dict[str, pd.DataFrame],
    event_duration_seconds: float,
) -> dict[str, dict[str, Any]]:
    references = {}
    event_rows = orderbooks[["event_slug", "asset"]].drop_duplicates()
    for row in event_rows.to_dict("records"):
        event_slug = str(row["event_slug"])
        asset = str(row["asset"])
        event_start = _event_start_from_slug(event_slug)
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
            "source": "first_observed_coinbase_spot_in_event_window",
            "collected_at": reference_tick["collected_at"],
        }
    return references


def _spot_ticks_asof(
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


def _attach_resolution_labels(replay: pd.DataFrame, resolutions: pd.DataFrame) -> pd.DataFrame:
    if resolutions.empty:
        return replay

    label_columns = ["event_slug", "token_id", "outcome_name", "outcome_price", "is_winner"]
    available = [column for column in label_columns if column in resolutions.columns]
    labels = resolutions[available].copy()
    labels = labels.dropna(subset=["event_slug", "outcome_name"])
    merge_keys = ["event_slug", "outcome_name"]
    if "token_id" in labels.columns and "token_id" in replay.columns:
        merge_keys.insert(1, "token_id")
    labels = labels.drop_duplicates(subset=merge_keys, keep="last")

    return replay.merge(
        labels,
        on=merge_keys,
        how="left",
        suffixes=("", "_resolution"),
    )


def _add_brier_component(replay: pd.DataFrame) -> pd.DataFrame:
    if "is_winner" not in replay.columns:
        replay["brier_component"] = pd.NA
        return replay

    replay["is_winner"] = pd.to_numeric(replay["is_winner"], errors="coerce")
    replay["brier_component"] = (replay["fair_token_price"] - replay["is_winner"]) ** 2
    return replay


def _event_start_from_slug(event_slug: str) -> datetime | None:
    suffix = event_slug.rsplit("-", 1)[-1]
    if not suffix.isdigit():
        return None
    return datetime.fromtimestamp(int(suffix), tz=timezone.utc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay collected BTC/ETH 5m data through pricing models.")
    parser.add_argument("--orderbook-glob", default=DEFAULT_ORDERBOOK_GLOB, help="Orderbook summary parquet glob")
    parser.add_argument("--spot-glob", default=DEFAULT_SPOT_GLOB, help="Spot ticker parquet glob")
    parser.add_argument("--resolution-glob", default=DEFAULT_RESOLUTION_GLOB, help="Resolution label parquet glob")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory for replay parquet")
    parser.add_argument(
        "--pricing-method",
        choices=["monte_carlo", "importance_sampling", "stratified"],
        default="monte_carlo",
        help="Pricing estimator to replay",
    )
    parser.add_argument("--n-samples", type=int, default=1_000, help="Monte Carlo samples per event snapshot")
    parser.add_argument("--min-edge", type=float, default=0.02, help="Minimum executable edge required to signal")
    parser.add_argument("--max-toxicity", type=float, default=0.7, help="Maximum toxicity score allowed to signal")
    parser.add_argument("--min-depth", type=float, default=0.0, help="Minimum same-side depth required to signal")
    parser.add_argument("--spot-tolerance-seconds", type=float, default=2.0, help="Max age for as-of spot ticks")
    parser.add_argument("--event-duration-seconds", type=float, default=300.0, help="Event duration in seconds")
    parser.add_argument(
        "--fallback-volatility-per-sqrt-second",
        type=float,
        default=0.0005,
        help="Volatility used before enough spot history exists",
    )
    parser.add_argument("--use-particle-filter", action="store_true", help="Smooth raw probabilities over time")
    parser.add_argument("--include-latest", action="store_true", help="Include *_latest.parquet inputs")
    args = parser.parse_args()

    result = replay_pricing(
        orderbook_glob=args.orderbook_glob,
        spot_glob=args.spot_glob,
        resolution_glob=args.resolution_glob,
        output_dir=args.output_dir,
        pricing_method=args.pricing_method,
        n_samples=args.n_samples,
        min_edge=args.min_edge,
        max_toxicity=args.max_toxicity,
        min_depth=args.min_depth,
        spot_tolerance_seconds=args.spot_tolerance_seconds,
        event_duration_seconds=args.event_duration_seconds,
        fallback_volatility_per_sqrt_second=args.fallback_volatility_per_sqrt_second,
        use_particle_filter=args.use_particle_filter,
        include_latest=args.include_latest,
    )
    logger.info("Replay complete: %s", result)


if __name__ == "__main__":
    main()
