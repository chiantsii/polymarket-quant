import argparse
from datetime import datetime, timezone
from pathlib import Path

from polymarket_quant.state import (
    LatentMarkovStateBuilder,
    LatentMarkovStateConfig,
    build_market_state_dataset,
)
from polymarket_quant.state.dataset import (
    filter_complete_event_windows,
    load_optional_parquet_glob,
    load_parquet_glob,
)
from polymarket_quant.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_ORDERBOOK_GLOB = "data/*/processed/polymarket/crypto_5m_orderbook_summary_*.parquet"
DEFAULT_ORDERBOOK_LEVELS_GLOB = "data/*/processed/polymarket/crypto_5m_orderbook_levels_*.parquet"
DEFAULT_SPOT_GLOB = "data/*/processed/spot/binance_spot_ticks_*.parquet"
WINDOW_COVERAGE_TOLERANCE_SECONDS = 10.0


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
    include_latest: bool = False,
    run_timestamp: str | None = None,
) -> dict[str, str | int]:
    """Build market_state from processed orderbook and spot parquet only."""
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

    state_config = _build_state_config(
        event_duration_seconds=event_duration_seconds,
        fallback_volatility_per_sqrt_second=fallback_volatility_per_sqrt_second,
        volatility_window_seconds=volatility_window_seconds,
        latent_anchor_weight=latent_anchor_weight,
        latent_observation_std=latent_observation_std,
        latent_observation_spread_scale=latent_observation_spread_scale,
    )
    builder = LatentMarkovStateBuilder(state_config)
    state = build_market_state_dataset(
        orderbooks=orderbooks,
        spot=spot,
        orderbook_levels=orderbook_levels,
        state_builder=builder,
        spot_tolerance_seconds=spot_tolerance_seconds,
        event_duration_seconds=event_duration_seconds,
    )

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
        include_latest=args.include_latest,
    )


if __name__ == "__main__":
    main()
