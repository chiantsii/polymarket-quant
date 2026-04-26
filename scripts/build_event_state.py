import argparse
from datetime import datetime, timezone
from pathlib import Path

from polymarket_quant.state import build_event_state_dataset
from polymarket_quant.state.dataset import load_parquet_glob
from polymarket_quant.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_MARKET_STATE_GLOB = "data/processed/crypto_5m_market_state_latest.parquet"


def build_event_state(
    market_state_glob: str = DEFAULT_MARKET_STATE_GLOB,
    output_dir: str = "data/processed",
    include_latest: bool = False,
    run_timestamp: str | None = None,
) -> dict[str, str | int]:
    market_state = load_parquet_glob(market_state_glob, include_latest=include_latest)
    event_state = build_event_state_dataset(market_state)

    run_timestamp = run_timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    parquet_path = output_path / f"crypto_5m_event_state_{run_timestamp}.parquet"
    latest_path = output_path / "crypto_5m_event_state_latest.parquet"
    event_state.to_parquet(parquet_path, index=False)
    event_state.to_parquet(latest_path, index=False)

    logger.info("Saved %s event-state rows to %s", len(event_state), parquet_path)
    return {
        "rows": len(event_state),
        "output_path": str(parquet_path),
        "latest_path": str(latest_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build event-level continuous state from token-level market state.")
    parser.add_argument(
        "--market-state-glob",
        default=DEFAULT_MARKET_STATE_GLOB,
        help="Market-state parquet path or glob",
    )
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    parser.add_argument("--include-latest", action="store_true", help="Include *_latest.parquet inputs")
    args = parser.parse_args()

    build_event_state(
        market_state_glob=args.market_state_glob,
        output_dir=args.output_dir,
        include_latest=args.include_latest,
    )


if __name__ == "__main__":
    main()
