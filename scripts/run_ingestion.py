import yaml
import argparse
from polymarket_quant.utils.logger import get_logger
from polymarket_quant.ingestion.client import PolymarketRESTClient
from polymarket_quant.ingestion.pipeline import IngestionPipeline

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run Polymarket Data Ingestion")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config file")
    parser.add_argument("--pipeline", type=str, choices=["markets", "orderbooks", "crypto-5m-history"], default="markets", help="Pipeline to run")
    parser.add_argument("--market-limit", type=int, default=20, help="Number of active markets to use for orderbook ingestion")
    parser.add_argument("--event-limit", type=int, default=10, help="Number of recent events per crypto 5m series")
    parser.add_argument("--fidelity", type=int, default=1, help="Price history fidelity in minutes")
    parser.add_argument("--interval", type=str, default="max", help="Price history interval, e.g. 1d, 1w, 1m, max")
    parser.add_argument("--include-open", action="store_true", help="Include unresolved/open crypto 5m events")
    args = parser.parse_args()

    logger.info(f"Loading configuration from {args.config}")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    client = PolymarketRESTClient(
        gamma_url=config["api"]["gamma_url"],
        clob_url=config["api"]["clob_url"]
    )

    pipeline = IngestionPipeline(
        client=client,
        raw_dir=config["data"]["raw_dir"],
        processed_dir=config["data"]["processed_dir"]
    )

    if args.pipeline == "markets":
        pipeline.run_market_metadata_ingestion()
    elif args.pipeline == "orderbooks":
        pipeline.run_orderbook_snapshot_ingestion(market_limit=args.market_limit)
    elif args.pipeline == "crypto-5m-history":
        pipeline.run_crypto_5m_history_ingestion(
            series_slugs=["btc-up-or-down-5m", "eth-up-or-down-5m"],
            event_limit=args.event_limit,
            interval=args.interval,
            fidelity=args.fidelity,
            closed_only=not args.include_open,
        )
    else:
        logger.error(f"Unknown pipeline: {args.pipeline}")

if __name__ == "__main__":
    main()
