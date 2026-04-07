import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml

from polymarket_quant.ingestion.client import PolymarketRESTClient
from polymarket_quant.ingestion.pipeline import IngestionPipeline
from polymarket_quant.ingestion.spot import CoinbaseSpotPriceClient
from polymarket_quant.signals.mispricing import MispricingDetectorConfig, RealTimeMispricingDetector
from polymarket_quant.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the live BTC/ETH 5m mispricing detector.")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config file")
    parser.add_argument("--interval-seconds", type=float, default=2.0, help="Seconds between detector polls")
    parser.add_argument("--duration-seconds", type=float, default=300.0, help="Total detector duration")
    parser.add_argument("--event-limit", type=int, default=1, help="Current events per BTC/ETH series to scan")
    parser.add_argument("--min-edge", type=float, default=0.02, help="Minimum executable edge required to signal")
    parser.add_argument("--max-toxicity", type=float, default=0.7, help="Maximum toxicity score allowed to signal")
    parser.add_argument("--min-depth", type=float, default=0.0, help="Minimum same-side book depth required to signal")
    parser.add_argument("--n-samples", type=int, default=5_000, help="Monte Carlo samples per fair-price estimate")
    parser.add_argument(
        "--pricing-method",
        choices=["monte_carlo", "importance_sampling", "stratified"],
        default="monte_carlo",
        help="Fair probability estimator to use",
    )
    parser.add_argument(
        "--disable-particle-filter",
        action="store_true",
        help="Use raw pricing probabilities without particle-filter smoothing",
    )
    parser.add_argument(
        "--event-slug-prefixes",
        nargs="+",
        default=["btc-updown-5m", "eth-updown-5m"],
        help="Only collect events whose slugs start with these prefixes",
    )
    parser.add_argument(
        "--series-slugs",
        nargs="+",
        default=["btc-up-or-down-5m", "eth-up-or-down-5m"],
        help="Gamma series slugs to collect",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    api_config = config.get("api", {})
    data_config = config.get("data", {})
    spot_products = config.get("spot", {}).get("products", {"BTC": "BTC-USD", "ETH": "ETH-USD"})

    polymarket_client = PolymarketRESTClient(
        gamma_url=api_config["gamma_url"],
        clob_url=api_config["clob_url"],
    )
    spot_client = CoinbaseSpotPriceClient(base_url=api_config.get("coinbase_url", "https://api.exchange.coinbase.com"))
    pipeline = IngestionPipeline(
        client=polymarket_client,
        raw_dir=data_config["raw_dir"],
        processed_dir=data_config["processed_dir"],
    )
    detector = RealTimeMispricingDetector(
        MispricingDetectorConfig(
            min_edge=args.min_edge,
            max_toxicity=args.max_toxicity,
            min_depth=args.min_depth,
            pricing_method=args.pricing_method,
            n_samples=args.n_samples,
            use_particle_filter=not args.disable_particle_filter,
        )
    )

    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    deadline = time.monotonic() + args.duration_seconds
    raw_snapshots: List[Dict[str, Any]] = []
    level_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    spot_rows: List[Dict[str, Any]] = []
    signal_rows: List[Dict[str, Any]] = []
    reference_cache: Dict[str, Dict[str, Any]] = {}
    poll_count = 0

    logger.info(
        "Starting live mispricing detector for %s seconds at %s second intervals",
        args.duration_seconds,
        args.interval_seconds,
    )

    try:
        while time.monotonic() < deadline:
            poll_started = time.monotonic()
            spot_ticks = spot_client.fetch_spot_tickers(spot_products)
            batch_raw, batch_levels, batch_summary = pipeline.collect_crypto_5m_orderbooks_once(
                series_slugs=args.series_slugs,
                event_limit=args.event_limit,
                event_slug_prefixes=args.event_slug_prefixes,
            )
            reference_prices = _fetch_reference_prices(
                rows=batch_summary,
                spot_client=spot_client,
                spot_products=spot_products,
                reference_cache=reference_cache,
            )
            batch_signals = detector.detect(
                batch_summary,
                spot_ticks,
                reference_prices_by_event=reference_prices,
            )

            raw_snapshots.extend(batch_raw)
            level_rows.extend(batch_levels)
            summary_rows.extend(batch_summary)
            spot_rows.extend([tick for tick in spot_ticks.values() if tick])
            signal_rows.extend(batch_signals)
            poll_count += 1

            signal_counts = pd.Series([row["signal"] for row in batch_signals]).value_counts().to_dict()
            logger.info(
                "Poll %s collected %s token books, %s level rows, %s signals %s",
                poll_count,
                len(batch_raw),
                len(batch_levels),
                len(batch_signals),
                signal_counts,
            )

            elapsed = time.monotonic() - poll_started
            time.sleep(max(0.0, args.interval_seconds - elapsed))
    except KeyboardInterrupt:
        logger.info("Mispricing detector interrupted; saving partial data.")
    finally:
        pipeline.save_crypto_5m_orderbook_collection(
            raw_snapshots=raw_snapshots,
            level_rows=level_rows,
            summary_rows=summary_rows,
            run_timestamp=run_timestamp,
        )
        _save_rows(
            rows=spot_rows,
            raw_dir=Path(data_config["raw_dir"]),
            processed_dir=Path(data_config["processed_dir"]),
            raw_name=f"crypto_spot_ticks_raw_{run_timestamp}.json",
            latest_raw_name="crypto_spot_ticks_raw_latest.json",
            parquet_name=f"crypto_spot_ticks_{run_timestamp}.parquet",
            latest_parquet_name="crypto_spot_ticks_latest.parquet",
        )
        _save_rows(
            rows=signal_rows,
            raw_dir=Path(data_config["raw_dir"]),
            processed_dir=Path(data_config["processed_dir"]),
            raw_name=f"crypto_5m_mispricing_signals_raw_{run_timestamp}.json",
            latest_raw_name="crypto_5m_mispricing_signals_raw_latest.json",
            parquet_name=f"crypto_5m_mispricing_signals_{run_timestamp}.parquet",
            latest_parquet_name="crypto_5m_mispricing_signals_latest.parquet",
        )


def _save_rows(
    rows: List[Dict[str, Any]],
    raw_dir: Path,
    processed_dir: Path,
    raw_name: str,
    latest_raw_name: str,
    parquet_name: str,
    latest_parquet_name: str,
) -> None:
    if not rows:
        logger.warning("No rows to save for %s", parquet_name)
        return

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    with open(raw_dir / raw_name, "w") as f:
        json.dump(rows, f)
    with open(raw_dir / latest_raw_name, "w") as f:
        json.dump(rows, f)

    df = pd.DataFrame(rows)
    df.to_parquet(processed_dir / parquet_name, index=False)
    df.to_parquet(processed_dir / latest_parquet_name, index=False)
    logger.info("Saved %s rows to %s", len(df), processed_dir / parquet_name)


def _fetch_reference_prices(
    rows: List[Dict[str, Any]],
    spot_client: CoinbaseSpotPriceClient,
    spot_products: Dict[str, str],
    reference_cache: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    references: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        event_slug = str(row.get("event_slug", ""))
        asset = str(row.get("asset", ""))
        if not event_slug or not asset or event_slug in references:
            continue
        if event_slug in reference_cache:
            references[event_slug] = reference_cache[event_slug]
            continue

        event_start = _event_start_from_slug(event_slug) or _parse_datetime(row.get("market_start_time"))
        product_id = spot_products.get(asset)
        if event_start is None or product_id is None:
            continue

        reference = spot_client.fetch_reference_price(
            asset=asset,
            product_id=product_id,
            reference_time=event_start,
        )
        if reference:
            reference_cache[event_slug] = reference
            references[event_slug] = reference
    return references


def _event_start_from_slug(event_slug: str) -> datetime | None:
    suffix = event_slug.rsplit("-", 1)[-1]
    if not suffix.isdigit():
        return None
    return datetime.fromtimestamp(int(suffix), tz=timezone.utc)


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


if __name__ == "__main__":
    main()
