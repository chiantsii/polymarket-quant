import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from datetime import datetime, timezone

from polymarket_quant.ingestion.client import BasePolymarketClient
from polymarket_quant.schemas.market import MarketMetadata
from polymarket_quant.utils.logger import get_logger

logger = get_logger(__name__)

class IngestionPipeline:
    """Orchestrates fetching, normalizing, and storing data."""
    
    def __init__(self, client: BasePolymarketClient, raw_dir: str, processed_dir: str):
        self.client = client
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def run_market_metadata_ingestion(self):
        """Fetches active markets, saves raw payload, and processes to canonical schemas."""
        run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        logger.info("Starting market metadata ingestion run.")

        # 1. Fetch Raw Data
        raw_markets = self.client.fetch_active_markets()
        if not raw_markets:
            logger.warning("No markets fetched. Aborting pipeline.")
            return

        # 2. Save Raw Data (Immutable record of API responses)
        raw_path = self.raw_dir / f"markets_raw_{run_timestamp}.json"
        self._write_json(raw_path, raw_markets)
        self._write_json(self.raw_dir / "markets_raw_latest.json", raw_markets)
        logger.info(f"Saved {len(raw_markets)} raw records to {raw_path}")

        # 3. Normalize to Canonical Schemas
        normalized_markets = []
        for raw_mkt in raw_markets:
            try:
                # Mapping adapter logic: translating raw API keys to our schema
                # TODO: Adjust these mappings based on exact live API payload structures
                market = MarketMetadata(
                    id=raw_mkt.get("id", raw_mkt.get("condition_id", "unknown")),
                    condition_id=raw_mkt.get("condition_id", raw_mkt.get("conditionId", "")),
                    title=raw_mkt.get("question", raw_mkt.get("title", "")),
                    category=raw_mkt.get("category", "Unknown"),
                    resolution_date=raw_mkt.get("endDate", datetime.now(timezone.utc).isoformat()),
                    is_active=raw_mkt.get("active", True),
                    contracts=self._extract_contracts(raw_mkt),
                )
                normalized_markets.append(market)
            except Exception as e:
                logger.debug(f"Skipping malformed market {raw_mkt.get('id', 'unknown')}: {e}")

        # 4. Save Processed Data (Parquet for analytics/research readiness)
        if normalized_markets:
            df = pd.DataFrame([m.model_dump() for m in normalized_markets])
            # Save derived features dynamically 
            df['time_to_resolution_seconds'] = [m.time_to_resolution_seconds for m in normalized_markets]
            
            processed_path = self.processed_dir / f"markets_normalized_{run_timestamp}.parquet"
            df.to_parquet(processed_path, index=False)
            df.to_parquet(self.processed_dir / "markets_normalized_latest.parquet", index=False)
            logger.info(f"Successfully processed and saved {len(df)} markets to {processed_path}")

    def run_orderbook_snapshot_ingestion(self, market_limit: int = 20):
        """Fetch current orderbook snapshots for active market outcome tokens."""
        run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        logger.info("Starting orderbook snapshot ingestion run.")

        raw_markets = self.client.fetch_active_markets(limit=market_limit)
        token_rows = self._extract_token_rows(raw_markets)
        if not token_rows:
            logger.warning("No CLOB token ids found. Aborting orderbook pipeline.")
            return

        raw_snapshots = []
        processed_rows = []
        for token_row in token_rows:
            book = self.client.fetch_orderbook(token_row["token_id"])
            if not book:
                continue

            raw_snapshots.append({**token_row, "orderbook": book})
            processed_rows.append(self._summarize_orderbook(token_row, book))

        if not raw_snapshots:
            logger.warning("No orderbooks fetched. Aborting orderbook pipeline.")
            return

        raw_path = self.raw_dir / f"orderbooks_raw_{run_timestamp}.json"
        self._write_json(raw_path, raw_snapshots)
        self._write_json(self.raw_dir / "orderbooks_raw_latest.json", raw_snapshots)
        logger.info(f"Saved {len(raw_snapshots)} raw orderbook records to {raw_path}")

        df = pd.DataFrame(processed_rows)
        processed_path = self.processed_dir / f"orderbooks_snapshot_{run_timestamp}.parquet"
        df.to_parquet(processed_path, index=False)
        df.to_parquet(self.processed_dir / "orderbooks_snapshot_latest.parquet", index=False)
        logger.info(f"Saved {len(df)} processed orderbook summaries to {processed_path}")

    def run_crypto_5m_history_ingestion(
        self,
        series_slugs: List[str],
        event_limit: int = 10,
        interval: str = "max",
        fidelity: int = 1,
        closed_only: bool = True,
    ) -> None:
        """Fetch historical CLOB prices for BTC/ETH Up or Down 5m markets."""
        run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        logger.info("Starting crypto 5m historical price ingestion run.")

        raw_records = []
        processed_rows = []
        for series_slug in series_slugs:
            for event in self._get_series_events(series_slug, event_limit, closed_only=closed_only):
                event_detail = self.client.fetch_event_by_slug(event["slug"])
                if not event_detail:
                    continue

                for market in event_detail.get("markets", []):
                    for token in self._extract_contracts(market):
                        token_id = token["token_id"]
                        if not token_id:
                            continue

                        price_history = self.client.fetch_price_history(
                            token_id=token_id,
                            interval=interval,
                            fidelity=fidelity,
                        )
                        raw_records.append(
                            {
                                "series_slug": series_slug,
                                "event": event_detail,
                                "market": market,
                                "token": token,
                                "price_history": price_history,
                            }
                        )
                        processed_rows.extend(
                            self._price_history_rows(
                                series_slug=series_slug,
                                event=event_detail,
                                market=market,
                                token=token,
                                price_history=price_history,
                            )
                        )

        if not raw_records:
            logger.warning("No historical price records fetched. Aborting crypto 5m history pipeline.")
            return

        raw_path = self.raw_dir / f"crypto_5m_history_raw_{run_timestamp}.json"
        self._write_json(raw_path, raw_records)
        self._write_json(self.raw_dir / "crypto_5m_history_raw_latest.json", raw_records)
        logger.info(f"Saved {len(raw_records)} raw crypto 5m history records to {raw_path}")

        df = pd.DataFrame(processed_rows)
        processed_path = self.processed_dir / f"crypto_5m_price_history_{run_timestamp}.parquet"
        df.to_parquet(processed_path, index=False)
        df.to_parquet(self.processed_dir / "crypto_5m_price_history_latest.parquet", index=False)
        logger.info(f"Saved {len(df)} crypto 5m historical price rows to {processed_path}")

    def collect_crypto_5m_orderbooks_once(
        self,
        series_slugs: List[str],
        event_limit: int = 1,
        event_slug_prefixes: Optional[List[str]] = None,
        event_slugs: Optional[List[str]] = None,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Collect one live orderbook snapshot batch for BTC/ETH Up or Down 5m markets."""
        collected_at = datetime.now(timezone.utc).isoformat()
        raw_snapshots = []
        level_rows = []
        summary_rows = []
        now = datetime.now(timezone.utc)

        for series_slug in series_slugs:
            if event_slugs:
                events = [
                    {"slug": event_slug, "closed": False}
                    for event_slug in event_slugs
                    if self._series_slug_from_event_slug(event_slug) == series_slug
                ]
            else:
                events = self._get_series_events(
                    series_slug,
                    event_limit=event_limit,
                    closed_only=False,
                    current_only=True,
                    event_slug_prefixes=event_slug_prefixes,
                    now=now,
                )
            open_events = [event for event in events if event.get("closed") is not True]
            for event in open_events:
                event_detail = self.client.fetch_event_by_slug(event["slug"])
                if not event_detail:
                    continue

                for market in event_detail.get("markets", []):
                    if market.get("closed") is True:
                        continue
                    if not self._is_current_time_window(market, now):
                        continue

                    for token in self._extract_contracts(market):
                        token_id = token["token_id"]
                        if not token_id:
                            continue

                        token_row = self._crypto_5m_token_row(
                            series_slug=series_slug,
                            event=event_detail,
                            market=market,
                            token=token,
                        )
                        book = self.client.fetch_orderbook(token_id)
                        if not book:
                            continue

                        raw_snapshots.append(
                            {
                                **token_row,
                                "collected_at": collected_at,
                                "orderbook": book,
                            }
                        )
                        summary_rows.append(
                            self._summarize_orderbook(
                                {**token_row, "collected_at": collected_at},
                                book,
                            )
                        )
                        level_rows.extend(
                            self._orderbook_level_rows(
                                token_row={**token_row, "collected_at": collected_at},
                                book=book,
                            )
                        )

        return raw_snapshots, level_rows, summary_rows

    def save_crypto_5m_orderbook_collection(
        self,
        raw_snapshots: List[Dict[str, Any]],
        level_rows: List[Dict[str, Any]],
        summary_rows: List[Dict[str, Any]],
        run_timestamp: Optional[str] = None,
    ) -> None:
        """Persist collected live orderbook snapshots as raw JSON plus parquet datasets."""
        if not raw_snapshots:
            logger.warning("No crypto 5m orderbook snapshots to save.")
            return

        run_timestamp = run_timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        raw_path = self.raw_dir / f"crypto_5m_orderbooks_raw_{run_timestamp}.json"
        self._write_json(raw_path, raw_snapshots)
        self._write_json(self.raw_dir / "crypto_5m_orderbooks_raw_latest.json", raw_snapshots)

        levels_path = self.processed_dir / f"crypto_5m_orderbook_levels_{run_timestamp}.parquet"
        levels_df = pd.DataFrame(level_rows)
        levels_df.to_parquet(levels_path, index=False)
        levels_df.to_parquet(self.processed_dir / "crypto_5m_orderbook_levels_latest.parquet", index=False)

        summary_path = self.processed_dir / f"crypto_5m_orderbook_summary_{run_timestamp}.parquet"
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_parquet(summary_path, index=False)
        summary_df.to_parquet(self.processed_dir / "crypto_5m_orderbook_summary_latest.parquet", index=False)

        logger.info(f"Saved {len(raw_snapshots)} raw crypto 5m orderbooks to {raw_path}")
        logger.info(f"Saved {len(levels_df)} orderbook level rows to {levels_path}")
        logger.info(f"Saved {len(summary_df)} orderbook summary rows to {summary_path}")

    def collect_crypto_5m_resolutions_for_event_slugs(
        self,
        event_slugs: List[str],
        event_slug_prefixes: Optional[List[str]] = None,
        resolved_only: bool = True,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Collect winner labels for explicitly requested BTC/ETH 5m event slugs."""
        resolved_at = datetime.now(timezone.utc).isoformat()
        raw_records = []
        resolution_rows = []
        seen_event_slugs = set()

        for event_slug in event_slugs:
            if not event_slug or event_slug in seen_event_slugs:
                continue
            if event_slug_prefixes and not self._matches_event_slug_prefix(event_slug, event_slug_prefixes):
                continue
            seen_event_slugs.add(event_slug)

            event_detail = self.client.fetch_event_by_slug(event_slug)
            if not event_detail:
                continue

            series_slug = self._series_slug_from_event_slug(event_slug)
            raw_records.append({"series_slug": series_slug, "event": event_detail, "resolved_at": resolved_at})
            for market in event_detail.get("markets", []):
                market_rows = self._crypto_5m_resolution_rows(
                    series_slug=series_slug,
                    event=event_detail,
                    market=market,
                    resolved_at=resolved_at,
                )
                if resolved_only:
                    market_rows = [row for row in market_rows if row["is_winner"] is not None]
                resolution_rows.extend(market_rows)

        return raw_records, resolution_rows

    def save_crypto_5m_resolutions(
        self,
        raw_records: List[Dict[str, Any]],
        resolution_rows: List[Dict[str, Any]],
        run_timestamp: Optional[str] = None,
    ) -> None:
        """Persist recently collected resolution labels."""
        if not raw_records:
            logger.warning("No crypto 5m resolution records to save.")
            return
        if not resolution_rows:
            logger.warning("No crypto 5m resolution labels to save.")
            return

        run_timestamp = run_timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        raw_path = self.raw_dir / f"crypto_5m_resolutions_raw_{run_timestamp}.json"
        self._write_json(raw_path, raw_records)
        self._write_json(self.raw_dir / "crypto_5m_resolutions_raw_latest.json", raw_records)

        df = pd.DataFrame(resolution_rows)
        processed_path = self.processed_dir / f"crypto_5m_resolutions_{run_timestamp}.parquet"
        df.to_parquet(processed_path, index=False)
        df.to_parquet(self.processed_dir / "crypto_5m_resolutions_latest.parquet", index=False)
        logger.info(f"Saved {len(df)} crypto 5m resolution rows to {processed_path}")

    def _write_json(self, path: Path, payload: Any) -> None:
        with open(path, "w") as f:
            json.dump(payload, f)

    def _extract_contracts(self, raw_market: Dict[str, Any]) -> List[Dict[str, str]]:
        tokens = raw_market.get("tokens") or []
        if tokens:
            return [
                {
                    "token_id": str(token.get("token_id", token.get("tokenId", ""))),
                    "outcome_name": str(token.get("outcome", token.get("outcome_name", ""))),
                }
                for token in tokens
            ]

        token_ids = self._loads_json_list(raw_market.get("clobTokenIds"))
        outcomes = self._loads_json_list(raw_market.get("outcomes"))
        return [
            {"token_id": str(token_id), "outcome_name": str(outcome)}
            for token_id, outcome in zip(token_ids, outcomes)
        ]

    def _extract_token_rows(self, raw_markets: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        rows = []
        for raw_market in raw_markets:
            for contract in self._extract_contracts(raw_market):
                token_id = contract["token_id"]
                if not token_id:
                    continue
                rows.append(
                    {
                        "market_id": str(raw_market.get("id", "")),
                        "condition_id": str(raw_market.get("condition_id", raw_market.get("conditionId", ""))),
                        "title": str(raw_market.get("question", raw_market.get("title", ""))),
                        "outcome_name": contract["outcome_name"],
                        "token_id": token_id,
                    }
                )
        return rows

    def _loads_json_list(self, value: Any) -> List[Any]:
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                return parsed if isinstance(parsed, list) else []
            except json.JSONDecodeError:
                return []
        return []

    def _get_series_events(
        self,
        series_slug: str,
        event_limit: int,
        closed_only: bool = True,
        current_only: bool = False,
        event_slug_prefixes: Optional[List[str]] = None,
        now: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        series_payloads = self.client.fetch_series(series_slug)
        if not series_payloads:
            logger.warning(f"No series payload found for {series_slug}")
            return []

        events = series_payloads[0].get("events", [])
        if closed_only:
            events = [event for event in events if event.get("closed") is True]
        if event_slug_prefixes:
            events = [
                event
                for event in events
                if self._matches_event_slug_prefix(event.get("slug"), event_slug_prefixes)
            ]
        if current_only:
            current_time = now or datetime.now(timezone.utc)
            events = [
                event
                for event in events
                if event.get("closed") is not True and self._is_current_time_window(event, current_time)
            ]
        events = sorted(events, key=lambda event: event.get("startTime") or event.get("startDate") or "")
        return events[-event_limit:]

    def _matches_event_slug_prefix(self, slug: Any, prefixes: List[str]) -> bool:
        if not isinstance(slug, str):
            return False
        return any(slug.startswith(prefix) for prefix in prefixes)

    def _series_slug_from_event_slug(self, event_slug: str) -> str:
        if event_slug.startswith("btc"):
            return "btc-up-or-down-5m"
        if event_slug.startswith("eth"):
            return "eth-up-or-down-5m"
        return ""

    def _is_current_time_window(self, payload: Dict[str, Any], now: datetime) -> bool:
        start_time = self._parse_iso_datetime(
            payload.get("eventStartTime") or payload.get("startTime") or payload.get("startDate")
        )
        end_time = self._parse_iso_datetime(payload.get("endDate"))
        if start_time is None or end_time is None:
            return False
        return start_time <= now <= end_time

    def _parse_iso_datetime(self, value: Any) -> Optional[datetime]:
        if value is None:
            return None
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

    def _price_history_rows(
        self,
        series_slug: str,
        event: Dict[str, Any],
        market: Dict[str, Any],
        token: Dict[str, str],
        price_history: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        outcome_prices = self._loads_json_list(market.get("outcomePrices"))
        outcomes = self._loads_json_list(market.get("outcomes"))
        outcome_name = token["outcome_name"]
        outcome_price = self._lookup_outcome_price(outcomes, outcome_prices, outcome_name)
        is_winner = self._infer_winner(outcome_price)

        rows = []
        for point in price_history.get("history", []):
            timestamp = datetime.fromtimestamp(int(point["t"]), tz=timezone.utc).isoformat()
            rows.append(
                {
                    "series_slug": series_slug,
                    "asset": "BTC" if series_slug.startswith("btc") else "ETH",
                    "event_id": event.get("id"),
                    "event_slug": event.get("slug"),
                    "event_title": event.get("title"),
                    "market_id": market.get("id"),
                    "condition_id": market.get("conditionId", market.get("condition_id")),
                    "token_id": token["token_id"],
                    "outcome_name": outcome_name,
                    "timestamp": timestamp,
                    "price": float(point["p"]),
                    "market_start_time": market.get("eventStartTime", event.get("startTime")),
                    "market_end_time": market.get("endDate", event.get("endDate")),
                    "closed": market.get("closed", event.get("closed")),
                    "outcome_price": outcome_price,
                    "is_winner": is_winner,
                }
            )
        return rows

    def _crypto_5m_token_row(
        self,
        series_slug: str,
        event: Dict[str, Any],
        market: Dict[str, Any],
        token: Dict[str, str],
    ) -> Dict[str, Any]:
        return {
            "series_slug": series_slug,
            "asset": "BTC" if series_slug.startswith("btc") else "ETH",
            "event_id": event.get("id"),
            "event_slug": event.get("slug"),
            "event_title": event.get("title"),
            "market_id": market.get("id"),
            "condition_id": market.get("conditionId", market.get("condition_id")),
            "token_id": token["token_id"],
            "outcome_name": token["outcome_name"],
            "market_start_time": market.get("eventStartTime", event.get("startTime")),
            "market_end_time": market.get("endDate", event.get("endDate")),
            "closed": market.get("closed", event.get("closed")),
            "accepting_orders": market.get("acceptingOrders"),
        }

    def _crypto_5m_resolution_rows(
        self,
        series_slug: str,
        event: Dict[str, Any],
        market: Dict[str, Any],
        resolved_at: str,
    ) -> List[Dict[str, Any]]:
        outcome_prices = self._loads_json_list(market.get("outcomePrices"))
        outcomes = self._loads_json_list(market.get("outcomes"))
        rows = []
        for token in self._extract_contracts(market):
            token_id = token["token_id"]
            if not token_id:
                continue

            outcome_name = token["outcome_name"]
            outcome_price = self._lookup_outcome_price(outcomes, outcome_prices, outcome_name)
            rows.append(
                {
                    "series_slug": series_slug,
                    "asset": "BTC" if series_slug.startswith("btc") else "ETH",
                    "event_id": event.get("id"),
                    "event_slug": event.get("slug"),
                    "event_title": event.get("title"),
                    "market_id": market.get("id"),
                    "condition_id": market.get("conditionId", market.get("condition_id")),
                    "token_id": token_id,
                    "outcome_name": outcome_name,
                    "market_start_time": market.get("eventStartTime", event.get("startTime")),
                    "market_end_time": market.get("endDate", event.get("endDate")),
                    "closed": market.get("closed", event.get("closed")),
                    "outcome_price": outcome_price,
                    "is_winner": self._infer_winner(outcome_price),
                    "resolved_at": resolved_at,
                }
            )
        return rows

    def _orderbook_level_rows(
        self,
        token_row: Dict[str, Any],
        book: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        rows = []
        book_timestamp = self._parse_clob_timestamp(book.get("timestamp"))
        for side in ("bids", "asks"):
            orders = book.get(side, [])
            for level, order in enumerate(orders, start=1):
                price = float(order["price"])
                size = float(order["size"])
                rows.append(
                    {
                        **token_row,
                        "book_timestamp": book_timestamp,
                        "book_hash": book.get("hash"),
                        "side": "bid" if side == "bids" else "ask",
                        "level": level,
                        "price": price,
                        "size": size,
                        "notional": price * size,
                    }
                )
        return rows

    def _lookup_outcome_price(
        self,
        outcomes: List[Any],
        outcome_prices: List[Any],
        outcome_name: str,
    ) -> Optional[float]:
        for raw_outcome, raw_price in zip(outcomes, outcome_prices):
            if str(raw_outcome) == outcome_name:
                return float(raw_price)
        return None

    def _infer_winner(self, outcome_price: Optional[float]) -> Optional[int]:
        if outcome_price is None:
            return None
        if outcome_price >= 0.999:
            return 1
        if outcome_price <= 0.001:
            return 0
        return None

    def _summarize_orderbook(self, token_row: Dict[str, str], book: Dict[str, Any]) -> Dict[str, Any]:
        bids = book.get("bids", [])
        asks = book.get("asks", [])
        best_bid = self._best_price(bids, max)
        best_ask = self._best_price(asks, min)
        timestamp = self._parse_clob_timestamp(book.get("timestamp"))

        return {
            **token_row,
            "book_timestamp": timestamp,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": (best_ask - best_bid) if best_bid is not None and best_ask is not None else None,
            "mid_price": ((best_bid + best_ask) / 2) if best_bid is not None and best_ask is not None else None,
            "bid_depth": self._sum_sizes(bids),
            "ask_depth": self._sum_sizes(asks),
            "bid_depth_top_5": self._sum_sizes(bids[:5]),
            "ask_depth_top_5": self._sum_sizes(asks[:5]),
            "orderbook_imbalance": self._orderbook_imbalance(bids, asks),
            "bid_levels": len(bids),
            "ask_levels": len(asks),
            "book_hash": book.get("hash"),
        }

    def _best_price(self, orders: List[Dict[str, Any]], reducer) -> Optional[float]:
        prices = [float(order["price"]) for order in orders if "price" in order]
        return reducer(prices) if prices else None

    def _sum_sizes(self, orders: List[Dict[str, Any]]) -> float:
        return sum(float(order.get("size", 0.0)) for order in orders)

    def _orderbook_imbalance(
        self,
        bids: List[Dict[str, Any]],
        asks: List[Dict[str, Any]],
    ) -> Optional[float]:
        bid_depth = self._sum_sizes(bids)
        ask_depth = self._sum_sizes(asks)
        total_depth = bid_depth + ask_depth
        if total_depth == 0:
            return None
        return (bid_depth - ask_depth) / total_depth

    def _parse_clob_timestamp(self, timestamp: Any) -> str:
        if timestamp is None:
            return datetime.now(timezone.utc).isoformat()
        ts = int(timestamp)
        if ts > 10_000_000_000:
            ts = ts / 1000
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
