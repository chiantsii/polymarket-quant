import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from datetime import datetime, timezone

from polymarket_quant.ingestion.client import BasePolymarketClient
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

    def _parse_clob_timestamp(self, timestamp: Any) -> str | None:
        if timestamp is None:
            return None
        ts = int(timestamp)
        if ts > 10_000_000_000:
            ts = ts / 1000
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
