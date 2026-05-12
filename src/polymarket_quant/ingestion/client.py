import abc
import asyncio
import threading
import requests
from typing import Coroutine
from typing import Dict, Any, List

try:
    import httpx
except ModuleNotFoundError:  # pragma: no cover - optional runtime dependency
    httpx = None  # type: ignore[assignment]

from polymarket_quant.utils.logger import get_logger

logger = get_logger(__name__)

class BasePolymarketClient(abc.ABC):
    """Abstract boundary for fetching raw Polymarket data."""
    def fetch_series(self, slug: str) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def fetch_event_by_slug(self, slug: str) -> Dict[str, Any]:
        raise NotImplementedError

    def fetch_orderbook(self, token_id: str) -> Dict[str, Any]:
        raise NotImplementedError

    def fetch_orderbooks(self, token_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        books: Dict[str, Dict[str, Any]] = {}
        for token_id in token_ids:
            book = self.fetch_orderbook(token_id)
            if book:
                books[token_id] = book
        return books

class PolymarketRESTClient(BasePolymarketClient):
    """
    Adapter for Polymarket's Gamma (Events/Markets) and CLOB APIs.
    TODO: Integrate official auth signatures for authenticated CLOB endpoints.
    """
    def __init__(self, gamma_url: str, clob_url: str):
        self.gamma_url = gamma_url
        self.clob_url = clob_url
        self.session = requests.Session()
        self._async_timeout_seconds = 10.0

    def fetch_series(self, slug: str) -> List[Dict[str, Any]]:
        """Fetch Gamma series metadata, including event slugs."""
        url = f"{self.gamma_url}/series"
        params = {"slug": slug}

        logger.info(f"Fetching series {slug}")
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data if isinstance(data, list) else []
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch series {slug}: {e}")
            return []

    def fetch_event_by_slug(self, slug: str) -> Dict[str, Any]:
        """Fetch a Gamma event detail payload by slug."""
        url = f"{self.gamma_url}/events/slug/{slug}"

        logger.info(f"Fetching event {slug}")
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data if isinstance(data, dict) else {}
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch event {slug}: {e}")
            return {}

    def fetch_orderbook(self, token_id: str) -> Dict[str, Any]:
        """Fetch the current CLOB L2 book for a token id."""
        url = f"{self.clob_url}/book"
        params = {"token_id": token_id}

        logger.info(f"Fetching orderbook for token {token_id}")
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch orderbook for token {token_id}: {e}")
            return {}

    def fetch_orderbooks(self, token_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        unique_token_ids = [str(token_id) for token_id in dict.fromkeys(token_ids) if str(token_id)]
        if not unique_token_ids:
            return {}
        if httpx is None:
            logger.warning("httpx is not installed; falling back to sequential orderbook fetches")
            return super().fetch_orderbooks(unique_token_ids)
        return _run_coro_blocking(self._fetch_orderbooks_async(unique_token_ids))

    async def _fetch_orderbooks_async(self, token_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        async with httpx.AsyncClient(timeout=self._async_timeout_seconds) as client:
            tasks = [self._fetch_single_orderbook_async(client, token_id) for token_id in token_ids]
            results = await asyncio.gather(*tasks)
        books: Dict[str, Dict[str, Any]] = {}
        for token_id, book in results:
            if book:
                books[token_id] = book
        return books

    async def _fetch_single_orderbook_async(
        self,
        client: "httpx.AsyncClient",
        token_id: str,
    ) -> tuple[str, Dict[str, Any]]:
        url = f"{self.clob_url}/book"
        params = {"token_id": token_id}
        logger.info(f"Fetching orderbook for token {token_id}")
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            return token_id, data if isinstance(data, dict) else {}
        except Exception as exc:
            logger.error(f"Failed to fetch orderbook for token {token_id}: {exc}")
            return token_id, {}


def _run_coro_blocking(coro: Coroutine[Any, Any, Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: Dict[str, Dict[str, Any]] = {}
    error: BaseException | None = None

    def _runner() -> None:
        nonlocal result, error
        try:
            result = asyncio.run(coro)
        except BaseException as exc:  # pragma: no cover - defensive thread handoff
            error = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if error is not None:
        raise error
    return result
