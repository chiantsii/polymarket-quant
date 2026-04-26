import abc
import requests
from typing import Dict, Any, List

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

class PolymarketRESTClient(BasePolymarketClient):
    """
    Adapter for Polymarket's Gamma (Events/Markets) and CLOB APIs.
    TODO: Integrate official auth signatures for authenticated CLOB endpoints.
    """
    def __init__(self, gamma_url: str, clob_url: str):
        self.gamma_url = gamma_url
        self.clob_url = clob_url
        self.session = requests.Session()

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
