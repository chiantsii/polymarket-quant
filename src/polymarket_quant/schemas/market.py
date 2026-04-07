from pydantic import BaseModel, Field, computed_field
from datetime import datetime, timezone
from typing import List, Optional, Literal

class Contract(BaseModel):
    """Represents a specific outcome asset (e.g., 'Yes' or 'No')."""
    token_id: str
    outcome_name: str
    
class MarketMetadata(BaseModel):
    """Canonical representation of a market's static/semi-static properties."""
    market_id: str = Field(..., alias="id")
    condition_id: str
    title: str
    category: Optional[str] = "Unknown"
    resolution_date: datetime
    contracts: List[Contract] = Field(default_factory=list)
    is_active: bool
    
    @computed_field
    @property
    def time_to_resolution_seconds(self) -> float:
        """Derived feature: Time remaining until scheduled resolution."""
        now = datetime.now(timezone.utc)
        delta = self.resolution_date - now
        return max(0.0, delta.total_seconds())

class Trade(BaseModel):
    """A single executed transaction on the CLOB."""
    market_id: str
    timestamp: datetime
    price: float
    size: float
    side: Literal["BUY", "SELL"]
    maker_address: Optional[str] = None
    taker_address: Optional[str] = None

class Order(BaseModel):
    price: float
    size: float

class OrderbookSnapshot(BaseModel):
    """L2 Book state at a specific discrete timestamp."""
    market_id: str
    timestamp: datetime
    bids: List[Order]
    asks: List[Order]


OrderBook = OrderbookSnapshot

class MarketStateSnapshot(BaseModel):
    """Periodic snapshots of liquidity, volume, and pricing."""
    market_id: str
    timestamp: datetime
    last_price: Optional[float] = None
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    volume_24h_usd: float = 0.0
    open_interest_usd: float = 0.0
    liquidity_usd: float = 0.0

class Resolution(BaseModel):
    """The final resolved state of a market."""
    market_id: str
    timestamp: datetime
    winning_token_id: str
