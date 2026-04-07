from datetime import datetime, timezone
from polymarket_quant.schemas.market import Order, OrderBook

def test_orderbook_schema():
    ob = OrderBook(
        market_id="test_mkt",
        timestamp=datetime.now(timezone.utc),
        bids=[Order(price=0.45, size=100.0)],
        asks=[Order(price=0.48, size=50.0)]
    )
    assert len(ob.bids) == 1
    assert ob.bids[0].price == 0.45