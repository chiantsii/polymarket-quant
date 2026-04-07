from datetime import datetime, timedelta, timezone

from polymarket_quant.signals.mispricing import MispricingDetectorConfig, RealTimeMispricingDetector


def _summary_row(outcome_name: str, best_bid: float, best_ask: float) -> dict:
    now = datetime.now(timezone.utc)
    return {
        "series_slug": "btc-up-or-down-5m",
        "asset": "BTC",
        "event_id": "event_1",
        "event_slug": "btc-updown-5m-test",
        "event_title": "Bitcoin Up or Down - Test",
        "market_id": "mkt_1",
        "condition_id": "cond_1",
        "token_id": f"tok_{outcome_name.lower()}",
        "outcome_name": outcome_name,
        "market_start_time": (now - timedelta(minutes=1)).isoformat(),
        "market_end_time": (now + timedelta(minutes=4)).isoformat(),
        "closed": False,
        "accepting_orders": True,
        "collected_at": now.isoformat(),
        "book_timestamp": now.isoformat(),
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": best_ask - best_bid,
        "mid_price": (best_bid + best_ask) / 2,
        "bid_depth": 100.0,
        "ask_depth": 100.0,
        "bid_depth_top_5": 100.0,
        "ask_depth_top_5": 100.0,
        "orderbook_imbalance": 0.0,
        "bid_levels": 1,
        "ask_levels": 1,
        "book_hash": "hash",
    }


def test_mispricing_detector_flags_underpriced_up_token() -> None:
    detector = RealTimeMispricingDetector(
        MispricingDetectorConfig(
            min_edge=0.05,
            fallback_volatility_per_sqrt_second=0.0005,
            n_samples=2_000,
            use_particle_filter=False,
            seed=42,
        )
    )

    spot_ticks = {
        "BTC": {
            "asset": "BTC",
            "product_id": "BTC-USD",
            "source": "coinbase",
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "exchange_time": datetime.now(timezone.utc).isoformat(),
            "price": 101.0,
            "bid": 100.9,
            "ask": 101.1,
        }
    }
    rows = [
        _summary_row("Up", best_bid=0.50, best_ask=0.60),
        _summary_row("Down", best_bid=0.39, best_ask=0.40),
    ]
    reference_prices = {
        "btc-updown-5m-test": {
            "price": 100.0,
            "source": "coinbase_1m_candle_open",
        }
    }

    signals = detector.detect(rows, spot_ticks, reference_prices_by_event=reference_prices)

    assert len(signals) == 2
    up_signal = next(row for row in signals if row["outcome_name"] == "Up")
    down_signal = next(row for row in signals if row["outcome_name"] == "Down")
    assert up_signal["signal"] == "BUY"
    assert up_signal["buy_edge"] > 0.05
    assert up_signal["reference_source"] == "coinbase_1m_candle_open"
    assert 0.0 <= up_signal["fair_up_probability"] <= 1.0
    assert down_signal["signal"] in {"HOLD", "SELL"}
    assert down_signal["fair_token_price"] == 1.0 - up_signal["fair_up_probability"]


def test_mispricing_detector_holds_when_toxicity_is_too_high() -> None:
    detector = RealTimeMispricingDetector(
        MispricingDetectorConfig(
            min_edge=0.01,
            max_toxicity=0.0,
            n_samples=1_000,
            use_particle_filter=False,
            seed=42,
        )
    )
    detector.reference_spot_prices["btc-updown-5m-test"] = 100.0

    now = datetime.now(timezone.utc)
    spot_ticks = {
        "BTC": {
            "asset": "BTC",
            "product_id": "BTC-USD",
            "source": "coinbase",
            "collected_at": now.isoformat(),
            "price": 101.0,
        }
    }
    rows = [_summary_row("Up", best_bid=0.50, best_ask=0.60)]

    detector.detect(rows, spot_ticks)
    spot_ticks["BTC"]["collected_at"] = (now + timedelta(seconds=2)).isoformat()
    spot_ticks["BTC"]["price"] = 103.0
    signals = detector.detect(rows, spot_ticks)

    assert signals[0]["toxicity_score"] > 0.0
    assert signals[0]["signal"] == "HOLD"
