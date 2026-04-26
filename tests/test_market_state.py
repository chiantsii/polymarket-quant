import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from polymarket_quant.state import (
    LatentMarkovStateBuilder,
    LatentMarkovStateConfig,
    build_event_state_dataset,
    build_market_state_dataset,
    load_orderbook_raw_glob,
    load_spot_raw_glob,
)


def _orderbook_row(timestamp, event_slug: str, outcome_name: str, token_id: str, best_bid: float, best_ask: float):
    event_start = datetime.fromtimestamp(1775578800, tz=timezone.utc)
    event_end = event_start + timedelta(minutes=5)
    return {
        "series_slug": "btc-up-or-down-5m",
        "asset": "BTC",
        "event_id": "event_1",
        "event_slug": event_slug,
        "event_title": "Bitcoin Up or Down - Test",
        "market_id": "mkt_1",
        "condition_id": "cond_1",
        "token_id": token_id,
        "outcome_name": outcome_name,
        "market_start_time": event_start.isoformat(),
        "market_end_time": event_end.isoformat(),
        "closed": False,
        "accepting_orders": True,
        "collected_at": timestamp.isoformat(),
        "book_timestamp": timestamp.isoformat(),
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": best_ask - best_bid,
        "mid_price": (best_bid + best_ask) / 2,
        "bid_depth": 120.0,
        "ask_depth": 110.0,
        "bid_depth_top_5": 90.0,
        "ask_depth_top_5": 85.0,
        "orderbook_imbalance": 0.05,
        "bid_levels": 2,
        "ask_levels": 2,
        "book_hash": "hash",
    }


def _spot_row(timestamp, price: float):
    return {
        "asset": "BTC",
        "product_id": "BTC-USD",
        "source": "coinbase",
        "collected_at": timestamp.isoformat(),
        "exchange_time": timestamp.isoformat(),
        "price": price,
        "bid": price - 0.01,
        "ask": price + 0.01,
        "size": 1.0,
        "volume": 100.0,
        "trade_id": 1,
    }


def _orderbook_levels_rows(timestamp, event_slug: str, outcome_name: str, token_id: str):
    rows = []
    bid_prices = [0.50, 0.49, 0.48]
    ask_prices = [0.51, 0.52, 0.54]
    bid_sizes = [60.0, 40.0, 30.0]
    ask_sizes = [55.0, 35.0, 20.0]
    for level, (price, size) in enumerate(zip(bid_prices, bid_sizes), start=1):
        rows.append(
            {
                "event_slug": event_slug,
                "collected_at": timestamp.isoformat(),
                "token_id": token_id,
                "outcome_name": outcome_name,
                "side": "bid",
                "level": level,
                "price": price,
                "size": size,
            }
        )
    for level, (price, size) in enumerate(zip(ask_prices, ask_sizes), start=1):
        rows.append(
            {
                "event_slug": event_slug,
                "collected_at": timestamp.isoformat(),
                "token_id": token_id,
                "outcome_name": outcome_name,
                "side": "ask",
                "level": level,
                "price": price,
                "size": size,
            }
        )
    return rows


def test_build_market_state_dataset_adds_latent_and_observation_columns() -> None:
    event_slug = "btc-updown-5m-1775578800"
    start = datetime.fromtimestamp(1775578800, tz=timezone.utc)
    ts_1 = start + timedelta(seconds=1)
    ts_2 = start + timedelta(seconds=2)

    orderbooks = pd.DataFrame(
        [
            _orderbook_row(ts_1, event_slug, "Up", "tok_up", 0.50, 0.51),
            _orderbook_row(ts_1, event_slug, "Down", "tok_down", 0.48, 0.49),
            _orderbook_row(ts_2, event_slug, "Up", "tok_up", 0.52, 0.53),
            _orderbook_row(ts_2, event_slug, "Down", "tok_down", 0.46, 0.47),
        ]
    )
    spot = pd.DataFrame(
        [
            _spot_row(start, 100.0),
            _spot_row(ts_1, 101.0),
            _spot_row(ts_2, 101.5),
        ]
    )
    orderbook_levels = pd.DataFrame(
        _orderbook_levels_rows(ts_1, event_slug, "Up", "tok_up")
        + _orderbook_levels_rows(ts_1, event_slug, "Down", "tok_down")
        + _orderbook_levels_rows(ts_2, event_slug, "Up", "tok_up")
        + _orderbook_levels_rows(ts_2, event_slug, "Down", "tok_down")
    )
    state = build_market_state_dataset(
        orderbooks=orderbooks,
        spot=spot,
        orderbook_levels=orderbook_levels,
        state_builder=LatentMarkovStateBuilder(LatentMarkovStateConfig()),
        spot_tolerance_seconds=2.0,
        event_duration_seconds=300.0,
    )

    assert len(state) == 4
    assert state["market_implied_up_probability"].between(0.0, 1.0).all()
    assert state["fundamental_up_probability"].between(0.0, 1.0).all()
    assert state["latent_up_probability"].between(0.0, 1.0).all()
    assert state["state_observation_variance"].gt(0.0).all()
    assert state["spot_price"].notna().all()
    assert state["micro_price"].notna().all()
    assert state["weighted_imbalance"].notna().all()
    assert state["depth_slope"].notna().all()
    assert state["tick_density"].notna().all()
    assert "spread_native" not in state.columns
    assert "relative_spread" not in state.columns
    assert state["market_fundamental_basis"].notna().all()
    assert state["latent_market_basis"].notna().all()
    assert state["latent_fundamental_basis"].notna().all()
    assert state["normalized_time_to_end"].between(0.0, 1.0).all()
    assert "book_velocity" in state.columns
    assert "toxicity_score" not in state.columns
    assert "is_winner" not in state.columns
    assert "outcome_price" not in state.columns


def test_build_market_state_dataset_handles_single_sided_quotes_with_nan_mid_price() -> None:
    event_slug = "btc-updown-5m-1775578800"
    start = datetime.fromtimestamp(1775578800, tz=timezone.utc)
    ts_1 = start + timedelta(seconds=1)

    up_row = _orderbook_row(ts_1, event_slug, "Up", "tok_up", 0.02, 0.03)
    down_row = _orderbook_row(ts_1, event_slug, "Down", "tok_down", 0.97, 0.98)
    up_row.update(
        {
            "best_bid": float("nan"),
            "best_ask": 0.01,
            "spread": float("nan"),
            "mid_price": float("nan"),
            "bid_depth": 0.0,
            "ask_depth": 100.0,
            "bid_depth_top_5": 0.0,
            "ask_depth_top_5": 80.0,
            "orderbook_imbalance": -1.0,
            "bid_levels": 0,
            "ask_levels": 99,
        }
    )
    down_row.update(
        {
            "best_bid": 0.99,
            "best_ask": float("nan"),
            "spread": float("nan"),
            "mid_price": float("nan"),
            "bid_depth": 100.0,
            "ask_depth": 0.0,
            "bid_depth_top_5": 80.0,
            "ask_depth_top_5": 0.0,
            "orderbook_imbalance": 1.0,
            "bid_levels": 99,
            "ask_levels": 0,
        }
    )
    orderbooks = pd.DataFrame([up_row, down_row])
    spot = pd.DataFrame([_spot_row(start, 100.0), _spot_row(ts_1, 99.9)])

    state = build_market_state_dataset(
        orderbooks=orderbooks,
        spot=spot,
        state_builder=LatentMarkovStateBuilder(LatentMarkovStateConfig()),
        spot_tolerance_seconds=2.0,
        event_duration_seconds=300.0,
    )

    assert len(state) == 2
    assert state["market_implied_up_probability"].notna().all()
    assert state["market_implied_up_probability"].between(0.0, 1.0).all()
    assert state["latent_up_probability"].notna().all()


def test_load_orderbook_and_spot_raw_glob_builds_frames(tmp_path) -> None:
    start = datetime.fromtimestamp(1775578800, tz=timezone.utc)
    end = start + timedelta(minutes=5)
    orderbook_payload = [
        {
            "series_slug": "btc-up-or-down-5m",
            "asset": "BTC",
            "event_id": "event_1",
            "event_slug": "btc-updown-5m-1775578800",
            "event_title": "Bitcoin Up or Down - Test",
            "market_id": "mkt_1",
            "condition_id": "cond_1",
            "token_id": "tok_up",
            "outcome_name": "Up",
            "market_start_time": start.isoformat(),
            "market_end_time": end.isoformat(),
            "closed": False,
            "accepting_orders": True,
            "collected_at": (start + timedelta(seconds=1)).isoformat(),
            "orderbook": {
                "timestamp": str(int((start + timedelta(seconds=1)).timestamp() * 1000)),
                "hash": "hash_1",
                "bids": [{"price": "0.50", "size": "100"}],
                "asks": [{"price": "0.51", "size": "90"}],
            },
        }
    ]
    spot_payload = [
        {
            "asset": "BTC",
            "product_id": "BTC-USD",
            "source": "coinbase",
            "collected_at": (start + timedelta(seconds=1)).isoformat(),
            "exchange_time": (start + timedelta(seconds=1)).isoformat(),
            "price": 101.0,
            "bid": 100.99,
            "ask": 101.01,
            "size": 1.0,
            "volume": 100.0,
            "trade_id": 1,
        }
    ]

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "crypto_5m_orderbooks_raw_20260408_173000.json").write_text(json.dumps(orderbook_payload))
    (raw_dir / "crypto_spot_ticks_raw_20260408_173000.json").write_text(json.dumps(spot_payload))

    orderbooks = load_orderbook_raw_glob(str(raw_dir / "crypto_5m_orderbooks_raw_*.json"))
    spot = load_spot_raw_glob(str(raw_dir / "crypto_spot_ticks_raw_*.json"))

    assert len(orderbooks) == 1
    assert len(spot) == 1
    assert orderbooks.loc[0, "best_bid"] == 0.50
    assert orderbooks.loc[0, "best_ask"] == 0.51
    assert orderbooks.loc[0, "bid_depth_top_5"] == 100.0
    assert orderbooks.loc[0, "ask_depth_top_5"] == 90.0
    assert spot.loc[0, "asset"] == "BTC"


def test_build_event_state_dataset_collapses_up_and_down_rows() -> None:
    event_slug = "btc-updown-5m-1775578800"
    start = datetime.fromtimestamp(1775578800, tz=timezone.utc)
    ts_1 = start + timedelta(seconds=1)
    ts_2 = start + timedelta(seconds=2)

    market_state = pd.DataFrame(
        [
            {
                **_orderbook_row(ts_1, event_slug, "Up", "tok_up", 0.50, 0.51),
                "spot_price": 101.0,
                "reference_spot_price": 100.0,
                "seconds_to_end": 299.0,
                "market_implied_up_probability": 0.51,
                "fundamental_up_probability": 0.52,
                "latent_up_probability": 0.518,
                "latent_logit_probability": 0.01,
                "state_observation_variance": 0.02,
                "micro_price": 0.5054,
                "weighted_imbalance": 0.07,
                "depth_slope": 12.0,
                "tick_density": 0.8,
                "book_velocity": 0.0,
                "state_timestamp": ts_1.isoformat(),
                "book_age_seconds": 0.0,
                "spot_source": "coinbase",
                "spot_product_id": "BTC-USD",
                "spot_exchange_time": ts_1.isoformat(),
                "spot_bid": 100.99,
                "spot_ask": 101.01,
                "reference_source": "test",
                "spot_return_since_reference": 0.01,
                "volatility_per_sqrt_second": 0.001,
            },
            {
                **_orderbook_row(ts_1, event_slug, "Down", "tok_down", 0.48, 0.49),
                "spot_price": 101.0,
                "reference_spot_price": 100.0,
                "seconds_to_end": 299.0,
                "market_implied_up_probability": 0.51,
                "fundamental_up_probability": 0.52,
                "latent_up_probability": 0.518,
                "latent_logit_probability": 0.01,
                "state_observation_variance": 0.02,
                "micro_price": 0.4846,
                "weighted_imbalance": -0.06,
                "depth_slope": 11.0,
                "tick_density": 0.75,
                "book_velocity": 0.0,
                "state_timestamp": ts_1.isoformat(),
                "book_age_seconds": 0.0,
                "spot_source": "coinbase",
                "spot_product_id": "BTC-USD",
                "spot_exchange_time": ts_1.isoformat(),
                "spot_bid": 100.99,
                "spot_ask": 101.01,
                "reference_source": "test",
                "spot_return_since_reference": 0.01,
                "volatility_per_sqrt_second": 0.001,
            },
            {
                **_orderbook_row(ts_2, event_slug, "Up", "tok_up", 0.52, 0.53),
                "spot_price": 101.5,
                "reference_spot_price": 100.0,
                "seconds_to_end": 298.0,
                "market_implied_up_probability": 0.53,
                "fundamental_up_probability": 0.54,
                "latent_up_probability": 0.538,
                "latent_logit_probability": 0.03,
                "state_observation_variance": 0.021,
                "micro_price": 0.5253,
                "weighted_imbalance": 0.08,
                "depth_slope": 13.0,
                "tick_density": 0.82,
                "book_velocity": 0.2,
                "state_timestamp": ts_2.isoformat(),
                "book_age_seconds": 0.0,
                "spot_source": "coinbase",
                "spot_product_id": "BTC-USD",
                "spot_exchange_time": ts_2.isoformat(),
                "spot_bid": 101.49,
                "spot_ask": 101.51,
                "reference_source": "test",
                "spot_return_since_reference": 0.015,
                "volatility_per_sqrt_second": 0.0012,
            },
            {
                **_orderbook_row(ts_2, event_slug, "Down", "tok_down", 0.46, 0.47),
                "spot_price": 101.5,
                "reference_spot_price": 100.0,
                "seconds_to_end": 298.0,
                "market_implied_up_probability": 0.53,
                "fundamental_up_probability": 0.54,
                "latent_up_probability": 0.538,
                "latent_logit_probability": 0.03,
                "state_observation_variance": 0.021,
                "micro_price": 0.4647,
                "weighted_imbalance": -0.07,
                "depth_slope": 10.5,
                "tick_density": 0.78,
                "book_velocity": 0.18,
                "state_timestamp": ts_2.isoformat(),
                "book_age_seconds": 0.0,
                "spot_source": "coinbase",
                "spot_product_id": "BTC-USD",
                "spot_exchange_time": ts_2.isoformat(),
                "spot_bid": 101.49,
                "spot_ask": 101.51,
                "reference_source": "test",
                "spot_return_since_reference": 0.015,
                "volatility_per_sqrt_second": 0.0012,
            },
        ]
    )

    event_state = build_event_state_dataset(market_state)

    assert len(event_state) == 2
    assert event_state["latent_up_probability"].between(0.0, 1.0).all()
    assert event_state["up_token_id"].eq("tok_up").all()
    assert event_state["down_token_id"].eq("tok_down").all()
    assert event_state["up_best_bid"].tolist() == [0.50, 0.52]
    assert event_state["down_best_bid"].tolist() == [0.48, 0.46]
    assert event_state["has_full_book_pair"].all()
    assert event_state["cross_book_basis"].notna().all()
    assert event_state["cross_book_bid_basis"].notna().all()
    assert event_state["cross_book_ask_basis"].notna().all()
    assert event_state["spread_divergence"].notna().all()
    assert event_state["dist_to_boundary"].notna().all()
    assert event_state["book_age_max"].notna().all()
    assert event_state["external_spot_drift"].notna().all()
    assert event_state["market_fundamental_basis"].notna().all()
    assert event_state["latent_market_basis"].notna().all()
    assert event_state["latent_fundamental_basis"].notna().all()
    assert event_state["normalized_time_to_end"].between(0.0, 1.0).all()
    assert event_state["regime_normal_posterior"].between(0.0, 1.0).all()
    assert event_state["regime_shock_posterior"].between(0.0, 1.0).all()
    assert event_state["regime_convergence_posterior"].between(0.0, 1.0).all()
    posterior_sum = (
        event_state["regime_normal_posterior"]
        + event_state["regime_shock_posterior"]
        + event_state["regime_convergence_posterior"]
    )
    assert posterior_sum.round(8).eq(1.0).all()
    assert "toxicity_score" not in event_state.columns
    assert "up_is_winner" not in event_state.columns
    assert "down_is_winner" not in event_state.columns


def test_build_market_state_dataset_skips_rows_without_trustworthy_timestamp() -> None:
    event_slug = "btc-updown-5m-1775578800"
    start = datetime.fromtimestamp(1775578800, tz=timezone.utc)

    orderbooks = pd.DataFrame(
        [
            {
                **_orderbook_row(start + timedelta(seconds=1), event_slug, "Up", "tok_up", 0.50, 0.51),
                "collected_at": None,
                "book_timestamp": None,
            },
            {
                **_orderbook_row(start + timedelta(seconds=1), event_slug, "Down", "tok_down", 0.48, 0.49),
                "collected_at": None,
                "book_timestamp": None,
            },
        ]
    )
    spot = pd.DataFrame([_spot_row(start + timedelta(seconds=1), 101.0)])

    with pytest.raises(ValueError, match="No market-state rows were generated"):
        build_market_state_dataset(
            orderbooks=orderbooks,
            spot=spot,
            state_builder=LatentMarkovStateBuilder(LatentMarkovStateConfig()),
            spot_tolerance_seconds=2.0,
            event_duration_seconds=300.0,
        )


def test_fundamental_probability_uses_gbm_terminal_crossing_probability() -> None:
    builder = LatentMarkovStateBuilder(LatentMarkovStateConfig())

    at_the_money = builder._spot_distance_probability(
        spot_price=100.0,
        reference_price=100.0,
        volatility=0.01,
        seconds_to_end=300.0,
    )
    in_the_money = builder._spot_distance_probability(
        spot_price=101.0,
        reference_price=100.0,
        volatility=0.01,
        seconds_to_end=300.0,
    )
    out_of_the_money = builder._spot_distance_probability(
        spot_price=99.0,
        reference_price=100.0,
        volatility=0.01,
        seconds_to_end=300.0,
    )

    assert 0.0 < at_the_money < 0.5
    assert in_the_money > at_the_money
    assert out_of_the_money < at_the_money


def test_fundamental_probability_handles_degenerate_terminal_paths() -> None:
    builder = LatentMarkovStateBuilder(LatentMarkovStateConfig())

    assert (
        builder._spot_distance_probability(
            spot_price=100.0,
            reference_price=100.0,
            volatility=0.0,
            seconds_to_end=300.0,
        )
        == 1.0
    )
    assert (
        builder._spot_distance_probability(
            spot_price=99.0,
            reference_price=100.0,
            volatility=0.0,
            seconds_to_end=300.0,
        )
        == 0.0
    )


def test_observation_variance_increases_with_quote_width_and_cross_side_disagreement() -> None:
    builder = LatentMarkovStateBuilder(LatentMarkovStateConfig())
    timestamp = datetime.fromtimestamp(1775578801, tz=timezone.utc)

    tight_consistent_rows = [
        _orderbook_row(timestamp, "evt", "Up", "tok_up", 0.50, 0.51),
        _orderbook_row(timestamp, "evt", "Down", "tok_down", 0.49, 0.50),
    ]
    wide_consistent_rows = [
        _orderbook_row(timestamp, "evt", "Up", "tok_up", 0.45, 0.55),
        _orderbook_row(timestamp, "evt", "Down", "tok_down", 0.45, 0.55),
    ]
    tight_disagreeing_rows = [
        _orderbook_row(timestamp, "evt", "Up", "tok_up", 0.50, 0.51),
        _orderbook_row(timestamp, "evt", "Down", "tok_down", 0.39, 0.40),
    ]

    tight_variance = builder._observation_variance(tight_consistent_rows, timestamp)
    wide_variance = builder._observation_variance(wide_consistent_rows, timestamp)
    disagreeing_variance = builder._observation_variance(tight_disagreeing_rows, timestamp)

    assert wide_variance > tight_variance
    assert disagreeing_variance > tight_variance


def test_observation_variance_penalizes_thin_or_stale_books() -> None:
    builder = LatentMarkovStateBuilder(LatentMarkovStateConfig())
    timestamp = datetime.fromtimestamp(1775578801, tz=timezone.utc)

    fresh_liquid_rows = [
        {
            **_orderbook_row(timestamp, "evt", "Up", "tok_up", 0.50, 0.51),
            "bid_depth_top_5": 5_000.0,
            "ask_depth_top_5": 5_000.0,
        },
        {
            **_orderbook_row(timestamp, "evt", "Down", "tok_down", 0.49, 0.50),
            "bid_depth_top_5": 5_000.0,
            "ask_depth_top_5": 5_000.0,
        },
    ]
    stale_or_thin_rows = [
        {
            **_orderbook_row(timestamp, "evt", "Up", "tok_up", 0.50, 0.51),
            "book_timestamp": (timestamp - timedelta(seconds=4)).isoformat(),
            "bid_depth_top_5": 5.0,
            "ask_depth_top_5": 5.0,
        },
        {
            **_orderbook_row(timestamp, "evt", "Down", "tok_down", 0.49, 0.50),
            "book_timestamp": (timestamp - timedelta(seconds=4)).isoformat(),
            "bid_depth_top_5": 5.0,
            "ask_depth_top_5": 5.0,
        },
    ]

    fresh_liquid_variance = builder._observation_variance(fresh_liquid_rows, timestamp)
    stale_or_thin_variance = builder._observation_variance(stale_or_thin_rows, timestamp)

    assert stale_or_thin_variance > fresh_liquid_variance
    assert (
        builder._spot_distance_probability(
            spot_price=99.0,
            reference_price=100.0,
            volatility=0.01,
            seconds_to_end=0.0,
        )
        == 0.0
    )
