from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from scripts.replay_pricing import replay_pricing
from polymarket_quant.pricing import (
    DEFAULT_MANUAL_SPOT_JUMP_INTENSITY_PER_SECOND,
    DEFAULT_MANUAL_SPOT_JUMP_STD_MULTIPLIER_ON_LOCAL_SIGMA,
)
from polymarket_quant.state import (
    LatentMarkovStateBuilder,
    LatentMarkovStateConfig,
    build_event_state_dataset,
    build_market_state_dataset,
)


class _StubTransitionBundle:
    default_step_seconds = 15.0
    spot_mu_model = object()
    spot_sigma_model = object()

    def predict_spot_kernel_from_event_state(self, event_state_row):
        return {
            "mu_hat_log_spot_ratio": 0.0,
            "sigma_hat_log_spot_ratio": float(event_state_row.get("volatility_per_sqrt_second", 0.0005)),
        }


class _ShiftedStubTransitionBundle(_StubTransitionBundle):
    def __init__(self, sigma: float) -> None:
        self._sigma = sigma

    def predict_spot_kernel_from_event_state(self, event_state_row):
        return {
            "mu_hat_log_spot_ratio": 0.0,
            "sigma_hat_log_spot_ratio": self._sigma,
        }


def _orderbook_row(
    timestamp,
    event_slug: str,
    outcome_name: str,
    token_id: str,
    best_bid: float,
    best_ask: float,
    *,
    asset: str = "BTC",
):
    event_start = datetime.fromtimestamp(1775578800, tz=timezone.utc)
    event_end = event_start + timedelta(minutes=5)
    asset_key = str(asset).upper()
    return {
        "series_slug": f"{asset_key.lower()}-up-or-down-5m",
        "asset": asset_key,
        "event_id": "event_1",
        "event_slug": event_slug,
        "event_title": f"{asset_key} Up or Down - Test",
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
        "bid_depth": 100.0,
        "ask_depth": 100.0,
        "bid_depth_top_5": 100.0,
        "ask_depth_top_5": 100.0,
        "orderbook_imbalance": 0.0,
        "bid_levels": 1,
        "ask_levels": 1,
        "book_hash": "hash",
    }


def _spot_row(timestamp, price: float, *, asset: str = "BTC"):
    asset_key = str(asset).upper()
    return {
        "asset": asset_key,
        "product_id": f"{asset_key}USDT",
        "source": "binance_book_ticker",
        "collected_at": timestamp.isoformat(),
        "exchange_time": timestamp.isoformat(),
        "price": price,
        "bid": price - 0.01,
        "ask": price + 0.01,
        "size": 1.0,
        "volume": 100.0,
        "trade_id": 1,
    }


def _write_event_state_input(input_dir, *, event_slug: str, start: datetime, asset: str = "BTC") -> None:
    ts_1 = start + timedelta(seconds=1)
    ts_2 = start + timedelta(seconds=2)

    orderbooks = pd.DataFrame(
        [
            _orderbook_row(ts_1, event_slug, "Up", "tok_up", 0.50, 0.51, asset=asset),
            _orderbook_row(ts_1, event_slug, "Down", "tok_down", 0.48, 0.49, asset=asset),
            _orderbook_row(ts_2, event_slug, "Up", "tok_up", 0.52, 0.53, asset=asset),
            _orderbook_row(ts_2, event_slug, "Down", "tok_down", 0.46, 0.47, asset=asset),
        ]
    )
    spot = pd.DataFrame(
        [
            _spot_row(start, 100.0, asset=asset),
            _spot_row(ts_1, 101.0, asset=asset),
            _spot_row(ts_2, 101.5, asset=asset),
        ]
    )

    state_builder = LatentMarkovStateBuilder(
        LatentMarkovStateConfig(
            fallback_volatility_per_sqrt_second=0.0005,
            event_duration_seconds=300.0,
        )
    )
    market_state = build_market_state_dataset(
        orderbooks=orderbooks,
        spot=spot,
        state_builder=state_builder,
        spot_tolerance_seconds=2.0,
        event_duration_seconds=300.0,
    )
    event_state = build_event_state_dataset(market_state)
    asset = str(event_state["asset"].iloc[0]).upper()
    shard_dir = input_dir / asset / "processed" / "event_state" / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    event_state.to_parquet(shard_dir / f"{event_slug}.parquet", index=False)


def test_replay_pricing_generates_edge_rows(tmp_path):
    event_slug = "btc-updown-5m-1775578800"
    start = datetime.fromtimestamp(1775578800, tz=timezone.utc)

    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "processed"
    input_dir.mkdir()
    _write_event_state_input(input_dir, event_slug=event_slug, start=start)

    result = replay_pricing(
        event_state_glob=str(input_dir / "*" / "processed" / "event_state" / "shards" / "*.parquet"),
        output_dir=str(tmp_path / "out"),
        n_samples=200,
        transition_bundle=_StubTransitionBundle(),
    )

    replay_path = tmp_path / "out" / "BTC" / "processed" / "pricing_replay" / "shards" / f"{event_slug}.parquet"
    replay = pd.read_parquet(replay_path)

    assert result["rows"] == 4
    assert result["shard_paths"][event_slug] == str(replay_path)
    assert set(replay["outcome_name"]) == {"Up", "Down"}
    assert replay["fair_token_price"].between(0.0, 1.0).all()
    assert "brier_component" not in replay.columns
    assert "is_winner" not in replay.columns


def test_replay_pricing_keeps_spot_terminal_payoff_mode_with_step_override(tmp_path):
    event_slug = "btc-updown-5m-1775578800"
    start = datetime.fromtimestamp(1775578800, tz=timezone.utc)

    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "processed"
    input_dir.mkdir()
    _write_event_state_input(input_dir, event_slug=event_slug, start=start)

    replay_pricing(
        event_state_glob=str(input_dir / "*" / "processed" / "event_state" / "shards" / "*.parquet"),
        output_dir=str(tmp_path / "out"),
        n_samples=128,
        rollout_horizon_seconds=15.0,
        transition_bundle=_StubTransitionBundle(),
    )

    replay = pd.read_parquet(tmp_path / "out" / "BTC" / "processed" / "pricing_replay" / "shards" / f"{event_slug}.parquet")
    assert set(replay["simulation_mode"]) == {"spot_terminal_binary_payoff_rollout"}
    assert set(replay["rollout_kernel"]) == {"spot_jump_diffusion"}
    assert np.allclose(replay["rollout_horizon_seconds"], replay["seconds_to_end"])
    assert replay["simulation_step_seconds"].eq(15.0).all()


def test_replay_pricing_defaults_to_selected_manual_jump_prior(tmp_path):
    event_slug = "btc-updown-5m-1775578800"
    start = datetime.fromtimestamp(1775578800, tz=timezone.utc)

    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "processed"
    input_dir.mkdir()
    _write_event_state_input(input_dir, event_slug=event_slug, start=start)

    replay_pricing(
        event_state_glob=str(input_dir / "*" / "processed" / "event_state" / "shards" / "*.parquet"),
        output_dir=str(tmp_path / "out"),
        n_samples=64,
        transition_bundle=_StubTransitionBundle(),
    )

    replay = pd.read_parquet(tmp_path / "out" / "BTC" / "processed" / "pricing_replay" / "shards" / f"{event_slug}.parquet")
    up = replay[replay["outcome_name"] == "Up"].reset_index(drop=True)

    assert not up.empty
    assert up["conditioned_spot_jump_intensity_per_second"].iloc[0] == pytest.approx(
        DEFAULT_MANUAL_SPOT_JUMP_INTENSITY_PER_SECOND
    )
    assert up["conditioned_spot_jump_log_return_std"].iloc[0] == pytest.approx(
        DEFAULT_MANUAL_SPOT_JUMP_STD_MULTIPLIER_ON_LOCAL_SIGMA * 0.0005
    )


def test_replay_pricing_routes_assets_to_separate_transition_bundles(tmp_path):
    start = datetime.fromtimestamp(1775578800, tz=timezone.utc)
    btc_event = "btc-updown-5m-1775578800"
    eth_event = "eth-updown-5m-1775579100"

    input_dir = tmp_path / "inputs"
    _write_event_state_input(input_dir, event_slug=btc_event, start=start, asset="BTC")
    _write_event_state_input(input_dir, event_slug=eth_event, start=start, asset="ETH")

    replay_pricing(
        event_state_glob=str(input_dir / "*" / "processed" / "event_state" / "shards" / "*.parquet"),
        output_dir=str(tmp_path / "out"),
        n_samples=64,
        transition_bundles_by_asset={
            "BTC": _ShiftedStubTransitionBundle(0.0005),
            "ETH": _ShiftedStubTransitionBundle(0.0050),
        },
    )

    btc_replay = pd.read_parquet(tmp_path / "out" / "BTC" / "processed" / "pricing_replay" / "shards" / f"{btc_event}.parquet")
    eth_replay = pd.read_parquet(tmp_path / "out" / "ETH" / "processed" / "pricing_replay" / "shards" / f"{eth_event}.parquet")
    assert btc_replay["conditioned_spot_volatility_per_sqrt_second"].iloc[0] == pytest.approx(0.0005)
    assert eth_replay["conditioned_spot_volatility_per_sqrt_second"].iloc[0] == pytest.approx(0.0050)
