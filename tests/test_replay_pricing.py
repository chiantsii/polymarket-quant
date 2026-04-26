from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from scripts.replay_pricing import replay_pricing
from polymarket_quant.state import (
    LatentMarkovStateBuilder,
    LatentMarkovStateConfig,
    build_event_state_dataset,
    build_market_state_dataset,
)


class _StubTransitionBundle:
    rollout_feature_columns = (
        "current_latent_up_probability",
        "current_latent_logit_probability",
        "current_seconds_to_end",
        "current_normalized_time_to_end",
        "current_dist_to_boundary",
        "current_boundary_leverage_ratio",
        "current_up_book_velocity",
        "current_down_book_velocity",
        "target_horizon_seconds",
    )
    feature_columns = (
        "current_market_implied_up_probability",
        "current_fundamental_up_probability",
        "current_latent_up_probability",
        "current_latent_logit_probability",
        "current_up_micro_price",
        "current_down_micro_price",
        "current_up_bid_depth_top_5",
        "current_up_ask_depth_top_5",
        "current_down_bid_depth_top_5",
        "current_down_ask_depth_top_5",
        "current_up_weighted_imbalance",
        "current_down_weighted_imbalance",
        "current_cross_book_basis",
        "current_regime_normal_posterior",
        "current_regime_shock_posterior",
        "current_regime_convergence_posterior",
        "current_normalized_time_to_end",
        "target_horizon_seconds",
    )
    primitive_target_columns = (
        "latent_logit_probability",
        "regime_normal_posterior",
        "regime_shock_posterior",
        "regime_convergence_posterior",
        "market_implied_up_probability",
        "up_micro_price",
        "down_micro_price",
        "up_weighted_imbalance",
        "down_weighted_imbalance",
        "up_bid_depth_top_5",
        "up_ask_depth_top_5",
        "down_bid_depth_top_5",
        "down_ask_depth_top_5",
        "cross_book_basis",
    )

    def predict_latent_step(self, rows: pd.DataFrame) -> pd.DataFrame:
        predictions = rows.copy()
        current_latent = pd.to_numeric(predictions["current_latent_logit_probability"], errors="coerce").fillna(0.0)
        predictions["drift_hat_latent_logit_probability"] = 0.02
        predictions["diffusion_hat_latent_logit_probability"] = 1e-4
        predictions["future_hat_latent_logit_probability"] = current_latent + 0.02
        predictions["future_hat_latent_up_probability"] = 1.0 / (
            1.0 + np.exp(-predictions["future_hat_latent_logit_probability"])
        )
        predictions["jump_intensity_hat"] = 0.02
        return predictions

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        predictions = rows.copy()
        current_latent = pd.to_numeric(predictions["current_latent_logit_probability"], errors="coerce").fillna(0.0)
        current_market = pd.to_numeric(predictions["current_market_implied_up_probability"], errors="coerce").fillna(0.5)
        current_up_micro = pd.to_numeric(predictions["current_up_micro_price"], errors="coerce").fillna(current_market)
        current_down_micro = pd.to_numeric(predictions["current_down_micro_price"], errors="coerce").fillna(1.0 - current_market)
        depth = pd.Series(100.0, index=predictions.index, dtype=float)

        predictions["future_hat_latent_logit_probability"] = current_latent + 0.02
        predictions["diffusion_hat_latent_logit_probability"] = 1e-4
        predictions["future_hat_market_implied_up_probability"] = np.clip(current_market + 0.01, 0.0, 1.0)
        predictions["diffusion_hat_market_implied_up_probability"] = 1e-5
        predictions["future_hat_up_micro_price"] = np.clip(current_up_micro + 0.01, 0.0, 1.0)
        predictions["future_hat_down_micro_price"] = np.clip(current_down_micro - 0.01, 0.0, 1.0)
        predictions["diffusion_hat_up_micro_price"] = 1e-5
        predictions["diffusion_hat_down_micro_price"] = 1e-5
        predictions["future_hat_up_weighted_imbalance"] = 0.05
        predictions["future_hat_down_weighted_imbalance"] = -0.05
        predictions["diffusion_hat_up_weighted_imbalance"] = 1e-4
        predictions["diffusion_hat_down_weighted_imbalance"] = 1e-4
        for column in (
            "future_hat_up_bid_depth_top_5",
            "future_hat_up_ask_depth_top_5",
            "future_hat_down_bid_depth_top_5",
            "future_hat_down_ask_depth_top_5",
        ):
            predictions[column] = depth
            predictions[column.replace("future_hat", "diffusion_hat")] = 1e-4
        predictions["future_hat_cross_book_basis"] = predictions["future_hat_up_micro_price"] + predictions["future_hat_down_micro_price"] - 1.0
        predictions["diffusion_hat_cross_book_basis"] = 1e-5
        predictions["future_hat_regime_normal_posterior"] = 0.85
        predictions["future_hat_regime_shock_posterior"] = 0.10
        predictions["future_hat_regime_convergence_posterior"] = 0.05
        predictions["diffusion_hat_regime_normal_posterior"] = 1e-6
        predictions["diffusion_hat_regime_shock_posterior"] = 1e-6
        predictions["diffusion_hat_regime_convergence_posterior"] = 1e-6
        predictions["jump_intensity_hat"] = 0.02
        return predictions


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
        "bid_depth": 100.0,
        "ask_depth": 100.0,
        "bid_depth_top_5": 100.0,
        "ask_depth_top_5": 100.0,
        "orderbook_imbalance": 0.0,
        "bid_levels": 1,
        "ask_levels": 1,
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


def _write_event_state_input(input_dir, *, event_slug: str, start: datetime) -> None:
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
    event_state.to_parquet(input_dir / "crypto_5m_event_state_test.parquet", index=False)


def test_replay_pricing_generates_edge_rows(tmp_path):
    event_slug = "btc-updown-5m-1775578800"
    start = datetime.fromtimestamp(1775578800, tz=timezone.utc)

    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "processed"
    input_dir.mkdir()
    _write_event_state_input(input_dir, event_slug=event_slug, start=start)

    result = replay_pricing(
        event_state_glob=str(input_dir / "crypto_5m_event_state_*.parquet"),
        output_dir=str(output_dir),
        n_samples=200,
        run_timestamp="test",
    )

    replay = pd.read_parquet(output_dir / "crypto_5m_pricing_replay_test.parquet")

    assert result["rows"] == 4
    assert (output_dir / "crypto_5m_pricing_replay_latest.parquet").exists()
    assert set(replay["outcome_name"]) == {"Up", "Down"}
    assert replay["fair_token_price"].between(0.0, 1.0).all()
    assert "brier_component" not in replay.columns
    assert "is_winner" not in replay.columns


def test_replay_pricing_uses_repeated_next_state_rollout_when_transition_bundle_is_provided(tmp_path):
    event_slug = "btc-updown-5m-1775578800"
    start = datetime.fromtimestamp(1775578800, tz=timezone.utc)

    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "processed"
    input_dir.mkdir()
    _write_event_state_input(input_dir, event_slug=event_slug, start=start)

    replay_pricing(
        event_state_glob=str(input_dir / "crypto_5m_event_state_*.parquet"),
        output_dir=str(output_dir),
        n_samples=128,
        rollout_horizon_seconds=15.0,
        transition_bundle=_StubTransitionBundle(),
        run_timestamp="rollout_test",
    )

    replay = pd.read_parquet(output_dir / "crypto_5m_pricing_replay_rollout_test.parquet")
    assert set(replay["simulation_mode"]) == {"repeated_next_state_rollout"}
