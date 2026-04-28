from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from polymarket_quant.state import TransitionTargetConfig, build_transition_target_dataset


def _event_state_row(
    timestamp: datetime,
    *,
    event_slug: str,
    latent_logit: float,
    latent_probability: float,
    market_probability: float,
    fundamental_probability: float,
    up_micro_price: float,
    down_micro_price: float,
    normalized_time_to_end: float,
    seconds_to_end: float,
    regime_normal: float,
    regime_shock: float,
    regime_convergence: float,
) -> dict[str, object]:
    return {
        "event_slug": event_slug,
        "asset": "BTC",
        "series_slug": "btc-up-or-down-5m",
        "event_id": "evt_1",
        "event_title": "BTC Up/Down",
        "market_id": "mkt_1",
        "condition_id": "cond_1",
        "collected_at": timestamp.isoformat(),
        "state_timestamp": timestamp.isoformat(),
        "seconds_to_end": seconds_to_end,
        "normalized_time_to_end": normalized_time_to_end,
        "spot_price": 100.0,
        "spot_bid": 99.9,
        "spot_ask": 100.1,
        "spot_return_since_reference": 0.01,
        "spot_vol_multiplier": 1.2,
        "external_spot_drift": 0.01,
        "up_best_bid": up_micro_price - 0.01,
        "up_best_ask": up_micro_price + 0.01,
        "down_best_bid": down_micro_price - 0.01,
        "down_best_ask": down_micro_price + 0.01,
        "up_mid_price": up_micro_price,
        "down_mid_price": down_micro_price,
        "up_micro_price": up_micro_price,
        "down_micro_price": down_micro_price,
        "up_spread": 0.02,
        "down_spread": 0.02,
        "up_bid_depth_top_5": 120.0 + 10.0 * latent_probability,
        "up_ask_depth_top_5": 100.0,
        "down_bid_depth_top_5": 110.0,
        "down_ask_depth_top_5": 105.0,
        "up_orderbook_imbalance": 0.1,
        "down_orderbook_imbalance": -0.1,
        "up_weighted_imbalance": 0.12,
        "down_weighted_imbalance": -0.08,
        "up_depth_slope": 12.0,
        "down_depth_slope": 10.0,
        "up_tick_density": 0.8,
        "down_tick_density": 0.85,
        "up_book_velocity": 0.01,
        "down_book_velocity": 0.02,
        "cross_book_basis": up_micro_price + down_micro_price - 1.0,
        "cross_book_bid_basis": up_micro_price + down_micro_price - 1.02,
        "cross_book_ask_basis": up_micro_price + down_micro_price - 0.98,
        "spread_divergence": 0.0,
        "dist_to_boundary": min(up_micro_price, 1.0 - up_micro_price),
        "boundary_leverage_ratio": 1.0 / max(1.0 - up_micro_price, 1e-6),
        "asymmetric_depth_ratio": 120.0 / 110.0,
        "book_age_max": 0.5,
        "has_full_book_pair": True,
        "market_implied_up_probability": market_probability,
        "fundamental_up_probability": fundamental_probability,
        "latent_up_probability": latent_probability,
        "latent_logit_probability": latent_logit,
        "market_fundamental_basis": market_probability - fundamental_probability,
        "latent_market_basis": latent_probability - market_probability,
        "latent_fundamental_basis": latent_probability - fundamental_probability,
        "abs_market_fundamental_basis": abs(market_probability - fundamental_probability),
        "abs_latent_market_basis": abs(latent_probability - market_probability),
        "abs_latent_fundamental_basis": abs(latent_probability - fundamental_probability),
        "volatility_per_sqrt_second": 0.001,
        "state_observation_variance": 0.02,
        "regime_normal_posterior": regime_normal,
        "regime_shock_posterior": regime_shock,
        "regime_convergence_posterior": regime_convergence,
    }


def test_build_transition_target_dataset_pairs_current_and_future_state() -> None:
    start = datetime(2026, 4, 11, 10, 0, 0, tzinfo=timezone.utc)
    event_slug = "btc-updown-5m-1775578800"
    event_state = pd.DataFrame(
        [
            _event_state_row(
                start,
                event_slug=event_slug,
                latent_logit=0.10,
                latent_probability=0.525,
                market_probability=0.53,
                fundamental_probability=0.51,
                up_micro_price=0.52,
                down_micro_price=0.47,
                normalized_time_to_end=1.0,
                seconds_to_end=300.0,
                regime_normal=0.7,
                regime_shock=0.2,
                regime_convergence=0.1,
            ),
            _event_state_row(
                start + timedelta(seconds=15),
                event_slug=event_slug,
                latent_logit=0.22,
                latent_probability=0.555,
                market_probability=0.56,
                fundamental_probability=0.53,
                up_micro_price=0.55,
                down_micro_price=0.44,
                normalized_time_to_end=0.95,
                seconds_to_end=285.0,
                regime_normal=0.55,
                regime_shock=0.35,
                regime_convergence=0.10,
            ),
            _event_state_row(
                start + timedelta(seconds=30),
                event_slug=event_slug,
                latent_logit=0.18,
                latent_probability=0.545,
                market_probability=0.54,
                fundamental_probability=0.52,
                up_micro_price=0.54,
                down_micro_price=0.45,
                normalized_time_to_end=0.90,
                seconds_to_end=270.0,
                regime_normal=0.50,
                regime_shock=0.25,
                regime_convergence=0.25,
            ),
        ]
    )

    targets = build_transition_target_dataset(event_state)

    assert len(targets) == 2
    assert targets["target_status"].eq("matched").all()
    assert targets["has_future_target"].all()
    assert targets["target_horizon_seconds"].eq(15.0).all()
    assert targets["realized_horizon_seconds"].eq(15.0).all()

    first = targets.iloc[0]
    assert first["current_latent_logit_probability"] == 0.10
    assert first["future_latent_logit_probability"] == 0.22
    assert first["target_delta_latent_logit_probability"] == 0.12
    assert first["current_cross_book_basis"] == pytest.approx(-0.01)
    assert first["future_cross_book_basis"] == pytest.approx(-0.01)
    assert first["target_delta_up_micro_price"] == pytest.approx(0.03)
    assert first["future_normalized_time_to_end"] == 0.95


def test_build_transition_target_dataset_can_keep_unmatched_rows() -> None:
    start = datetime(2026, 4, 11, 10, 0, 0, tzinfo=timezone.utc)
    event_slug = "btc-updown-5m-1775578800"
    event_state = pd.DataFrame(
        [
            _event_state_row(
                start,
                event_slug=event_slug,
                latent_logit=0.10,
                latent_probability=0.525,
                market_probability=0.53,
                fundamental_probability=0.51,
                up_micro_price=0.52,
                down_micro_price=0.47,
                normalized_time_to_end=1.0,
                seconds_to_end=300.0,
                regime_normal=0.7,
                regime_shock=0.2,
                regime_convergence=0.1,
            ),
            _event_state_row(
                start + timedelta(seconds=15),
                event_slug=event_slug,
                latent_logit=0.22,
                latent_probability=0.555,
                market_probability=0.56,
                fundamental_probability=0.53,
                up_micro_price=0.55,
                down_micro_price=0.44,
                normalized_time_to_end=0.95,
                seconds_to_end=285.0,
                regime_normal=0.55,
                regime_shock=0.35,
                regime_convergence=0.10,
            ),
        ]
    )

    targets = build_transition_target_dataset(
        event_state,
        config=TransitionTargetConfig(include_unmatched=True),
    )

    assert len(targets) == 2
    matched = targets[targets["target_status"] == "matched"]
    missing = targets[targets["target_status"] == "missing_future_snapshot"]

    assert len(matched) == 1
    assert len(missing) == 1
    assert missing.iloc[0]["has_future_target"] == False
    assert pd.isna(missing.iloc[0]["future_latent_logit_probability"])


def test_build_transition_target_dataset_handles_multiple_events_with_global_time_sorting() -> None:
    start = datetime(2026, 4, 11, 10, 0, 0, tzinfo=timezone.utc)
    event_a = "btc-updown-5m-1775578800"
    event_b = "eth-updown-5m-1775578800"
    event_state = pd.DataFrame(
        [
            _event_state_row(
                start + timedelta(seconds=5),
                event_slug=event_b,
                latent_logit=0.05,
                latent_probability=0.512,
                market_probability=0.51,
                fundamental_probability=0.50,
                up_micro_price=0.51,
                down_micro_price=0.48,
                normalized_time_to_end=0.98,
                seconds_to_end=295.0,
                regime_normal=0.7,
                regime_shock=0.2,
                regime_convergence=0.1,
            ),
            _event_state_row(
                start,
                event_slug=event_a,
                latent_logit=0.10,
                latent_probability=0.525,
                market_probability=0.53,
                fundamental_probability=0.51,
                up_micro_price=0.52,
                down_micro_price=0.47,
                normalized_time_to_end=1.0,
                seconds_to_end=300.0,
                regime_normal=0.7,
                regime_shock=0.2,
                regime_convergence=0.1,
            ),
            _event_state_row(
                start + timedelta(seconds=20),
                event_slug=event_a,
                latent_logit=0.18,
                latent_probability=0.545,
                market_probability=0.54,
                fundamental_probability=0.52,
                up_micro_price=0.54,
                down_micro_price=0.45,
                normalized_time_to_end=0.93,
                seconds_to_end=280.0,
                regime_normal=0.6,
                regime_shock=0.25,
                regime_convergence=0.15,
            ),
            _event_state_row(
                start + timedelta(seconds=25),
                event_slug=event_b,
                latent_logit=0.12,
                latent_probability=0.53,
                market_probability=0.525,
                fundamental_probability=0.505,
                up_micro_price=0.525,
                down_micro_price=0.465,
                normalized_time_to_end=0.92,
                seconds_to_end=275.0,
                regime_normal=0.65,
                regime_shock=0.25,
                regime_convergence=0.10,
            ),
        ]
    )

    targets = build_transition_target_dataset(event_state)

    assert len(targets) == 2
    assert set(targets["event_slug"]) == {event_a, event_b}
    assert targets["target_status"].eq("matched").all()
    assert list(targets["realized_horizon_seconds"]) == [20.0, 20.0]


def test_build_transition_target_dataset_defaults_to_next_observation_pairing() -> None:
    start = datetime(2026, 4, 11, 10, 0, 0, tzinfo=timezone.utc)
    event_slug = "btc-updown-5m-1775578800"
    event_state = pd.DataFrame(
        [
            _event_state_row(
                start,
                event_slug=event_slug,
                latent_logit=0.10,
                latent_probability=0.525,
                market_probability=0.53,
                fundamental_probability=0.51,
                up_micro_price=0.52,
                down_micro_price=0.47,
                normalized_time_to_end=1.0,
                seconds_to_end=300.0,
                regime_normal=0.7,
                regime_shock=0.2,
                regime_convergence=0.1,
            ),
            _event_state_row(
                start + timedelta(seconds=7),
                event_slug=event_slug,
                latent_logit=0.15,
                latent_probability=0.537,
                market_probability=0.54,
                fundamental_probability=0.52,
                up_micro_price=0.53,
                down_micro_price=0.46,
                normalized_time_to_end=0.976,
                seconds_to_end=293.0,
                regime_normal=0.65,
                regime_shock=0.25,
                regime_convergence=0.10,
            ),
            _event_state_row(
                start + timedelta(seconds=19),
                event_slug=event_slug,
                latent_logit=0.20,
                latent_probability=0.550,
                market_probability=0.55,
                fundamental_probability=0.53,
                up_micro_price=0.54,
                down_micro_price=0.45,
                normalized_time_to_end=0.937,
                seconds_to_end=281.0,
                regime_normal=0.60,
                regime_shock=0.30,
                regime_convergence=0.10,
            ),
        ]
    )

    targets = build_transition_target_dataset(event_state)

    assert len(targets) == 2
    assert targets["target_status"].eq("matched").all()
    assert list(targets["realized_horizon_seconds"]) == [7.0, 12.0]
    assert list(targets["target_horizon_seconds"]) == [7.0, 12.0]
