from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from polymarket_quant.signals.mispricing import MispricingDetectorConfig, RealTimeMispricingDetector


def _state_row(
    outcome_name: str,
    best_bid: float,
    best_ask: float,
    *,
    market_implied_up_probability: float = 0.50,
    fundamental_up_probability: float = 0.50,
    latent_up_probability: float = 0.50,
    orderbook_imbalance: float = 0.0,
    weighted_imbalance: float | None = None,
    bid_depth_top_5: float = 100.0,
    ask_depth_top_5: float = 100.0,
    book_velocity: float = 0.0,
    spot_vol_multiplier: float = 1.0,
    cross_book_basis: float = 0.0,
    dist_to_boundary: float | None = None,
) -> dict:
    now = datetime.now(timezone.utc)
    mid_price = (best_bid + best_ask) / 2
    if weighted_imbalance is None:
        weighted_imbalance = orderbook_imbalance
    if dist_to_boundary is None:
        dist_to_boundary = min(latent_up_probability, 1.0 - latent_up_probability)

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
        "mid_price": mid_price,
        "bid_depth": 100.0,
        "ask_depth": 100.0,
        "bid_depth_top_5": bid_depth_top_5,
        "ask_depth_top_5": ask_depth_top_5,
        "orderbook_imbalance": orderbook_imbalance,
        "weighted_imbalance": weighted_imbalance,
        "book_velocity": book_velocity,
        "bid_levels": 1,
        "ask_levels": 1,
        "book_hash": "hash",
        "spot_price": 101.0,
        "reference_spot_price": 100.0,
        "reference_source": "coinbase_1m_candle_open",
        "seconds_to_end": 240.0,
        "market_implied_up_probability": market_implied_up_probability,
        "fundamental_up_probability": fundamental_up_probability,
        "latent_up_probability": latent_up_probability,
        "spot_vol_multiplier": spot_vol_multiplier,
        "cross_book_basis": cross_book_basis,
        "dist_to_boundary": dist_to_boundary,
    }


def test_mispricing_detector_prices_paths_and_flags_underpriced_up_token() -> None:
    detector = RealTimeMispricingDetector(
        MispricingDetectorConfig(
            n_samples=512,
            fallback_spot_volatility_per_sqrt_second=0.0,
            spot_jump_intensity_per_second=0.0,
            edge_threshold=0.01,
            max_allowed_risk=0.05,
            seed=42,
        )
    )

    state_rows = [
        _state_row(
            "Up",
            best_bid=0.50,
            best_ask=0.60,
            market_implied_up_probability=0.55,
            fundamental_up_probability=0.74,
            latent_up_probability=0.72,
            orderbook_imbalance=0.10,
            weighted_imbalance=0.10,
        ),
        _state_row(
            "Down",
            best_bid=0.39,
            best_ask=0.40,
            market_implied_up_probability=0.55,
            fundamental_up_probability=0.74,
            latent_up_probability=0.72,
            orderbook_imbalance=-0.10,
            weighted_imbalance=-0.10,
        ),
    ]

    valuations = detector.detect(state_rows)

    assert len(valuations) == 2
    up_valuation = next(row for row in valuations if row["outcome_name"] == "Up")
    down_valuation = next(row for row in valuations if row["outcome_name"] == "Down")
    assert up_valuation["pricing_method"] == "markov_mcmc"
    assert up_valuation["buy_edge"] > 0.10
    assert up_valuation["buy_signal"] is True
    assert 0.0 <= up_valuation["risk_score"] < detector.config.max_allowed_risk
    assert 0.0 <= up_valuation["fair_up_probability"] <= 1.0
    assert down_valuation["fair_token_price"] == pytest.approx(1.0 - up_valuation["fair_up_probability"])
    assert "signal" not in up_valuation
    assert "net_buy_edge" not in up_valuation
    assert "inventory_penalty" not in up_valuation


def test_mispricing_detector_applies_risk_gate_to_buy_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    detector = RealTimeMispricingDetector(
        MispricingDetectorConfig(
            edge_threshold=0.01,
            max_allowed_risk=0.05,
            seed=42,
        )
    )

    class _FakeDistribution:
        def __init__(self, expected_fair_price: float, risk_score: float) -> None:
            self.expected_fair_price = expected_fair_price
            self.risk_score = risk_score
            self.n_paths = 1000
            self.diagnostics = {}

    class _FakeSimulation:
        expected_terminal_probability = 0.62
        terminal_probability_std = 0.20
        n_paths = 1000
        diagnostics = {
            "n_steps": 10,
            "dt_seconds": 1.0,
            "conditioned_spot_log_drift_per_second": 0.0,
            "conditioned_spot_volatility_per_sqrt_second": 0.2,
            "conditioned_spot_jump_intensity_per_second": 0.05,
        }

        def aggregate(self, pricing_model=None, invert_probability: bool = False):
            if invert_probability:
                return _FakeDistribution(0.38, 0.20)
            return _FakeDistribution(0.62, 0.20)

    monkeypatch.setattr(detector.engine, "simulate", lambda **_: _FakeSimulation())

    valuations = detector.detect(
        [
            _state_row("Up", best_bid=0.50, best_ask=0.55, latent_up_probability=0.60),
            _state_row("Down", best_bid=0.44, best_ask=0.45, latent_up_probability=0.60),
        ]
    )

    up_valuation = next(row for row in valuations if row["outcome_name"] == "Up")
    assert up_valuation["buy_edge"] > 0.01
    assert up_valuation["risk_score"] == pytest.approx(0.20)
    assert up_valuation["buy_signal"] is False


def test_mispricing_detector_does_not_emit_toxicity_from_valuation_layer() -> None:
    detector = RealTimeMispricingDetector(
        MispricingDetectorConfig(
            n_samples=256,
            fallback_spot_volatility_per_sqrt_second=0.0,
            spot_jump_intensity_per_second=0.0,
            seed=42,
        )
    )
    valuations = detector.detect(
        [
            _state_row(
                "Up",
                best_bid=0.50,
                best_ask=0.60,
                market_implied_up_probability=0.55,
                fundamental_up_probability=0.60,
                latent_up_probability=0.58,
            )
        ]
    )

    assert "toxicity_score" not in valuations[0]
    assert "signal" not in valuations[0]


def test_only_markov_mcmc_pricing_method_is_supported() -> None:
    with pytest.raises(ValueError, match="pricing_method must be 'markov_mcmc'"):
        RealTimeMispricingDetector(
            MispricingDetectorConfig(
                pricing_method="latent",
            )
        )


def test_mispricing_detector_prices_when_spot_inputs_exist_even_if_latent_probability_is_nan() -> None:
    detector = RealTimeMispricingDetector(MispricingDetectorConfig(seed=42))

    valuations = detector.detect(
        [
            _state_row("Up", best_bid=0.50, best_ask=0.55, latent_up_probability=np.nan),
            _state_row("Down", best_bid=0.44, best_ask=0.45, latent_up_probability=np.nan),
        ]
    )

    assert len(valuations) == 2
    assert all(0.0 <= row["fair_up_probability"] <= 1.0 for row in valuations)


def test_simulation_market_state_handles_missing_side_signals_without_warning() -> None:
    detector = RealTimeMispricingDetector(MispricingDetectorConfig(seed=42))

    state = detector._simulation_market_state(
        [
            _state_row(
                "Up",
                best_bid=0.50,
                best_ask=0.55,
                latent_up_probability=0.60,
                weighted_imbalance=np.nan,
                bid_depth_top_5=np.nan,
                ask_depth_top_5=np.nan,
                book_velocity=np.nan,
                cross_book_basis=np.nan,
                dist_to_boundary=np.nan,
            )
        ]
    )

    assert state.liquidity_depth == 0.0
    assert state.book_velocity == 0.0
    assert state.spot_vol_multiplier == pytest.approx(1.0)
    assert state.spot_volatility_per_sqrt_second > 0.0
