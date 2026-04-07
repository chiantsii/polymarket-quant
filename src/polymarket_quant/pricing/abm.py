"""Agent-based price formation simulator for binary prediction markets."""

from dataclasses import dataclass

import numpy as np

from polymarket_quant.utils.math import logit, sigmoid


@dataclass(frozen=True)
class ABMResult:
    """Simulated path from the binary market agent-based model."""

    prices: np.ndarray
    net_order_flow: np.ndarray
    mean_beliefs: np.ndarray


class BinaryMarketABM:
    """Simple binary-market price formation model for research simulations."""

    def __init__(
        self,
        n_agents: int = 100,
        belief_dispersion: float = 0.08,
        price_impact: float = 0.05,
        seed: int | None = None,
    ) -> None:
        if n_agents <= 0:
            raise ValueError("n_agents must be positive")
        if belief_dispersion < 0:
            raise ValueError("belief_dispersion must be non-negative")
        if price_impact < 0:
            raise ValueError("price_impact must be non-negative")

        self.n_agents = n_agents
        self.belief_dispersion = belief_dispersion
        self.price_impact = price_impact
        self.rng = np.random.default_rng(seed)

    def simulate(
        self,
        n_steps: int,
        initial_price: float = 0.5,
        fundamental_probability: float = 0.5,
    ) -> ABMResult:
        """Simulate a bounded binary-contract price path."""
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        if not 0.0 < initial_price < 1.0:
            raise ValueError("initial_price must be strictly between 0 and 1")
        if not 0.0 < fundamental_probability < 1.0:
            raise ValueError("fundamental_probability must be strictly between 0 and 1")

        price_logit = float(logit(initial_price))
        prices = []
        net_order_flow = []
        mean_beliefs = []

        for _ in range(n_steps):
            current_price = float(sigmoid(price_logit))
            beliefs = np.clip(
                self.rng.normal(fundamental_probability, self.belief_dispersion, size=self.n_agents),
                1e-6,
                1 - 1e-6,
            )
            desired_flow = beliefs - current_price
            flow = float(np.mean(desired_flow))
            price_logit += self.price_impact * flow

            prices.append(float(sigmoid(price_logit)))
            net_order_flow.append(flow)
            mean_beliefs.append(float(np.mean(beliefs)))

        return ABMResult(
            prices=np.asarray(prices),
            net_order_flow=np.asarray(net_order_flow),
            mean_beliefs=np.asarray(mean_beliefs),
        )
