# Polymarket Quant Research Environment

This repository is a research codebase for short-horizon Polymarket BTC/ETH binary markets.

The current scope is:

```text
data capture
-> market_state
-> event_state
-> latent probability / regime Markov modeling
-> fair-price model
-> edge extraction
```

## Ultimate Goal

The long-run objective of this repository is:

```text
to model binary prediction markets as a latent probability / regime Markov system
```

This means:

- market quotes are observations, not the true state
- the core hidden object is a latent probability state
- the latent state may evolve under jump and regime-switching dynamics
- pricing should be based on current or predicted latent state
- edge should come from model fair value versus executable Polymarket prices

## State Design

The working state definition is:

```text
S_t = (M_t, L_t, T_t)
```

Where:

- `M_t` is the market-observation block
  - quote, spread, micro-price, depth, imbalance, cross-book consistency, boundary pressure, data freshness, and external spot context
- `L_t` is the latent/mechanism block
  - market-implied probability
  - fundamental probability
  - latent probability in both probability and logit space
  - basis terms between those layers
  - current observation uncertainty
  - regime posterior
- `T_t` is the temporal block
  - currently `normalized_time_to_end`

`L_t` should describe the current latent belief only. Drift, diffusion, and jump belong to the transition kernel and should be learned in the transition layer rather than hard-coded into state construction.

`event_state` should be treated as the serialized row-level representation of the current `S_t`.

## First Transition Objective

The first transition model should not try to predict repair outcomes or final winners.

Its first job is:

```text
predict the conditional drift, diffusion, and jump intensity
of future latent_logit_probability given current S_t
```

Equivalently, the first supervised target should be built around:

```text
X_t = latent_logit_probability
P(X_{t+Δ} | S_t)
```

In practice this means the first model should learn:

- where future latent logit is centered
- how uncertain the next state is
- when the transition kernel becomes jump-like instead of smooth

These transition quantities are not part of `L_t`; they belong to the learned kernel used by `fit_transition_model.py` and `MarkovSimulationEngine`.

## Current Architecture

```text
raw data
(orderbook + spot + reference + time)
    ↓
build_market_state.py
    ↓
market_implied_up_probability
fundamental_up_probability
latent_up_probability
latent_logit_probability
    ↓
build_event_state.py
    ↓
event_state = S_t
    ↓
build_transition_targets.py
    ↓
structured transition targets for P(S_{n+1} | S_n)
    ↓
fit_transition_model.py
    ↓
drift / diffusion / jump heads
    ↓
mispricing.py + MarkovSimulationEngine
    ↓
fair_up_probability
fair_token_price
    ↓
buy_edge / hold_edge / sell_edge
```

## Core Principles

1. State first
   We first construct a latent Markov-style state from aligned orderbook and spot data.

2. Pricing consumes state
   Pricing methods should read state outputs, not rebuild state from raw inputs inside the pricing layer.

3. Edge comes from the model
   Edge is the difference between model fair value and Polymarket executable prices.

4. Keep early layers clean
   Toxicity, inventory, fee adjustments, execution policy, and strategy gating do not belong in the state or pricing core.

## Current Mainline

Implemented:

- aligned 5-minute data capture
- token-level `market_state`
- event-level `event_state`
- `M_t v2` market-observation features
- `L_t v1` latent/mechanism features
- structured transition targets for `P(S_{t+Δ} | S_t)`
- first transition model with drift / diffusion / jump outputs
- MCMC-based fair pricing replay
  - now using repeated next-state rollout by default when a trained transition model is available

Current next-step focus:

- strengthen the full transition kernel beyond the current v1 model
- improve regime and jump modeling
- connect the learned transition model more tightly to pricing

## Quick Start

Create the environment:

```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

Capture one full 5-minute window:

```bash
venv/bin/python scripts/run_window_capture.py --windows 1 --interval-seconds 1
```

Build market state:

```bash
venv/bin/python scripts/build_market_state.py
```

Build event state:

```bash
venv/bin/python scripts/build_event_state.py
```

Build transition targets:

```bash
venv/bin/python scripts/build_transition_targets.py
```

Fit the transition model:

```bash
venv/bin/python scripts/fit_transition_model.py
```

Replay pricing:

```bash
venv/bin/python scripts/replay_pricing.py --pricing-method markov_mcmc
```

`replay_pricing.py` now consumes serialized `event_state` rows directly and reads
`data/processed/crypto_5m_event_state_latest.parquet` by default.

Run tests:

```bash
venv/bin/pytest -q
```
