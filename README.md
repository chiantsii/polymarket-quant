# Polymarket Quant Research Environment

This repository is a research codebase for short-horizon Polymarket BTC/ETH binary markets.

The current scope is:

```text
data capture
-> market_state
-> event_state
-> structured next-state / spot-kernel modeling
-> binary payoff fair-price model
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

## Current Pricing Object

The active pricing layer values the Polymarket binary contract directly.

For an `Up` market with reference opening price `K = reference_spot_price`, the
current fair value is defined as:

```text
fair_up_probability(t) = P(spot_T >= K | current information at t)
```

The replay engine estimates this quantity by simulating terminal spot paths and
applying the binary payoff path by path:

```text
payoff_i = 1{spot_T_i >= reference_spot_price}
fair_up_probability = mean(payoff_i)
fair_token_price(Up) = fair_up_probability
fair_token_price(Down) = 1 - fair_up_probability
```

Edges are then defined against executable Polymarket prices:

- `buy_edge = fair_token_price - best_ask`
- `hold_edge = fair_token_price - best_bid`
- `sell_edge = best_bid - fair_token_price`

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

## Transition Research Objective

The current transition layer is built from:

```text
event_state rows
-> next-observation pairing within the same event_slug
-> current_* / future_* / target_delta_* transition targets
-> learned drift / diffusion / jump heads
```

The goal is not to predict repair outcomes or final winner labels directly.
Instead, the transition model learns a structured approximation to:

```text
P(S_{n+1} | S_n)
```

In the current codebase, this serves two practical purposes:

- learn next-state structure for the full event state
- learn a state-conditioned log-spot jump-diffusion kernel used by pricing replay

That pricing-relevant kernel is the object consumed by
`MarkovSimulationEngine`:

```text
log spot_{t+Δ}
= log spot_t
+ drift(S_t) * Δt
+ diffusion(S_t) * sqrt(Δt) * ε
+ jump(S_t)
```

These transition quantities are not part of `L_t`; they belong to the learned
kernel produced by `fit_transition_model.py`.

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
mispricing.py + MarkovSimulationEngine
    ↓
simulate terminal spot paths
    ↓
apply binary payoff 1{spot_T >= reference_spot_price}
    ↓
fair_up_probability
fair_token_price
    ↓
buy_edge / hold_edge / sell_edge
```

There is also a parallel transition-modeling track:

```text
event_state = S_t
    ↓
build_transition_targets.py
    ↓
structured next-observation transition targets for P(S_{n+1} | S_n)
    ↓
fit_transition_model.py
    ↓
drift / diffusion / jump heads for state evolution and spot-kernel pricing
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

- continuous aligned data capture with 5-minute chunk flushes
- token-level `market_state`
- event-level `event_state`
- `M_t v2` market-observation features
- `L_t v1` latent/mechanism features
- structured next-observation transition targets for `P(S_{n+1} | S_n)`
- first transition model with learned drift / diffusion / jump outputs
- Monte Carlo spot-terminal binary payoff pricing replay
  - replay pricing currently consumes `event_state`
  - each path is priced by whether terminal spot finishes above the market reference opening price
  - current pricing replay loads a trained transition-model artifact to obtain the learned spot kernel

Current next-step focus:

- strengthen the structured state-transition research track
- improve regime and jump modeling for state evolution and spot dynamics
- continue validating spot-terminal binary payoff pricing against market edge behavior

## Quick Start

Create the environment:

```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

Capture continuously for one 5-minute chunk of wall-clock time:

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

`replay_pricing.py` expects a trained transition-model artifact at:

```text
artifacts/transition_model/transition_model_latest.joblib
```

If it is missing, first run:

```bash
venv/bin/python scripts/build_transition_targets.py
venv/bin/python scripts/fit_transition_model.py
```

Useful replay options:

```bash
venv/bin/python scripts/replay_pricing.py \
  --n-samples 500 \
  --simulation-dt-seconds 1.0 \
  --rollout-horizon-seconds 1.0 \
  --fallback-spot-volatility-per-sqrt-second 0.0005
```

The replay output includes:

- `fair_up_probability`
- `fair_token_price`
- `buy_edge`
- `hold_edge`
- `sell_edge`
- `simulation_mode = spot_terminal_binary_payoff_rollout`
- `rollout_kernel = spot_jump_diffusion`

Run tests:

```bash
venv/bin/pytest -q
```
