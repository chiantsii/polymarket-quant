# COMMEND

This file is a copy-paste command list for the current mainline workflow.

## 0. Activate Environment

```bash
cd /Users/chiantsii/Desktop/polymarket_quant
source venv/bin/activate
```

## 0.5. Oracle Remote Connection

### SSH into the Oracle remote machine

This is the previously used Oracle remote login command found from local shell history.

```bash
ssh -i '/Users/chiantsii/Downloads/ssh-key-2026-04-07.key' ubuntu@141.147.161.26
```

### SSH into the Oracle remote machine without specifying the key explicitly

This only works if your SSH agent or SSH config is already set up.

```bash
ssh ubuntu@141.147.161.26
```

## 1. Capture Raw Market Data

### Run the full 5-minute capture window pipeline

This command is valid because the script has defaults:
- config: `configs/base.yaml`
- interval: `1.0`
- event duration: `300`
- windows: `1`
- event slug prefixes: `btc-updown-5m eth-updown-5m`
- series slugs: `btc-up-or-down-5m eth-up-or-down-5m`

```bash
venv/bin/python scripts/run_window_capture.py
```

### Recommended explicit full-window run

```bash
venv/bin/python scripts/run_window_capture.py --interval-seconds 1.0 --windows 2
```

### Run from a specific window start

```bash
venv/bin/python scripts/run_window_capture.py --config configs/base.yaml --interval-seconds 1.0 --event-duration-seconds 300 --window-start 1775688000 --windows 1 --event-slug-prefixes btc-updown-5m eth-updown-5m --series-slugs btc-up-or-down-5m eth-up-or-down-5m
```

### Collect Polymarket orderbooks only

```bash
venv/bin/python scripts/collect_orderbooks.py
```

### Collect Coinbase spot ticks only

```bash
venv/bin/python scripts/collect_spot_prices.py
```

### Build per-event reference prices from processed orderbooks + processed spot ticks

This uses the same reference-price builder that `build_market_state.py` consumes.

```bash
venv/bin/python -c "import pandas as pd; from polymarket_quant.state.dataset import build_reference_prices, prepare_orderbooks, prepare_spot, load_parquet_glob; orderbooks=prepare_orderbooks(load_parquet_glob('data/processed/crypto_5m_orderbook_summary_*.parquet', include_latest=True)); spot=prepare_spot(load_parquet_glob('data/processed/crypto_spot_ticks_*.parquet', include_latest=True)); refs=build_reference_prices(orderbooks=orderbooks, spot_by_asset={'BTC': spot[spot['asset']=='BTC'].copy(), 'ETH': spot[spot['asset']=='ETH'].copy()}, event_duration_seconds=300.0); print(refs)"
```

### Show reference-related columns from the latest `market_state`

```bash
venv/bin/python -c "import pandas as pd; df=pd.read_parquet('data/processed/crypto_5m_market_state_latest.parquet'); cols=['event_slug','asset','collected_at','reference_spot_price','reference_source','spot_price','spot_return_since_reference']; print(df[cols].head(30).to_string())"
```

## 2. Build Current State

### Build token-level `market_state`

```bash
venv/bin/python scripts/build_market_state.py
```

`build_market_state.py` now defaults to file-batched processing to reduce peak RAM on cloud instances.

If you want the old load-everything behavior for parity/debugging:

```bash
venv/bin/python scripts/build_market_state.py --batch-mode full
```

### Build event-level `event_state = S_t`

```bash
venv/bin/python scripts/build_event_state.py
```

## 3. Build Markov Transition Training Data

### Default mainline: pair each row with the next observation

```bash
venv/bin/python scripts/build_transition_targets.py
```

This mainline output now also includes jump-label research fields such as:
- `is_jump`
- `spot_log_return_delta`
- `jump_threshold`

### Keep unmatched rows in the output

```bash
venv/bin/python scripts/build_transition_targets.py --include-unmatched
```

## 4. Fit the Transition Model

### Train the current tree-based transition model

```bash
venv/bin/python scripts/fit_transition_model.py
```

### Train with custom minimum sample count

```bash
venv/bin/python scripts/fit_transition_model.py --min-training-rows 64 --random-state 42
```

## 5. Run Markov Pricing Replay

### Mainline pricing replay

```bash
venv/bin/python scripts/replay_pricing.py --pricing-method markov_mcmc
```

This reads `data/processed/crypto_5m_event_state_latest.parquet` by default.

Current default assumptions:
- drift decay uses a `5-second half-life`
- replay consumes the learned `mu_t`
- replay uses `event_state["volatility_per_sqrt_second"]` as `sigma_i`
- jump is dormant unless manually overridden or supported by learned data

### Smoke test: fewer paths and fewer rows

```bash
venv/bin/python scripts/replay_pricing.py --pricing-method markov_mcmc --n-samples 5 --max-rows 5
```

### Faster debug run without progress bar

```bash
venv/bin/python scripts/replay_pricing.py --pricing-method markov_mcmc --n-samples 10 --max-rows 10 --no-progress
```

### Use an explicit trained transition model artifact

```bash
venv/bin/python scripts/replay_pricing.py --pricing-method markov_mcmc --transition-model-path artifacts/transition_model/transition_model_latest.joblib
```

### Replay a specific event-state artifact

```bash
venv/bin/python scripts/replay_pricing.py --pricing-method markov_mcmc --event-state-glob data/processed/crypto_5m_event_state_latest.parquet
```

### Explicitly set drift half-life

`5s half-life`:

```bash
venv/bin/python scripts/replay_pricing.py --pricing-method markov_mcmc --spot-drift-decay-kappa-per-second 0.13862943611198905
```

`10s half-life`:

```bash
venv/bin/python scripts/replay_pricing.py --pricing-method markov_mcmc --spot-drift-decay-kappa-per-second 0.06931471805599453
```

### Manual rare-jump sensitivity test

```bash
venv/bin/python scripts/replay_pricing.py --pricing-method markov_mcmc --force-manual-jump-parameters --spot-jump-intensity-per-second 0.0002314814814814815 --spot-jump-log-return-mean 0.0 --spot-jump-log-return-std 0.001
```

## 6. Study Decay / Horizon Behavior

### Horizon-decay study for `mu_t` and microstructure signals

```bash
venv/bin/python scripts/study_horizon_decay.py --include-latest
```

### Pricing drift-decay sensitivity grid

This is the cloud-friendly command to compare replay outputs across multiple drift half-lives.

```bash
venv/bin/python scripts/study_pricing_decay_sensitivity.py --include-latest --half-lives-seconds 5 10 15 30 60 --n-samples 1000 --force-manual-jump-parameters --spot-jump-intensity-per-second 0.0002314814814814815 --spot-jump-log-return-mean 0.0 --spot-jump-log-return-std 0.001
```

### Small cloud smoke test for the sensitivity script

```bash
venv/bin/python scripts/study_pricing_decay_sensitivity.py --include-latest --half-lives-seconds 5 10 --n-samples 50 --max-rows 200 --force-manual-jump-parameters --spot-jump-intensity-per-second 0.0002314814814814815 --spot-jump-log-return-mean 0.0 --spot-jump-log-return-std 0.001
```

## 7. Inspect Key Outputs

### Show transition model summary

```bash
cat artifacts/transition_model/transition_model_summary_latest.json
```

### Preview transition predictions parquet

```bash
venv/bin/python -c "import pandas as pd; df=pd.read_parquet('data/processed/crypto_5m_transition_predictions_latest.parquet'); print(df.head(20).to_string())"
```

### Preview pricing replay parquet

```bash
venv/bin/python -c "import pandas as pd; df=pd.read_parquet('data/processed/crypto_5m_pricing_replay_latest.parquet'); print(df.head(20).to_string())"
```

### Inspect fair price, edge, and pricing-kernel columns

```bash
venv/bin/python -c "import pandas as pd; df=pd.read_parquet('data/processed/crypto_5m_pricing_replay_latest.parquet'); cols=['event_slug','outcome_name','latent_up_probability','fair_up_probability','fair_token_price','buy_edge','buy_signal','conditioned_spot_log_drift_per_second','conditioned_spot_volatility_per_sqrt_second','conditioned_spot_jump_intensity_per_second']; print(df[cols].head(30).to_string())"
```

### Inspect pricing and edge distributions

```bash
venv/bin/python -c "import pandas as pd; df=pd.read_parquet('data/processed/crypto_5m_pricing_replay_latest.parquet'); print(df[['fair_up_probability','fair_token_price','buy_edge','hold_edge','sell_edge','conditioned_spot_log_drift_per_second','conditioned_spot_volatility_per_sqrt_second']].describe().to_string())"
```

### Inspect horizon-decay study outputs

```bash
venv/bin/python -c "import pandas as pd; df=pd.read_parquet('artifacts/horizon_decay/crypto_5m_horizon_decay_summary_latest.parquet'); print(df.to_string(index=False))"
```

### Inspect pricing-decay sensitivity outputs

```bash
venv/bin/python -c "import pandas as pd; df=pd.read_parquet('artifacts/pricing_decay_sensitivity/crypto_5m_pricing_decay_sensitivity_latest.parquet'); print(df.to_string(index=False))"
```

## 8. Run Core Tests

### Run the mainline test set

```bash
venv/bin/pytest -q tests/test_market_state.py tests/test_transition_targets.py tests/test_transition_model.py tests/test_markov_simulation.py tests/test_mispricing.py tests/test_replay_pricing.py tests/test_study_horizon_decay.py tests/test_study_pricing_decay_sensitivity.py
```

### Run a single test file

```bash
venv/bin/pytest -q tests/test_market_state.py
```

## 9. Full Mainline Workflow

### Rebuild everything from processed data to pricing replay

```bash
venv/bin/python scripts/build_market_state.py
venv/bin/python scripts/build_event_state.py
venv/bin/python scripts/build_transition_targets.py
venv/bin/python scripts/fit_transition_model.py
venv/bin/python scripts/replay_pricing.py --pricing-method markov_mcmc --include-latest
```

## 10. Main Output Files

### Current state outputs

```bash
ls -1 data/processed/crypto_5m_market_state_*.parquet
ls -1 data/processed/crypto_5m_event_state_*.parquet
```

### Transition outputs

```bash
ls -1 data/processed/crypto_5m_transition_targets_*.parquet
ls -1 data/processed/crypto_5m_transition_predictions_*.parquet
ls -1 artifacts/transition_model/*.joblib
ls -1 artifacts/transition_model/*.json
```

### Pricing outputs

```bash
ls -1 data/processed/crypto_5m_pricing_replay_*.parquet
ls -1 artifacts/horizon_decay/*.parquet
ls -1 artifacts/pricing_decay_sensitivity/*.parquet
```

## 11. What Each Main Script Does

### Build token-level market state

```bash
printf '%s\n' "scripts/build_market_state.py -> orderbook + spot -> token-level market_state"
```

### Build event-level Markov state

```bash
printf '%s\n' "scripts/build_event_state.py -> market_state -> event_state = S_t"
```

### Build next-step transition targets

```bash
printf '%s\n' "scripts/build_transition_targets.py -> event_state -> structured P(S_{n+1} | S_n) targets"
```

### Fit transition dynamics

```bash
printf '%s\n' "scripts/fit_transition_model.py -> learn drift / diffusion / jump heads"
```

### Replay Markov pricing

```bash
printf '%s\n' "scripts/replay_pricing.py -> terminal spot rollout with drift decay -> fair price / edge"
```

### Study signal horizon decay

```bash
printf '%s\n' "scripts/study_horizon_decay.py -> clock-time horizon study for mu_t / microstructure signal predictive decay"
```

### Study pricing drift-decay sensitivity

```bash
printf '%s\n' "scripts/study_pricing_decay_sensitivity.py -> replay pricing sensitivity across drift half-lives"
```
