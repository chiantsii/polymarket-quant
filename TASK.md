# TASK

## Current Focus

Current pricing output already includes:

- `fair_up_probability`
- `fair_token_price`
- `buy_edge`
- `hold_edge`
- `sell_edge`

But the current strategy layer is still incomplete:

- `edge` is implemented
- full exit logic is not implemented yet
- true trading risk is not implemented yet

## Work List

### 1. Define first-version exit philosophy

We need to confirm the main strategy principle:

- we are trading mispricing, not necessarily holding to final settlement
- primary exit should be based on mispricing / edge convergence

Recommended direction:

- exit when `hold_edge` converges back to a normal range

### 2. Define the primary exit rule

Candidate first-version main exit:

- `hold_edge <= exit_threshold`

Where:

- `hold_edge = fair_token_price - best_bid`

Meaning:

- the original reason to keep holding is gone
- or the market has converged back toward fair value

### 3. Decide whether each position should have its own target price

Per-position fields at entry:

- `entry_price`
- `entry_fair_price`
- `entry_edge`

Candidate target:

- `target_price = entry_price + alpha * entry_edge`

Recommended direction:

- yes, each position should have its own target price
- but target price should be a secondary take-profit rule, not the only exit rule

### 4. Add maximum holding time

We need a timeout rule because short-horizon mispricing that does not converge is probably weak.

Candidate rule:

- exit when `holding_time >= max_holding_time`

Candidate values to test:

- `30s`
- `60s`
- `120s`

### 5. Add near-expiry forced exit

We need a rule for very late market states where orderbook structure becomes distorted.

Candidate rule:

- exit when `seconds_to_end <= forced_exit_time`

Candidate values to test:

- `5s`
- `10s`

### 6. Add stop-loss

We need a basic protective exit even in the first version.

Candidate rules:

- price-based:
  - `best_bid - entry_price <= -stop_loss`
- edge-based:
  - `hold_edge <= -stop_threshold`

Recommended direction:

- first implement a simple price/PnL stop-loss

### 7. Build first position-level replay

After entry/exit rules are defined, we need a replay that tracks actual positions.

It should support:

- entry time
- entry price
- current holding state
- exit reason
- exit price
- holding time
- realized PnL

### 8. Add true strategy-layer risk metrics

After position-level replay exists, add true strategy-layer risk metrics such as:

- liquidity risk
- holding risk
- exit risk
- slippage risk
- drawdown risk

## Recommended First-Version Strategy Spec

### Entry

- `buy_edge >= entry_threshold`

### Exit

Exit if any of the following is true:

1. `hold_edge <= exit_threshold`
2. `best_bid >= target_price`
3. `holding_time >= max_holding_time`
4. `seconds_to_end <= forced_exit_time`
5. `best_bid - entry_price <= -stop_loss`

## Suggested Initial Parameters For Discussion

- `entry_threshold = 0.02`
- `exit_threshold = 0.005`
- `alpha = 0.7`
- `max_holding_time = 60s`
- `forced_exit_time = 10s`
- `stop_loss = 0.03`

## Suggested Execution Order

1. finalize first-version exit logic
2. implement position-level replay
3. add true strategy risk metrics
