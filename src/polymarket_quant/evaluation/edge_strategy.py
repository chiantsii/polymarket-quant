"""Dynamic exit replay for executable mispricing edge-capture strategies."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

DEFAULT_POLYMARKET_CRYPTO_TAKER_FEE_RATE = 0.07


def replay_baseline_up_strategy(
    rows: pd.DataFrame,
    entry_edge_threshold: float = 0.03,
    entry_confirmation_snapshots: int = 2,
    min_entry_seconds_to_end: float = 120.0,
    max_entry_seconds_to_end: float = 300.0,
    no_new_entry_seconds_to_end: float = 60.0,
    max_holding_seconds: float = 60.0,
    max_edge_cap: float = 0.12,
    exit_hold_edge_threshold: float = 0.0,
    timestamp_col: str = "collected_at",
    event_col: str = "event_slug",
    outcome_col: str = "outcome_name",
    fair_price_col: str = "fair_token_price",
    buy_edge_col: str = "buy_edge",
    hold_edge_col: str = "hold_edge",
    bid_price_col: str = "best_bid",
    ask_price_col: str = "best_ask",
    entry_price_col: str = "best_ask",
    exit_price_col: str = "best_bid",
    seconds_to_end_col: str = "seconds_to_end",
    target_outcome: str = "Up",
    estimated_fee_rate: float = DEFAULT_POLYMARKET_CRYPTO_TAKER_FEE_RATE,
) -> pd.DataFrame:
    """Replay a simple Up-only taker strategy on pricing-replay rows.

    Baseline rules:
    - trade only the target outcome (default: ``Up``)
    - enter only when ``buy_edge`` exceeds ``entry_edge_threshold``
    - require ``entry_confirmation_snapshots`` consecutive qualifying rows
    - block oversized edges above ``max_edge_cap``
    - buy at current ``best_ask`` and exit at future ``best_bid``
    - exit when hold edge reverses, max holding time is hit, or time to end is short
    """
    if rows.empty:
        return pd.DataFrame()
    if entry_confirmation_snapshots <= 0:
        raise ValueError("entry_confirmation_snapshots must be positive")
    if max_holding_seconds <= 0:
        raise ValueError("max_holding_seconds must be positive")
    if min_entry_seconds_to_end < no_new_entry_seconds_to_end:
        raise ValueError("min_entry_seconds_to_end must be >= no_new_entry_seconds_to_end")
    if max_entry_seconds_to_end < min_entry_seconds_to_end:
        raise ValueError("max_entry_seconds_to_end must be >= min_entry_seconds_to_end")
    if max_edge_cap < entry_edge_threshold:
        raise ValueError("max_edge_cap must be >= entry_edge_threshold")

    prepared = rows.copy()
    required = {
        timestamp_col,
        event_col,
        outcome_col,
        fair_price_col,
        buy_edge_col,
        hold_edge_col,
        bid_price_col,
        ask_price_col,
        entry_price_col,
        exit_price_col,
        seconds_to_end_col,
    }
    missing = required - set(prepared.columns)
    if missing:
        raise ValueError(f"rows is missing columns: {sorted(missing)}")

    prepared = prepared.loc[prepared[outcome_col] == target_outcome].copy()
    if prepared.empty:
        return pd.DataFrame()

    prepared["_timestamp"] = pd.to_datetime(prepared[timestamp_col], utc=True)
    for column in [
        fair_price_col,
        buy_edge_col,
        hold_edge_col,
        entry_price_col,
        exit_price_col,
        seconds_to_end_col,
    ]:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    prepared = prepared.sort_values([event_col, "_timestamp"]).reset_index(drop=True)
    episodes: list[dict[str, Any]] = []

    for _, group in prepared.groupby(event_col, sort=False):
        records = group.to_dict("records")
        confirmation_count = 0
        entry: dict[str, Any] | None = None
        pending_signal_row: dict[str, Any] | None = None

        for current in records:
            if entry is None:
                if pending_signal_row is not None:
                    if _finite(current.get(entry_price_col)):
                        entry = current
                        signal_row = pending_signal_row
                        pending_signal_row = None
                        confirmation_count = 0
                        entry["_signal_confirmation_time"] = signal_row["_timestamp"]
                        entry["_signal_confirmation_buy_edge"] = signal_row.get(buy_edge_col)
                        entry["_signal_confirmation_seconds_to_end"] = signal_row.get(seconds_to_end_col)
                        continue
                    pending_signal_row = None

                if _is_valid_baseline_entry(
                    current=current,
                    buy_edge_col=buy_edge_col,
                    entry_price_col=entry_price_col,
                    seconds_to_end_col=seconds_to_end_col,
                    entry_edge_threshold=entry_edge_threshold,
                    min_entry_seconds_to_end=min_entry_seconds_to_end,
                    max_entry_seconds_to_end=max_entry_seconds_to_end,
                    no_new_entry_seconds_to_end=no_new_entry_seconds_to_end,
                    max_edge_cap=max_edge_cap,
                ):
                    confirmation_count += 1
                else:
                    confirmation_count = 0

                if confirmation_count >= entry_confirmation_snapshots:
                    pending_signal_row = current
                    confirmation_count = 0
                continue

            holding_seconds = (current["_timestamp"] - entry["_timestamp"]).total_seconds()
            exit_reason = _baseline_exit_reason(
                current=current,
                hold_edge_col=hold_edge_col,
                seconds_to_end_col=seconds_to_end_col,
                exit_hold_edge_threshold=exit_hold_edge_threshold,
                holding_seconds=holding_seconds,
                max_holding_seconds=max_holding_seconds,
                no_new_entry_seconds_to_end=no_new_entry_seconds_to_end,
            )
            if exit_reason is None:
                continue

            episodes.append(
                _baseline_episode_row(
                    entry=entry,
                    exit_row=current,
                    entry_price_col=entry_price_col,
                    exit_price_col=exit_price_col,
                    fair_price_col=fair_price_col,
                    buy_edge_col=buy_edge_col,
                    hold_edge_col=hold_edge_col,
                    bid_price_col=bid_price_col,
                    ask_price_col=ask_price_col,
                    seconds_to_end_col=seconds_to_end_col,
                    exit_reason=exit_reason,
                    confirmation_snapshots=entry_confirmation_snapshots,
                    estimated_fee_rate=estimated_fee_rate,
                )
            )
            entry = None

        if entry is not None:
            episodes.append(
                _baseline_episode_row(
                    entry=entry,
                    exit_row=records[-1],
                    entry_price_col=entry_price_col,
                    exit_price_col=exit_price_col,
                    fair_price_col=fair_price_col,
                    buy_edge_col=buy_edge_col,
                    hold_edge_col=hold_edge_col,
                    bid_price_col=bid_price_col,
                    ask_price_col=ask_price_col,
                    seconds_to_end_col=seconds_to_end_col,
                    exit_reason="data_end",
                    confirmation_snapshots=entry_confirmation_snapshots,
                    estimated_fee_rate=estimated_fee_rate,
                )
            )

    return pd.DataFrame(episodes)


def replay_dynamic_buy_strategy(
    rows: pd.DataFrame,
    entry_edge: float = 0.03,
    exit_edge: float = 0.005,
    take_profit: float = 0.03,
    stop_loss: float = 0.03,
    max_toxicity: float = 0.7,
    min_seconds_to_end: float = 15.0,
    max_holding_seconds: float = 120.0,
    timestamp_col: str = "collected_at",
    group_cols: tuple[str, ...] = ("event_slug", "token_id"),
    signal_col: str = "signal",
    entry_edge_col: str | None = None,
    hold_edge_col: str | None = None,
    fair_price_col: str = "fair_token_price",
    entry_price_col: str = "best_ask",
    exit_price_col: str = "best_bid",
    entry_fee_col: str = "entry_fee_penalty",
    exit_fee_col: str = "exit_fee_penalty",
    adverse_selection_penalty_col: str = "adverse_selection_penalty",
    inventory_penalty_col: str = "inventory_penalty",
    toxicity_col: str = "toxicity_score",
    seconds_to_end_col: str = "seconds_to_end",
) -> pd.DataFrame:
    """Replay non-overlapping BUY episodes with dynamic exit rules.

    Entry is conservative: buy at current best ask. Exit is also conservative:
    sell only at a future best bid. The replay is grouped by token so it does
    not open overlapping positions in the same contract.
    """
    if rows.empty:
        return pd.DataFrame()
    if max_holding_seconds <= 0:
        raise ValueError("max_holding_seconds must be positive")
    if stop_loss < 0 or take_profit < 0:
        raise ValueError("take_profit and stop_loss must be non-negative")

    prepared = rows.copy()
    resolved_entry_edge_col = _resolve_column(prepared, entry_edge_col, "net_buy_edge", "buy_edge")
    resolved_hold_edge_col = _resolve_column(prepared, hold_edge_col, "net_hold_edge", "hold_edge")

    required = {
        timestamp_col,
        signal_col,
        resolved_entry_edge_col,
        entry_price_col,
        exit_price_col,
        *group_cols,
    }
    missing = required - set(prepared.columns)
    if missing:
        raise ValueError(f"rows is missing columns: {sorted(missing)}")

    prepared["_timestamp"] = pd.to_datetime(prepared[timestamp_col], utc=True)
    for column in [
        resolved_entry_edge_col,
        resolved_hold_edge_col,
        fair_price_col,
        entry_price_col,
        exit_price_col,
        entry_fee_col,
        exit_fee_col,
        adverse_selection_penalty_col,
        inventory_penalty_col,
        toxicity_col,
        seconds_to_end_col,
    ]:
        if column in prepared.columns:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    for column, default in [
        (toxicity_col, 0.0),
        (seconds_to_end_col, np.inf),
        (entry_fee_col, 0.0),
        (exit_fee_col, 0.0),
        (adverse_selection_penalty_col, 0.0),
        (inventory_penalty_col, 0.0),
    ]:
        if column not in prepared.columns:
            prepared[column] = default

    prepared = prepared.sort_values([*group_cols, "_timestamp"]).reset_index(drop=True)
    episodes: list[dict[str, Any]] = []

    for _, group in prepared.groupby(list(group_cols), sort=False):
        records = group.to_dict("records")
        i = 0
        while i < len(records):
            entry = records[i]
            if not _is_valid_entry(
                entry,
                signal_col=signal_col,
                entry_edge_col=resolved_entry_edge_col,
                entry_price_col=entry_price_col,
                toxicity_col=toxicity_col,
                seconds_to_end_col=seconds_to_end_col,
                entry_edge=entry_edge,
                max_toxicity=max_toxicity,
                min_seconds_to_end=min_seconds_to_end,
            ):
                i += 1
                continue

            episode, exit_index = _scan_exit(
                records=records,
                entry_index=i,
                timestamp_col="_timestamp",
                group_cols=group_cols,
                entry_edge_col=resolved_entry_edge_col,
                hold_edge_col=resolved_hold_edge_col,
                fair_price_col=fair_price_col,
                entry_price_col=entry_price_col,
                exit_price_col=exit_price_col,
                entry_fee_col=entry_fee_col,
                exit_fee_col=exit_fee_col,
                adverse_selection_penalty_col=adverse_selection_penalty_col,
                inventory_penalty_col=inventory_penalty_col,
                toxicity_col=toxicity_col,
                seconds_to_end_col=seconds_to_end_col,
                exit_edge=exit_edge,
                take_profit=take_profit,
                stop_loss=stop_loss,
                max_toxicity=max_toxicity,
                min_seconds_to_end=min_seconds_to_end,
                max_holding_seconds=max_holding_seconds,
            )
            episodes.append(episode)
            i = max(exit_index + 1, i + 1)

    return pd.DataFrame(episodes)


def edge_strategy_summary(episodes: pd.DataFrame) -> dict[str, Any]:
    """Summarize dynamic edge strategy episodes."""
    if episodes.empty:
        return {
            "episodes": 0,
            "mean_pnl": np.nan,
            "median_pnl": np.nan,
            "hit_rate": np.nan,
            "total_pnl": 0.0,
            "avg_holding_seconds": np.nan,
            "exit_reasons": {},
        }

    if "net_pnl_with_fee" in episodes.columns:
        pnl_col = "net_pnl_with_fee"
    elif "net_pnl" in episodes.columns:
        pnl_col = "net_pnl"
    else:
        pnl_col = "pnl"
    pnl = pd.to_numeric(episodes[pnl_col], errors="coerce").dropna()
    holding = pd.to_numeric(episodes["holding_seconds"], errors="coerce").dropna()
    gross = pd.to_numeric(episodes["gross_pnl"], errors="coerce").dropna() if "gross_pnl" in episodes.columns else pd.Series(dtype=float)
    mid = pd.to_numeric(episodes["mid_pnl"], errors="coerce").dropna() if "mid_pnl" in episodes.columns else pd.Series(dtype=float)
    execution_cost = (
        pd.to_numeric(episodes["execution_cost"], errors="coerce").dropna()
        if "execution_cost" in episodes.columns
        else pd.Series(dtype=float)
    )
    estimated_fee = (
        pd.to_numeric(episodes["estimated_fee"], errors="coerce").dropna()
        if "estimated_fee" in episodes.columns
        else pd.Series(dtype=float)
    )
    return {
        "episodes": int(len(episodes)),
        "pnl_basis": pnl_col,
        "mean_pnl": float(pnl.mean()) if not pnl.empty else np.nan,
        "median_pnl": float(pnl.median()) if not pnl.empty else np.nan,
        "hit_rate": float((pnl > 0).mean()) if not pnl.empty else np.nan,
        "total_pnl": float(pnl.sum()) if not pnl.empty else 0.0,
        "mean_gross_pnl": float(gross.mean()) if not gross.empty else np.nan,
        "mean_mid_pnl": float(mid.mean()) if not mid.empty else np.nan,
        "mean_execution_cost": float(execution_cost.mean()) if not execution_cost.empty else np.nan,
        "mean_estimated_fee": float(estimated_fee.mean()) if not estimated_fee.empty else np.nan,
        "avg_holding_seconds": float(holding.mean()) if not holding.empty else np.nan,
        "exit_reasons": episodes["exit_reason"].value_counts(dropna=False).to_dict(),
    }


def _is_valid_baseline_entry(
    current: dict[str, Any],
    buy_edge_col: str,
    entry_price_col: str,
    seconds_to_end_col: str,
    entry_edge_threshold: float,
    min_entry_seconds_to_end: float,
    max_entry_seconds_to_end: float,
    no_new_entry_seconds_to_end: float,
    max_edge_cap: float,
) -> bool:
    seconds_to_end = _to_float(current.get(seconds_to_end_col), default=np.nan)
    if not np.isfinite(seconds_to_end):
        return False
    if seconds_to_end < no_new_entry_seconds_to_end:
        return False
    if seconds_to_end < min_entry_seconds_to_end or seconds_to_end > max_entry_seconds_to_end:
        return False

    buy_edge = _to_float(current.get(buy_edge_col), default=np.nan)
    return (
        np.isfinite(buy_edge)
        and buy_edge >= entry_edge_threshold
        and buy_edge <= max_edge_cap
        and _finite(current.get(entry_price_col))
    )


def _baseline_exit_reason(
    current: dict[str, Any],
    hold_edge_col: str,
    seconds_to_end_col: str,
    exit_hold_edge_threshold: float,
    holding_seconds: float,
    max_holding_seconds: float,
    no_new_entry_seconds_to_end: float,
) -> str | None:
    hold_edge = _to_float(current.get(hold_edge_col), default=np.nan)
    if np.isfinite(hold_edge) and hold_edge < exit_hold_edge_threshold:
        return "hold_edge_reversal"
    if holding_seconds >= max_holding_seconds:
        return "max_holding_time"
    seconds_to_end = _to_float(current.get(seconds_to_end_col), default=np.inf)
    if seconds_to_end < no_new_entry_seconds_to_end:
        return "time_to_end"
    return None


def _baseline_episode_row(
    entry: dict[str, Any],
    exit_row: dict[str, Any],
    entry_price_col: str,
    exit_price_col: str,
    fair_price_col: str,
    buy_edge_col: str,
    hold_edge_col: str,
    bid_price_col: str,
    ask_price_col: str,
    seconds_to_end_col: str,
    exit_reason: str,
    confirmation_snapshots: int,
    estimated_fee_rate: float,
) -> dict[str, Any]:
    entry_time = entry["_timestamp"]
    exit_time = exit_row["_timestamp"]
    entry_price = _to_float(entry.get(entry_price_col), default=np.nan)
    exit_price = _to_float(exit_row.get(exit_price_col), default=np.nan)
    gross_pnl = exit_price - entry_price if np.isfinite(entry_price) and np.isfinite(exit_price) else np.nan
    entry_bid = _to_float(entry.get(bid_price_col), default=np.nan)
    entry_ask = _to_float(entry.get(ask_price_col), default=np.nan)
    exit_bid = _to_float(exit_row.get(bid_price_col), default=np.nan)
    exit_ask = _to_float(exit_row.get(ask_price_col), default=np.nan)
    entry_mid = (entry_bid + entry_ask) / 2 if np.isfinite(entry_bid) and np.isfinite(entry_ask) else np.nan
    exit_mid = (exit_bid + exit_ask) / 2 if np.isfinite(exit_bid) and np.isfinite(exit_ask) else np.nan
    mid_pnl = exit_mid - entry_mid if np.isfinite(entry_mid) and np.isfinite(exit_mid) else np.nan
    execution_cost = mid_pnl - gross_pnl if np.isfinite(mid_pnl) and np.isfinite(gross_pnl) else np.nan
    fair_at_exit = _to_float(exit_row.get(fair_price_col), default=np.nan)
    exit_edge = fair_at_exit - exit_price if np.isfinite(fair_at_exit) and np.isfinite(exit_price) else np.nan
    entry_estimated_fee = _polymarket_taker_fee_usdc(entry_price, estimated_fee_rate)
    exit_estimated_fee = _polymarket_taker_fee_usdc(exit_price, estimated_fee_rate)
    estimated_fee = (
        entry_estimated_fee + exit_estimated_fee
        if np.isfinite(entry_estimated_fee) and np.isfinite(exit_estimated_fee)
        else np.nan
    )
    net_pnl_with_fee = gross_pnl - estimated_fee if np.isfinite(gross_pnl) and np.isfinite(estimated_fee) else np.nan
    signal_confirmation_time = entry.get("_signal_confirmation_time")
    signal_confirmation_buy_edge = entry.get("_signal_confirmation_buy_edge")
    signal_confirmation_seconds_to_end = entry.get("_signal_confirmation_seconds_to_end")
    entry_execution_delay_seconds = (
        (entry_time - signal_confirmation_time).total_seconds() if signal_confirmation_time is not None else np.nan
    )

    row = {
        column: entry.get(column)
        for column in ["asset", "event_slug", "market_id", "condition_id", "token_id", "outcome_name"]
        if column in entry
    }
    row.update(
        {
            "entry_time": entry_time.isoformat(),
            "exit_time": exit_time.isoformat(),
            "holding_seconds": (exit_time - entry_time).total_seconds(),
            "signal_confirmation_time": signal_confirmation_time.isoformat()
            if signal_confirmation_time is not None
            else None,
            "entry_execution_delay_seconds": entry_execution_delay_seconds,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "entry_mid_price": entry_mid,
            "exit_mid_price": exit_mid,
            "gross_pnl": gross_pnl,
            "pnl": gross_pnl,
            "mid_pnl": mid_pnl,
            "execution_cost": execution_cost,
            "entry_estimated_fee": entry_estimated_fee,
            "exit_estimated_fee": exit_estimated_fee,
            "estimated_fee": estimated_fee,
            "net_pnl": net_pnl_with_fee,
            "net_pnl_with_fee": net_pnl_with_fee,
            "entry_edge": entry.get(buy_edge_col),
            "entry_buy_edge": entry.get(buy_edge_col),
            "exit_buy_edge": exit_row.get(buy_edge_col),
            "entry_hold_edge": entry.get(hold_edge_col),
            "exit_hold_edge": exit_row.get(hold_edge_col),
            "signal_confirmation_buy_edge": signal_confirmation_buy_edge,
            "entry_fair_token_price": entry.get(fair_price_col),
            "exit_fair_token_price": exit_row.get(fair_price_col),
            "signal_confirmation_seconds_to_end": signal_confirmation_seconds_to_end,
            "entry_seconds_to_end": entry.get(seconds_to_end_col),
            "exit_seconds_to_end": exit_row.get(seconds_to_end_col),
            "exit_edge": exit_edge,
            "exit_reason": exit_reason,
            "entry_confirmation_snapshots": confirmation_snapshots,
        }
    )
    return row


def _polymarket_taker_fee_usdc(price: float | None, fee_rate: float) -> float:
    if price is None or not np.isfinite(price):
        return np.nan
    fee = round(float(fee_rate) * float(price) * (1.0 - float(price)), 5)
    return fee if fee >= 0 else np.nan


def _is_valid_entry(
    row: dict[str, Any],
    signal_col: str,
    entry_edge_col: str,
    entry_price_col: str,
    toxicity_col: str,
    seconds_to_end_col: str,
    entry_edge: float,
    max_toxicity: float,
    min_seconds_to_end: float,
) -> bool:
    return (
        row.get(signal_col) == "BUY"
        and _finite(row.get(entry_edge_col), min_value=entry_edge)
        and _finite(row.get(entry_price_col))
        and _to_float(row.get(toxicity_col), default=0.0) <= max_toxicity
        and _to_float(row.get(seconds_to_end_col), default=np.inf) > min_seconds_to_end
    )


def _scan_exit(
    records: list[dict[str, Any]],
    entry_index: int,
    timestamp_col: str,
    group_cols: tuple[str, ...],
    entry_edge_col: str,
    hold_edge_col: str | None,
    fair_price_col: str,
    entry_price_col: str,
    exit_price_col: str,
    entry_fee_col: str,
    exit_fee_col: str,
    adverse_selection_penalty_col: str,
    inventory_penalty_col: str,
    toxicity_col: str,
    seconds_to_end_col: str,
    exit_edge: float,
    take_profit: float,
    stop_loss: float,
    max_toxicity: float,
    min_seconds_to_end: float,
    max_holding_seconds: float,
) -> tuple[dict[str, Any], int]:
    entry = records[entry_index]
    entry_time = entry[timestamp_col]
    entry_price = float(entry[entry_price_col])
    entry_edge_value = float(entry[entry_edge_col])
    entry_fee_penalty = _to_float(entry.get(entry_fee_col), default=0.0)
    entry_adverse_selection_penalty = _to_float(entry.get(adverse_selection_penalty_col), default=0.0)
    entry_inventory_penalty = _to_float(entry.get(inventory_penalty_col), default=0.0)
    max_favorable = -np.inf
    max_adverse = np.inf
    last_executable_index = None
    last_executable_net_pnl = np.nan

    for exit_index in range(entry_index + 1, len(records)):
        current = records[exit_index]
        holding_seconds = (current[timestamp_col] - entry_time).total_seconds()
        exit_price = _to_float(current.get(exit_price_col))
        gross_pnl = (exit_price - entry_price) if exit_price is not None else np.nan
        exit_fee_penalty = _to_float(current.get(exit_fee_col), default=0.0)
        total_fees = entry_fee_penalty + exit_fee_penalty
        net_pnl = (
            gross_pnl - total_fees - entry_adverse_selection_penalty - entry_inventory_penalty
            if np.isfinite(gross_pnl)
            else np.nan
        )

        if np.isfinite(net_pnl):
            max_favorable = max(max_favorable, net_pnl)
            max_adverse = min(max_adverse, net_pnl)
            last_executable_index = exit_index
            last_executable_net_pnl = net_pnl

        reason = _exit_reason(
            current=current,
            pnl=net_pnl,
            fair_price_col=fair_price_col,
            hold_edge_col=hold_edge_col,
            toxicity_col=toxicity_col,
            seconds_to_end_col=seconds_to_end_col,
            exit_edge=exit_edge,
            take_profit=take_profit,
            stop_loss=stop_loss,
            max_toxicity=max_toxicity,
            min_seconds_to_end=min_seconds_to_end,
            holding_seconds=holding_seconds,
            max_holding_seconds=max_holding_seconds,
        )
        if reason:
            return (
                _episode_row(
                    entry=entry,
                    exit_row=current,
                    group_cols=group_cols,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    entry_edge=entry_edge_value,
                    hold_edge_col=hold_edge_col,
                    entry_fee_penalty=entry_fee_penalty,
                    exit_fee_penalty=exit_fee_penalty,
                    adverse_selection_penalty=entry_adverse_selection_penalty,
                    inventory_penalty=entry_inventory_penalty,
                    exit_reason=reason,
                    max_favorable=max_favorable,
                    max_adverse=max_adverse,
                    pnl=net_pnl,
                    timestamp_col=timestamp_col,
                    entry_edge_col=entry_edge_col,
                    fair_price_col=fair_price_col,
                    exit_price_col=exit_price_col,
                    toxicity_col=toxicity_col,
                    seconds_to_end_col=seconds_to_end_col,
                ),
                exit_index,
            )

    if last_executable_index is not None:
        current = records[last_executable_index]
        exit_price = _to_float(current.get(exit_price_col))
        return (
            _episode_row(
                entry=entry,
                exit_row=current,
                group_cols=group_cols,
                entry_price=entry_price,
                exit_price=exit_price,
                entry_edge=entry_edge_value,
                hold_edge_col=hold_edge_col,
                entry_fee_penalty=entry_fee_penalty,
                exit_fee_penalty=_to_float(current.get(exit_fee_col), default=0.0),
                adverse_selection_penalty=entry_adverse_selection_penalty,
                inventory_penalty=entry_inventory_penalty,
                exit_reason="data_end",
                max_favorable=max_favorable,
                max_adverse=max_adverse,
                pnl=last_executable_net_pnl,
                timestamp_col=timestamp_col,
                entry_edge_col=entry_edge_col,
                fair_price_col=fair_price_col,
                exit_price_col=exit_price_col,
                toxicity_col=toxicity_col,
                seconds_to_end_col=seconds_to_end_col,
            ),
            last_executable_index,
        )

    return (
        _episode_row(
            entry=entry,
            exit_row=entry,
            group_cols=group_cols,
            entry_price=entry_price,
            exit_price=np.nan,
            entry_edge=entry_edge_value,
            hold_edge_col=hold_edge_col,
            entry_fee_penalty=entry_fee_penalty,
            exit_fee_penalty=0.0,
            adverse_selection_penalty=entry_adverse_selection_penalty,
            inventory_penalty=entry_inventory_penalty,
            exit_reason="no_executable_exit",
            max_favorable=np.nan,
            max_adverse=np.nan,
            pnl=np.nan,
            timestamp_col=timestamp_col,
            entry_edge_col=entry_edge_col,
            fair_price_col=fair_price_col,
            exit_price_col=exit_price_col,
            toxicity_col=toxicity_col,
            seconds_to_end_col=seconds_to_end_col,
        ),
        entry_index,
    )


def _exit_reason(
    current: dict[str, Any],
    pnl: float,
    fair_price_col: str,
    hold_edge_col: str | None,
    toxicity_col: str,
    seconds_to_end_col: str,
    exit_edge: float,
    take_profit: float,
    stop_loss: float,
    max_toxicity: float,
    min_seconds_to_end: float,
    holding_seconds: float,
    max_holding_seconds: float,
) -> str | None:
    if not np.isfinite(pnl):
        return None
    if _to_float(current.get(toxicity_col), default=0.0) >= max_toxicity:
        return "toxicity"
    if pnl <= -stop_loss:
        return "stop_loss"
    if pnl >= take_profit:
        return "take_profit"

    current_hold_edge = _to_float(current.get(hold_edge_col), default=np.nan) if hold_edge_col else np.nan
    if not np.isfinite(current_hold_edge):
        fair_price = _to_float(current.get(fair_price_col))
        exit_price = _to_float(current.get("best_bid"))
        current_hold_edge = (fair_price - exit_price) if fair_price is not None and exit_price is not None else np.nan
    if np.isfinite(current_hold_edge) and current_hold_edge <= exit_edge:
        return "edge_repaired"
    if _to_float(current.get(seconds_to_end_col), default=np.inf) <= min_seconds_to_end:
        return "time_to_end"
    if holding_seconds >= max_holding_seconds:
        return "max_holding"
    return None


def _episode_row(
    entry: dict[str, Any],
    exit_row: dict[str, Any],
    group_cols: tuple[str, ...],
    entry_price: float,
    exit_price: float | None,
    entry_edge: float,
    hold_edge_col: str | None,
    entry_fee_penalty: float,
    exit_fee_penalty: float,
    adverse_selection_penalty: float,
    inventory_penalty: float,
    exit_reason: str,
    max_favorable: float,
    max_adverse: float,
    pnl: float,
    timestamp_col: str,
    entry_edge_col: str,
    fair_price_col: str,
    exit_price_col: str,
    toxicity_col: str,
    seconds_to_end_col: str,
) -> dict[str, Any]:
    entry_time = entry[timestamp_col]
    exit_time = exit_row[timestamp_col]
    holding_seconds = (exit_time - entry_time).total_seconds()
    fair_at_exit = _to_float(exit_row.get(fair_price_col))
    executable_exit = _to_float(exit_row.get(exit_price_col))
    exit_edge_value = (
        fair_at_exit - executable_exit if fair_at_exit is not None and executable_exit is not None else np.nan
    )
    gross_pnl = (exit_price - entry_price) if exit_price is not None and np.isfinite(exit_price) else np.nan
    total_fees = entry_fee_penalty + exit_fee_penalty

    row = {
        column: entry.get(column)
        for column in ["asset", "event_slug", "market_id", "condition_id", "token_id", "outcome_name"]
        if column in entry
    }
    row.update(
        {
            **{column: entry.get(column) for column in group_cols if column not in row},
            "entry_time": entry_time.isoformat(),
            "exit_time": exit_time.isoformat(),
            "holding_seconds": holding_seconds,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "entry_edge": entry_edge,
            "exit_edge": exit_edge_value,
            "gross_pnl": gross_pnl,
            "pnl": pnl,
            "net_pnl": pnl,
            "entry_fee_penalty": entry_fee_penalty,
            "exit_fee_penalty": exit_fee_penalty,
            "total_fees": total_fees,
            "entry_adverse_selection_penalty": adverse_selection_penalty,
            "entry_inventory_penalty": inventory_penalty,
            "exit_reason": exit_reason,
            "max_favorable_excursion": max_favorable if np.isfinite(max_favorable) else np.nan,
            "max_adverse_excursion": max_adverse if np.isfinite(max_adverse) else np.nan,
            "entry_fair_token_price": entry.get(fair_price_col),
            "exit_fair_token_price": exit_row.get(fair_price_col),
            "entry_buy_edge": entry.get(entry_edge_col),
            "exit_buy_edge": exit_row.get(entry_edge_col),
            "exit_hold_edge": exit_row.get(hold_edge_col) if hold_edge_col else np.nan,
            "entry_toxicity_score": entry.get(toxicity_col),
            "exit_toxicity_score": exit_row.get(toxicity_col),
            "entry_seconds_to_end": entry.get(seconds_to_end_col),
            "exit_seconds_to_end": exit_row.get(seconds_to_end_col),
        }
    )
    return row


def _finite(value, min_value: float | None = None) -> bool:
    parsed = _to_float(value)
    if parsed is None or not np.isfinite(parsed):
        return False
    return parsed >= min_value if min_value is not None else True


def _to_float(value, default=None):
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if np.isfinite(parsed) else default


def _resolve_column(rows: pd.DataFrame, explicit: str | None, *preferred: str) -> str:
    if explicit is not None:
        return explicit
    for column in preferred:
        if column in rows.columns:
            return column
    return preferred[-1]
