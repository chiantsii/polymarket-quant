from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def build_live_dashboard_snapshot(
    *,
    event_state_path: str | Path,
    live_signal_path: str | Path,
    paper_signal_path: str | Path,
    paper_order_path: str | Path,
    paper_trade_path: str | Path,
    max_points: int = 80,
    max_rows: int = 12,
) -> dict[str, Any]:
    event_state = _read_latest_event_state(event_state_path)
    live_signals = _read_jsonl(live_signal_path)
    paper_signals = _read_jsonl(paper_signal_path)
    paper_orders = _read_jsonl(paper_order_path)
    paper_trades = _read_jsonl(paper_trade_path)

    signal_rows = paper_signals or live_signals
    latest_signal = signal_rows[-1] if signal_rows else {}
    latest_trade = paper_trades[-1] if paper_trades else {}
    latest_order = paper_orders[-1] if paper_orders else {}
    latest_state_row = event_state.iloc[-1].to_dict() if not event_state.empty else {}
    latest_signal_by_asset = _latest_rows_by_asset(signal_rows)
    latest_order_by_asset = _latest_rows_by_asset(paper_orders)
    latest_trade_by_asset = _latest_rows_by_asset(paper_trades)
    latest_state_by_asset = _latest_event_state_rows_by_asset(event_state)
    asset_order = _ordered_assets(
        latest_signal_by_asset.keys(),
        latest_state_by_asset.keys(),
        latest_order_by_asset.keys(),
        latest_trade_by_asset.keys(),
    )
    latest_exit_reason = _coalesce(
        latest_trade.get("exit_reason") if isinstance(latest_trade, dict) else None,
        latest_order.get("reason") if isinstance(latest_order, dict) and str(latest_order.get("side", "")).upper() == "SELL" else None,
        latest_signal.get("closed_trade_exit_reason"),
        latest_signal.get("exit_reason"),
    )

    state_panel = {
        "event_slug": _coalesce(latest_signal.get("event_slug"), latest_state_row.get("event_slug")),
        "collected_at": _coalesce(latest_signal.get("collected_at"), latest_state_row.get("collected_at")),
        "market_implied_up_probability": _coalesce(
            latest_signal.get("market_implied_up_probability"),
            latest_state_row.get("market_implied_up_probability"),
        ),
        "latent_up_probability": _coalesce(
            latest_signal.get("latent_up_probability"),
            latest_state_row.get("latent_up_probability"),
        ),
        "fair_up_probability": latest_signal.get("fair_up_probability"),
        "fundamental_up_probability": _coalesce(
            latest_signal.get("fundamental_up_probability"),
            latest_state_row.get("fundamental_up_probability"),
        ),
    }

    edge_panel = {
        "fair_token_price": latest_signal.get("fair_token_price"),
        "best_bid": latest_signal.get("best_bid"),
        "best_ask": latest_signal.get("best_ask"),
        "buy_edge": latest_signal.get("buy_edge"),
        "hold_edge": latest_signal.get("hold_edge"),
        "sell_edge": latest_signal.get("sell_edge"),
        "decision": latest_signal.get("decision"),
        "seconds_to_end": latest_signal.get("seconds_to_end"),
    }

    execution_panel = {
        "position_state": latest_signal.get("position_state", "FLAT"),
        "pending_entry": bool(latest_signal.get("pending_entry", False)),
        "has_position": bool(latest_signal.get("has_position", False)),
        "latest_order_state": latest_order.get("status", "idle"),
        "latest_order_side": latest_order.get("side"),
        "latest_order_price": latest_order.get("price"),
        "latest_order_reason": latest_order.get("reason"),
        "exit_reason": latest_exit_reason,
        "filled_orders": len(paper_orders),
        "closed_trades": len(paper_trades),
        "latest_trade": latest_trade,
    }

    pnl_panel = _build_pnl_panel(paper_trades)

    probability_series = _build_probability_series(signal_rows[-max_points:])
    probability_series_by_asset = _build_probability_series_by_asset(signal_rows[-max_points:])
    pnl_curve = _build_pnl_curve(paper_trades[-max_points:])

    recent_signals = [_compact_signal_row(row) for row in signal_rows[-max_rows:]]
    recent_orders = [_compact_order_row(row) for row in paper_orders[-max_rows:]]
    recent_trades = [_compact_trade_row(row) for row in paper_trades[-max_rows:]]
    asset_snapshots = {
        asset: _build_asset_snapshot(
            asset=asset,
            latest_signal=latest_signal_by_asset.get(asset, {}),
            latest_state_row=latest_state_by_asset.get(asset, {}),
            latest_order=latest_order_by_asset.get(asset, {}),
            latest_trade=latest_trade_by_asset.get(asset, {}),
        )
        for asset in asset_order
    }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "current_snapshot": {
            "label": "Current Snapshot",
            "signal_timestamp": latest_signal.get("collected_at"),
            "event_slug": state_panel["event_slug"],
            "latest_signal_decision": latest_signal.get("decision"),
            "latest_signal_state": latest_signal.get("position_state"),
        },
        "history_summary": {
            "label": "Cumulative Paper History",
            "latest_order_time": latest_order.get("filled_at") or latest_order.get("submitted_at"),
            "latest_trade_time": latest_trade.get("exit_time") or latest_trade.get("entry_time"),
            "filled_orders": len(paper_orders),
            "closed_trades": len(paper_trades),
        },
        "asset_order": asset_order,
        "asset_snapshots": asset_snapshots,
        "paths": {
            "event_state": str(event_state_path),
            "live_signal": str(live_signal_path),
            "paper_signal": str(paper_signal_path),
            "paper_order": str(paper_order_path),
            "paper_trade": str(paper_trade_path),
        },
        "state_panel": state_panel,
        "edge_panel": edge_panel,
        "execution_panel": execution_panel,
        "pnl_panel": pnl_panel,
        "probability_series": probability_series,
        "probability_series_by_asset": probability_series_by_asset,
        "pnl_curve": pnl_curve,
        "recent_signals": recent_signals,
        "recent_orders": recent_orders,
        "recent_trades": recent_trades,
    }


def _read_latest_event_state(path: str | Path) -> pd.DataFrame:
    parquet_path = Path(path)
    if not parquet_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(parquet_path)
    except Exception:
        return pd.DataFrame()


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError:
                continue
    return rows


def _build_pnl_panel(trades: list[dict[str, Any]]) -> dict[str, Any]:
    if not trades:
        return {
            "trade_count": 0,
            "realized_pnl": 0.0,
            "gross_pnl": 0.0,
            "net_pnl": 0.0,
            "fee_cost": 0.0,
            "mean_gross_pnl": 0.0,
            "mean_fee_cost": 0.0,
            "win_rate": 0.0,
            "mean_holding_seconds": 0.0,
        }

    frame = pd.DataFrame(trades)
    net_col = "net_pnl_with_fee" if "net_pnl_with_fee" in frame.columns else "net_pnl"
    gross_col = "gross_pnl" if "gross_pnl" in frame.columns else "pnl"
    fee_series = frame.get("estimated_fee", pd.Series([0.0] * len(frame), index=frame.index))
    holding_series = frame.get("holding_seconds", pd.Series([0.0] * len(frame), index=frame.index))
    net_series = pd.to_numeric(frame.get(net_col, pd.Series([0.0] * len(frame), index=frame.index)), errors="coerce").fillna(0.0)
    gross_series = pd.to_numeric(frame.get(gross_col, pd.Series([0.0] * len(frame), index=frame.index)), errors="coerce").fillna(0.0)
    fee_series = pd.to_numeric(fee_series, errors="coerce").fillna(0.0)
    holding_series = pd.to_numeric(holding_series, errors="coerce").fillna(0.0)
    return {
        "trade_count": int(len(frame)),
        "realized_pnl": float(net_series.sum()),
        "gross_pnl": float(gross_series.sum()),
        "net_pnl": float(net_series.sum()),
        "fee_cost": float(fee_series.sum()),
        "mean_gross_pnl": float(gross_series.mean()),
        "mean_fee_cost": float(fee_series.mean()),
        "win_rate": float((net_series > 0).mean()),
        "mean_holding_seconds": float(holding_series.mean()),
    }


def _build_probability_series(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "labels": [str(row.get("collected_at", "")) for row in rows],
        "market": [_to_float(row.get("market_implied_up_probability")) for row in rows],
        "latent": [_to_float(row.get("latent_up_probability")) for row in rows],
        "fair": [_to_float(row.get("fair_up_probability")) for row in rows],
    }


def _build_probability_series_by_asset(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        asset = _row_asset(row)
        if asset is None:
            continue
        grouped.setdefault(asset, []).append(row)
    return {asset: _build_probability_series(asset_rows) for asset, asset_rows in grouped.items()}


def _build_pnl_curve(trades: list[dict[str, Any]]) -> dict[str, Any]:
    if not trades:
        return {"labels": [], "cumulative_net": []}
    frame = pd.DataFrame(trades)
    net_col = "net_pnl_with_fee" if "net_pnl_with_fee" in frame.columns else "net_pnl"
    net_series = pd.to_numeric(frame.get(net_col, pd.Series([0.0] * len(frame), index=frame.index)), errors="coerce").fillna(0.0)
    timestamps = frame.get("exit_time", frame.get("entry_time", pd.Series([""] * len(frame), index=frame.index)))
    cumulative = net_series.cumsum()
    return {
        "labels": [str(value) for value in timestamps.tolist()],
        "cumulative_net": [float(value) for value in cumulative.tolist()],
    }


def _latest_rows_by_asset(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        asset = _row_asset(row)
        if asset is None:
            continue
        latest[asset] = row
    return latest


def _latest_event_state_rows_by_asset(event_state: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if event_state.empty:
        return {}
    frame = event_state.copy()
    if "asset" not in frame.columns:
        frame["asset"] = frame.get("event_slug", pd.Series(dtype=object)).astype(str).map(_event_slug_asset)
    frame = frame.dropna(subset=["asset"]).copy()
    if frame.empty:
        return {}
    frame["_collected_at_dt"] = pd.to_datetime(frame["collected_at"], utc=True, errors="coerce")
    frame = frame.sort_values(["asset", "_collected_at_dt"]).reset_index(drop=True)
    latest: dict[str, dict[str, Any]] = {}
    for asset, group in frame.groupby("asset", sort=False):
        latest[str(asset).strip().upper()] = group.iloc[-1].to_dict()
    return latest


def _ordered_assets(*asset_key_iterables) -> list[str]:
    preferred = ["BTC", "ETH"]
    discovered: list[str] = []
    for iterable in asset_key_iterables:
        for asset in iterable:
            normalized = str(asset).strip().upper()
            if normalized and normalized not in discovered:
                discovered.append(normalized)
    ordered = [asset for asset in preferred if asset in discovered]
    ordered.extend(asset for asset in discovered if asset not in ordered)
    return ordered


def _build_asset_snapshot(
    *,
    asset: str,
    latest_signal: dict[str, Any],
    latest_state_row: dict[str, Any],
    latest_order: dict[str, Any],
    latest_trade: dict[str, Any],
) -> dict[str, Any]:
    exit_reason = _coalesce(
        latest_trade.get("exit_reason") if isinstance(latest_trade, dict) else None,
        latest_order.get("reason") if isinstance(latest_order, dict) and str(latest_order.get("side", "")).upper() == "SELL" else None,
        latest_signal.get("closed_trade_exit_reason"),
        latest_signal.get("exit_reason"),
    )
    state_panel = {
        "asset": asset,
        "event_slug": _coalesce(latest_signal.get("event_slug"), latest_state_row.get("event_slug")),
        "collected_at": _coalesce(latest_signal.get("collected_at"), latest_state_row.get("collected_at")),
        "market_implied_up_probability": _coalesce(
            latest_signal.get("market_implied_up_probability"),
            latest_state_row.get("market_implied_up_probability"),
        ),
        "latent_up_probability": _coalesce(
            latest_signal.get("latent_up_probability"),
            latest_state_row.get("latent_up_probability"),
        ),
        "fair_up_probability": latest_signal.get("fair_up_probability"),
        "fundamental_up_probability": _coalesce(
            latest_signal.get("fundamental_up_probability"),
            latest_state_row.get("fundamental_up_probability"),
        ),
    }
    edge_panel = {
        "fair_token_price": latest_signal.get("fair_token_price"),
        "best_bid": latest_signal.get("best_bid"),
        "best_ask": latest_signal.get("best_ask"),
        "buy_edge": latest_signal.get("buy_edge"),
        "hold_edge": latest_signal.get("hold_edge"),
        "sell_edge": latest_signal.get("sell_edge"),
        "decision": latest_signal.get("decision"),
        "seconds_to_end": latest_signal.get("seconds_to_end"),
    }
    execution_panel = {
        "position_state": latest_signal.get("position_state", "FLAT"),
        "pending_entry": bool(latest_signal.get("pending_entry", False)),
        "has_position": bool(latest_signal.get("has_position", False)),
        "latest_order_state": latest_order.get("status", "idle"),
        "latest_order_side": latest_order.get("side"),
        "latest_order_price": latest_order.get("price"),
        "latest_order_reason": latest_order.get("reason"),
        "exit_reason": exit_reason,
    }
    current = {
        "signal_timestamp": latest_signal.get("collected_at"),
        "event_slug": state_panel["event_slug"],
        "latest_signal_decision": latest_signal.get("decision"),
        "latest_signal_state": latest_signal.get("position_state"),
    }
    return {
        "asset": asset,
        "current_snapshot": current,
        "state_panel": state_panel,
        "edge_panel": edge_panel,
        "execution_panel": execution_panel,
    }


def _compact_signal_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "Time": _short_time(row.get("collected_at")),
        "Asset": _row_asset(row) or "-",
        "Event": row.get("event_slug"),
        "Decision": row.get("decision"),
        "Edge": _format_px(row.get("buy_edge")),
        "State": row.get("position_state"),
    }


def _compact_order_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "Time": _short_time(row.get("filled_at") or row.get("submitted_at")),
        "Asset": _row_asset(row) or "-",
        "Side": row.get("side"),
        "Status": row.get("status"),
        "Price": _format_px(row.get("price")),
        "Reason": row.get("reason"),
    }


def _compact_trade_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "Asset": _row_asset(row) or "-",
        "Exit": _short_time(row.get("exit_time")),
        "Reason": row.get("exit_reason"),
        "Net PnL": _format_px(row.get("net_pnl_with_fee", row.get("net_pnl"))),
        "Gross": _format_px(row.get("gross_pnl", row.get("pnl"))),
        "Hold": f"{_format_num(row.get('holding_seconds'), 1)}s",
    }


def _short_time(value: Any) -> str:
    if value in (None, ""):
        return "-"
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.strftime("%H:%M:%S")
    except ValueError:
        return str(value)


def _format_px(value: Any) -> str:
    number = _to_float(value)
    return "-" if number is None else f"{number:.4f}"


def _format_num(value: Any, digits: int = 2) -> str:
    number = _to_float(value)
    return "-" if number is None else f"{number:.{digits}f}"


def _to_float(value: Any) -> float | None:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, float) and pd.isna(value):
            continue
        return value
    return None


def _row_asset(row: dict[str, Any]) -> str | None:
    asset = str(row.get("asset", "")).strip().upper()
    if asset:
        return asset
    return _event_slug_asset(row.get("event_slug"))


def _event_slug_asset(event_slug: Any) -> str | None:
    slug = str(event_slug or "").strip().lower()
    if slug.startswith("btc-"):
        return "BTC"
    if slug.startswith("eth-"):
        return "ETH"
    return None
