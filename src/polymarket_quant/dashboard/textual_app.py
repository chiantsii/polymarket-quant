from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from polymarket_quant.dashboard.data import build_live_dashboard_snapshot
from polymarket_quant.dashboard.runtime import EmbeddedLiveRuntime, EmbeddedLiveRuntimeConfig

try:
    from textual.app import App, ComposeResult
    from textual.containers import Grid, Horizontal, Vertical
    from textual.widgets import Footer, Header, Static
except ModuleNotFoundError:  # pragma: no cover - runtime dependency
    App = object  # type: ignore[assignment]
    ComposeResult = object  # type: ignore[assignment]
    Grid = object  # type: ignore[assignment]
    Horizontal = object  # type: ignore[assignment]
    Vertical = object  # type: ignore[assignment]
    Footer = object  # type: ignore[assignment]
    Header = object  # type: ignore[assignment]
    Static = object  # type: ignore[assignment]


@dataclass(frozen=True)
class TextualDashboardConfig:
    event_state_path: str
    live_signal_path: str
    paper_signal_path: str
    paper_order_path: str
    paper_trade_path: str
    refresh_seconds: float = 2.0
    max_points: int = 80
    max_rows: int = 12
    runtime: EmbeddedLiveRuntimeConfig | None = None


def run_textual_dashboard_app(config: TextualDashboardConfig) -> None:
    try:
        from textual.app import App as _TextualApp  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "Textual is not installed. Install it with `pip install textual` or add it to the project environment before running the dashboard."
        ) from exc

    PolymarketDashboardApp(config).run()


class PolymarketDashboardApp(App):  # pragma: no cover - UI runtime
    TITLE = "Polymarket Quant Terminal Board"
    SUB_TITLE = "State -> Fair -> Edge -> Paper Execution"
    CSS = """
    Screen {
        background: #05070c;
        color: #f6f8fc;
    }

    #root {
        layout: vertical;
        padding: 1 2;
    }

    #top-grid {
        layout: grid;
        grid-size: 2 1;
        grid-gutter: 1 2;
        height: auto;
        margin-bottom: 1;
    }

    #history-top-grid {
        layout: grid;
        grid-size: 3 1;
        grid-gutter: 1 2;
        height: 14;
        margin-bottom: 1;
    }

    #stream-grid {
        layout: grid;
        grid-size: 3 1;
        grid-gutter: 1 2;
        height: 16;
    }

    .panel {
        border: round #2f3445;
        background: #0d1118;
        padding: 1 2;
    }

    .section-banner {
        border: round #24424b;
        background: #081018;
        padding: 0 2;
        color: #8ea0bd;
        text-style: bold;
        margin-bottom: 1;
    }

    .panel-title {
        color: #8ea0bd;
        text-style: bold;
    }

    .metric {
        color: #f6f8fc;
    }
    """

    def __init__(self, config: TextualDashboardConfig) -> None:
        super().__init__()
        self.config = config
        self.runtime = EmbeddedLiveRuntime(config.runtime) if config.runtime is not None else None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="root"):
            yield Static(id="hero", classes="panel")
            yield Static(id="snapshot-banner", classes="section-banner")
            with Grid(id="top-grid"):
                yield Static(id="btc-snapshot-panel", classes="panel")
                yield Static(id="eth-snapshot-panel", classes="panel")
            yield Static(id="history-banner", classes="section-banner")
            with Grid(id="history-top-grid"):
                yield Static(id="pnl-panel", classes="panel")
                yield Static(id="probability-panel", classes="panel")
                yield Static(id="pnl-curve-panel", classes="panel")
            with Grid(id="stream-grid"):
                yield Static(id="signal-stream-panel", classes="panel")
                yield Static(id="order-stream-panel", classes="panel")
                yield Static(id="trade-stream-panel", classes="panel")
        yield Footer()

    def on_mount(self) -> None:
        self.set_interval(max(self.config.refresh_seconds, 1.0), self.refresh_dashboard)
        self.refresh_dashboard()

    def refresh_dashboard(self) -> None:
        if self.runtime is not None:
            self.runtime.poll_once()
            runtime_paths = self.runtime.snapshot_paths()
            event_state_path = runtime_paths["event_state_path"]
            live_signal_path = runtime_paths["live_signal_path"]
            paper_signal_path = runtime_paths["paper_signal_path"]
            paper_order_path = runtime_paths["paper_order_path"]
            paper_trade_path = runtime_paths["paper_trade_path"]
        else:
            event_state_path = self.config.event_state_path
            live_signal_path = self.config.live_signal_path
            paper_signal_path = self.config.paper_signal_path
            paper_order_path = self.config.paper_order_path
            paper_trade_path = self.config.paper_trade_path
        snapshot = build_live_dashboard_snapshot(
            event_state_path=event_state_path,
            live_signal_path=live_signal_path,
            paper_signal_path=paper_signal_path,
            paper_order_path=paper_order_path,
            paper_trade_path=paper_trade_path,
            max_points=self.config.max_points,
            max_rows=self.config.max_rows,
        )
        self.query_one("#hero", Static).update(_render_hero(snapshot))
        self.query_one("#snapshot-banner", Static).update(_render_snapshot_banner(snapshot))
        self.query_one("#history-banner", Static).update(_render_history_banner(snapshot))
        self.query_one("#btc-snapshot-panel", Static).update(_render_asset_snapshot_panel(snapshot, "BTC"))
        self.query_one("#eth-snapshot-panel", Static).update(_render_asset_snapshot_panel(snapshot, "ETH"))
        self.query_one("#pnl-panel", Static).update(_render_pnl_panel(snapshot))
        self.query_one("#probability-panel", Static).update(_render_probability_panel(snapshot))
        self.query_one("#pnl-curve-panel", Static).update(_render_curve_panel(snapshot))
        self.query_one("#signal-stream-panel", Static).update(_render_stream_panel("Signal Stream", snapshot.get("recent_signals", [])))
        self.query_one("#order-stream-panel", Static).update(_render_stream_panel("Order Stream", snapshot.get("recent_orders", [])))
        self.query_one("#trade-stream-panel", Static).update(_render_stream_panel("Trade Ledger", snapshot.get("recent_trades", [])))


def _render_hero(snapshot: dict[str, Any]) -> str:
    asset_snapshots = snapshot.get("asset_snapshots", {})
    btc = asset_snapshots.get("BTC", {}).get("state_panel", {})
    eth = asset_snapshots.get("ETH", {}).get("state_panel", {})
    return (
        "[b cyan]POLYMARKET QUANT TERMINAL BOARD[/b cyan]\n"
        f"[dim]Generated[/dim] {snapshot.get('generated_at', '-')}\n"
        f"[dim]BTC[/dim] {btc.get('event_slug', '-')}  [dim]ts[/dim] {btc.get('collected_at', '-')}\n"
        f"[dim]ETH[/dim] {eth.get('event_slug', '-')}  [dim]ts[/dim] {eth.get('collected_at', '-')}"
    )


def _render_snapshot_banner(snapshot: dict[str, Any]) -> str:
    asset_snapshots = snapshot.get("asset_snapshots", {})
    btc = asset_snapshots.get("BTC", {}).get("current_snapshot", {})
    eth = asset_snapshots.get("ETH", {}).get("current_snapshot", {})
    return (
        "[bold cyan]CURRENT SNAPSHOT[/bold cyan]  "
        f"[dim]BTC[/dim] {btc.get('signal_timestamp') or '-'} / {btc.get('latest_signal_decision') or '-'} / {btc.get('latest_signal_state') or '-'}  "
        f"[dim]ETH[/dim] {eth.get('signal_timestamp') or '-'} / {eth.get('latest_signal_decision') or '-'} / {eth.get('latest_signal_state') or '-'}"
    )


def _render_history_banner(snapshot: dict[str, Any]) -> str:
    history = snapshot.get("history_summary", {})
    return (
        "[bold magenta]CUMULATIVE PAPER HISTORY[/bold magenta]  "
        f"[dim]latest_order[/dim] {history.get('latest_order_time') or '-'}  "
        f"[dim]latest_trade[/dim] {history.get('latest_trade_time') or '-'}  "
        f"[dim]orders[/dim] {history.get('filled_orders') or 0}  "
        f"[dim]trades[/dim] {history.get('closed_trades') or 0}"
    )


def _render_state_panel(snapshot: dict[str, Any]) -> str:
    state = snapshot.get("state_panel", {})
    return _panel(
        "STATE PANEL",
        [
            _metric("Latent Probability", _fmt_pct(state.get("latent_up_probability"))),
            _metric("Market Implied", _fmt_pct(state.get("market_implied_up_probability"))),
            _metric("Fair Probability", _fmt_pct(state.get("fair_up_probability"))),
            _metric("Fundamental", _fmt_pct(state.get("fundamental_up_probability"))),
        ],
    )


def _render_asset_snapshot_panel(snapshot: dict[str, Any], asset: str) -> str:
    asset_snapshot = snapshot.get("asset_snapshots", {}).get(asset, {})
    state = asset_snapshot.get("state_panel", {})
    edge = asset_snapshot.get("edge_panel", {})
    execution = asset_snapshot.get("execution_panel", {})
    if not state and not edge and not execution:
        return _panel(f"{asset} CURRENT SNAPSHOT", ["No live rows yet"])
    return _panel(
        f"{asset} CURRENT SNAPSHOT",
        [
            _metric("Event", str(state.get("event_slug") or "-")),
            _metric("Latent / Market / Fair", f"{_fmt_pct(state.get('latent_up_probability'))} / {_fmt_pct(state.get('market_implied_up_probability'))} / {_fmt_pct(state.get('fair_up_probability'))}"),
            _metric("Best Bid / Ask", f"{_fmt_num(edge.get('best_bid'), 4)} / {_fmt_num(edge.get('best_ask'), 4)}"),
            _metric("Buy / Hold / Sell", f"{_fmt_num(edge.get('buy_edge'), 4)} / {_fmt_num(edge.get('hold_edge'), 4)} / {_fmt_num(edge.get('sell_edge'), 4)}"),
            _metric("Decision / State", f"{edge.get('decision') or '-'} / {execution.get('position_state') or '-'}"),
            _metric("Exit Reason", str(execution.get("exit_reason") or "-")),
        ],
    )


def _render_edge_panel(snapshot: dict[str, Any]) -> str:
    edge = snapshot.get("edge_panel", {})
    return _panel(
        "EDGE PANEL",
        [
            _metric("Fair", _fmt_num(edge.get("fair_token_price"), 4)),
            _metric("Best Bid / Ask", f"{_fmt_num(edge.get('best_bid'), 4)} / {_fmt_num(edge.get('best_ask'), 4)}"),
            _metric(
                "Buy / Hold / Sell",
                f"{_fmt_num(edge.get('buy_edge'), 4)} / {_fmt_num(edge.get('hold_edge'), 4)} / {_fmt_num(edge.get('sell_edge'), 4)}",
            ),
            _metric("Decision", str(edge.get("decision") or "-")),
            _metric("Seconds To End", _fmt_num(edge.get("seconds_to_end"), 1)),
        ],
    )


def _render_execution_panel(snapshot: dict[str, Any]) -> str:
    execution = snapshot.get("execution_panel", {})
    return _panel(
        "PAPER EXECUTION PANEL",
        [
            _metric("Position State", str(execution.get("position_state") or "FLAT")),
            _metric("Pending Entry", "YES" if execution.get("pending_entry") else "NO"),
            _metric("Filled Orders", str(int(execution.get("filled_orders") or 0))),
            _metric(
                "Latest Order",
                f"{execution.get('latest_order_side') or '-'} @ {_fmt_num(execution.get('latest_order_price'), 4)}",
            ),
            _metric("Order State", str(execution.get("latest_order_state") or "idle")),
            _metric("Exit Reason", str(execution.get("exit_reason") or "-")),
        ],
    )


def _render_pnl_panel(snapshot: dict[str, Any]) -> str:
    pnl = snapshot.get("pnl_panel", {})
    return _panel(
        "CUMULATIVE PNL PANEL",
        [
            _metric("Realized PnL", _fmt_num(pnl.get("realized_pnl"), 4)),
            _metric("Gross / Net / Fee", f"{_fmt_num(pnl.get('gross_pnl'), 4)} / {_fmt_num(pnl.get('net_pnl'), 4)} / {_fmt_num(pnl.get('fee_cost'), 4)}"),
            _metric("Win Rate", _fmt_pct(pnl.get("win_rate"))),
            _metric("Holding Time", f"{_fmt_num(pnl.get('mean_holding_seconds'), 1)}s"),
            _metric("Trade Count", str(int(pnl.get("trade_count") or 0))),
        ],
    )


def _render_probability_panel(snapshot: dict[str, Any]) -> str:
    series = snapshot.get("probability_series", {})
    market = _sparkline(series.get("market", []), minimum=0.0, maximum=1.0)
    latent = _sparkline(series.get("latent", []), minimum=0.0, maximum=1.0)
    fair = _sparkline(series.get("fair", []), minimum=0.0, maximum=1.0)
    return _panel(
        "RECENT SIGNAL PROBABILITY TRAJECTORY",
        [
            "[cyan]Market[/cyan] " + market,
            "[green]Latent[/green] " + latent,
            "[magenta]Fair  [/magenta] " + fair,
        ],
    )


def _render_curve_panel(snapshot: dict[str, Any]) -> str:
    curve = snapshot.get("pnl_curve", {})
    values = curve.get("cumulative_net", [])
    minimum = min(values) if values else 0.0
    maximum = max(values) if values else 1.0
    if minimum == maximum:
        maximum = minimum + 1.0
    return _panel(
        "CUMULATIVE PNL CURVE",
        [
            _sparkline(values, minimum=minimum, maximum=maximum),
            "",
            f"Latest Net PnL: {_fmt_num(values[-1], 4) if values else '-'}",
        ],
    )


def _render_stream_panel(title: str, rows: list[dict[str, Any]]) -> str:
    if not rows:
        return _panel(title, ["No rows yet"])
    headers = list(rows[0].keys())
    widths = {
        header: max(len(header), *(len(str(row.get(header, ""))) for row in rows))
        for header in headers
    }
    header_line = "  ".join(header.ljust(widths[header]) for header in headers)
    separator = "  ".join("-" * widths[header] for header in headers)
    body = [
        "  ".join(str(row.get(header, "-")).ljust(widths[header]) for header in headers)
        for row in rows
    ]
    return _panel(title, [header_line, separator, *body])


def _panel(title: str, lines: list[str]) -> str:
    return f"[bold #8ea0bd]{title}[/bold #8ea0bd]\n" + "\n".join(lines)


def _metric(label: str, value: str) -> str:
    return f"[#8ea0bd]{label:<18}[/#8ea0bd] [white]{value}[/white]"


def _fmt_pct(value: Any) -> str:
    number = _to_float(value)
    return "-" if number is None else f"{number * 100:.2f}%"


def _fmt_num(value: Any, digits: int) -> str:
    number = _to_float(value)
    return "-" if number is None else f"{number:.{digits}f}"


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _sparkline(values: list[Any], *, minimum: float, maximum: float, width: int = 48) -> str:
    bars = "▁▂▃▄▅▆▇█"
    finite = [_to_float(value) for value in values]
    finite = [value for value in finite if value is not None]
    if not finite:
        return "·" * min(width, 16)
    sampled = _sample_values(finite, width)
    span = max(maximum - minimum, 1e-9)
    chars: list[str] = []
    for value in sampled:
        clipped = max(min(value, maximum), minimum)
        normalized = (clipped - minimum) / span
        index = min(int(normalized * (len(bars) - 1)), len(bars) - 1)
        chars.append(bars[index])
    return "".join(chars)


def _sample_values(values: list[float], width: int) -> list[float]:
    if len(values) <= width:
        return values
    step = (len(values) - 1) / max(width - 1, 1)
    return [values[int(round(i * step))] for i in range(width)]
