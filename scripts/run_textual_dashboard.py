import argparse

from polymarket_quant.dashboard.runtime import EmbeddedLiveRuntimeConfig
from polymarket_quant.dashboard.textual_app import TextualDashboardConfig, run_textual_dashboard_app
from polymarket_quant.live import (
    DEFAULT_LIVE_ORDERBOOK_LEVELS_GLOB,
    DEFAULT_LIVE_ORDERBOOK_SUMMARY_GLOB,
    DEFAULT_LIVE_SPOT_GLOB,
    DEFAULT_LIVE_STATE_OUTPUT_DIR,
    DEFAULT_TRANSITION_MODEL_BTC_PATH,
    DEFAULT_TRANSITION_MODEL_ETH_PATH,
)
from polymarket_quant.dashboard.runtime import DEFAULT_PAPER_OUTPUT_DIR, DEFAULT_LIVE_SIGNAL_OUTPUT_DIR, DEFAULT_TRANSITION_MODEL_PATH

DEFAULT_EVENT_STATE_PATH = f"{DEFAULT_LIVE_STATE_OUTPUT_DIR}/live_event_state_latest.parquet"
DEFAULT_LIVE_SIGNAL_PATH = f"{DEFAULT_LIVE_SIGNAL_OUTPUT_DIR}/live_signal_events_latest.jsonl"
DEFAULT_PAPER_SIGNAL_PATH = f"{DEFAULT_PAPER_OUTPUT_DIR}/paper_signal_events_latest.jsonl"
DEFAULT_PAPER_ORDER_PATH = f"{DEFAULT_PAPER_OUTPUT_DIR}/paper_order_events_latest.jsonl"
DEFAULT_PAPER_TRADE_PATH = f"{DEFAULT_PAPER_OUTPUT_DIR}/paper_trade_ledger_latest.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Textual live dashboard for state/fair/edge/paper execution.")
    parser.add_argument("--runtime-mode", choices=["embedded-live", "artifact-view"], default="embedded-live", help="Run the live engine inside the Textual app or only render existing artifacts")
    parser.add_argument("--source-mode", choices=["direct-live", "live-capture", "event-state"], default="direct-live", help="Embedded live source mode")
    parser.add_argument("--transition-model-path", default=DEFAULT_TRANSITION_MODEL_PATH, help="Fallback shared transition model artifact for embedded-live mode")
    parser.add_argument("--transition-model-btc-path", default=DEFAULT_TRANSITION_MODEL_BTC_PATH, help="BTC transition model artifact for embedded-live mode")
    parser.add_argument("--transition-model-eth-path", default=DEFAULT_TRANSITION_MODEL_ETH_PATH, help="ETH transition model artifact for embedded-live mode")
    parser.add_argument("--config-path", default="configs/base.yaml", help="Repo config path for direct-live mode")
    parser.add_argument("--orderbook-summary-glob", default=DEFAULT_LIVE_ORDERBOOK_SUMMARY_GLOB, help="Live-capture orderbook summary glob")
    parser.add_argument("--orderbook-levels-glob", default=DEFAULT_LIVE_ORDERBOOK_LEVELS_GLOB, help="Live-capture orderbook levels glob")
    parser.add_argument("--spot-glob", default=DEFAULT_LIVE_SPOT_GLOB, help="Live-capture spot glob")
    parser.add_argument("--live-state-output-dir", default=DEFAULT_LIVE_STATE_OUTPUT_DIR, help="Output dir for live state parquet")
    parser.add_argument("--live-signal-output-dir", default=DEFAULT_LIVE_SIGNAL_OUTPUT_DIR, help="Output dir for embedded live signal artifacts")
    parser.add_argument("--paper-output-dir", default=DEFAULT_PAPER_OUTPUT_DIR, help="Output dir for embedded paper artifacts")
    parser.add_argument("--event-duration-seconds", type=int, default=300, help="Active market window size")
    parser.add_argument("--event-slug-prefixes", nargs="+", default=["btc-updown-5m", "eth-updown-5m"], help="Direct-live event slug prefixes")
    parser.add_argument("--series-slugs", nargs="+", default=["btc-up-or-down-5m", "eth-up-or-down-5m"], help="Direct-live Gamma series slugs")
    parser.add_argument("--event-state-path", default=DEFAULT_EVENT_STATE_PATH, help="Latest event_state parquet path")
    parser.add_argument("--live-signal-path", default=DEFAULT_LIVE_SIGNAL_PATH, help="Latest live-signal JSONL path")
    parser.add_argument("--paper-signal-path", default=DEFAULT_PAPER_SIGNAL_PATH, help="Latest paper-signal JSONL path")
    parser.add_argument("--paper-order-path", default=DEFAULT_PAPER_ORDER_PATH, help="Latest paper-order JSONL path")
    parser.add_argument("--paper-trade-path", default=DEFAULT_PAPER_TRADE_PATH, help="Latest paper-trade ledger JSONL path")
    parser.add_argument("--refresh-seconds", type=float, default=0.25, help="UI refresh cadence")
    parser.add_argument("--poll-seconds", type=float, default=1.0, help="Embedded live market poll cadence")
    parser.add_argument("--max-points", type=int, default=80, help="Maximum history points for sparkline panels")
    parser.add_argument("--max-rows", type=int, default=12, help="Maximum rows for stream panels")
    parser.add_argument("--n-samples", type=int, default=1000, help="Pricing simulation paths")
    parser.add_argument("--simulation-dt-seconds", type=float, default=1.0, help="Simulation dt in seconds")
    parser.add_argument("--rollout-horizon-seconds", type=float, default=0.0, help="Optional rollout step override")
    parser.add_argument("--edge-threshold", type=float, default=0.0, help="Detector buy-signal threshold")
    parser.add_argument("--open-cooldown-seconds", type=float, default=30.0, help="Skip new entries during the first N seconds after market open")
    parser.add_argument("--entry-edge-threshold", type=float, default=0.03, help="Baseline strategy entry threshold")
    parser.add_argument("--exit-hold-edge-threshold", type=float, default=0.0, help="Baseline strategy exit hold-edge threshold")
    parser.add_argument("--max-holding-seconds", type=float, default=None, help="Optional baseline strategy max holding time; disabled by default")
    parser.add_argument("--forced-exit-seconds-to-end", type=float, default=None, help="Optional forced exit time-to-end threshold; disabled by default")
    parser.add_argument("--max-edge-cap", type=float, default=0.12, help="Ignore oversized entry edges above this cap")
    parser.add_argument("--market-probability-exclusion-low", type=float, default=0.40, help="Lower bound of market implied probability chop zone exclusion")
    parser.add_argument("--market-probability-exclusion-high", type=float, default=0.60, help="Upper bound of market implied probability chop zone exclusion")
    parser.add_argument("--confident-exit-fair-probability-threshold", type=float, default=0.86, help="Relax exit threshold when fair probability is above this level")
    parser.add_argument("--confident-exit-window-seconds", type=float, default=60.0, help="Apply dynamic confident exit logic inside this time-to-end window")
    parser.add_argument("--confident-exit-hold-edge-floor", type=float, default=-0.02, help="Most negative hold-edge threshold allowed near settlement for high-confidence states")
    parser.add_argument(
        "--verbose-runtime-logs",
        action="store_true",
        help="Print live runtime timing directly to the terminal and run Textual inline so the logs remain in scrollback",
    )
    args = parser.parse_args()

    run_textual_dashboard_app(
        TextualDashboardConfig(
            event_state_path=args.event_state_path,
            live_signal_path=args.live_signal_path,
            paper_signal_path=args.paper_signal_path,
            paper_order_path=args.paper_order_path,
            paper_trade_path=args.paper_trade_path,
            refresh_seconds=args.refresh_seconds,
            poll_seconds=args.poll_seconds,
            max_points=args.max_points,
            max_rows=args.max_rows,
            inline_mode=args.verbose_runtime_logs,
            inline_no_clear=args.verbose_runtime_logs,
            runtime=(
                EmbeddedLiveRuntimeConfig(
                    runtime_mode=args.runtime_mode,
                    source_mode=args.source_mode,
                    transition_model_path=args.transition_model_path,
                    transition_model_btc_path=args.transition_model_btc_path,
                    transition_model_eth_path=args.transition_model_eth_path,
                    config_path=args.config_path,
                    event_state_path=args.event_state_path,
                    orderbook_summary_glob=args.orderbook_summary_glob,
                    orderbook_levels_glob=args.orderbook_levels_glob,
                    spot_glob=args.spot_glob,
                    live_state_output_dir=args.live_state_output_dir,
                    live_signal_output_dir=args.live_signal_output_dir,
                    paper_output_dir=args.paper_output_dir,
                    event_duration_seconds=args.event_duration_seconds,
                    event_slug_prefixes=tuple(args.event_slug_prefixes),
                    series_slugs=tuple(args.series_slugs),
                    n_samples=args.n_samples,
                    simulation_dt_seconds=args.simulation_dt_seconds,
                    rollout_horizon_seconds=args.rollout_horizon_seconds,
                    edge_threshold=args.edge_threshold,
                    open_cooldown_seconds=args.open_cooldown_seconds,
                    entry_edge_threshold=args.entry_edge_threshold,
                    exit_hold_edge_threshold=args.exit_hold_edge_threshold,
                    max_holding_seconds=args.max_holding_seconds,
                    forced_exit_seconds_to_end=args.forced_exit_seconds_to_end,
                    max_edge_cap=args.max_edge_cap,
                    market_probability_exclusion_low=args.market_probability_exclusion_low,
                    market_probability_exclusion_high=args.market_probability_exclusion_high,
                    confident_exit_fair_probability_threshold=args.confident_exit_fair_probability_threshold,
                    confident_exit_window_seconds=args.confident_exit_window_seconds,
                    confident_exit_hold_edge_floor=args.confident_exit_hold_edge_floor,
                    quiet_runtime_logs=not args.verbose_runtime_logs,
                    print_timing_to_terminal=args.verbose_runtime_logs,
                )
                if args.runtime_mode == "embedded-live"
                else None
            ),
        )
    )


if __name__ == "__main__":
    main()
