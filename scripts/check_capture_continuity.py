import argparse
import glob
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


DEFAULT_ORDERBOOK_GLOB = "data/*/processed/polymarket/crypto_5m_orderbook_summary_*.parquet"
DEFAULT_SPOT_GLOB = "data/*/processed/spot/binance_spot_ticks_*.parquet"


def _matching_paths(pattern: str, include_latest: bool) -> list[Path]:
    paths = [Path(path) for path in sorted(glob.glob(pattern))]
    if not include_latest:
        paths = [path for path in paths if "latest" not in path.name]
    return paths


def _parse_iso8601_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, format="ISO8601", errors="coerce")


def _event_start_from_slug(event_slug: str) -> datetime | None:
    suffix = str(event_slug).rsplit("-", 1)[-1]
    if not suffix.isdigit():
        return None
    return datetime.fromtimestamp(int(suffix), tz=timezone.utc)


@dataclass(frozen=True)
class FileScanResult:
    path: str
    source: str
    asset: str
    rows: int
    bad_timestamp_rows: int
    event_slugs: int
    min_collected_at: str | None
    max_collected_at: str | None


def _file_asset(path: Path) -> str:
    return path.parents[2].name


def _scan_file(path: Path, *, source: str, required_columns: list[str]) -> tuple[FileScanResult, pd.DataFrame]:
    frame = pd.read_parquet(path, columns=required_columns)
    parsed = _parse_iso8601_utc(frame["collected_at"].astype(str))
    bad_rows = int(parsed.isna().sum())
    valid = frame.loc[parsed.notna()].copy()
    valid["_collected_at_dt"] = parsed.loc[parsed.notna()].to_numpy()
    result = FileScanResult(
        path=str(path),
        source=source,
        asset=_file_asset(path),
        rows=len(frame),
        bad_timestamp_rows=bad_rows,
        event_slugs=int(valid.get("event_slug", pd.Series(dtype=object)).nunique()),
        min_collected_at=valid["_collected_at_dt"].min().isoformat() if not valid.empty else None,
        max_collected_at=valid["_collected_at_dt"].max().isoformat() if not valid.empty else None,
    )
    return result, valid


def _summarize_event_source(frame: pd.DataFrame, *, source: str, event_duration_seconds: float) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "asset",
                "event_slug",
                "source",
                "rows",
                "observed_start",
                "observed_end",
                "event_start",
                "event_end",
                "start_offset_seconds",
                "end_shortfall_seconds",
                "max_internal_gap_seconds",
                "median_internal_gap_seconds",
            ]
        )

    summaries: list[dict[str, object]] = []
    for (asset, event_slug), group in frame.groupby(["asset", "event_slug"], sort=False):
        ordered = group.sort_values("_collected_at_dt").reset_index(drop=True)
        event_start = _event_start_from_slug(event_slug)
        if event_start is None:
            continue
        event_end = event_start + timedelta(seconds=event_duration_seconds)
        observed_start = ordered["_collected_at_dt"].iloc[0]
        observed_end = ordered["_collected_at_dt"].iloc[-1]
        gaps = ordered["_collected_at_dt"].diff().dt.total_seconds().dropna()
        summaries.append(
            {
                "asset": asset,
                "event_slug": event_slug,
                "source": source,
                "rows": len(ordered),
                "observed_start": observed_start.isoformat(),
                "observed_end": observed_end.isoformat(),
                "event_start": event_start.isoformat(),
                "event_end": event_end.isoformat(),
                "start_offset_seconds": float((observed_start - event_start).total_seconds()),
                "end_shortfall_seconds": float((event_end - observed_end).total_seconds()),
                "max_internal_gap_seconds": float(gaps.max()) if not gaps.empty else 0.0,
                "median_internal_gap_seconds": float(gaps.median()) if not gaps.empty else 0.0,
            }
        )
    return pd.DataFrame(summaries)


def _summarize_event_slug_sequence(frame: pd.DataFrame, *, event_duration_seconds: float) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["asset", "event_slug", "event_start_epoch", "prev_epoch", "gap_seconds", "missing_windows"])

    unique_events = (
        frame[["asset", "event_slug"]]
        .dropna()
        .drop_duplicates()
        .assign(
            event_start_epoch=lambda rows: pd.to_numeric(
                rows["event_slug"].astype(str).str.rsplit("-", n=1).str[-1],
                errors="coerce",
            )
        )
        .dropna(subset=["event_start_epoch"])
        .copy()
    )
    unique_events["event_start_epoch"] = unique_events["event_start_epoch"].astype(int)
    unique_events = unique_events.sort_values(["asset", "event_start_epoch"]).reset_index(drop=True)
    unique_events["prev_epoch"] = unique_events.groupby("asset", sort=False)["event_start_epoch"].shift(1)
    unique_events["gap_seconds"] = unique_events["event_start_epoch"] - unique_events["prev_epoch"]
    unique_events["missing_windows"] = (
        ((unique_events["gap_seconds"] / event_duration_seconds) - 1)
        .clip(lower=0)
        .fillna(0)
        .astype(int)
    )
    return unique_events


def _print_console_summary(
    *,
    file_summary: pd.DataFrame,
    event_summary: pd.DataFrame,
    sequence_summary: pd.DataFrame,
    max_gap_seconds: float,
    boundary_tolerance_seconds: float,
) -> None:
    total_bad = int(file_summary["bad_timestamp_rows"].sum()) if not file_summary.empty else 0
    print(f"bad_timestamp_rows={total_bad}")

    if not sequence_summary.empty:
        missing = sequence_summary[sequence_summary["missing_windows"] > 0].copy()
        print(f"event_sequence_missing_runs={len(missing)}")
        if not missing.empty:
            print("event_sequence_examples:")
            print(
                missing[["asset", "event_slug", "prev_epoch", "event_start_epoch", "missing_windows"]]
                .head(10)
                .to_string(index=False)
            )

    if not event_summary.empty:
        flagged = event_summary[
            (event_summary["max_internal_gap_seconds"] > max_gap_seconds)
            | (event_summary["start_offset_seconds"] > boundary_tolerance_seconds)
            | (event_summary["end_shortfall_seconds"] > boundary_tolerance_seconds)
        ].copy()
        print(f"flagged_event_windows={len(flagged)}")
        if not flagged.empty:
            print("flagged_event_examples:")
            print(
                flagged[
                    [
                        "asset",
                        "event_slug",
                        "source",
                        "rows",
                        "start_offset_seconds",
                        "end_shortfall_seconds",
                        "max_internal_gap_seconds",
                    ]
                ]
                .head(20)
                .to_string(index=False)
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Check capture continuity for processed orderbook and spot parquet files.")
    parser.add_argument("--orderbook-glob", default=DEFAULT_ORDERBOOK_GLOB, help="Orderbook summary parquet glob")
    parser.add_argument("--spot-glob", default=DEFAULT_SPOT_GLOB, help="Spot parquet glob")
    parser.add_argument("--output-dir", default="artifacts/continuity", help="Directory for continuity reports")
    parser.add_argument("--event-duration-seconds", type=float, default=300.0, help="Expected event duration")
    parser.add_argument("--max-gap-seconds", type=float, default=2.0, help="Flag event windows whose internal gap exceeds this threshold")
    parser.add_argument("--boundary-tolerance-seconds", type=float, default=2.0, help="Flag event windows whose start/end coverage misses this tolerance")
    parser.add_argument("--include-latest", action="store_true", help="Include *_latest.parquet inputs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    orderbook_paths = _matching_paths(args.orderbook_glob, include_latest=args.include_latest)
    spot_paths = _matching_paths(args.spot_glob, include_latest=args.include_latest)
    if not orderbook_paths:
        raise FileNotFoundError(f"No parquet files matched {args.orderbook_glob}")
    if not spot_paths:
        raise FileNotFoundError(f"No parquet files matched {args.spot_glob}")

    file_results: list[FileScanResult] = []
    orderbook_frames: list[pd.DataFrame] = []
    spot_frames: list[pd.DataFrame] = []

    for path in orderbook_paths:
        result, frame = _scan_file(
            path,
            source="orderbook_summary",
            required_columns=["collected_at", "event_slug", "asset"],
        )
        file_results.append(result)
        orderbook_frames.append(frame)

    for path in spot_paths:
        result, frame = _scan_file(
            path,
            source="spot",
            required_columns=["collected_at", "event_slug", "asset", "price"],
        )
        file_results.append(result)
        spot_frames.append(frame)

    file_summary = pd.DataFrame([result.__dict__ for result in file_results])
    orderbooks = pd.concat(orderbook_frames, ignore_index=True) if orderbook_frames else pd.DataFrame()
    spots = pd.concat(spot_frames, ignore_index=True) if spot_frames else pd.DataFrame()

    event_summary = pd.concat(
        [
            _summarize_event_source(orderbooks, source="orderbook_summary", event_duration_seconds=args.event_duration_seconds),
            _summarize_event_source(spots, source="spot", event_duration_seconds=args.event_duration_seconds),
        ],
        ignore_index=True,
    )
    sequence_summary = _summarize_event_slug_sequence(orderbooks, event_duration_seconds=args.event_duration_seconds)

    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    file_summary_path = output_dir / f"crypto_5m_capture_continuity_files_{run_timestamp}.parquet"
    event_summary_path = output_dir / f"crypto_5m_capture_continuity_events_{run_timestamp}.parquet"
    sequence_summary_path = output_dir / f"crypto_5m_capture_continuity_sequence_{run_timestamp}.parquet"
    latest_file_summary_path = output_dir / "crypto_5m_capture_continuity_files_latest.parquet"
    latest_event_summary_path = output_dir / "crypto_5m_capture_continuity_events_latest.parquet"
    latest_sequence_summary_path = output_dir / "crypto_5m_capture_continuity_sequence_latest.parquet"
    latest_metadata_path = output_dir / "crypto_5m_capture_continuity_latest.json"

    file_summary.to_parquet(file_summary_path, index=False)
    event_summary.to_parquet(event_summary_path, index=False)
    sequence_summary.to_parquet(sequence_summary_path, index=False)
    file_summary.to_parquet(latest_file_summary_path, index=False)
    event_summary.to_parquet(latest_event_summary_path, index=False)
    sequence_summary.to_parquet(latest_sequence_summary_path, index=False)

    metadata = {
        "files_rows": int(len(file_summary)),
        "event_rows": int(len(event_summary)),
        "sequence_rows": int(len(sequence_summary)),
        "bad_timestamp_rows": int(file_summary["bad_timestamp_rows"].sum()) if not file_summary.empty else 0,
        "output_paths": {
            "files": str(file_summary_path),
            "events": str(event_summary_path),
            "sequence": str(sequence_summary_path),
        },
        "latest_paths": {
            "files": str(latest_file_summary_path),
            "events": str(latest_event_summary_path),
            "sequence": str(latest_sequence_summary_path),
        },
    }
    latest_metadata_path.write_text(json.dumps(metadata, indent=2))

    _print_console_summary(
        file_summary=file_summary,
        event_summary=event_summary,
        sequence_summary=sequence_summary,
        max_gap_seconds=args.max_gap_seconds,
        boundary_tolerance_seconds=args.boundary_tolerance_seconds,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
