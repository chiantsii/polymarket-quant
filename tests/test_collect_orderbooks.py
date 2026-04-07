from datetime import datetime, timedelta, timezone

from scripts.collect_orderbooks import _next_events_by_series


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


class MockSeriesClient:
    def __init__(self, now: datetime):
        self.now = now

    def fetch_series(self, slug: str):
        prefix = "btc-updown-5m" if slug.startswith("btc") else "eth-updown-5m"
        return [
            {
                "slug": slug,
                "events": [
                    {
                        "slug": f"{prefix}-current",
                        "startTime": _iso(self.now - timedelta(minutes=1)),
                        "endDate": _iso(self.now + timedelta(minutes=4)),
                        "closed": False,
                    },
                    {
                        "slug": f"{prefix}-next",
                        "startTime": _iso(self.now + timedelta(minutes=4)),
                        "endDate": _iso(self.now + timedelta(minutes=9)),
                        "closed": False,
                    },
                    {
                        "slug": "xrp-updown-5m-next",
                        "startTime": _iso(self.now + timedelta(minutes=4)),
                        "endDate": _iso(self.now + timedelta(minutes=9)),
                        "closed": False,
                    },
                ],
            }
        ]


def test_next_events_by_series_selects_future_complete_window() -> None:
    now = datetime.now(timezone.utc)
    client = MockSeriesClient(now)

    events = _next_events_by_series(
        client=client,
        series_slugs=["btc-up-or-down-5m", "eth-up-or-down-5m"],
        event_slug_prefixes=["btc-updown-5m", "eth-updown-5m"],
        now=now,
    )

    assert events["btc-up-or-down-5m"]["slug"] == "btc-updown-5m-next"
    assert events["eth-up-or-down-5m"]["slug"] == "eth-updown-5m-next"
