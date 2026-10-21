"""Live NFL totals from The Odds API (https://the-odds-api.com)."""
import pandas as pd
import requests

from data import FULL_TO_ABBR

API_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"


def parse_events(events: list) -> pd.DataFrame:
    """Flatten the API payload to one row per (game, sportsbook)."""
    rows = []
    for ev in events:
        home = FULL_TO_ABBR.get(ev.get("home_team"), ev.get("home_team"))
        away = FULL_TO_ABBR.get(ev.get("away_team"), ev.get("away_team"))
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                if mk.get("key") != "totals":
                    continue
                over = next((o for o in mk["outcomes"] if o["name"] == "Over"), None)
                under = next((o for o in mk["outcomes"] if o["name"] == "Under"), None)
                if not over or not under or over.get("point") is None:
                    continue
                rows.append({
                    "event_id": ev.get("id"),
                    "commence": pd.to_datetime(ev.get("commence_time"), utc=True),
                    "home": home, "away": away,
                    "book": bk.get("title"),
                    "line": float(over["point"]),
                    "over_odds": int(over["price"]),
                    "under_odds": int(under["price"]),
                })
    cols = ["event_id", "commence", "home", "away", "book", "line", "over_odds", "under_odds"]
    return pd.DataFrame(rows, columns=cols)


def fetch_totals(api_key: str, regions: str = "us"):
    """Return (odds_df, requests_remaining). Raises on HTTP errors."""
    resp = requests.get(
        API_URL,
        params={"apiKey": api_key, "regions": regions, "markets": "totals",
                "oddsFormat": "american"},
        timeout=20,
    )
    if resp.status_code == 401:
        raise RuntimeError("Odds API rejected the key (401). Check it was pasted correctly.")
    if resp.status_code == 429:
        raise RuntimeError("Odds API rate/quota limit reached (429).")
    resp.raise_for_status()
    return parse_events(resp.json()), resp.headers.get("x-requests-remaining", "unknown")
