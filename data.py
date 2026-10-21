"""Data loading and team-name helpers.

History comes from the free nflverse games file (scores, closing totals,
roof/temp/wind, rest days). No API key needed.
"""
import os

import numpy as np
import pandas as pd

GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
# Override with a local CSV path (used for testing): NFL_GAMES_PATH=/path/to/games.csv
GAMES_SOURCE = os.environ.get("NFL_GAMES_PATH", GAMES_URL)

# nflverse has used several abbreviations over the years; normalize to one set.
ABBR_FIX = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA", "WSH": "WAS", "JAC": "JAX"}

# The Odds API returns full team names.
FULL_TO_ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}

NEEDED = [
    "game_id", "season", "game_type", "week", "gameday",
    "away_team", "away_score", "home_team", "home_score",
    "total_line", "roof", "temp", "wind", "home_rest", "away_rest", "div_game",
    "gametime", "location",
]

# Used only when the data has no roof value for a game (common for upcoming games).
DEFAULT_ROOF = {t: "dome" for t in
                ("ARI", "ATL", "DAL", "DET", "HOU", "IND", "LA", "LAC", "LV", "MIN", "NO")}


def load_games(source: str = None) -> pd.DataFrame:
    """Load and tidy the nflverse games table."""
    df = pd.read_csv(source or GAMES_SOURCE)
    for col in NEEDED:
        if col not in df.columns:
            df[col] = np.nan
    df = df[NEEDED].copy()
    df["gameday"] = pd.to_datetime(df["gameday"])
    for col in ("home_team", "away_team"):
        df[col] = df[col].replace(ABBR_FIX)
    for col in ("season", "week", "home_score", "away_score", "total_line",
                "temp", "wind", "home_rest", "away_rest", "div_game"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    roof = df["roof"].astype("string").str.lower()
    roof = roof.fillna(df["home_team"].map(DEFAULT_ROOF)).fillna("outdoors")
    df["roof"] = roof.astype(object)

    # Kickoff as a UTC timestamp (nflverse lists local-to-ET kickoff times).
    clock = df["gametime"].astype("string").fillna("13:00")
    local = pd.to_datetime(df["gameday"].dt.strftime("%Y-%m-%d") + " " + clock, errors="coerce")
    df["kickoff_utc"] = (local.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
                         .dt.tz_convert("UTC"))
    return df.sort_values(["gameday", "game_id"]).reset_index(drop=True)
