"""Game-time forecast weather from Open-Meteo (free, no API key).

Looks up the forecast at the home team's stadium and averages wind and
temperature over the roughly three hours of the game.
"""
import pandas as pd
import requests

# (latitude, longitude) of each team's home stadium. Approximate is fine for weather.
STADIUMS = {
    "ARI": (33.5276, -112.2626), "ATL": (33.7554, -84.4010), "BAL": (39.2780, -76.6227),
    "BUF": (42.7738, -78.7870), "CAR": (35.2258, -80.8528), "CHI": (41.8623, -87.6167),
    "CIN": (39.0955, -84.5161), "CLE": (41.5061, -81.6995), "DAL": (32.7473, -97.0945),
    "DEN": (39.7439, -105.0201), "DET": (42.3400, -83.0456), "GB": (44.5013, -88.0622),
    "HOU": (29.6847, -95.4107), "IND": (39.7601, -86.1639), "JAX": (30.3239, -81.6373),
    "KC": (39.0489, -94.4839), "LAC": (33.9535, -118.3392), "LA": (33.9535, -118.3392),
    "LV": (36.0909, -115.1833), "MIA": (25.9580, -80.2389), "MIN": (44.9735, -93.2575),
    "NE": (42.0909, -71.2643), "NO": (29.9511, -90.0812), "NYG": (40.8135, -74.0745),
    "NYJ": (40.8135, -74.0745), "PHI": (39.9008, -75.1675), "PIT": (40.4468, -80.0158),
    "SF": (37.4030, -121.9700), "SEA": (47.5952, -122.3316), "TB": (27.9759, -82.5033),
    "TEN": (36.1665, -86.7713), "WAS": (38.9076, -76.8645),
}

API_URL = "https://api.open-meteo.com/v1/forecast"
GAME_HOURS = 3  # average conditions from kickoff through ~3 hours later


def parse_forecast(payload: dict, kickoff_utc: pd.Timestamp):
    """Return (wind_mph, temp_f) averaged over the game window, or None if not covered."""
    hourly = payload.get("hourly") or {}
    times = pd.to_datetime(hourly.get("time", []), utc=True)
    if len(times) == 0:
        return None
    start = pd.Timestamp(kickoff_utc).floor("h")
    end = start + pd.Timedelta(hours=GAME_HOURS)
    mask = (times >= start) & (times <= end)
    if not mask.any():
        return None
    wind = pd.Series(hourly["wind_speed_10m"])[mask].astype(float).mean()
    temp = pd.Series(hourly["temperature_2m"])[mask].astype(float).mean()
    if pd.isna(wind) or pd.isna(temp):
        return None
    return float(wind), float(temp)


def fetch_game_weather(home_team: str, kickoff_utc, raise_errors: bool = False):
    """Forecast (wind_mph, temp_f) at kickoff, or None if unavailable.

    With raise_errors=True, failures raise ValueError with a readable reason
    (the app uses this so failed lookups are never cached and the cause is visible).
    """
    def fail(msg):
        if raise_errors:
            raise ValueError(msg)
        return None

    coords = STADIUMS.get(home_team)
    if coords is None:
        return fail(f"no stadium coordinates for {home_team}")
    if kickoff_utc is None or pd.isna(kickoff_utc):
        return fail("kickoff time missing (is data.py up to date?)")
    try:
        resp = requests.get(
            API_URL,
            params={
                "latitude": coords[0], "longitude": coords[1],
                "hourly": "temperature_2m,wind_speed_10m",
                "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
                "timezone": "GMT", "forecast_days": 10,
            },
            timeout=15,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        if raise_errors:
            raise ValueError(f"forecast request failed: {exc}") from exc
        return None
    result = parse_forecast(payload, kickoff_utc)
    if result is None:
        return fail("kickoff is outside the forecast window")
    return result
