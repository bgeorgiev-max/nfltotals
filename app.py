"""NFL Totals Edge Finder - Streamlit app.

Tabs: weekly slate (model vs. market), backtest, and bet tracker.
Projections only: sportsbooks price totals efficiently, so check the Backtest
tab before trusting any pick.
"""
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

import model as M
from data import load_games
from odds import fetch_totals
from weather import fetch_game_weather

st.set_page_config(page_title="NFL Totals Edge Finder", page_icon="🏈", layout="wide")


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def get_secret(name: str) -> str:
    try:
        return str(st.secrets[name])
    except Exception:
        return ""


DATA_VERSION = 2  # bump when data.py changes, so Streamlit rebuilds its cached data


@st.cache_data(ttl=6 * 3600, show_spinner="Loading NFL history and fitting the model...")
def load_everything(data_version: int = DATA_VERSION):
    games = load_games()
    feat, _ = M.build_features(games)
    oos = M.run_backtest(feat)
    live_model = M.fit_live_model(feat)
    sigma = M.estimate_sigma(oos)
    return feat, oos, live_model, sigma


@st.cache_data(ttl=1800, show_spinner="Pulling live totals...")
def get_odds(api_key: str):
    return fetch_totals(api_key)


@st.cache_data(ttl=3600, show_spinner=False)
def get_forecast(home_team: str, kickoff_utc):
    # Raises on failure, and Streamlit does not cache exceptions, so a bad lookup is retried.
    return fetch_game_weather(home_team, kickoff_utc, raise_errors=True)


def edge_verdict(oos: pd.DataFrame):
    """Plain-English read of whether the backtest supports trusting the model."""
    tbl = M.summarize_thresholds(oos, thresholds=(1.5,))
    if tbl.empty:
        return "warning", "Not enough backtest bets at a 1.5-point edge to judge the model."
    row = tbl.iloc[0]
    if row["ci_low"] > M.BREAKEVEN:
        return "success", (
            f"Backtest: at a 1.5+ point edge the model won {row['win_pct']:.1%} "
            f"({int(row['bets'])} bets, 95% range {row['ci_low']:.1%}-{row['ci_high']:.1%}), "
            f"above the {M.BREAKEVEN:.1%} breakeven at -110. Past results don't guarantee future ones."
        )
    return "warning", (
        f"Backtest: at a 1.5+ point edge the model won {row['win_pct']:.1%} "
        f"({int(row['bets'])} bets, 95% range {row['ci_low']:.1%}-{row['ci_high']:.1%}). "
        f"That range includes the {M.BREAKEVEN:.1%} breakeven, so there is no statistically reliable edge. "
        "Treat the slate as projections, not picks."
    )


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.title("NFL Totals Edge Finder")
st.caption("Game over/under projections vs. the market, with a backtest and bet tracker. "
           "For entertainment and research. No model guarantees profit.")

with st.sidebar:
    st.header("Settings")
    bankroll = st.number_input("Bankroll ($)", min_value=0.0, value=1000.0, step=100.0)
    kelly_mult = st.slider("Kelly fraction", 0.05, 0.50, 0.25, 0.05,
                           help="Fraction of the full-Kelly stake to bet. 0.25 is a common cautious choice.")
    max_pct = st.slider("Max stake (% of bankroll)", 0.5, 5.0, 2.0, 0.5)
    min_ev = st.slider("Minimum EV to flag a bet (%)", 0.0, 10.0, 3.0, 0.5) / 100
    key_from_secrets = get_secret("ODDS_API_KEY")
    api_key = key_from_secrets or st.text_input("The Odds API key", type="password",
                                                help="Free key at the-odds-api.com. Or set ODDS_API_KEY in app Secrets.")
    if st.button("Refresh data"):
        st.cache_data.clear()
        st.rerun()

try:
    feat, oos, live_model, sigma = load_everything(DATA_VERSION)
except Exception as exc:  # network or schema problems
    st.error(f"Could not load NFL history: {exc}")
    st.stop()

tab_slate, tab_backtest, tab_tracker = st.tabs(["This week's slate", "Backtest", "Bet tracker"])


# ----------------------------------------------------------------------------
# Slate
# ----------------------------------------------------------------------------
with tab_slate:
    level, msg = edge_verdict(oos) if not oos.empty else ("warning", "Backtest unavailable.")
    (st.success if level == "success" else st.warning)(msg)

    odds_df = None
    if api_key:
        try:
            odds_df, remaining = get_odds(api_key)
            st.caption(f"Live odds loaded. Odds API requests remaining this month: {remaining}.")
        except Exception as exc:
            st.warning(f"Couldn't load live odds ({exc}). Falling back to the nflverse line at -110.")
    else:
        st.info("No Odds API key set. Using the nflverse line and assuming -110 on both sides. "
                "Add a key in the sidebar for live prices across sportsbooks.")

    window = M.upcoming_games(feat)
    overrides = {}
    if window.empty:
        st.info("No upcoming games in the next 8 days (bye week or offseason).")
    else:
        with st.expander("Game-time weather (auto-filled, editable)"):
            st.caption("Wind and temperature come from Open-Meteo forecasts at each stadium, averaged "
                       "over the game. Domes are skipped. Edit any value to override it. If a forecast "
                       "isn't available, the default is neutral (5 mph, 60 F).")
            for _, g in window.iterrows():
                if pd.notna(g["roof"]) and g["roof"] in ("dome", "closed"):
                    continue
                forecast, why = None, ""
                if str(g.get("location")) == "Neutral":
                    why = "neutral-site game"
                else:
                    try:
                        forecast = get_forecast(g["home_team"], g.get("kickoff_utc"))
                    except Exception as exc:
                        why = str(exc)
                if forecast:
                    def_wind, def_temp, src = forecast[0], forecast[1], "forecast"
                else:
                    def_wind, def_temp, src = 5.0, 60.0, f"neutral default: {why}"
                def_wind = float(min(max(round(def_wind), 0), 50))
                def_temp = float(min(max(round(def_temp), -10), 110))
                c1, c2, c3 = st.columns([2, 1, 1])
                c1.write(f'{g["away_team"]} @ {g["home_team"]} ({src})')
                wind = c2.number_input("Wind mph", 0.0, 50.0, def_wind, 1.0,
                                       key=f'w_{g["game_id"]}_{def_wind:.0f}')
                temp = c3.number_input("Temp F", -10.0, 110.0, def_temp, 1.0,
                                       key=f't_{g["game_id"]}_{def_temp:.0f}')
                overrides[g["game_id"]] = (wind, temp)

        best, detail = M.build_slate(feat, live_model, sigma, odds_df, overrides)
        if best.empty:
            st.info("Upcoming games found, but none have a market line yet.")
        else:
            best = best.copy()
            best["action"] = np.where(best["ev"] >= min_ev, "BET", "pass")
            best["stake"] = np.where(
                best["action"] == "BET",
                np.minimum(bankroll * kelly_mult * best["kelly"], bankroll * max_pct / 100).round(0),
                0.0,
            )
            show = best[["kickoff", "matchup", "consensus_line", "model_total", "edge", "action",
                         "side", "line", "odds", "book", "p_win", "ev", "stake"]].copy()
            show["kickoff"] = show["kickoff"].dt.strftime("%a %b %d")
            show["p_win"] = show["p_win"] * 100
            st.dataframe(
                show, hide_index=True,
                column_config={
                    "kickoff": "Day", "matchup": "Game",
                    "consensus_line": st.column_config.NumberColumn("Market", format="%.1f"),
                    "model_total": st.column_config.NumberColumn("Model", format="%.1f"),
                    "edge": st.column_config.NumberColumn("Edge (pts)", format="%+.1f"),
                    "action": "Action", "side": "Side",
                    "line": st.column_config.NumberColumn("Line", format="%.1f"),
                    "odds": st.column_config.NumberColumn("Odds", format="%d"),
                    "book": "Book",
                    "p_win": st.column_config.NumberColumn("Win prob", format="%.1f%%",
                                                           help="Model probability the side wins (pushes excluded)."),
                    "ev": st.column_config.NumberColumn("EV / $1", format="%+.3f"),
                    "stake": st.column_config.NumberColumn("Stake ($)", format="%.0f"),
                },
            )
            st.caption(f"Market = median line across books. The model assumes totals are roughly normal "
                       f"with a {sigma:.1f}-point standard deviation, and 'Win prob' excludes pushes.")
            with st.expander("Every book and side"):
                d = detail.copy()
                d["kickoff"] = d["kickoff"].dt.strftime("%a %b %d")
                d["p_win"] = (d["p_win"] * 100).round(1)
                st.dataframe(d[["kickoff", "matchup", "book", "side", "line", "odds", "p_win", "ev"]],
                             hide_index=True)


# ----------------------------------------------------------------------------
# Backtest
# ----------------------------------------------------------------------------
with tab_backtest:
    st.subheader("Walk-forward backtest")
    if oos.empty:
        st.info("No backtest data available.")
    else:
        st.caption(f"Each season from {int(oos['season'].min())} on is predicted using only earlier seasons, "
                   "then bets are settled at the closing total at -110. Breakeven win rate is 52.4%.")
        acc = M.accuracy_summary(oos)
        c1, c2, c3 = st.columns(3)
        c1.metric("Games tested", f'{acc["games"]:,}')
        c2.metric("Model avg miss (pts)", f'{acc["mae_model"]:.2f}',
                  delta=f'{acc["mae_model"] - acc["mae_market"]:+.2f} vs market', delta_color="inverse")
        c3.metric("Market avg miss (pts)", f'{acc["mae_market"]:.2f}')

        tbl = M.summarize_thresholds(oos)
        t = tbl.rename(columns={"min_edge_pts": "Min edge (pts)", "bets": "Bets", "win_pct": "Win %",
                                "ci_low": "95% low", "ci_high": "95% high", "roi": "ROI", "units": "Units"})
        for col in ("Win %", "95% low", "95% high", "ROI"):
            t[col] = (t[col] * 100).round(1)
        t["Units"] = t["Units"].round(1)
        st.dataframe(t, hide_index=True)
        st.caption("Win %, 95% range and ROI are in percent. If the 95% range straddles 52.4, "
                   "the result can't be told apart from luck.")

        thr = st.select_slider("Show cumulative units for min edge (pts)", options=[0.5, 1.0, 1.5, 2.0, 3.0], value=1.5)
        bets = M.grade_bets(oos, thr)
        if not bets.empty:
            curve = bets["units"].cumsum().reset_index(drop=True)
            st.line_chart(curve)
            by_season = (bets.groupby("season")
                         .agg(bets=("units", "size"), units=("units", "sum"), win_rate=("won", "mean"))
                         .round(3).reset_index())
            st.dataframe(by_season, hide_index=True)


# ----------------------------------------------------------------------------
# Bet tracker
# ----------------------------------------------------------------------------
TRACKER_COLS = ["date", "game", "side", "line", "odds", "stake", "close_line", "result"]

with tab_tracker:
    st.subheader("Bet tracker")
    st.warning("Streamlit Community Cloud doesn't keep files between restarts. "
               "Download your CSV after each session and re-upload it next time.")

    if "bets" not in st.session_state:
        st.session_state["bets"] = pd.DataFrame({
            "date": pd.Series(dtype="object"), "game": pd.Series(dtype="object"),
            "side": pd.Series(dtype="object"), "line": pd.Series(dtype="float"),
            "odds": pd.Series(dtype="float"), "stake": pd.Series(dtype="float"),
            "close_line": pd.Series(dtype="float"), "result": pd.Series(dtype="object"),
        })

    up_file = st.file_uploader("Restore a saved tracker CSV", type="csv")
    if up_file is not None and st.button("Load CSV"):
        loaded = pd.read_csv(up_file)
        st.session_state["bets"] = loaded.reindex(columns=TRACKER_COLS)
        st.rerun()

    with st.form("add_bet", clear_on_submit=True):
        a, b, c, d = st.columns(4)
        game = a.text_input("Game", placeholder="PIT @ JAX")
        side = b.selectbox("Side", ["Over", "Under"])
        line = c.number_input("Line", 20.0, 80.0, 44.5, 0.5)
        odds = d.number_input("Odds (American)", -1000, 1000, -110, 5)
        e, f = st.columns(2)
        stake_in = e.number_input("Stake ($)", 0.0, 100000.0, 20.0, 5.0)
        close_in = f.number_input("Closing line (optional, 0 = unknown)", 0.0, 80.0, 0.0, 0.5)
        if st.form_submit_button("Add bet") and game:
            row = {"date": date.today().isoformat(), "game": game, "side": side, "line": line, "odds": odds,
                   "stake": stake_in, "close_line": close_in if close_in > 0 else np.nan, "result": "Pending"}
            st.session_state["bets"] = pd.concat([st.session_state["bets"], pd.DataFrame([row])], ignore_index=True)

    edited = st.data_editor(
        st.session_state["bets"], num_rows="dynamic", hide_index=True,
        column_config={
            "line": st.column_config.NumberColumn("line", format="%.1f"),
            "odds": st.column_config.NumberColumn("odds", format="%d"),
            "stake": st.column_config.NumberColumn("stake", format="%.2f"),
            "close_line": st.column_config.NumberColumn("close_line", format="%.1f"),
            "side": st.column_config.SelectboxColumn("side", options=["Over", "Under"]),
            "result": st.column_config.SelectboxColumn("result", options=["Pending", "Win", "Loss", "Push"]),
        },
    )
    st.session_state["bets"] = edited

    df = edited.copy()
    for col in ("line", "odds", "stake", "close_line"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    graded = df[df["result"].isin(["Win", "Loss", "Push"]) & df["stake"].notna() & df["odds"].notna()].copy()

    if graded.empty:
        st.info("Settle a bet (set its result) to see ROI and closing line value.")
    else:
        graded["profit"] = np.where(
            graded["result"] == "Win", graded["stake"] * graded["odds"].apply(M.american_profit),
            np.where(graded["result"] == "Loss", -graded["stake"], 0.0))
        risked = graded["stake"].sum()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Settled bets", len(graded))
        m2.metric("Profit ($)", f'{graded["profit"].sum():+,.0f}')
        m3.metric("ROI", f'{graded["profit"].sum() / risked:+.1%}' if risked else "n/a")
        has_close = graded.dropna(subset=["close_line", "line"])
        if has_close.empty:
            m4.metric("Avg CLV (pts)", "n/a")
        else:
            clv = np.where(has_close["side"] == "Over",
                           has_close["close_line"] - has_close["line"],
                           has_close["line"] - has_close["close_line"])
            m4.metric("Avg CLV (pts)", f"{clv.mean():+.2f}",
                      help="Positive means you got a better number than the closing line, "
                           "the best early sign of a real edge.")
        st.line_chart(graded["profit"].cumsum().reset_index(drop=True))

    st.download_button("Download tracker CSV", edited.to_csv(index=False), "bet_tracker.csv", "text/csv")
