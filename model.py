"""NFL game-total model, walk-forward backtest, and betting math.

Approach (deliberately simple and market-anchored):
  1. Build pre-game team ratings (exponentially weighted points scored/allowed).
  2. Turn them into a raw projected total.
  3. Fit a heavily regularized ridge regression that predicts how far the actual
     total lands from the closing line, using the rating gap and weather/rest.
     Predictions therefore start at the market and only move when the data
     gives a reason.
  4. Backtest walk-forward: each season is predicted using only earlier seasons.
"""
import math

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ALPHA = 0.15          # how fast ratings react to new games
SEASON_CARRY = 0.65   # share of a team's rating carried into next season
RIDGE_ALPHA = 200.0   # strong shrinkage toward "the market is right"
DEFAULT_SIGMA = 13.0  # std dev of (actual total - projection), refined from backtest
MIN_TRAIN_SEASON = 2003
FIRST_TEST_SEASON = 2010
BREAKEVEN = 110 / 210  # win rate needed at -110 (52.38%)


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
def build_features(games: pd.DataFrame):
    """Add pre-game rating features. Uses only information available before kickoff.

    Returns (frame, final_state). Unplayed games get features from the latest ratings.
    """
    g = games.sort_values(["gameday", "game_id"]).reset_index(drop=True).copy()
    state = {}
    lg_sum, lg_n = 0.0, 0
    last_season = None
    pred_home, pred_away = [], []

    for r in g.itertuples(index=False):
        lg = lg_sum / lg_n if lg_n >= 100 else 22.0
        if last_season is not None and r.season != last_season:
            for t, (o, d) in list(state.items()):
                state[t] = (lg + SEASON_CARRY * (o - lg), lg + SEASON_CARRY * (d - lg))
        last_season = r.season

        ho, hd = state.get(r.home_team, (lg, lg))
        ao, ad = state.get(r.away_team, (lg, lg))
        pred_home.append(lg + (ho - lg) + (ad - lg))
        pred_away.append(lg + (ao - lg) + (hd - lg))

        if pd.notna(r.home_score) and pd.notna(r.away_score):
            hs, aw = float(r.home_score), float(r.away_score)
            state[r.home_team] = (ho + ALPHA * (hs - ho), hd + ALPHA * (aw - hd))
            state[r.away_team] = (ao + ALPHA * (aw - ao), ad + ALPHA * (hs - ad))
            lg_sum += hs + aw
            lg_n += 2

    g["pred_home"] = pred_home
    g["pred_away"] = pred_away
    g["model_total"] = g["pred_home"] + g["pred_away"]
    return g, state


def feature_matrix(df: pd.DataFrame, line) -> pd.DataFrame:
    """Regression inputs. `line` is the market total (scalar or per-row)."""
    line = pd.Series(np.broadcast_to(np.asarray(line, dtype=float), len(df)), index=df.index)
    indoors = df["roof"].isin(["dome", "closed"]).astype(bool)
    outdoors = ~indoors
    wind = df["wind"].fillna(5.0)
    temp = df["temp"].fillna(60.0)
    return pd.DataFrame({
        "mkt_gap": df["model_total"] - line,
        "wind_eff": np.where(outdoors, wind, 0.0),
        "cold_eff": np.where(outdoors, np.clip(40.0 - temp, 0, None), 0.0),
        "dome": indoors.astype(float),
        "rest_sum": (df["home_rest"].fillna(7) + df["away_rest"].fillna(7) - 14).clip(-6, 8),
        "div_game": df["div_game"].fillna(0).astype(float),
    }, index=df.index)


# --------------------------------------------------------------------------
# Fit / project
# --------------------------------------------------------------------------
def _training_frame(feat: pd.DataFrame, before_season=None) -> pd.DataFrame:
    d = feat[(feat["season"] >= MIN_TRAIN_SEASON)
             & feat["home_score"].notna() & feat["away_score"].notna()
             & feat["total_line"].notna()].copy()
    if before_season is not None:
        d = d[d["season"] < before_season]
    d["actual_total"] = d["home_score"] + d["away_score"]
    return d


def fit_model(train: pd.DataFrame):
    X = feature_matrix(train, train["total_line"])
    y = train["actual_total"] - train["total_line"]
    model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
    model.fit(X, y)
    return model


def fit_live_model(feat: pd.DataFrame):
    """Model for upcoming games: trained on every completed game with a line."""
    return fit_model(_training_frame(feat))


def project(model, df: pd.DataFrame, line) -> np.ndarray:
    """Projected total = market line + model's predicted gap."""
    line_arr = np.broadcast_to(np.asarray(line, dtype=float), len(df))
    return line_arr + model.predict(feature_matrix(df, line))


# --------------------------------------------------------------------------
# Backtest
# --------------------------------------------------------------------------
def run_backtest(feat: pd.DataFrame, first_test: int = FIRST_TEST_SEASON) -> pd.DataFrame:
    """Walk-forward: predict each season using only prior seasons."""
    d = _training_frame(feat)
    out = []
    for season in sorted(d["season"].unique()):
        if season < first_test:
            continue
        train = d[d["season"] < season]
        test = d[d["season"] == season].copy()
        if len(train) < 500 or test.empty:
            continue
        model = fit_model(train)
        test["pred_total"] = project(model, test, test["total_line"])
        out.append(test)
    if not out:
        return pd.DataFrame()
    res = pd.concat(out).sort_values(["gameday", "game_id"]).reset_index(drop=True)
    res["edge"] = res["pred_total"] - res["total_line"]
    return res


def estimate_sigma(oos: pd.DataFrame) -> float:
    if oos is None or oos.empty:
        return DEFAULT_SIGMA
    return float((oos["actual_total"] - oos["pred_total"]).std())


def grade_bets(oos: pd.DataFrame, min_edge: float, price: int = -110) -> pd.DataFrame:
    """Bet Over/Under on every game whose |edge| >= min_edge; settle at the closing line."""
    b = oos[oos["edge"].abs() >= max(min_edge, 1e-9)].copy()
    b["side"] = np.where(b["edge"] > 0, "Over", "Under")
    diff = b["actual_total"] - b["total_line"]
    won = ((b["side"] == "Over") & (diff > 0)) | ((b["side"] == "Under") & (diff < 0))
    push = diff == 0
    profit = american_profit(price)
    b["units"] = np.where(push, 0.0, np.where(won, profit, -1.0))
    b["won"] = won
    b["push"] = push
    return b


def summarize_thresholds(oos: pd.DataFrame, thresholds=(0.0, 0.5, 1.0, 1.5, 2.0, 3.0)) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        b = grade_bets(oos, t)
        n_dec = int((~b["push"]).sum())
        if n_dec == 0:
            continue
        wins = int(b["won"].sum())
        wp = wins / n_dec
        se = math.sqrt(wp * (1 - wp) / n_dec)
        rows.append({
            "min_edge_pts": t,
            "bets": len(b),
            "win_pct": wp,
            "ci_low": wp - 1.96 * se,
            "ci_high": wp + 1.96 * se,
            "roi": b["units"].sum() / len(b),
            "units": b["units"].sum(),
        })
    return pd.DataFrame(rows)


def accuracy_summary(oos: pd.DataFrame) -> dict:
    err_model = oos["actual_total"] - oos["pred_total"]
    err_mkt = oos["actual_total"] - oos["total_line"]
    return {
        "games": len(oos),
        "mae_model": float(err_model.abs().mean()),
        "mae_market": float(err_mkt.abs().mean()),
        "rmse_model": float(np.sqrt((err_model ** 2).mean())),
        "rmse_market": float(np.sqrt((err_mkt ** 2).mean())),
    }


# --------------------------------------------------------------------------
# Odds / probability math
# --------------------------------------------------------------------------
def american_profit(odds: float) -> float:
    """Profit on a 1-unit stake if the bet wins."""
    return odds / 100.0 if odds > 0 else 100.0 / abs(odds)


def implied_prob(odds: float) -> float:
    return 100.0 / (odds + 100.0) if odds > 0 else abs(odds) / (abs(odds) + 100.0)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def over_under_probs(mu: float, line: float, sigma: float):
    """P(Over), P(Under) for an integer-valued total ~ Normal(mu, sigma).

    Uses a continuity correction, so a whole-number line leaves room for a push.
    """
    p_over = 1.0 - _norm_cdf((math.floor(line) + 0.5 - mu) / sigma)
    p_under = _norm_cdf((math.ceil(line) - 0.5 - mu) / sigma)
    return p_over, p_under


def ev_per_unit(p_win: float, p_lose: float, odds: float) -> float:
    return p_win * american_profit(odds) - p_lose


def kelly_fraction(p_win: float, p_lose: float, odds: float) -> float:
    b = american_profit(odds)
    return max(0.0, (b * p_win - p_lose) / b)


# --------------------------------------------------------------------------
# Weekly slate
# --------------------------------------------------------------------------
def upcoming_games(feat: pd.DataFrame, today=None, days_ahead: int = 8) -> pd.DataFrame:
    today = pd.Timestamp(today) if today is not None else pd.Timestamp.today().normalize()
    mask = (feat["home_score"].isna()
            & (feat["gameday"] >= today - pd.Timedelta(days=1))
            & (feat["gameday"] <= today + pd.Timedelta(days=days_ahead)))
    return feat[mask].copy()


def build_slate(feat, model, sigma, odds_df=None, overrides=None, today=None):
    """Return (best_bet_per_game, every_book_and_side) for upcoming games.

    odds_df: output of odds.fetch_totals (optional). Without it, the nflverse
    line is used and both sides are assumed to pay -110.
    overrides: {game_id: (wind_mph, temp_f)} manual weather inputs.
    """
    up = upcoming_games(feat, today)
    if up.empty:
        return pd.DataFrame(), pd.DataFrame()
    for gid, (wind, temp) in (overrides or {}).items():
        up.loc[up["game_id"] == gid, ["wind", "temp"]] = [wind, temp]

    detail_rows = []
    for idx, r in up.iterrows():
        books = pd.DataFrame()
        if odds_df is not None and not odds_df.empty:
            books = odds_df[(odds_df["home"] == r["home_team"]) & (odds_df["away"] == r["away_team"])]
        if len(books):
            consensus = float(books["line"].median())
            books = books[["book", "line", "over_odds", "under_odds"]]
        elif pd.notna(r["total_line"]):
            consensus = float(r["total_line"])
            books = pd.DataFrame([{"book": "nflverse line", "line": consensus,
                                   "over_odds": -110, "under_odds": -110}])
        else:
            continue

        mu = float(project(model, up.loc[[idx]], consensus)[0])
        for b in books.itertuples(index=False):
            p_over, p_under = over_under_probs(mu, b.line, sigma)
            for side, p_win, p_lose, odds in (("Over", p_over, p_under, b.over_odds),
                                              ("Under", p_under, p_over, b.under_odds)):
                detail_rows.append({
                    "game_id": r["game_id"], "kickoff": r["gameday"],
                    "matchup": f'{r["away_team"]} @ {r["home_team"]}',
                    "consensus_line": consensus, "model_total": mu, "edge": mu - consensus,
                    "side": side, "book": b.book, "line": b.line, "odds": odds,
                    "p_win": p_win, "ev": ev_per_unit(p_win, p_lose, odds),
                    "kelly": kelly_fraction(p_win, p_lose, odds),
                })
    detail = pd.DataFrame(detail_rows)
    if detail.empty:
        return pd.DataFrame(), detail
    best = (detail.sort_values("ev", ascending=False)
            .drop_duplicates("game_id")
            .reset_index(drop=True))
    return best, detail.sort_values(["kickoff", "matchup", "ev"], ascending=[True, True, False])
