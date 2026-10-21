# NFL Totals Edge Finder

A Streamlit app that projects NFL game totals (over/under), compares them to the market,
backtests the approach, and tracks your bets. Projections only: sportsbooks price totals
efficiently, so read the Backtest tab before trusting any pick.

## Files
- `app.py` - the web app (slate, backtest, bet tracker)
- `model.py` - team ratings, ridge model, walk-forward backtest, odds/EV/Kelly math
- `data.py` - loads free nflverse history (scores, closing totals, weather, rest)
- `odds.py` - pulls live totals from The Odds API
- `requirements.txt` - dependencies for the host

## Deploy (Streamlit Community Cloud, free)
1. Create a **personal** GitHub account/repo and upload these files to it (browser: Add file > Upload files).
2. Go to https://share.streamlit.io, sign in with GitHub, choose **New app**, pick the repo, main file `app.py`.
3. Under **Advanced settings > Secrets**, paste: `ODDS_API_KEY = "your-key"` (see `secrets.toml.example`).
4. Deploy. The first load takes a minute or two while it downloads history and fits the model.
5. In the app's **Share** settings, restrict viewers to your own email. New apps can be public by default.

Never put the API key in a file in the repo.

## How the model works
1. Pre-game team ratings: exponentially weighted points scored and allowed, carried partly across seasons.
2. A heavily regularized ridge regression predicts how far the actual total lands from the closing line,
   using the rating gap, wind, cold, roof, rest, and division games. Projections start at the market
   and only move when the data supports it.
3. Win probability assumes the total is roughly normal around the projection (spread estimated from the
   backtest), with a continuity correction for pushes.
4. Stakes use fractional Kelly with a per-bet cap.

## Backtest
Walk-forward: each season is predicted using only earlier seasons, then bets are settled at the closing
total at -110 (breakeven 52.4%). The app prints a 95% range for each win rate; if that range includes 52.4%,
the result can't be told apart from luck.

## Limits to know about
- Historical `total_line` is the closing line. Opening lines and injuries are not modeled.
- Forecast weather isn't available; enter wind/temp manually in the slate tab.
- The bet tracker lives in the browser session. Download the CSV after each session and re-upload it later.
- The free Odds API tier has a monthly request cap; odds are cached for 30 minutes.
- Closing line value (CLV) in the tracker is the best early signal of a real edge: log the closing number and watch the average.
