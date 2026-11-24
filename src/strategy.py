"""Pairs trading strategy: pair selection -> spread and z-score -> positions -> backtest -> metrics.

Idea: two coins A and B that move together form a "spread" = log(A) - beta * log(B) that goes back to
its mean. When the spread is unusually far from its mean (|z-score| large) we bet that it comes back:
    z very low  -> spread too low  -> LONG the spread  = buy A, short B   (position +1)
    z very high -> spread too high -> SHORT the spread = short A, buy B   (position -1)
All prices are hourly, so one bar = one hour.
"""

from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint

from .data import split


# =============================================================================
# 1. PAIR SELECTION (train data only)
# =============================================================================

def half_life(spread: pd.Series) -> float:
    """How many hours it takes the spread to close half of its distance to the mean.

    Fit  change(t) = c + phi * spread(t-1). phi < 0 means the spread is pulled back to the mean.
    Half-life = ln(2) / -ln(1 + phi). If phi >= 0 the spread does not revert: infinite half-life.
    """
    spread = spread.dropna()
    phi = sm.OLS(spread.diff().iloc[1:], sm.add_constant(spread.shift(1).iloc[1:])).fit().params.iloc[1]
    return np.log(2) / -np.log(1 + phi) if phi < 0 else np.inf


def rank_pairs(prices: pd.DataFrame) -> pd.DataFrame:
    """Engle-Granger cointegration test for every pair of coins, best (lowest p-value) first.

    Engle-Granger: regress log(A) on log(B) -> the residual is the spread -> ADF test whether the
    spread is stationary (mean-reverting). Low p-value = probably cointegrated.

    Extra columns to judge the result:
      p_bonferroni:   p-value x number of tests. We run 66 tests, so ~3 pairs look "significant"
                      at 5% by pure luck. Bonferroni corrects for that (conservative).
      adf_p_a/b:      ADF p-value of each coin ALONE. Should be high (the coin itself trends).
                      If a coin is already stationary on its own, the "pair" is fake: it is just that coin.
    """
    logp = np.log(prices)
    returns_corr = logp.diff().corr()
    alone = {c: adfuller(logp[c], maxlag=24, result_object=False)[1] for c in prices.columns}
    pairs = list(combinations(prices.columns, 2))

    rows = []
    for a, b in pairs:
        p_value = coint(logp[a], logp[b], maxlag=24)[1]
        beta = sm.OLS(logp[a], sm.add_constant(logp[b])).fit().params.iloc[1]
        rows.append({
            "a": a, "b": b,
            "correlation": returns_corr.loc[a, b],
            "p_value": p_value,
            "p_bonferroni": min(1.0, p_value * len(pairs)),
            "beta": beta,
            "half_life_h": half_life(logp[a] - beta * logp[b]),
            "adf_p_a": alone[a], "adf_p_b": alone[b],
        })
    return pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)


# =============================================================================
# 2. SPREAD AND Z-SCORE
# =============================================================================

def spread_and_zscore(a: pd.Series, b: pd.Series, cfg: dict) -> pd.DataFrame:
    """Rolling hedge ratio beta, the spread and its z-score. Every value only uses PAST data.

    beta:   slope of log(A) on log(B) over the last `hedge_window` hours = cov(logA, logB) / var(logB)
    alpha:  matching intercept, keeps the spread centred around 0 when beta changes
    spread: log(A) - beta * log(B) - alpha
    z:      (spread - rolling mean) / rolling std over the last `zscore_lookback` hours
    """
    log_a, log_b = np.log(a), np.log(b)
    w = cfg["hedge_window"]
    beta = log_a.rolling(w).cov(log_b) / log_b.rolling(w).var()
    alpha = log_a.rolling(w).mean() - beta * log_b.rolling(w).mean()
    spread = log_a - beta * log_b - alpha

    n = cfg["zscore_lookback"]
    z = (spread - spread.rolling(n).mean()) / spread.rolling(n).std()
    return pd.DataFrame({"beta": beta, "spread": spread, "z": z})


# =============================================================================
# 3. TRADING RULES -> POSITIONS
# =============================================================================

def positions(z: pd.Series, cfg: dict) -> tuple[pd.Series, pd.Series]:
    """Go through the z-score hour by hour and decide the position (+1, 0, -1) at the close of each hour.

    Entry (when flat):  z < -entry_z -> +1 (long spread),  z > +entry_z -> -1 (short spread),
                        but not if |z| > stop_z already (we would be stopped out immediately).
    Exit (when open):   1. stop loss:  |z| > stop_z               (spread keeps running away)
                        2. time stop:  open for time_stop_hours    (spread does not come back in time)
                        3. take profit: z is back to exit_z        (the normal case)
    After a stop we wait until |z| < entry_z before trading again, so we don't jump right back
    into the same move that just stopped us out.
    """
    entry_z, exit_z, stop_z = cfg["entry_z"], cfg["exit_z"], cfg["stop_z"]
    pos = np.zeros(len(z))
    reason = np.full(len(z), None, dtype=object)
    state, entry_hour, wait = 0, 0, False

    for t, zt in enumerate(z.to_numpy()):
        if np.isnan(zt):                      # warm-up: not enough history yet
            continue
        if state != 0:                        # --- a trade is open: check the exits ---
            if abs(zt) > stop_z:
                reason[t], wait = "stop_loss", True
            elif t - entry_hour >= cfg["time_stop_hours"]:
                reason[t], wait = "time_stop", True
            elif (state == 1 and zt >= -exit_z) or (state == -1 and zt <= exit_z):
                reason[t] = "take_profit"
            if reason[t]:
                state = 0
        elif wait:                            # --- after a stop: wait until z is calm again ---
            wait = abs(zt) >= entry_z
        elif entry_z < abs(zt) <= stop_z:     # --- flat: check the entry ---
            state = 1 if zt < 0 else -1
            entry_hour = t
        pos[t] = state

    if pos[-1] != 0:                          # close any open trade at the end of the data
        pos[-1], reason[-1] = 0, "end_of_data"
    return pd.Series(pos, index=z.index), pd.Series(reason, index=z.index)


# =============================================================================
# 4. BACKTEST
# =============================================================================

def backtest(prices: pd.DataFrame, funding: pd.DataFrame | None, cfg: dict, period: str = "train") -> dict:
    """Run the whole strategy for cfg["pair"] on the train or test period.

    Money: we always use all capital. Position +1 puts 1 / (1 + |beta|) of it long A and
    beta / (1 + |beta|) short B (so 1:beta, the hedge ratio). beta is FROZEN at entry for the whole trade.

    Timing (no lookahead): the position decided at the close of hour t-1 earns the return of hour t.

    Per hour t:
        gross   = position(t-1) * (w_a * return_a(t) + w_b * return_b(t))
        costs   = (fee + slippage) * |position(t) - position(t-1)|     (paid when we trade)
        funding = -position(t-1) * (w_a * rate_a + w_b * rate_b)        (long pays a positive rate)
        net     = gross - costs + funding

    Returns a dict with: bars (DataFrame per hour), trades (DataFrame per trade), metrics (dict).
    """
    name_a, name_b = cfg["pair"]
    a, b = prices[name_a], prices[name_b]

    # z-score on the FULL history (it only looks backwards), so the test period starts warmed up.
    sz = spread_and_zscore(a, b, cfg)
    train, test = split(prices, cfg)
    index = train.index if period == "train" else test.index
    sz, a, b = sz.loc[index], a.loc[index], b.loc[index]

    pos, reason = positions(sz["z"], cfg)

    # beta frozen at entry: take beta on the bar where the position changes, carry it forward
    beta = sz["beta"].where(pos.diff() != 0).ffill().fillna(sz["beta"])
    w_a, w_b = 1 / (1 + beta.abs()), -beta / (1 + beta.abs())
    held = pos.shift(1, fill_value=0)                 # position held DURING each hour
    w_a_held, w_b_held = w_a.shift(1), w_b.shift(1)

    gross = held * (w_a_held * a.pct_change() + w_b_held * b.pct_change())
    costs = (cfg["fee"] + cfg["slippage"]) * pos.diff().abs().fillna(pos.abs())
    if funding is not None:
        # The row of hour t covers t..t+1; the payment stamped t+1 is paid by the position held in it.
        rate_a, rate_b = funding[name_a].shift(-1).loc[index], funding[name_b].shift(-1).loc[index]
        fund = -held * (w_a_held * rate_a + w_b_held * rate_b)
    else:
        fund = 0.0 * held
    net = (gross.fillna(0) - costs + fund.fillna(0))

    bars = pd.DataFrame({
        "price_a": a, "price_b": b, "beta": sz["beta"], "spread": sz["spread"], "z": sz["z"],
        "position": pos, "exit_reason": reason,
        "gross": gross.fillna(0), "costs": -costs, "funding": fund.fillna(0), "net": net,
        "equity": (1 + net).cumprod(),
    })
    trades = list_trades(bars)
    return {"bars": bars, "trades": trades, "metrics": metrics(bars, trades), "period": period}


def list_trades(bars: pd.DataFrame) -> pd.DataFrame:
    """One row per trade: from the hour we open (pay entry costs) to the hour we close (last return + exit costs)."""
    pos = bars["position"]
    entries = bars.index[(pos != 0) & (pos.shift(1, fill_value=0) == 0)]
    exits = bars.index[bars["exit_reason"].notna()]
    rows = []
    for entry, exit_ in zip(entries, exits):
        trade = bars.loc[entry:exit_]
        rows.append({
            "entry": entry, "exit": exit_,
            "direction": "long spread" if pos[entry] > 0 else "short spread",
            "hours": len(trade) - 1,
            "entry_z": trade["z"].iloc[0], "exit_z": trade["z"].iloc[-1],
            "return": (1 + trade["net"]).prod() - 1,
            "exit_reason": trade["exit_reason"].iloc[-1],
        })
    return pd.DataFrame(rows)


def metrics(bars: pd.DataFrame, trades: pd.DataFrame) -> dict:
    """Key numbers. Crypto trades 24/7, so one year = 24 * 365 = 8760 hours."""
    net, equity = bars["net"], bars["equity"]
    years = len(bars) / 8760
    return {
        "total_return": equity.iloc[-1] - 1,
        "annual_return": equity.iloc[-1] ** (1 / years) - 1,
        "sharpe": net.mean() / net.std() * np.sqrt(8760),
        "max_drawdown": (equity / equity.cummax().clip(lower=1) - 1).min(),
        "n_trades": len(trades),
        "win_rate": (trades["return"] > 0).mean() if len(trades) else np.nan,
        "avg_trade_return": trades["return"].mean() if len(trades) else np.nan,
        "avg_hours_in_trade": trades["hours"].mean() if len(trades) else np.nan,
        "time_in_market": (bars["position"] != 0).mean(),
        "sum_gross": bars["gross"].sum(),
        "sum_costs": bars["costs"].sum(),
        "sum_funding": bars["funding"].sum(),
    }
