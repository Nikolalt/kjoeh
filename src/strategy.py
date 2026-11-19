"""Strategy logic in three steps: pair selection, signal generation, risk management.

Conventions:
    a, b        PRICE series of the two coins (a is the dependent leg); functions take logs themselves
    spread      log(a) - beta * log(b) - alpha
    position +1 long the spread  = long a, short b   (we expect the spread to rise back to its mean)
    position -1 short the spread = short a, long b   (we expect the spread to fall back to its mean)
    time        hourly bars, so "bars" and "hours" are the same unit

Every number that is a decision (thresholds, windows, switches) is a function argument or comes
from the params dict; nothing is hard-coded here.
"""

from collections.abc import Callable
from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint


# =============================================================================
# 1. PAIR SELECTION
#    Which two coins move together closely enough to trade the spread?
#    Always run on TRAIN data only.
# =============================================================================

def correlation_screen(prices: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Rank all coin pairs by the correlation of their hourly log returns, as a fast first filter.

    We use returns, not price levels: two trending prices look correlated even when unrelated
    (spurious correlation).

    Input:  price DataFrame (one column per coin), number of pairs to keep.
    Output: DataFrame with columns a, b, correlation, sorted descending, top_n rows.
    """
    corr = np.log(prices).diff().corr()
    rows = [(a, b, corr.loc[a, b]) for a, b in combinations(prices.columns, 2)]
    ranked = pd.DataFrame(rows, columns=["a", "b", "correlation"])
    return ranked.sort_values("correlation", ascending=False).head(top_n).reset_index(drop=True)


def half_life(spread: pd.Series) -> float:
    """Mean-reversion half-life of a spread in bars (hours), from an AR(1) fit.

    Regress the change of the spread on its previous level:  d_spread(t) = c + phi * spread(t-1).
    phi < 0 means the spread is pulled back to its mean; each bar a share phi of the distance
    disappears, so the distance halves after ln(2) / -ln(1 + phi) bars. phi >= 0 -> no mean
    reversion -> infinite half-life.

    Input:  spread Series.
    Output: half-life in bars (float, np.inf if not mean-reverting).
    """
    spread = spread.dropna()
    lagged = spread.shift(1).iloc[1:]
    delta = spread.diff().iloc[1:]
    phi = sm.OLS(delta, sm.add_constant(lagged)).fit().params.iloc[1]
    return float(np.log(2) / -np.log(1 + phi)) if phi < 0 else np.inf


def test_cointegration(a: pd.Series, b: pd.Series, adf_maxlag: int = 24) -> dict:
    """Engle-Granger cointegration test of one pair, plus hedge ratio and half-life.

    Step 1: OLS of log(a) on log(b):  log(a) = alpha + beta * log(b) + residual.
    Step 2: ADF test on the residual (= the spread). If it is stationary, the pair is cointegrated.
    The p-value uses Engle-Granger (MacKinnon) critical values, which are stricter than plain ADF
    ones: beta was fitted to make the residual look as stationary as possible, so a plain ADF
    p-value would be too optimistic. statsmodels' coint() does exactly this.

    Input:  price series a and b (same index), max ADF lags (chosen by AIC up to this many hours).
    Output: dict with alpha, beta, adf_stat, p_value, half_life (hours).
    """
    log_a, log_b = np.log(a), np.log(b)
    ols = sm.OLS(log_a, sm.add_constant(log_b)).fit()
    alpha, beta = ols.params.iloc[0], ols.params.iloc[1]
    spread = log_a - beta * log_b - alpha

    adf_stat, p_value, _ = coint(log_a, log_b, maxlag=adf_maxlag, autolag="aic")
    return {
        "alpha": alpha,
        "beta": beta,
        "adf_stat": adf_stat,
        "p_value": p_value,
        "half_life": half_life(spread),
    }


def adf_levels(prices: pd.DataFrame) -> pd.Series:
    """ADF p-value of each coin's own log price (is the single coin already mean-reverting?).

    Why this matters: if log(a) is stationary on its own (e.g. a coin stuck in a range), the
    regression of a on b finds a small beta and the "spread" is basically log(a) itself. The pair
    then looks cointegrated although b adds nothing. A real pair has non-stationary legs (high
    p-values here) and a stationary spread.

    Input:  price DataFrame.
    Output: Series of ADF p-values, one per coin.
    """
    pvalues = {}
    for coin in prices.columns:
        log_price = np.log(prices[coin].dropna())
        pvalues[coin] = adfuller(log_price, maxlag=24, autolag="aic", result_object=False)[1]
    return pd.Series(pvalues, name="adf_p_value_levels")


def rolling_coint_pvalues(a: pd.Series, b: pd.Series, window: int, step: int) -> pd.Series:
    """Engle-Granger p-value on rolling windows of `window` bars, recomputed every `step` bars.

    The value at time t only uses data up to and including t (no lookahead). We use a fixed ADF
    lag of 1 here instead of an AIC search, because this runs many times and must be fast.

    Input:  price series a and b, window length and step in bars of the input data.
    Output: Series of p-values indexed by the end of each window.
    """
    log_a, log_b = np.log(a), np.log(b)
    pvalues = {}
    for end in range(window, len(log_a) + 1, step):
        window_a, window_b = log_a.iloc[end - window:end], log_b.iloc[end - window:end]
        pvalues[log_a.index[end - 1]] = coint(window_a, window_b, maxlag=1, autolag=None)[1]
    return pd.Series(pvalues, name="p_value", dtype=float)


def cointegration_stability(
    a: pd.Series, b: pd.Series, window_days: int = 180, step_days: int = 30, alpha: float = 0.05
) -> float:
    """Share of rolling sub-windows in which the pair is cointegrated (p < alpha).

    A pair that is cointegrated over the full train period but only in a few sub-windows is a
    one-off, not a stable relationship. Runs on daily closes, which is fast and enough for this check.

    Input:  hourly price series a and b, window length and step in days, significance level.
    Output: share of windows with p < alpha (0..1).
    """
    daily_a, daily_b = a.resample("1D").last(), b.resample("1D").last()
    pvalues = rolling_coint_pvalues(daily_a, daily_b, window=window_days, step=step_days)
    return float((pvalues < alpha).mean())


def rank_pairs(
    prices: pd.DataFrame,
    adf_maxlag: int = 24,
    window_days: int = 180,
    step_days: int = 30,
) -> pd.DataFrame:
    """Test every pair of coins for cointegration and rank them by p-value.

    Engle-Granger is not symmetric (regressing a on b or b on a gives slightly different results),
    so each pair is tested in both directions and the direction with the lower p-value is kept.

    Multiple testing: with 12 coins there are 66 pairs, i.e. 132 tests. At a 5% level we would
    expect about 6-7 "significant" results by pure chance even if no pair were cointegrated.
    p_bonferroni = p x number of tests (capped at 1) is the simplest, conservative correction:
    only pairs with p_bonferroni < 0.05 are significant after accounting for all tests.

    Input:  price DataFrame (TRAIN data only), ADF lag limit, stability window and step in days.
    Output: DataFrame, one row per pair: a, b, correlation, p_value, p_bonferroni, adf_stat,
            beta, half_life (hours), stability, sorted by p_value.
    """
    corr = np.log(prices).diff().corr()
    pairs = list(combinations(prices.columns, 2))
    n_tests = 2 * len(pairs)  # both directions

    rows = []
    for x, y in pairs:
        # test both directions, keep the one with the lower p-value
        results = [(x, y, test_cointegration(prices[x], prices[y], adf_maxlag)),
                   (y, x, test_cointegration(prices[y], prices[x], adf_maxlag))]
        a, b, res = min(results, key=lambda r: r[2]["p_value"])
        rows.append({
            "a": a,
            "b": b,
            "correlation": corr.loc[a, b],
            "p_value": res["p_value"],
            "p_bonferroni": min(res["p_value"] * n_tests, 1.0),
            "adf_stat": res["adf_stat"],
            "beta": res["beta"],
            "half_life": res["half_life"],
            "stability": cointegration_stability(prices[a], prices[b], window_days, step_days),
        })
    return pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)


# =============================================================================
# 2. SIGNALS
#    Hedge ratio -> spread -> z-score -> desired direction (+1 / 0 / -1)
# =============================================================================

def hedge_ratio(a: pd.Series, b: pd.Series, window: int | None = None) -> float | pd.Series:
    """Hedge ratio beta from an OLS regression of log(a) on log(b) (with intercept).

    window=None: one static beta estimated on all the input data (pass TRAIN data only!).
    window=n:    rolling beta over the last n bars; the value at t uses data up to and including t.
                 The OLS slope equals cov(log a, log b) / var(log b), which pandas computes fast.

    Input:  price series a and b, window in bars or None.
    Output: float (static) or Series (rolling, NaN during the first window-1 bars).
    """
    log_a, log_b = np.log(a), np.log(b)
    if window is None:
        return float(sm.OLS(log_a, sm.add_constant(log_b)).fit().params.iloc[1])
    return (log_a.rolling(window).cov(log_b) / log_b.rolling(window).var()).rename("beta")


def hedge_intercept(a: pd.Series, b: pd.Series, beta: float | pd.Series, window: int | None = None) -> float | pd.Series:
    """OLS intercept alpha that belongs to a hedge ratio: alpha = mean(log a) - beta * mean(log b).

    Why this matters for a ROLLING beta: log(BTC) is about 11, so a change of beta by only 0.01
    shifts log(a) - beta * log(b) by 0.11, i.e. by a jump much larger than normal spread moves.
    Subtracting the rolling intercept turns the spread into the residual of the rolling regression,
    which stays centred around zero when beta moves. (Measured on ETH/BTC train data: the half-life
    drops from ~22,700 h without the intercept to ~200 h with it.)

    Input:  price series a and b, hedge ratio (float or Series), window (None = all input data).
    Output: float (static) or Series (rolling, same past-only window as hedge_ratio).
    """
    log_a, log_b = np.log(a), np.log(b)
    if window is None:
        return float(log_a.mean() - beta * log_b.mean())
    return (log_a.rolling(window).mean() - beta * log_b.rolling(window).mean()).rename("alpha")


def compute_spread(a: pd.Series, b: pd.Series, beta: float | pd.Series, alpha: float = 0.0) -> pd.Series:
    """Spread = log(a) - beta * log(b) - alpha.

    With a static beta, alpha only shifts the level and the z-score removes it anyway. With a
    rolling beta, pass the rolling alpha from hedge_intercept (see there why).

    Input:  price series a and b, hedge ratio (float or Series), intercept (float or Series).
    Output: spread Series.
    """
    return (np.log(a) - beta * np.log(b) - alpha).rename("spread")


def zscore(spread: pd.Series, lookback: int) -> pd.Series:
    """Standardise the spread with its rolling mean and standard deviation over the past `lookback` bars.

    The rolling window ends at t (inclusive), so z(t) only uses data known at the close of bar t.

    Input:  spread Series, lookback in bars.
    Output: z-score Series (NaN during the first lookback-1 bars).
    """
    mean = spread.rolling(lookback).mean()
    std = spread.rolling(lookback).std()
    return ((spread - mean) / std).rename("z")


def frozen_zscore(
    a: pd.Series, b: pd.Series, beta: pd.Series, alpha: pd.Series, lookback: int
) -> Callable[[int, int], float]:
    """Build a function z(entry_bar, t): the z-score at bar t of the spread with the hedge FROZEN at entry.

    Why: once a trade is open we hold fixed units of A and B, bought with the beta of the entry bar.
    The spread we are actually exposed to is therefore log(a) - beta_entry * log(b) - alpha_entry,
    not the spread with today's rolling beta. Exits and stops must judge THIS spread; otherwise a
    moving beta could signal "reverted" while the position we hold has not made any money (or the
    other way round). While flat, the rolling beta and intercept keep updating the z-score as usual.

    The z-score of the frozen spread over the last `lookback` bars only needs rolling means,
    variances and the covariance of log(a) and log(b), which we compute once for all bars:
        mean(spread) = mean(log a) - beta_e * mean(log b) - alpha_e
        var(spread)  = var(log a) + beta_e^2 * var(log b) - 2 * beta_e * cov(log a, log b)
    (alpha_e is frozen too, but it shifts spread and mean equally, so it cancels in the z-score.)
    The result equals zscore(compute_spread(a, b, beta_e, alpha_e), lookback) at every bar.

    Input:  price series a and b, rolling (or constant) beta and alpha Series, z-score lookback in bars.
    Output: function (entry_bar, t) -> z-score at bar t (integer positions in the index).
    """
    log_a, log_b = np.log(a), np.log(b)
    la, lb = log_a.to_numpy(), log_b.to_numpy()
    mean_a, mean_b = log_a.rolling(lookback).mean().to_numpy(), log_b.rolling(lookback).mean().to_numpy()
    var_a, var_b = log_a.rolling(lookback).var().to_numpy(), log_b.rolling(lookback).var().to_numpy()
    cov_ab = log_a.rolling(lookback).cov(log_b).to_numpy()
    betas, alphas = beta.to_numpy(), alpha.to_numpy()

    def z_at(entry_bar: int, t: int) -> float:
        beta_e, alpha_e = betas[entry_bar], alphas[entry_bar]
        spread = la[t] - beta_e * lb[t] - alpha_e
        mean = mean_a[t] - beta_e * mean_b[t] - alpha_e
        std = np.sqrt(var_a[t] + beta_e ** 2 * var_b[t] - 2 * beta_e * cov_ab[t])
        return (spread - mean) / std

    return z_at


def generate_signals(z: pd.Series, entry_z: float, exit_z: float) -> pd.Series:
    """Turn the z-score into a desired direction with a simple state machine.

    flat  -> long  (+1) when z < -entry_z   (spread unusually low, expect it to rise)
    flat  -> short (-1) when z > +entry_z   (spread unusually high, expect it to fall)
    long  -> flat       when z >= -exit_z   (spread is back within exit_z of its mean)
    short -> flat       when z <= +exit_z
    After an exit the same bar is checked for a new entry, so a direct flip is possible.
    While z is NaN (warm-up) the signal is 0.

    Input:  z-score Series, entry threshold, exit threshold (exit_z < entry_z).
    Output: Series of +1, -1 or 0, decided at the close of each bar.
    """
    signal = np.zeros(len(z))
    state = 0
    for t, value in enumerate(z.to_numpy()):
        if np.isnan(value):
            state = 0
        else:
            # 1) exit an open position once z is back near the mean
            if state == 1 and value >= -exit_z:
                state = 0
            elif state == -1 and value <= exit_z:
                state = 0
            # 2) enter when flat and z is extreme
            if state == 0:
                if value < -entry_z:
                    state = 1
                elif value > entry_z:
                    state = -1
        signal[t] = state
    return pd.Series(signal, index=z.index, name="signal")


# =============================================================================
# 3. RISK MANAGEMENT
#    Cut trades that go wrong, block entries in bad regimes, size positions -> final positions
# =============================================================================

def apply_risk(
    signals: pd.Series,
    z: pd.Series,
    half_life: float,
    params: dict,
    coint_pvalues: pd.Series | None = None,
    trade_z: Callable[[int, int], float] | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Turn raw signals into positions: entries, exits, stops, the cointegration filter and sizing.

    Entries (while flat) use the rolling z-score `z` and the raw signals:
        a new trade needs a fresh signal: entry_z < |z| <= stop_z on the entry bar (beyond stop_z it
        would be stopped out at once and only cost fees).
    While a trade is open, the z-score is `trade_z(entry_bar, t)`, i.e. the spread with the hedge
    ratio frozen at entry (see frozen_zscore); without trade_z the rolling z is used. Exits, checked
    bar by bar (first match wins):
        z_stop       |z| > stop_z: the spread kept diverging -> close
        time_stop    open for >= time_stop_halflives x half_life hours -> close
        coint_break  rolling cointegration p-value > coint_max_pvalue (only if use_coint_filter)
        signal       z is back within exit_z of the mean -> close (a flip can follow on the same bar)
    After every exit, no new trade in the SAME direction until |z| has been back below entry_z
    (cooldown). This stops us from re-entering the same divergence right after a stop, or right
    after the frozen spread reverted while the rolling z-score still looks extreme.
    With use_coint_filter, new entries are also blocked while the p-value is above the threshold
    (or not yet known during the first window).
    An open trade is closed on the last bar ("end_of_data"), so its closing cost is counted.
    Every rule can be switched off: stop_z=None, time_stop_halflives=None, use_coint_filter=False.

    Input:  raw signals (+1/0/-1) from generate_signals, rolling z-score, half-life in hours,
            params dict with keys entry_z, exit_z, stop_z, time_stop_halflives, position_size,
            use_coint_filter, coint_max_pvalue; rolling cointegration p-values (needed only if
            use_coint_filter); optional frozen-hedge z function from frozen_zscore.
    Output: (positions, exit_reasons, z_used):
            positions    = direction x position_size
            exit_reasons = "signal", "z_stop", "time_stop", "coint_break" or "end_of_data" on the bar
                           where a trade is closed, None elsewhere
            z_used       = the z-score each decision was based on (frozen-hedge z while in a trade)
    """
    entry_z, exit_z = params["entry_z"], params["exit_z"]
    stop_z = params.get("stop_z")
    time_stop = params.get("time_stop_halflives")
    max_hours = time_stop * half_life if time_stop is not None and np.isfinite(half_life) else None
    use_filter = params.get("use_coint_filter", False)
    max_pvalue = params.get("coint_max_pvalue")
    if use_filter:
        # p-value known at t (recomputed every few hours, carried forward in between)
        pvalue = coint_pvalues.reindex(z.index, method="ffill").to_numpy()

    sig, zs = signals.to_numpy(), z.to_numpy()
    n = len(sig)
    direction = np.zeros(n)
    z_used = zs.copy()
    reasons = np.full(n, None, dtype=object)
    state, entry_bar = 0, 0
    blocked = 0          # direction of the last exit; no re-entry in it until |z| < entry_z

    for t in range(n):
        coint_ok = (not use_filter) or (not np.isnan(pvalue[t]) and pvalue[t] <= max_pvalue)

        # --- manage an open trade, judged on the spread we actually hold ---
        if state != 0:
            z_now = trade_z(entry_bar, t) if trade_z is not None else zs[t]
            z_used[t] = z_now
            reverted = (state == 1 and z_now >= -exit_z) or (state == -1 and z_now <= exit_z)
            exit_reason = None
            if stop_z is not None and abs(z_now) > stop_z:
                exit_reason = "z_stop"
            elif max_hours is not None and t - entry_bar >= max_hours:
                exit_reason = "time_stop"
            elif not coint_ok:
                exit_reason = "coint_break"
            elif reverted:
                exit_reason = "signal"
            if exit_reason:
                reasons[t], blocked, state = exit_reason, state, 0

        # --- entries use the rolling z-score ---
        abs_z = abs(zs[t]) if not np.isnan(zs[t]) else np.nan
        if blocked and abs_z < entry_z:          # cooldown over: the spread is back inside the band
            blocked = 0
        fresh_signal = sig[t] != 0 and abs_z > entry_z and (stop_z is None or abs_z <= stop_z)
        if state == 0 and fresh_signal and sig[t] != blocked and coint_ok:
            state, entry_bar = int(sig[t]), t
            z_used[t] = zs[t]

        direction[t] = state

    # close anything still open on the last bar
    if n and direction[-1] != 0:
        direction[-1], reasons[-1] = 0, "end_of_data"

    index = signals.index
    positions = pd.Series(direction * params["position_size"], index=index, name="position")
    exit_reasons = pd.Series(reasons, index=index, name="exit_reason")
    return positions, exit_reasons, pd.Series(z_used, index=index, name="z")
