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

from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint


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


def compute_spread(a: pd.Series, b: pd.Series, beta: float | pd.Series, alpha: float = 0.0) -> pd.Series:
    """Spread = log(a) - beta * log(b) - alpha.

    alpha only shifts the level; the z-score removes the level anyway, so alpha = 0 is fine there.

    Input:  price series a and b, hedge ratio (float or Series), intercept.
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
) -> tuple[pd.Series, pd.Series]:
    """Apply stops, the cointegration filter and position sizing to the raw signals.

    Rules, checked bar by bar while a trade is open (first match wins):
        z_stop       |z| > stop_z: the spread kept diverging -> close
        time_stop    open for >= time_stop_halflives x half_life hours -> close
        coint_break  rolling cointegration p-value > coint_max_pvalue (only if use_coint_filter)
        signal       the raw signal says flat (z reverted) or flips direction -> close (and maybe flip)
    After a forced exit (z_stop, time_stop, coint_break) there is a cooldown: no new entry until |z|
    has dropped back below entry_z. New entries need a fresh signal, i.e. |z| > entry_z on the entry
    bar, so we never jump back into an old, already stopped divergence.
    With use_coint_filter, new entries are also blocked while the p-value is above the threshold
    (or not yet known during the first window).
    An open trade is closed on the last bar ("end_of_data"), so its closing cost is counted.
    Every rule can be switched off: stop_z=None, time_stop_halflives=None, use_coint_filter=False.
    With all rules off, positions = signals x position_size.

    Input:  raw signals (+1/0/-1), z-score, half-life in hours, params dict with keys entry_z,
            stop_z, time_stop_halflives, position_size, use_coint_filter, coint_max_pvalue;
            hourly rolling cointegration p-values (needed only if use_coint_filter).
    Output: (positions, exit_reasons): positions = direction x position_size; exit_reasons holds
            "signal", "z_stop", "time_stop", "coint_break" or "end_of_data" on the bar where a trade
            is closed, None elsewhere.
    """
    entry_z = params["entry_z"]
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
    reasons = np.full(n, None, dtype=object)
    state, entry_bar, cooldown = 0, 0, False

    for t in range(n):
        abs_z = abs(zs[t]) if not np.isnan(zs[t]) else np.nan
        coint_ok = (not use_filter) or (not np.isnan(pvalue[t]) and pvalue[t] <= max_pvalue)

        # --- manage an open trade ---
        if state != 0:
            forced = None
            if stop_z is not None and abs_z > stop_z:
                forced = "z_stop"
            elif max_hours is not None and t - entry_bar >= max_hours:
                forced = "time_stop"
            elif not coint_ok:
                forced = "coint_break"

            if forced:
                state, reasons[t], cooldown = 0, forced, True
            elif sig[t] != state:            # raw signal exits or flips
                state, reasons[t] = 0, "signal"

        # --- cooldown ends once the spread is back inside the entry band ---
        if cooldown and abs_z < entry_z:
            cooldown = False

        # --- open a new trade (also covers the second half of a flip) ---
        fresh_signal = sig[t] != 0 and abs_z > entry_z
        if state == 0 and not cooldown and fresh_signal and coint_ok:
            state, entry_bar = int(sig[t]), t

        direction[t] = state

    # close anything still open on the last bar
    if n and direction[-1] != 0:
        direction[-1], reasons[-1] = 0, "end_of_data"

    positions = pd.Series(direction * params["position_size"], index=signals.index, name="position")
    exit_reasons = pd.Series(reasons, index=signals.index, name="exit_reason")
    return positions, exit_reasons
