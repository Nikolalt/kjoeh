"""Strategy logic in three steps: pair selection, signal generation, risk management.

Conventions (pair track; the ratio track in section 4 uses a = alt, b = BTC, beta = 1):
    a, b        log prices of the two coins (a is the dependent leg)
    spread      log(a) - beta * log(b)
    position +1 long the spread  = long a, short beta units of b
    position -1 short the spread = short a, long beta units of b
"""

import numpy as np
import pandas as pd


# =============================================================================
# 1. PAIR SELECTION
#    Which two coins move together closely enough to trade the spread?
# =============================================================================

def correlation_screen(prices: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Rank all coin pairs by the correlation of their returns as a fast first filter.

    Input:  price DataFrame (one column per coin), number of pairs to keep.
    Output: DataFrame with columns a, b, correlation, sorted descending, top_n rows.
    """
    # TODO: correlation of log returns (not price levels, which gives spurious correlation)
    raise NotImplementedError


def test_cointegration(a: pd.Series, b: pd.Series) -> dict:
    """Run an Engle-Granger cointegration test on one pair and estimate its mean-reversion speed.

    Input:  log price series a and b.
    Output: dict with p_value, beta (OLS hedge ratio) and half_life (in bars, from an AR(1) fit on the spread).
    """
    # TODO: statsmodels.tsa.stattools.coint, OLS for beta, regress d(spread) on lagged spread
    raise NotImplementedError


def rank_pairs(prices: pd.DataFrame) -> pd.DataFrame:
    """Test candidate pairs for cointegration and rank them by p-value and half-life.

    Input:  price DataFrame (train data only).
    Output: DataFrame with columns a, b, correlation, p_value, beta, half_life, one row per pair.
    """
    # TODO: correlation_screen -> test_cointegration on each candidate -> sort
    raise NotImplementedError


# =============================================================================
# 2. SIGNALS
#    Spread -> z-score -> desired direction (+1 / 0 / -1)
# =============================================================================

def hedge_ratio(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
    """Estimate a rolling hedge ratio beta by regressing a on b over the past `window` bars only.

    Input:  log price series a and b, window length in bars.
    Output: Series of beta values; at time t it uses data up to and including t, never later.
    """
    raise NotImplementedError


def compute_spread(a: pd.Series, b: pd.Series, beta: pd.Series) -> pd.Series:
    """Compute the spread a - beta * b.

    Input:  log price series a and b, hedge ratio (Series or float).
    Output: spread Series.
    """
    return a - beta * b


def rolling_mean_std(series: pd.Series, lookback: int, method: str = "rolling") -> tuple[pd.Series, pd.Series]:
    """Mean and standard deviation of a series using only data up to and including each bar.

    Input:  Series, lookback window in bars, method "rolling" (last `lookback` bars) or
            "expanding" (all bars so far, at least `lookback`).
    Output: (mean, std) Series; NaN until `lookback` bars are available.
    """
    if method == "rolling":
        window = series.rolling(lookback, min_periods=lookback)
    elif method == "expanding":
        window = series.expanding(min_periods=lookback)
    else:
        raise ValueError(f"unknown method {method!r}, use 'rolling' or 'expanding'")
    return window.mean(), window.std()


def zscore(spread: pd.Series, lookback: int, method: str = "rolling") -> pd.Series:
    """Standardise the spread with its rolling mean and standard deviation.

    Input:  spread Series, lookback window in bars, method "rolling" (default) or "expanding".
    Output: z-score Series, (spread - rolling mean) / rolling std; at time t it uses data up to t only.
    """
    mean, std = rolling_mean_std(spread, lookback, method)
    return (spread - mean) / std.replace(0.0, np.nan)


def generate_signals(z: pd.Series, entry_z: float, exit_z: float) -> pd.Series:
    """Turn the z-score into a desired direction: short the spread when z is high, long when low, flat after reversion.

    Input:  z-score Series, entry threshold, exit threshold.
    Output: Series of +1 (long spread), -1 (short spread) or 0 (flat).

    Rules (exit_z is mirrored for longs):
        flat  -> -1 when z > +entry_z,  -> +1 when z < -entry_z
        -1    -> exit when z <= exit_z, +1 -> exit when z >= -exit_z
    An exit and an entry on the other side can happen on the same bar. NaN z: flat stays flat, a
    position is held.
    """
    out = np.zeros(len(z))
    state = 0
    for i, value in enumerate(z.to_numpy()):
        if np.isnan(value):
            out[i] = state
            continue
        if (state == -1 and value <= exit_z) or (state == 1 and value >= -exit_z):
            state = 0
        if state == 0:
            state = -1 if value > entry_z else 1 if value < -entry_z else 0
        out[i] = state
    return pd.Series(out, index=z.index)


# =============================================================================
# 3. RISK MANAGEMENT
#    Cut losing trades and size positions -> final positions
# =============================================================================

EXIT_REASONS = ("mean", "stop_z", "stop_loss", "take_profit", "time", "end")  # "end" is set by the backtest for trades still open


def max_holding_bars(params: dict) -> float:
    """Time-stop length in bars: time_stop_halflives x half_life, capped by max_holding_days if given.

    Input:  params dict (time_stop_halflives, half_life, optional max_holding_days).
    Output: number of bars (inf if neither limit applies).
    """
    limit = params.get("time_stop_halflives", np.inf) * params.get("half_life", np.inf)
    return min(limit, params.get("max_holding_days") or np.inf)


def apply_risk(
    signals: pd.Series,
    z: pd.Series,
    params: dict,
    return_reasons: bool = False,
    spread: pd.Series | None = None,
) -> pd.Series | pd.DataFrame:
    """Apply the z-stop, the time stop and position sizing to the raw signals.

    Rules:
        z-stop       close when |z| > stop_z (the ratio keeps running away: mean reversion failed);
                     an entry signal that fires while |z| > stop_z is treated the same way
        stop loss    close when the spread has moved more than stop_loss against the trade since entry
                     (needs `spread`; for a log spread this is about the loss as a fraction of notional).
                     Unlike the z-stop it also catches slow trends, which the rolling mean follows.
        take profit  close when the spread has moved more than take_profit in favour of the trade
        time stop    close after max_holding_bars(params) bars in the trade
        after a stop, stay flat until the raw signal changes (back to 0 or to the other side)

    Input:  raw signals (+1/0/-1), z-score Series, params dict (stop_z, time_stop_halflives,
            position_size, half_life, optional max_holding_days, stop_loss, take_profit; missing or
            None = rule off), return_reasons flag, optional spread Series (e.g. the log ratio).
    Output: Series of final positions (signed fraction of capital); 0 after a stop until the next fresh entry.
            With return_reasons: DataFrame with columns position and exit_reason (one of EXIT_REASONS
            on the bar a trade is closed, "" otherwise).
    """
    stop_z = params.get("stop_z") or np.inf
    stop_loss = params.get("stop_loss") or np.inf
    take_profit = params.get("take_profit") or np.inf
    max_bars = max_holding_bars(params)
    sig, zs = signals.to_numpy(), z.to_numpy()
    sp = spread.to_numpy() if spread is not None else np.zeros(len(sig))
    entry_spread = 0.0
    position = np.zeros(len(sig))
    reason = np.full(len(sig), "", dtype=object)
    state, held, blocked, prev_sig = 0, 0, False, 0
    for i in range(len(sig)):
        if blocked and sig[i] != prev_sig:
            blocked = False
        breakout = abs(zs[i]) > stop_z  # False for NaN
        if state != 0:
            held += 1
            if sig[i] != state:
                reason[i] = "mean"
                state = 0
            elif breakout:
                reason[i], state, blocked = "stop_z", 0, True
            elif state * (sp[i] - entry_spread) < -stop_loss:
                reason[i], state, blocked = "stop_loss", 0, True
            elif state * (sp[i] - entry_spread) > take_profit:
                reason[i], state, blocked = "take_profit", 0, True
            elif held >= max_bars:
                reason[i], state, blocked = "time", 0, True
        if state == 0 and sig[i] != 0 and not blocked:
            if breakout:
                blocked = True  # signal fires during a breakout: treat like a stop, wait for a fresh signal
            else:
                state, held, entry_spread = int(sig[i]), 0, sp[i]
        position[i] = state
        prev_sig = sig[i]
    position = pd.Series(position * params.get("position_size", 1.0), index=signals.index)
    if return_reasons:
        return pd.DataFrame({"position": position, "exit_reason": reason}, index=signals.index)
    return position


SIZING_METHODS = ("equal_capital", "equal_risk", "correlation", "corr_risk", "sharpe", "corr_sharpe")
NEEDS_SHARPE = ("sharpe", "corr_sharpe")


def equal_risk_sizes(ratio_vol: pd.Series, max_gross_leverage: float, equal_capital: bool = False) -> pd.Series:
    """Size of each pair (per leg, as a fraction of capital) so every pair carries the same risk.

    size_i ~ 1 / ratio_vol_i, scaled so that with all pairs open the gross exposure
    sum_i 2 * size_i equals max_gross_leverage. Use train-period volatility only.

    Input:  annualised ratio volatility per alt (e.g. pair_diagnostics(train)["ratio_vol"]),
            gross leverage cap, equal_capital flag (same size for every pair instead).
    Output: Series of per-leg sizes indexed by alt.
    """
    method = "equal_capital" if equal_capital else "equal_risk"
    return pair_sizes(ratio_vol, None, max_gross_leverage, method)


def pair_sizes(
    ratio_vol: pd.Series,
    corr_btc: pd.Series | None,
    max_gross_leverage: float,
    method: str = "equal_risk",
    sharpe: pd.Series | None = None,
) -> pd.Series:
    """Size of each pair (per leg, as a fraction of capital), weighted by risk, correlation with BTC
    and/or the pair's own train Sharpe.

    Raw weight per alt, then scaled so that with all pairs open sum_i 2 * size_i = max_gross_leverage:
        equal_capital  1
        equal_risk     1 / ratio_vol                 (wilder ratios get less)
        correlation    corr_btc                      (alts that move more with BTC get more: the BTC leg
                                                      hedges them better, so less of the trade is
                                                      unhedged alt-specific risk)
        corr_risk      corr_btc / ratio_vol          (both)
        sharpe         max(sharpe, 0)                (pairs that made money on train get more)
        corr_sharpe    corr_btc * max(sharpe, 0)     (correlation and Sharpe together)
    Negative correlations and Sharpes are floored at 0, so those pairs get no position. Use train
    data only. Sharpe weighting picks pairs by their own past backtest, which tends to overfit.

    Input:  annualised ratio volatility per alt, correlation of daily returns with BTC per alt
            (both from pair_diagnostics(train); corr_btc may be None for the first two methods),
            gross leverage cap, sizing method, per-pair train Sharpe (needed for sharpe / corr_sharpe).
    Output: Series of per-leg sizes indexed by alt.
    """
    if method not in SIZING_METHODS:
        raise ValueError(f"sizing must be one of {SIZING_METHODS}, got {method!r}")
    if method in ("correlation", "corr_risk", "corr_sharpe"):
        if corr_btc is None:
            raise ValueError(f"sizing {method!r} needs corr_btc")
        corr = corr_btc.reindex(ratio_vol.index).clip(lower=0.0)
    if method in NEEDS_SHARPE:
        if sharpe is None:
            raise ValueError(f"sizing {method!r} needs sharpe")
        sr = sharpe.reindex(ratio_vol.index).clip(lower=0.0)
    raw = {
        "equal_capital": lambda: pd.Series(1.0, index=ratio_vol.index),
        "equal_risk": lambda: 1.0 / ratio_vol,
        "correlation": lambda: corr,
        "corr_risk": lambda: corr / ratio_vol,
        "sharpe": lambda: sr,
        "corr_sharpe": lambda: corr * sr,
    }[method]().astype(float)
    if raw.sum() <= 0:
        raise ValueError("all sizing weights are zero")
    return raw / raw.sum() * max_gross_leverage / 2


# =============================================================================
# 4. RATIO VS BTC
#    How far is each alt from its usual price in BTC, and how fast does it come back?
# =============================================================================

def compute_ratio(alt: pd.Series, btc: pd.Series, log: bool = True) -> pd.Series:
    """Price of the alt expressed in BTC: the ratio alt / BTC, by default in logs.

    The log ratio is the spread log(alt) - 1 * log(btc), so its change equals the return of a
    dollar-neutral position long the alt and short BTC.

    Input:  alt and BTC price Series (USD), log flag.
    Output: Series log(alt / btc) if log else alt / btc.
    """
    if log:
        return compute_spread(np.log(alt), np.log(btc), 1.0)
    return alt / btc


def fair_value(btc: pd.Series, ratio_mean: pd.Series, log: bool = True) -> pd.Series:
    """Fair value of the alt in USD if the ratio were at its mean: BTC price x mean ratio.

    Input:  BTC price Series, mean of the ratio (log or plain, matching `log`).
    Output: Series of fair alt prices in USD.
    """
    return btc * (np.exp(ratio_mean) if log else ratio_mean)


def ratio_features(
    alt: pd.Series, btc: pd.Series, lookback: int, method: str = "rolling"
) -> pd.DataFrame:
    """All indicators of one alt/BTC pair in one table, each using only data up to that bar.

    Input:  alt and BTC price Series, lookback window in bars, method "rolling" or "expanding".
    Output: DataFrame with columns ratio (plain alt/BTC), log_ratio, mean, std (of the log ratio),
            z, fair_value (USD), deviation (alt / fair_value - 1).
    """
    log_ratio = compute_ratio(alt, btc, log=True)
    mean, std = rolling_mean_std(log_ratio, lookback, method)
    fair = fair_value(btc, mean)
    return pd.DataFrame({
        "ratio": compute_ratio(alt, btc, log=False),
        "log_ratio": log_ratio,
        "mean": mean,
        "std": std,
        "z": zscore(log_ratio, lookback, method),
        "fair_value": fair,
        "deviation": alt / fair - 1,
    })


def estimate_half_life(series: pd.Series) -> float:
    """Mean-reversion half-life from an AR(1) fit: regress the change on the lagged level.

    d(s_t) = c + phi * s_{t-1} + e_t, half-life = -ln(2) / ln(1 + phi).

    Input:  Series (e.g. log ratio on train data).
    Output: half-life in bars; inf if the series does not revert (phi >= 0).
    """
    s = series.dropna()
    lagged, change = s.shift(1).iloc[1:], s.diff().iloc[1:]
    phi = np.polyfit(lagged.to_numpy(), change.to_numpy(), 1)[0]
    if phi >= 0 or phi <= -1:
        return float("inf") if phi >= 0 else 0.0
    return float(-np.log(2) / np.log1p(phi))


def pair_diagnostics(
    prices: pd.DataFrame, lookback: int, method: str = "rolling", numeraire: str = "BTC"
) -> pd.DataFrame:
    """Per-alt statistics that tell whether the ratio to BTC is worth trading (use train data only).

    Two views of mean reversion:
      level:     the log ratio itself (does it return to one long-run mean?)
      deviation: log ratio minus its rolling mean, which is what the signal trades. Note that removing
                 a rolling mean makes any series look more mean-reverting, so a low ADF p-value here
                 supports the time scale of the trades, not their profitability.

    Input:  price DataFrame with one column per coin including the numeraire, lookback and method
            of the rolling mean.
    Output: DataFrame indexed by alt with columns corr_btc (daily log returns), ratio_vol (annualised
            std of daily log-ratio changes), half_life_level, adf_level, half_life_dev, adf_dev
            (half-lives in days, ADF p-values; low = mean-reverting).
    """
    from statsmodels.tsa.stattools import adfuller

    returns = np.log(prices).diff()
    rows = {}
    for alt in prices.columns.drop(numeraire):
        f = ratio_features(prices[alt], prices[numeraire], lookback, method)
        dev = (f["log_ratio"] - f["mean"]).dropna()
        rows[alt] = {
            "corr_btc": returns[alt].corr(returns[numeraire]),
            "ratio_vol": f["log_ratio"].diff().std() * np.sqrt(365),
            "half_life_level": estimate_half_life(f["log_ratio"]),
            "adf_level": adfuller(f["log_ratio"].dropna(), autolag="AIC")[1],
            "half_life_dev": estimate_half_life(dev),
            "adf_dev": adfuller(dev, autolag="AIC")[1],
        }
    return pd.DataFrame(rows).T


def return_correlations(prices: pd.DataFrame) -> pd.DataFrame:
    """Correlation matrix of daily log returns (not price levels, which gives spurious correlation).

    Input:  price DataFrame (one column per coin).
    Output: square correlation DataFrame.
    """
    return np.log(prices).diff().corr()
