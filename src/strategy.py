"""Strategy logic in three steps: pair selection, signal generation, risk management.

Conventions:
    a, b        log prices of the two coins (a is the dependent leg)
    spread      log(a) - beta * log(b)
    position +1 long the spread  = long a, short beta units of b
    position -1 short the spread = short a, long beta units of b
"""

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
    raise NotImplementedError


def zscore(spread: pd.Series, lookback: int) -> pd.Series:
    """Standardise the spread with its rolling mean and standard deviation.

    Input:  spread Series, lookback window in bars.
    Output: z-score Series, (spread - rolling mean) / rolling std.
    """
    raise NotImplementedError


def generate_signals(z: pd.Series, entry_z: float, exit_z: float) -> pd.Series:
    """Turn the z-score into a desired direction: short the spread when z is high, long when low, flat after reversion.

    Input:  z-score Series, entry threshold, exit threshold.
    Output: Series of +1 (long spread), -1 (short spread) or 0 (flat).
    """
    # TODO: stateful loop: enter at |z| > entry_z, hold until z crosses exit_z
    raise NotImplementedError


# =============================================================================
# 3. RISK MANAGEMENT
#    Cut losing trades and size positions -> final positions
# =============================================================================

def apply_risk(signals: pd.Series, z: pd.Series, params: dict) -> pd.Series:
    """Apply the z-stop, the time stop and position sizing to the raw signals.

    Input:  raw signals (+1/0/-1), z-score Series, params dict (stop_z, time_stop_halflives,
            position_size, half_life).
    Output: Series of final positions (signed fraction of capital); 0 after a stop until the next fresh entry.
    """
    # TODO: z-stop if |z| > stop_z, time stop after time_stop_halflives * half_life bars, multiply by position_size
    raise NotImplementedError
