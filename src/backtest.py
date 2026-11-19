"""Backtest engine: turns positions into PnL, trades and an equity curve, then computes metrics.

Position convention
-------------------
`positions` is a Series in [-1, 1], one value per hourly bar, decided at the CLOSE of that bar:
    +1  long the spread  = long coin A, short coin B
    -1  short the spread = short coin A, long coin B
     0  flat
The magnitude is the fraction of capital used, e.g. 0.5 = half of the equity is deployed.

Leg weights (fractions of current equity) for hedge ratio beta:
    w_a =  p        / (1 + |beta|)
    w_b = -p * beta / (1 + |beta|)
so the gross exposure |w_a| + |w_b| = |p|, i.e. |p| x equity in total across both legs, split in
the ratio 1 : beta. With beta = 1 and p = 1 we hold 50% of equity long A and 50% short B.

No lookahead
------------
Weights decided at the close of bar t are held during bar t+1 and earn the return of bar t+1.
In code this is an explicit `.shift(1)` of the weights ("held weights").

Simplification to be aware of: weights are fractions of equity, so we implicitly rebalance to
constant weights every bar. Trading costs are only charged when the target weights change
(entries, exits, hedge-ratio updates), not for this small drift rebalancing.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

HOURS_PER_YEAR = 24 * 365          # crypto trades around the clock, every day of the year
HOURS_PER_MONTH = HOURS_PER_YEAR / 12


@dataclass
class Result:
    """Everything one backtest run produces.

    params:      the parameters used for this run (dict)
    positions:   target position per bar in [-1, 1], decided at the bar's close
    weights:     DataFrame with columns w_a, w_b (target leg weights, fractions of equity)
    returns:     net return per bar (fraction of equity)
    components:  DataFrame per bar with the signed parts of the net return:
                 gross (price PnL), fees (<= 0), slippage (<= 0), funding (+/-); they sum to `returns`
    equity:      equity curve in USDT, one value per bar
    trades:      one row per round trip (see extract_trades)
    metrics:     dict from compute_metrics
    spread, z, beta, prices: filled in by run_strategy, used by the plots (None if not available)
    """

    params: dict
    positions: pd.Series
    weights: pd.DataFrame
    returns: pd.Series
    components: pd.DataFrame
    equity: pd.Series
    trades: pd.DataFrame
    metrics: dict = field(default_factory=dict)
    spread: pd.Series | None = None
    z: pd.Series | None = None
    beta: pd.Series | None = None
    prices: pd.DataFrame | None = None


# =============================================================================
# 1. BACKTEST
# =============================================================================

def leg_weights(positions: pd.Series, beta: pd.Series | float) -> pd.DataFrame:
    """Convert spread positions into the two leg weights w_a and w_b (fractions of equity).

    Input:  positions in [-1, 1], hedge ratio beta (Series on the same index, or a constant).
    Output: DataFrame with columns w_a, w_b. Bars with an undefined beta (rolling warm-up) get 0.
    """
    beta = pd.Series(beta, index=positions.index) if np.isscalar(beta) else beta.reindex(positions.index)
    scale = 1 + beta.abs()
    w_a = positions / scale
    w_b = -positions * beta / scale
    return pd.DataFrame({"w_a": w_a, "w_b": w_b}).fillna(0.0)


def run_backtest(
    price_a: pd.Series,
    price_b: pd.Series,
    beta: pd.Series | float,
    positions: pd.Series,
    fee_per_leg: float,
    slippage_per_leg: float,
    funding_a: pd.Series | None = None,
    funding_b: pd.Series | None = None,
    initial_capital: float = 10_000,
    exit_reasons: pd.Series | None = None,
    z: pd.Series | None = None,
    params: dict | None = None,
) -> Result:
    """Simulate the two-leg trade bar by bar and return a Result with equity, trades and metrics.

    Per bar t (all as fractions of the equity at the start of the bar):
        gross(t)    = w_a(t-1) * r_a(t) + w_b(t-1) * r_b(t)        r = simple return of the bar
        costs(t)    = (fee + slippage) * (|w_a(t) - w_a(t-1)| + |w_b(t) - w_b(t-1)|)
        funding(t)  = -(w_a(t-1) * f_a(t+1) + w_b(t-1) * f_b(t+1))  f = funding rate (see below)
        net(t)      = gross(t) - costs(t) + funding(t)
        equity(t)   = equity(t-1) * (1 + net(t)), starting from initial_capital

    Costs are charged in the bar whose close the trade happens at, on every change of the target
    weights: entries, exits, sign flips and also hedge-ratio rebalancing while a trade is open.

    Funding timing:
        Candles are labelled by their OPEN time: bar 07:00 covers 07:00-08:00. Funding labelled 08:00
        is paid by whoever holds the position at 08:00, i.e. the position held DURING bar 07:00.
        That position is w.shift(1) at 07:00 (decided at the close of bar 06:00). So we move the
        funding one bar back with funding.shift(-1) and multiply it with the same shifted weights
        that earn that bar's price return.
        This is not lookahead: funding rates are published before they are paid, and the position
        that pays was decided one bar before the payment. A position opened at the close of bar
        07:00 (= 08:00) does not pay the 08:00 funding.
        Sign: a positive rate means longs pay shorts, so a long leg (w > 0) pays and a short leg
        (w < 0) receives: funding PnL = -w * rate.

    Input:  close prices of both legs, hedge ratio (Series or constant), target positions in [-1, 1],
            fee and slippage per leg (fractions of traded notional), optional hourly funding rates
            per leg (0 in hours without funding), initial capital, optional exit reasons per bar and
            z-score (only used to annotate trades), optional params dict to store in the Result.
    Output: Result.
    """
    index = positions.index
    weights = leg_weights(positions, beta)

    # NO LOOKAHEAD: weights decided at the close of bar t are only held from bar t+1 on.
    # shift(1) moves every weight one bar later, so bar t+1 uses the decision from bar t.
    held = weights.shift(1).fillna(0.0)

    # Simple returns of each leg within each bar; a missing price (data gap) counts as no move.
    r_a = price_a.reindex(index).pct_change().fillna(0.0)
    r_b = price_b.reindex(index).pct_change().fillna(0.0)
    gross = held["w_a"] * r_a + held["w_b"] * r_b

    # Turnover = how much of the equity is traded at the close of bar t (both legs together).
    # Comparing with the previous bar's weights (0 before the first bar) catches every change.
    turnover = (weights - weights.shift(1, fill_value=0.0)).abs().sum(axis=1)
    fees = -fee_per_leg * turnover
    slippage = -slippage_per_leg * turnover

    # Funding: shift(-1) puts the funding paid at 08:00 on bar 07:00, where it meets the held
    # weights of bar 07:00 (see docstring). The last bar has no "next" funding -> 0.
    funding = pd.Series(0.0, index=index)
    for leg, rates in (("w_a", funding_a), ("w_b", funding_b)):
        if rates is not None:
            next_rate = rates.reindex(index).fillna(0.0).shift(-1).fillna(0.0)
            funding -= held[leg] * next_rate

    components = pd.DataFrame({"gross": gross, "fees": fees, "slippage": slippage, "funding": funding})
    returns = components.sum(axis=1).rename("returns")
    equity = (initial_capital * (1 + returns).cumprod()).rename("equity")

    trades = extract_trades(
        positions, weights, components, fee_per_leg, slippage_per_leg, z=z, exit_reasons=exit_reasons
    )
    result = Result(
        params=dict(params or {}, initial_capital=initial_capital),
        positions=positions,
        weights=weights,
        returns=returns,
        components=components,
        equity=equity,
        trades=trades,
        z=z,
    )
    result.metrics = compute_metrics(result)
    return result


# =============================================================================
# 2. TRADES
# =============================================================================

TRADE_COLUMNS = [
    "entry_time", "exit_time", "direction", "holding_hours", "entry_z", "exit_z",
    "pnl_pct", "fees", "slippage", "funding", "exit_reason",
]


def extract_trades(
    positions: pd.Series,
    weights: pd.DataFrame,
    components: pd.DataFrame,
    fee_per_leg: float,
    slippage_per_leg: float,
    z: pd.Series | None = None,
    exit_reasons: pd.Series | None = None,
) -> pd.DataFrame:
    """Cut the position series into round trips and compute the PnL of each one.

    A trade starts at the bar where the position goes from 0 (or the opposite sign) to nonzero,
    and ends at the bar where it goes back to 0 or flips sign. A sign flip closes one trade and
    opens the next one at the same bar. A trade still open on the last bar ends there
    ("end_of_data", no closing cost because the data ends).

    Trade PnL (relative to the equity at entry) compounds over the trade's bars:
        entry bar:           - cost of opening both legs
        bars in between:     gross + funding - rebalancing costs
        exit bar:            gross + funding - cost of closing both legs
    Entry and exit costs are computed from the trade's own weights, so a sign flip splits its
    cost exactly between the closing and the opening trade.

    Input:  target positions, target leg weights, per-bar components from run_backtest,
            fee and slippage per leg, optional z-score and exit reasons (per bar).
    Output: DataFrame, one row per trade: entry_time, exit_time, direction ("long"/"short" spread),
            holding_hours, entry_z, exit_z, pnl_pct, fees, slippage, funding (all fractions of
            equity at entry, costs negative) and exit_reason.
    """
    sign = np.sign(positions.to_numpy())
    index = positions.index
    abs_w = weights.abs().sum(axis=1).to_numpy()   # gross weight of the open position per bar
    turnover = (weights - weights.shift(1, fill_value=0.0)).abs().sum(axis=1).to_numpy()
    gross = components["gross"].to_numpy()
    funding = components["funding"].to_numpy()
    cost_rate = fee_per_leg + slippage_per_leg

    # Find (entry bar, exit bar) pairs with one pass over the signs.
    segments = []
    entry = None
    for i in range(len(sign)):
        prev = sign[i - 1] if i > 0 else 0
        if entry is not None and sign[i] != prev:   # back to flat or sign flip: trade closes here
            segments.append((entry, i, True))
            entry = None
        if sign[i] != 0 and sign[i] != prev:        # new trade opens here
            entry = i
    if entry is not None:                           # still open at the end of the data
        segments.append((entry, len(sign) - 1, False))

    rows = []
    for i0, i1, closed in segments:
        # traded fraction of equity per bar of this trade: open, rebalance, (close)
        trade_turnover = np.zeros(i1 - i0 + 1)
        trade_turnover[0] = abs_w[i0]                    # open both legs from zero
        if i1 > i0:
            trade_turnover[1:-1] = turnover[i0 + 1:i1]   # hedge-ratio rebalancing while open
            # close both legs from the last held weights; an open trade only rebalances on its last bar
            trade_turnover[-1] = abs_w[i1 - 1] if closed else turnover[i1]

        # price PnL and funding start one bar after entry (held weights), see run_backtest
        trade_gross = np.r_[0.0, gross[i0 + 1:i1 + 1]]
        trade_funding = np.r_[0.0, funding[i0 + 1:i1 + 1]]
        net = trade_gross + trade_funding - cost_rate * trade_turnover

        exit_reason = exit_reasons.iloc[i1] if exit_reasons is not None else None
        if not closed and pd.isna(exit_reason):   # pd.isna covers both None and NaN
            exit_reason = "end_of_data"
        rows.append({
            "entry_time": index[i0],
            "exit_time": index[i1],
            "direction": "long" if sign[i0] > 0 else "short",
            "holding_hours": (index[i1] - index[i0]) / pd.Timedelta(hours=1),
            "entry_z": z.iloc[i0] if z is not None else np.nan,
            "exit_z": z.iloc[i1] if z is not None else np.nan,
            "pnl_pct": np.prod(1 + net) - 1,
            "fees": -fee_per_leg * trade_turnover.sum(),
            "slippage": -slippage_per_leg * trade_turnover.sum(),
            "funding": trade_funding.sum(),
            "exit_reason": exit_reason,
        })
    return pd.DataFrame(rows, columns=TRADE_COLUMNS)


# =============================================================================
# 3. METRICS
# =============================================================================

def compute_metrics(result: Result) -> dict:
    """Summarise a backtest in a few numbers.

    Conventions: hourly bars, annualised with 24 * 365 hours per year (crypto never closes),
    risk-free rate 0. Sharpe = mean(hourly return) / std(hourly return) * sqrt(24 * 365).
    Max drawdown is negative (e.g. -0.12 = 12% below the running peak). Cost and funding totals
    are in USDT and signed: negative = money paid.

    Input:  Result (uses returns, components, equity, positions, trades, params).
    Output: dict of metrics.
    """
    returns, equity, trades = result.returns, result.equity, result.trades
    initial = result.params["initial_capital"]
    n_bars = len(returns)

    total_return = equity.iloc[-1] / initial - 1
    years = n_bars / HOURS_PER_YEAR
    ann_return = (1 + total_return) ** (1 / years) - 1 if total_return > -1 else -1.0
    ann_vol = returns.std() * np.sqrt(HOURS_PER_YEAR)
    sharpe = returns.mean() / returns.std() * np.sqrt(HOURS_PER_YEAR) if returns.std() > 0 else np.nan

    # Drawdown: distance of equity below its running maximum (initial capital counts as a peak)
    running_peak = np.maximum(equity.cummax(), initial)
    max_drawdown = (equity / running_peak - 1).min()
    calmar = ann_return / abs(max_drawdown) if max_drawdown < 0 else np.nan

    # USDT amount of each component = fraction of equity x equity at the start of the bar
    equity_before = equity.shift(1).fillna(initial)
    usdt = result.components.mul(equity_before, axis=0).sum()

    n_trades = len(trades)
    return {
        "total_return": total_return,
        "annual_return": ann_return,
        "annual_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "n_trades": n_trades,
        "trades_per_month": n_trades / (n_bars / HOURS_PER_MONTH),
        "win_rate": (trades["pnl_pct"] > 0).mean() if n_trades else np.nan,
        "avg_trade_pnl": trades["pnl_pct"].mean() if n_trades else np.nan,
        "avg_holding_hours": trades["holding_hours"].mean() if n_trades else np.nan,
        "time_in_market": (result.positions != 0).mean(),
        "total_gross": usdt["gross"],
        "total_fees": usdt["fees"],
        "total_slippage": usdt["slippage"],
        "total_funding": usdt["funding"],
    }
