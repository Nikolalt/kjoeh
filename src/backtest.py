"""Backtest engine: turns positions into PnL, trades and an equity curve, then computes metrics.

Position convention
-------------------
`positions` is a Series in [-1, 1], one value per bar, decided at the CLOSE of that bar:
    +1  long the spread  = long coin A, short coin B
    -1  short the spread = short coin A, long coin B
     0  flat
The magnitude is the fraction of capital used, e.g. 0.5 = half of the equity is deployed.

Holding units, like a real account
----------------------------------
When a trade is opened at the close of bar t, we buy/sell a fixed NUMBER OF UNITS (coins) of each leg,
computed from the equity at that moment, the position p and the hedge ratio beta at entry:
    notional_a =  p        / (1 + |beta|) * equity      ->  units_a = notional_a / price_a
    notional_b = -p * beta / (1 + |beta|) * equity      ->  units_b = notional_b / price_b
so the gross exposure |notional_a| + |notional_b| = |p| x equity, split in the ratio 1 : beta.
With beta = 1 and p = 1 we start with 50% of equity long A and 50% short B.

These units are then held UNCHANGED until the position changes (exit or flip). The leg weights
(notional / equity) drift with prices during the trade, exactly as in a real account. Trading costs
are charged only when units change: at entry, exit and flip.

No lookahead
------------
Units decided at the close of bar t are held during bar t+1 and earn the price change of bar t+1:
    PnL(t+1) = units_a(t) * (price_a(t+1) - price_a(t)) + units_b(t) * (price_b(t+1) - price_b(t))
In the loop below, the PnL of a bar is always computed BEFORE the units are updated at its close.
"""

from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from . import strategy
from .data import split_train_test

HOURS_PER_YEAR = 24 * 365          # crypto trades around the clock, every day of the year


@dataclass
class Result:
    """Everything one backtest run produces.

    params:       the parameters used for this run (dict)
    positions:    target position per bar in [-1, 1], decided at the bar's close
    units:        DataFrame with columns units_a, units_b: coins held after the close of each bar
    weights:      DataFrame with columns w_a, w_b: actual leg notional / equity after each close (drifts)
    returns:      net return per bar (fraction of equity at the start of the bar)
    components:   DataFrame per bar with the signed parts of the net return (fractions of equity):
                  gross (price PnL), fees (<= 0), slippage (<= 0), funding (+/-); they sum to `returns`
    equity:       equity curve in USDT, one value per bar
    prices:       DataFrame with the two price series (columns = coin names, or A/B)
    trades:       one row per round trip (see extract_trades)
    metrics:      dict from compute_metrics
    exit_reasons: why each trade was closed (on the exit bar), or None
    z:            z-score used for the decisions (None if not given)
    spread, beta: filled in by run_strategy, used by the plots (None if not available)
    """

    params: dict
    positions: pd.Series
    units: pd.DataFrame
    weights: pd.DataFrame
    returns: pd.Series
    components: pd.DataFrame
    equity: pd.Series
    prices: pd.DataFrame
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    metrics: dict = field(default_factory=dict)
    exit_reasons: pd.Series | None = None
    z: pd.Series | None = None
    spread: pd.Series | None = None
    beta: pd.Series | None = None


# =============================================================================
# 1. BACKTEST
# =============================================================================

def target_units(equity: float, position: float, beta: float, price_a: float, price_b: float) -> tuple[float, float]:
    """Number of coins of A and B to hold for a spread position, from equity and hedge ratio at entry.

    Input:  current equity (USDT), position p in [-1, 1], hedge ratio beta, prices of A and B.
    Output: (units_a, units_b); positive = long, negative = short. (0, 0) if p = 0.
    """
    if position == 0:
        return 0.0, 0.0
    scale = equity * position / (1 + abs(beta))
    return scale / price_a, -scale * beta / price_b


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
    """Simulate the two-leg trade bar by bar with fixed units per trade; return a Result.

    For every bar t, in this order:
      1. price PnL of the units held during the bar (set at the close of t-1):
             gross(t) = units_a * (price_a(t) - price_a(t-1)) + units_b * (price_b(t) - price_b(t-1))
      2. funding paid at the end of the bar, on the actual notional of each leg (see below):
             funding(t) = -(units_a * price_a(t) * rate_a + units_b * price_b(t) * rate_b)
      3. if the target position changes at the close of t: close the old units, open new ones
         (units from target_units with the equity after steps 1-2 and beta(t), frozen for the trade)
             costs(t) = (fee + slippage) * notional traded = (fee + slippage) * sum over legs of
                        (|old units| + |new units|) * price(t)
      4. equity(t) = equity(t-1) + gross + funding - costs

    Funding timing:
        Candles are labelled by their OPEN time: bar 07:00 covers 07:00-08:00. Funding labelled 08:00
        is paid by whoever holds the position at 08:00, i.e. the units held DURING bar 07:00. So we
        move the funding one bar back with funding.shift(-1) and apply it to bar 07:00, on the notional
        at 08:00 (units x close price of bar 07:00).
        This is not lookahead: funding rates are published before they are paid, and the position
        that pays was decided one bar before the payment. A position opened at the close of bar
        07:00 (= 08:00) does not pay the 08:00 funding.
        Sign: a positive rate means longs pay shorts, so a long leg (units > 0) pays and a short leg
        (units < 0) receives: funding PnL = -units * price * rate.

    Input:  close prices of both legs, hedge ratio (Series or constant; only its value at each entry
            is used), target positions in [-1, 1], fee and slippage per leg (fractions of traded
            notional), optional funding rates per leg on the same index (0 in bars without funding),
            initial capital, optional exit reasons and z-score per bar (to annotate trades),
            optional params dict to store in the Result.
    Output: Result.
    """
    index = positions.index
    pa = price_a.reindex(index).ffill().to_numpy(dtype=float)
    pb = price_b.reindex(index).ffill().to_numpy(dtype=float)
    beta_values = (pd.Series(beta, index=index) if np.isscalar(beta) else beta.reindex(index)).to_numpy()
    pos = positions.to_numpy(dtype=float)

    # Funding paid at the END of each bar: shift(-1) puts the 08:00 payment on bar 07:00 (see docstring).
    def next_rate(rates):
        if rates is None:
            return np.zeros(len(index))
        return rates.reindex(index).fillna(0.0).shift(-1).fillna(0.0).to_numpy()
    rate_a, rate_b = next_rate(funding_a), next_rate(funding_b)

    n = len(index)
    gross, funding, fees, slippage = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    equity, units_a, units_b, equity_before = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)

    held_a, held_b = 0.0, 0.0          # units held right now
    current_position = 0.0             # target position those units belong to
    eq = initial_capital
    for t in range(n):
        equity_before[t] = eq

        # 1) price PnL of the bar, from the units decided at the previous close (no lookahead)
        if t > 0:
            gross[t] = held_a * (pa[t] - pa[t - 1]) + held_b * (pb[t] - pb[t - 1])
        # 2) funding paid at the end of the bar, on the notional at that moment
        funding[t] = -(held_a * pa[t] * rate_a[t] + held_b * pb[t] * rate_b[t])
        eq += gross[t] + funding[t]

        # 3) trade at the close only if the target position changed (entry, exit or flip):
        #    close the old units completely and open the new ones; beta is frozen until the next change
        if pos[t] != current_position:
            new_a, new_b = target_units(eq, pos[t], beta_values[t], pa[t], pb[t])
            traded_notional = (abs(held_a) + abs(new_a)) * pa[t] + (abs(held_b) + abs(new_b)) * pb[t]
            fees[t] = -fee_per_leg * traded_notional
            slippage[t] = -slippage_per_leg * traded_notional
            eq += fees[t] + slippage[t]
            held_a, held_b, current_position = new_a, new_b, pos[t]

        equity[t], units_a[t], units_b[t] = eq, held_a, held_b

    # Express everything as fractions of the equity at the start of each bar, so the parts sum
    # to the bar's net return and compound to the equity curve.
    usdt = pd.DataFrame({"gross": gross, "fees": fees, "slippage": slippage, "funding": funding}, index=index)
    components = usdt.div(equity_before, axis=0)
    returns = components.sum(axis=1).rename("returns")
    equity = pd.Series(equity, index=index, name="equity")
    units = pd.DataFrame({"units_a": units_a, "units_b": units_b}, index=index)
    weights = pd.DataFrame({"w_a": units_a * pa / equity, "w_b": units_b * pb / equity}, index=index)
    prices = pd.DataFrame({price_a.name or "A": pa, price_b.name or "B": pb}, index=index)

    result = Result(
        params=dict(params or {}, initial_capital=initial_capital, fee_per_leg=fee_per_leg,
                    slippage_per_leg=slippage_per_leg),
        positions=positions,
        units=units,
        weights=weights,
        returns=returns,
        components=components,
        equity=equity,
        prices=prices,
        exit_reasons=exit_reasons,
        z=z,
    )
    result.trades = extract_trades(result)
    result.metrics = compute_metrics(result)
    return result


# =============================================================================
# 2. TRADES
# =============================================================================

TRADE_COLUMNS = [
    "entry_time", "exit_time", "direction", "holding_hours", "entry_z", "exit_z",
    "pnl_pct", "pnl_usdt", "fees", "slippage", "funding", "exit_reason",
]


def extract_trades(result: Result) -> pd.DataFrame:
    """Cut the backtest into round trips and compute the PnL of each one.

    A trade is a run of bars with the same nonzero target position: it opens at the close of the
    bar where the position changes to that value and closes at the close of the bar where it changes
    again (back to 0, or a flip, which closes one trade and opens the next one at the same bar).
    A trade still open on the last bar ends there ("end_of_data", no closing cost).

    Trade PnL (USDT) = price PnL + funding over the bars the units are held
                       - costs of opening and of closing the trade's own units.
    pnl_pct is relative to the equity just before the trade was opened.

    Input:  Result from run_backtest (uses positions, units, prices, components, equity, params,
            z and exit_reasons).
    Output: DataFrame, one row per trade: entry_time, exit_time, direction ("long"/"short" spread),
            holding_hours, entry_z, exit_z, pnl_pct (fraction), pnl_usdt, fees, slippage, funding
            (USDT, negative = paid) and exit_reason.
    """
    pos = result.positions.to_numpy()
    index = result.positions.index
    pa, pb = result.prices.iloc[:, 0].to_numpy(), result.prices.iloc[:, 1].to_numpy()
    units_a, units_b = result.units["units_a"].to_numpy(), result.units["units_b"].to_numpy()
    equity = result.equity.to_numpy()
    equity_before = result.equity.shift(1).fillna(result.params["initial_capital"])
    usdt = result.components.mul(equity_before, axis=0)            # per-bar PnL parts in USDT
    gross, funding = usdt["gross"].to_numpy(), usdt["funding"].to_numpy()
    fee_rate, slip_rate = result.params["fee_per_leg"], result.params["slippage_per_leg"]

    # Find (entry bar, exit bar, closed?) with one pass over the positions.
    segments = []
    entry = None
    for i in range(len(pos)):
        prev = pos[i - 1] if i > 0 else 0.0
        if entry is not None and pos[i] != prev:   # position changed: the open trade closes here
            segments.append((entry, i, True))
            entry = None
        if pos[i] != 0 and pos[i] != prev:          # new trade opens here
            entry = i
    if entry is not None:                           # still open at the end of the data
        segments.append((entry, len(pos) - 1, False))

    rows = []
    for i0, i1, closed in segments:
        ua, ub = units_a[i0], units_b[i0]           # the trade's units, unchanged until i1
        open_notional = abs(ua) * pa[i0] + abs(ub) * pb[i0]
        close_notional = abs(ua) * pa[i1] + abs(ub) * pb[i1] if closed else 0.0
        traded = open_notional + close_notional
        price_pnl = gross[i0 + 1:i1 + 1].sum()      # PnL starts one bar after entry (no lookahead)
        trade_funding = funding[i0 + 1:i1 + 1].sum()
        trade_fees, trade_slippage = -fee_rate * traded, -slip_rate * traded
        pnl = price_pnl + trade_funding + trade_fees + trade_slippage
        equity_at_entry = equity[i0] + (fee_rate + slip_rate) * open_notional  # before opening costs

        exit_reason = result.exit_reasons.iloc[i1] if result.exit_reasons is not None else None
        if not closed and pd.isna(exit_reason):     # pd.isna covers both None and NaN
            exit_reason = "end_of_data"
        rows.append({
            "entry_time": index[i0],
            "exit_time": index[i1],
            "direction": "long" if pos[i0] > 0 else "short",
            "holding_hours": (index[i1] - index[i0]) / pd.Timedelta(hours=1),
            "entry_z": result.z.iloc[i0] if result.z is not None else np.nan,
            "exit_z": result.z.iloc[i1] if result.z is not None else np.nan,
            "pnl_pct": pnl / equity_at_entry,
            "pnl_usdt": pnl,
            "fees": trade_fees,
            "slippage": trade_slippage,
            "funding": trade_funding,
            "exit_reason": exit_reason,
        })
    return pd.DataFrame(rows, columns=TRADE_COLUMNS)


# =============================================================================
# 3. METRICS
# =============================================================================

def bar_hours(index: pd.DatetimeIndex) -> float:
    """Length of one bar in hours (1 for hourly data, 4 for 4h bars, 24 for daily bars)."""
    return pd.Series(index).diff().median() / pd.Timedelta(hours=1)


def compute_metrics(result: Result) -> dict:
    """Summarise a backtest in a few numbers.

    Conventions: annualised with 24 * 365 hours per year (crypto never closes), i.e. 8760 bars per
    year for hourly bars, 2190 for 4h bars, 365 for daily bars; risk-free rate 0.
    Sharpe = mean(bar return) / std(bar return) * sqrt(bars per year).
    Max drawdown is negative (e.g. -0.12 = 12% below the running peak). Cost and funding totals
    are in USDT and signed: negative = money paid.

    Input:  Result (uses returns, components, equity, positions, trades, params).
    Output: dict of metrics.
    """
    returns, equity, trades = result.returns, result.equity, result.trades
    initial = result.params["initial_capital"]
    bars_per_year = HOURS_PER_YEAR / bar_hours(returns.index)
    years = len(returns) / bars_per_year

    total_return = equity.iloc[-1] / initial - 1
    ann_return = (1 + total_return) ** (1 / years) - 1 if total_return > -1 else -1.0
    ann_vol = returns.std() * np.sqrt(bars_per_year)
    sharpe = returns.mean() / returns.std() * np.sqrt(bars_per_year) if returns.std() > 0 else np.nan

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
        "trades_per_month": n_trades / (years * 12),
        "win_rate": (trades["pnl_pct"] > 0).mean() if n_trades else np.nan,
        "avg_trade_pnl": trades["pnl_pct"].mean() if n_trades else np.nan,
        "avg_holding_hours": trades["holding_hours"].mean() if n_trades else np.nan,
        "time_in_market": (result.positions != 0).mean(),
        "total_gross": usdt["gross"],
        "total_fees": usdt["fees"],
        "total_slippage": usdt["slippage"],
        "total_funding": usdt["funding"],
    }


# =============================================================================
# 4. PIPELINE AND PARAMETER SWEEPS
# =============================================================================

def run_strategy(
    prices: pd.DataFrame,
    funding: pd.DataFrame | None,
    pair: tuple[str, str],
    params: dict,
    period: str = "train",
) -> Result:
    """Run the full pipeline for one pair: hedge ratio -> spread -> z-score -> signals -> risk -> backtest.

    This is the main entry point of the notebooks (also called repeatedly by `sweep`).

    How the train/test split is respected:
        - Everything estimated once (static beta, half-life for the time stop) uses TRAIN data only.
        - Rolling quantities (rolling beta, z-score, rolling cointegration p-value) are computed on
          the full price history, because at every bar they only look backwards. This gives the
          test period a proper warm-up from the end of the train period without using future data.
        - Signals, risk rules and the backtest then run only inside the requested period, starting flat.

    Input:  prices (full history, one column per coin), hourly funding rates (same layout) or None,
            pair (A, B) = trade A against B, params from data.get_params, period "train" or "test".
    Output: Result, including spread, z, beta and the two price series for plotting.
    """
    if period not in ("train", "test"):
        raise ValueError("period must be 'train' or 'test'")
    if period == "test":
        print("=" * 78)
        print("WARNING: running on the TEST period. Use this only in notebook 04, once,")
        print("with parameters already fixed in config.yaml. Never tune parameters on it.")
        print("=" * 78)

    name_a, name_b = pair
    a, b = prices[name_a], prices[name_b]
    train_part, test_part = split_train_test(prices[[name_a, name_b]], params["train_end"])
    train_index = train_part.index
    index = train_index if period == "train" else test_part.index   # the bars we trade on
    train_a, train_b = train_part[name_a], train_part[name_b]

    # 1) hedge ratio and intercept: static (estimated on train only) or rolling (backward-looking)
    window = params["hedge_window"]
    if window is None:
        static_beta = strategy.hedge_ratio(train_a, train_b)
        alpha = strategy.hedge_intercept(train_a, train_b, static_beta)
        beta = pd.Series(static_beta, index=prices.index, name="beta")
    else:
        beta = strategy.hedge_ratio(a, b, window)
        alpha = strategy.hedge_intercept(a, b, beta, window)

    # 2) spread and z-score on the full history (both only use past data at every bar)
    spread = strategy.compute_spread(a, b, beta, alpha)
    z = strategy.zscore(spread, params["zscore_lookback"])

    # 3) half-life of the traded spread, estimated on train only (used by the time stop)
    half_life = strategy.half_life(spread.loc[train_index])

    # 4) optional rolling cointegration p-values for the entry filter
    coint_pvalues = None
    if params.get("use_coint_filter", False):
        coint_pvalues = strategy.rolling_coint_pvalues(a, b, params["coint_window"], params["coint_step"])

    # 5) restrict to the requested period, then signals -> risk rules -> positions
    z_period = z.loc[index]
    signals = strategy.generate_signals(z_period, params["entry_z"], params["exit_z"])
    positions, exit_reasons = strategy.apply_risk(signals, z_period, half_life, params, coint_pvalues)

    # 6) backtest with costs and (optionally) funding
    use_funding = params.get("use_funding", True) and funding is not None
    result = run_backtest(
        a.loc[index], b.loc[index], beta.loc[index], positions,
        fee_per_leg=params["fee_per_leg"],
        slippage_per_leg=params["slippage_per_leg"],
        funding_a=funding[name_a].loc[index] if use_funding else None,
        funding_b=funding[name_b].loc[index] if use_funding else None,
        initial_capital=params["initial_capital"],
        exit_reasons=exit_reasons,
        z=z_period,
        params=dict(params, pair=tuple(pair), period=period, half_life=half_life),
    )
    result.spread = spread.loc[index]
    result.beta = beta.loc[index]
    return result


SWEEP_METRICS = ["sharpe", "total_return", "annual_return", "max_drawdown", "calmar",
                 "n_trades", "trades_per_month", "win_rate", "avg_holding_hours", "time_in_market"]


def sweep(
    prices: pd.DataFrame,
    funding: pd.DataFrame | None,
    pair: tuple[str, str],
    base_params: dict,
    grid: dict,
    period: str = "train",
) -> pd.DataFrame:
    """Run run_strategy for every combination of the parameter grid.

    Example: grid = {"entry_z": [1.5, 2.0], "zscore_lookback": [72, 168]} -> 4 runs. Parameters not
    in the grid keep their value from base_params.

    Input:  prices, funding, pair, base params (from data.get_params), grid as {name: list of values},
            period (always "train" for calibration).
    Output: DataFrame, one row per combination: the grid parameters followed by the main metrics.
    """
    names = list(grid)
    combos = list(product(*grid.values()))  # every combination of the grid values
    rows = []
    for values in tqdm(combos, desc="sweep"):
        params = dict(base_params, **dict(zip(names, values)))
        result = run_strategy(prices, funding, pair, params, period=period)
        rows.append({**dict(zip(names, values)), **{m: result.metrics[m] for m in SWEEP_METRICS}})
    return pd.DataFrame(rows)


def compare_metrics(results: dict) -> pd.DataFrame:
    """Put the metrics of several runs side by side (one column per run), e.g. train vs test.

    Input:  dict of label -> Result.
    Output: DataFrame, one row per metric, one column per run.
    """
    return pd.DataFrame({label: res.metrics for label, res in results.items()})
