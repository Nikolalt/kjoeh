"""Backtest engine: turns positions into PnL, trades and an equity curve, then computes metrics."""

import copy
import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import data, strategy


@dataclass
class Result:
    """Everything one backtest run produces.

    positions: final position per bar (signed fraction of capital); for a portfolio: gross exposure per bar
    trades:    one row per round trip (entry_time, exit_time, direction, pnl, holding_bars, exit_reason)
    equity:    equity curve in USDT, one value per bar
    metrics:   dict from compute_metrics
    daily:     optional per-bar detail (features, positions, exposure), used for plots
    """

    positions: pd.Series
    trades: pd.DataFrame
    equity: pd.Series
    metrics: dict
    daily: pd.DataFrame | None = None


TRADE_COLUMNS = [
    "entry_time", "exit_time", "direction", "size", "notional", "pnl", "return", "costs", "holding_bars", "exit_reason",
]


def run_backtest(
    price_a: pd.Series,
    price_b: pd.Series,
    beta: pd.Series,
    positions: pd.Series,
    fee_per_leg: float,
    *,
    slippage: float = 0.0,
    borrow_per_bar: float = 0.0,
    capital: float = 10000.0,
    execution_lag: int = 0,
    exit_reasons: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Simulate the two-leg trade: long/short a against beta units of b, paying fees on each leg.

    Positions are shifted by one bar, so a signal computed at the close of bar t is traded at bar t+1 (no lookahead).
    With execution_lag = k the fill is k bars later still: signal at close t, fill at close t+k, first return
    earned from close t+k to close t+k+1.

    Accounting (no compounding: notionals are fractions of the initial capital):
        exposure_t  = positions shifted by 1 + execution_lag (what is held during bar t)
        pnl_t       = exposure_t * capital * (ret_a_t - beta * ret_b_t)       simple returns
        trading     = (fee_per_leg + slippage) on the notional of each leg traded, booked on the first bar of the
                      new exposure (entry) and on the bar after the last one (exit); a flip pays both
        borrow      = borrow_per_bar on the notional of the short leg, every bar the position is held
    Trades still open on the last bar are closed at the last price with exit_reason "end" and no exit cost.

    Input:  prices of both legs, hedge ratio (Series or float), target positions, fee per leg as a fraction;
            optional slippage per leg, borrow cost per bar, capital, execution lag in bars and the exit
            reasons from strategy.apply_risk (aligned with positions).
    Output: (trades DataFrame, equity Series).
    """
    index = positions.index
    lag = 1 + execution_lag
    beta = beta.reindex(index) if isinstance(beta, pd.Series) else pd.Series(float(beta), index=index)
    exposure = positions.shift(lag).fillna(0.0)
    beta_held = beta.shift(lag).fillna(0.0)
    ret_a = price_a.reindex(index).pct_change().fillna(0.0)
    ret_b = price_b.reindex(index).pct_change().fillna(0.0)

    w, prev = exposure.to_numpy(), exposure.shift(1, fill_value=0.0).to_numpy()
    b, b_prev = beta_held.abs().to_numpy(), beta_held.abs().shift(1, fill_value=0.0).to_numpy()
    changed = w != prev
    rate = (fee_per_leg + slippage) * capital
    entry_cost = np.where(changed, np.abs(w) * (1 + b), 0.0) * rate
    exit_cost = np.where(changed, np.abs(prev) * (1 + b_prev), 0.0) * rate
    short_notional = np.where(w > 0, w * b, -w)
    carry = (w * capital * (ret_a - beta_held * ret_b)).to_numpy() - borrow_per_bar * capital * short_notional
    pnl = carry - entry_cost - exit_cost
    equity = pd.Series(capital + np.cumsum(pnl), index=index)

    reasons = (exit_reasons.reindex(index).shift(execution_lag) if exit_reasons is not None
               else pd.Series("", index=index)).fillna("").to_numpy()
    trades, start = [], None
    for t in range(len(w) + 1):
        current = w[t] if t < len(w) else 0.0
        previous = w[t - 1] if t > 0 else 0.0
        if t < len(w) and current == previous:
            continue
        if previous != 0:                                   # close the run start..t-1
            last, is_end = t - 1, t == len(w)
            exit_part = 0.0 if is_end else exit_cost[t]
            costs = entry_cost[start] + exit_part + borrow_per_bar * capital * short_notional[start:t].sum()
            notional = abs(previous) * capital
            trade_pnl = carry[start:t].sum() - entry_cost[start] - exit_part
            trades.append({
                "entry_time": index[max(start - 1, 0)],
                "exit_time": index[last],
                "direction": int(np.sign(previous)),
                "size": abs(previous),
                "notional": notional,
                "pnl": trade_pnl,
                "return": trade_pnl / notional,
                "costs": costs,
                "holding_bars": last - start + 1,
                "exit_reason": "end" if is_end else (reasons[last] or "signal"),
            })
        if t < len(w) and current != 0:
            start = t
    return pd.DataFrame(trades, columns=TRADE_COLUMNS), equity


def compute_metrics(
    equity: pd.Series, trades: pd.DataFrame, periods_per_year: int = 24 * 365, compounding: bool = False
) -> dict:
    """Summarise a backtest in a few numbers.

    compounding=False (the backtests here: notionals are fractions of the initial capital): returns are
    pnl_t / equity_0, the annual return is the simple average, and drawdown is the fall from the running
    peak as a fraction of the initial capital.
    compounding=True (buy-and-hold benchmarks): returns are equity_t / equity_t-1 - 1, the annual return
    is the CAGR, and drawdown is equity / running peak - 1.

    Input:  equity Series, trades DataFrame, bars per year (24*365 for hourly, 365 for daily crypto),
            compounding flag.
    Output: dict with total_return, ann_return, ann_vol, sharpe, sortino (risk-free rate 0),
            max_drawdown, calmar, n_trades, win_rate, profit_factor, avg_trade_pnl, avg_trade_return,
            avg_holding_bars.
    """
    capital = equity.iloc[0]
    total = equity.iloc[-1] / capital - 1
    if compounding:
        returns = equity.pct_change().dropna()
        years = len(returns) / periods_per_year
        ann_return = (1 + total) ** (1 / years) - 1
        max_dd = (equity / equity.cummax() - 1).min()
    else:
        returns = equity.diff().dropna() / capital
        years = len(returns) / periods_per_year
        ann_return = total / years
        max_dd = ((equity - equity.cummax()) / capital).min()
    std = returns.std()
    downside = np.sqrt((returns.clip(upper=0) ** 2).mean())
    pnl = trades["pnl"] if len(trades) else pd.Series(dtype=float)
    losses = -pnl[pnl < 0].sum()
    return {
        "total_return": total,
        "ann_return": ann_return,
        "ann_vol": std * np.sqrt(periods_per_year),
        "sharpe": returns.mean() / std * np.sqrt(periods_per_year) if std > 0 else np.nan,
        "sortino": returns.mean() / downside * np.sqrt(periods_per_year) if downside > 0 else np.nan,
        "max_drawdown": max_dd,
        "calmar": ann_return / -max_dd if max_dd < 0 else np.nan,
        "n_trades": len(trades),
        "win_rate": (pnl > 0).mean() if len(pnl) else np.nan,
        "profit_factor": pnl[pnl > 0].sum() / losses if losses > 0 else np.nan,
        "avg_trade_pnl": pnl.mean() if len(pnl) else np.nan,
        "avg_trade_return": trades["return"].mean() if len(trades) else np.nan,
        "avg_holding_bars": trades["holding_bars"].mean() if len(trades) else np.nan,
    }


def run_strategy(cfg: dict, prices: pd.DataFrame) -> Result:
    """Run the full pipeline for the pair and parameters in cfg: signals -> risk -> backtest -> metrics.

    This is the main entry point the notebooks call, also repeatedly with modified cfg for parameter sweeps.

    Input:  config dict, price DataFrame (train or test period).
    Output: Result.
    """
    # TODO: hedge_ratio -> compute_spread -> zscore -> generate_signals -> apply_risk
    #       -> run_backtest -> compute_metrics
    raise NotImplementedError


# =============================================================================
# RATIO VS BTC PIPELINE (config_ratio.yaml, notebook 05)
#    calibrate on train -> one backtest per alt/BTC pair -> sum into a portfolio
# =============================================================================

DAYS_PER_YEAR = 365  # crypto trades every day


def period_bounds(cfg: dict, period: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Start (inclusive) and end (exclusive) of a named period.

    Input:  config dict, period "train" [start, train_end), "test" [train_end, end) or "full" [start, end).
    Output: (start, end) as UTC Timestamps.
    """
    s = cfg["setup"]
    start, end = {"train": (s["start"], s["train_end"]), "test": (s["train_end"], s["end"]),
                  "full": (s["start"], s["end"])}[period]
    return pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")


def _params(cfg: dict, params: dict | None) -> dict:
    """Config parameters with optional overrides (used by sweeps)."""
    return {**cfg["parameters"], **(params or {})}


def calibrate_pairs(cfg: dict, prices: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """Everything estimated from history before train_end only: diagnostics, half-life and size per pair.

    Input:  config dict, full price DataFrame (all dates; only rows before train_end are used),
            optional parameter overrides.
    Output: strategy.pair_diagnostics table plus a column size (per-leg fraction of capital).
    """
    p = _params(cfg, params)
    train, _ = data.split_train_test(prices, cfg["setup"]["train_end"])
    diag = strategy.pair_diagnostics(
        train, p["zscore_lookback"], p["zscore_method"], data.coin_name(cfg["setup"]["numeraire"])
    )
    diag["size"] = strategy.equal_risk_sizes(
        diag["ratio_vol"], p["max_gross_leverage"], equal_capital=p["sizing"] == "equal_capital"
    )
    return diag


def run_ratio_pair(
    cfg: dict,
    prices: pd.DataFrame,
    alt: str,
    period: str = "test",
    params: dict | None = None,
    calibration: pd.DataFrame | None = None,
) -> Result:
    """Backtest one alt/BTC pair: features -> signals -> risk -> dollar-neutral two-leg backtest.

    Rolling statistics are computed on the full history, so the first days of a period use the
    preceding (past) days as warm-up; the position always starts flat at the start of the period.

    Input:  config dict, full price DataFrame, alt column name (e.g. "ETH"), period name, optional
            parameter overrides, optional calibration table from calibrate_pairs (computed if None).
    Output: Result; trades have extra columns pair, signal_time, entry_ratio, exit_ratio, entry_z, exit_z,
            fair_ratio_at_entry; daily holds the features, position, exit_reason, exposure and equity.
    """
    p, s = _params(cfg, params), cfg["setup"]
    calib = calibration if calibration is not None else calibrate_pairs(cfg, prices, params)
    btc = data.coin_name(s["numeraire"])
    start, end = period_bounds(cfg, period)

    features = strategy.ratio_features(prices[alt], prices[btc], p["zscore_lookback"], p["zscore_method"])
    f = features[(features.index >= start) & (features.index < end)]
    signals = strategy.generate_signals(f["z"], p["entry_z"], p["exit_z"])
    risk_params = {**p, "half_life": calib.loc[alt, "half_life_dev"], "position_size": calib.loc[alt, "size"]}
    risk = strategy.apply_risk(signals, f["z"], risk_params, return_reasons=True, spread=f["log_ratio"])

    capital, lag = s["initial_capital"], s["execution_lag"]
    trades, equity = run_backtest(
        prices.loc[f.index, alt], prices.loc[f.index, btc], 1.0, risk["position"],
        s["fee_bps_per_leg"] / 1e4,
        slippage=s["slippage_bps_per_leg"] / 1e4,
        borrow_per_bar=s["borrow_bps_per_day"] / 1e4,
        capital=capital,
        execution_lag=lag,
        exit_reasons=risk["exit_reason"],
    )

    pos_of = f.index.get_indexer
    signal_idx = np.maximum(pos_of(trades["entry_time"]) - lag, 0)
    trades.insert(0, "pair", f"{alt}/{btc}")
    trades.insert(1, "signal_time", f.index[signal_idx])
    trades["entry_ratio"] = f["ratio"].reindex(trades["entry_time"]).to_numpy()
    trades["exit_ratio"] = f["ratio"].reindex(trades["exit_time"]).to_numpy()
    trades["entry_z"] = f["z"].iloc[signal_idx].to_numpy()
    trades["exit_z"] = f["z"].reindex(trades["exit_time"]).to_numpy()
    trades["fair_ratio_at_entry"] = np.exp(f["mean"].iloc[signal_idx].to_numpy())

    daily = f.join(risk).assign(exposure=risk["position"].shift(1 + lag).fillna(0.0), equity=equity)
    return Result(risk["position"], trades, equity, compute_metrics(equity, trades, DAYS_PER_YEAR), daily)


def run_ratio_portfolio(
    cfg: dict, prices: pd.DataFrame, period: str = "test", params: dict | None = None
) -> tuple[Result, dict[str, Result]]:
    """Run every alt/BTC pair and add them up into one portfolio on one capital base.

    Input:  config dict, full price DataFrame, period name, optional parameter overrides.
    Output: (portfolio Result, dict alt -> pair Result). The portfolio's positions are the gross exposure
            (sum of |alt leg| + |BTC leg| as a fraction of capital), its daily holds per-pair exposure.
    """
    calib = calibrate_pairs(cfg, prices, params)
    pairs = {alt: run_ratio_pair(cfg, prices, alt, period, params, calib) for alt in calib.index}
    capital = cfg["setup"]["initial_capital"]
    equity = capital + sum(r.equity - capital for r in pairs.values())
    exposure = pd.DataFrame({alt: r.daily["exposure"] for alt, r in pairs.items()})
    gross = 2 * exposure.abs().sum(axis=1)
    trades = pd.concat([r.trades for r in pairs.values()], ignore_index=True).sort_values("entry_time")
    trades = trades.reset_index(drop=True)
    metrics = compute_metrics(equity, trades, DAYS_PER_YEAR)
    metrics |= {"avg_gross_exposure": gross.mean(), "max_gross_exposure": gross.max()}
    return Result(gross, trades, equity, metrics, exposure), pairs


def benchmark_equity(cfg: dict, prices: pd.DataFrame, period: str = "test") -> pd.DataFrame:
    """Buy-and-hold benchmarks on the same capital and dates as the strategy.

    Input:  config dict, full price DataFrame, period name.
    Output: DataFrame with columns buy_hold_btc and equal_weight_alts (equal dollars in each alt at the
            start, then held without rebalancing), in USD.
    """
    start, end = period_bounds(cfg, period)
    px = prices[(prices.index >= start) & (prices.index < end)]
    btc = data.coin_name(cfg["setup"]["numeraire"])
    capital = cfg["setup"]["initial_capital"]
    alts = px.drop(columns=btc)
    return pd.DataFrame({
        "buy_hold_btc": capital * px[btc] / px[btc].iloc[0],
        "equal_weight_alts": capital * (alts / alts.iloc[0]).mean(axis=1),
    })


def benchmark_metrics(cfg: dict, prices: pd.DataFrame, period: str = "test") -> pd.DataFrame:
    """Metrics of the buy-and-hold benchmarks (compounding returns) for the comparison table.

    Input:  config dict, full price DataFrame, period name.
    Output: DataFrame indexed by benchmark name with the compute_metrics columns (trade columns NaN).
    """
    no_trades = pd.DataFrame(columns=TRADE_COLUMNS)
    bench = benchmark_equity(cfg, prices, period)
    return pd.DataFrame({
        name: compute_metrics(equity, no_trades, DAYS_PER_YEAR, compounding=True) for name, equity in bench.items()
    }).T


def parameter_sweep(
    cfg: dict, prices: pd.DataFrame, grid: dict[str, list], period: str = "train"
) -> pd.DataFrame:
    """Run the portfolio for every combination of the grid (use the train period only for decisions).

    Input:  config dict, full price DataFrame, grid dict parameter name -> list of values, period name.
    Output: long-format DataFrame, one row per combination: the grid parameters plus all metrics.
    """
    rows = []
    for values in itertools.product(*grid.values()):
        params = dict(zip(grid.keys(), values))
        portfolio, _ = run_ratio_portfolio(cfg, prices, period, params)
        rows.append({**params, **portfolio.metrics})
    return pd.DataFrame(rows)


def cost_sensitivity(
    cfg: dict, prices: pd.DataFrame, multipliers: list[float], period: str = "test"
) -> pd.DataFrame:
    """Re-run the portfolio with fee, slippage and borrow cost all scaled by each multiplier.

    Input:  config dict, full price DataFrame, list of cost multipliers (e.g. [0, 1, 2, 3]), period name.
    Output: DataFrame indexed by multiplier with all portfolio metrics.
    """
    rows = {}
    for m in multipliers:
        scaled = copy.deepcopy(cfg)
        for key in ("fee_bps_per_leg", "slippage_bps_per_leg", "borrow_bps_per_day"):
            scaled["setup"][key] = cfg["setup"][key] * m
        rows[m] = run_ratio_portfolio(scaled, prices, period)[0].metrics
    return pd.DataFrame(rows).T.rename_axis("cost_multiplier")


def trade_frequency(trades: pd.DataFrame, cfg: dict, period: str = "test") -> dict:
    """How often the strategy trades and how long it holds.

    Input:  trades DataFrame (with a pair column), config dict, period name.
    Output: dict with n_trades, trades_per_month, trades_per_pair_per_year, avg_holding_days,
            median_holding_days.
    """
    start, end = period_bounds(cfg, period)
    months = (end - start).days / 30.44
    n_pairs = trades["pair"].nunique() if len(trades) else 1
    return {
        "n_trades": len(trades),
        "trades_per_month": len(trades) / months,
        "trades_per_pair_per_year": len(trades) / n_pairs / (months / 12),
        "avg_holding_days": trades["holding_bars"].mean(),
        "median_holding_days": trades["holding_bars"].median(),
    }


def performance_table(portfolios: dict[str, Result], cfg: dict, prices: pd.DataFrame) -> pd.DataFrame:
    """Strategy and buy-and-hold benchmarks side by side, for every period that was run.

    Input:  dict period name -> portfolio Result (e.g. {"train": ..., "test": ...}), config dict,
            full price DataFrame.
    Output: DataFrame with a (period, series) row index and the compute_metrics columns.
    """
    tables = {}
    for period, result in portfolios.items():
        strategy_row = pd.DataFrame({"strategy": result.metrics}).T
        tables[period] = pd.concat([strategy_row, benchmark_metrics(cfg, prices, period)])
    return pd.concat(tables).drop(columns=["avg_gross_exposure", "max_gross_exposure"], errors="ignore")


def exit_reason_summary(trades: pd.DataFrame) -> pd.DataFrame:
    """How trades end: count, share, win rate, total and average PnL and holding time per exit reason.

    Input:  trades DataFrame.
    Output: DataFrame indexed by exit_reason, sorted by count.
    """
    summary = trades.groupby("exit_reason").agg(
        n_trades=("pnl", "size"),
        win_rate=("pnl", lambda p: (p > 0).mean()),
        total_pnl=("pnl", "sum"),
        avg_pnl=("pnl", "mean"),
        avg_holding_days=("holding_bars", "mean"),
    )
    summary.insert(1, "share", summary["n_trades"] / len(trades))
    return summary.sort_values("n_trades", ascending=False)


def pick_example_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """Three trades worth walking through: a typical reversion win, the best trade and the worst trade.

    The typical win is the mean-reversion exit whose PnL is closest to the median of those exits.

    Input:  trades DataFrame (closed trades with exit reasons).
    Output: DataFrame of up to three trade rows with an extra column example.
    """
    closed = trades[trades["exit_reason"] != "end"]
    mean_exits = closed[closed["exit_reason"] == "mean"]
    picks = {}
    if len(mean_exits):
        picks["typical reversion"] = (mean_exits["pnl"] - mean_exits["pnl"].median()).abs().idxmin()
    picks["best"] = closed["pnl"].idxmax()
    picks["worst"] = closed["pnl"].idxmin()
    rows = [trades.loc[[i]].assign(example=label) for label, i in picks.items()]
    return pd.concat(rows)


def pnl_by_pair(trades: pd.DataFrame) -> pd.Series:
    """Total PnL per pair over all trades, largest first.

    Input:  trades DataFrame with a pair column.
    Output: Series pair -> total PnL.
    """
    return trades.groupby("pair")["pnl"].sum().sort_values(ascending=False)
