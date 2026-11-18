"""Backtest engine: turns positions into PnL, trades and an equity curve, then computes metrics."""

from dataclasses import dataclass

import pandas as pd

from . import strategy  # noqa: F401  (used by run_strategy once implemented)


@dataclass
class Result:
    """Everything one backtest run produces.

    positions: final position per bar (signed fraction of capital)
    trades:    one row per round trip (entry_time, exit_time, direction, pnl, holding_bars, exit_reason)
    equity:    equity curve in USDT, one value per bar
    metrics:   dict from compute_metrics
    """

    positions: pd.Series
    trades: pd.DataFrame
    equity: pd.Series
    metrics: dict


def run_backtest(
    price_a: pd.Series,
    price_b: pd.Series,
    beta: pd.Series,
    positions: pd.Series,
    fee_per_leg: float,
) -> tuple[pd.DataFrame, pd.Series]:
    """Simulate the two-leg trade: long/short a against beta units of b, paying fees on each leg.

    Positions are shifted by one bar, so a signal computed at the close of bar t is traded at bar t+1 (no lookahead).

    Input:  prices of both legs, hedge ratio, target positions, fee per leg as a fraction.
    Output: (trades DataFrame, equity Series).
    """
    # TODO: shift positions, leg returns, fees on position changes, collect round trips
    raise NotImplementedError


def compute_metrics(equity: pd.Series, trades: pd.DataFrame) -> dict:
    """Summarise a backtest in a few numbers.

    Input:  equity Series, trades DataFrame.
    Output: dict with total_return, sharpe (annualised for hourly bars), max_drawdown, win_rate, n_trades.
    """
    raise NotImplementedError


def run_strategy(cfg: dict, prices: pd.DataFrame) -> Result:
    """Run the full pipeline for the pair and parameters in cfg: signals -> risk -> backtest -> metrics.

    This is the main entry point the notebooks call, also repeatedly with modified cfg for parameter sweeps.

    Input:  config dict, price DataFrame (train or test period).
    Output: Result.
    """
    # TODO: hedge_ratio -> compute_spread -> zscore -> generate_signals -> apply_risk
    #       -> run_backtest -> compute_metrics
    raise NotImplementedError
