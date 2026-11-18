"""Sanity tests for the backtest engine. Run with `python -m pytest` from the repo root."""

import numpy as np
import pandas as pd
import pytest

from src import backtest


def _series(values) -> pd.Series:
    return pd.Series(values, index=pd.date_range("2024-01-01", periods=len(values), freq="D", tz="UTC"), dtype=float)


FLAT = _series([100.0] * 10)
JUMP = _series([100.0] * 5 + [110.0] * 5)       # +10% on bar 5


def test_no_lookahead():
    """A position decided at bar t must only earn the return from bar t to t+1 onwards (positions shifted by one bar)."""
    decided_after_jump = _series([0] * 5 + [1] * 5)  # signal on the close of the jump bar
    _, equity = backtest.run_backtest(JUMP, FLAT, 1.0, decided_after_jump, 0.0, capital=1000)
    assert equity.iloc[-1] == pytest.approx(1000)

    decided_before_jump = _series([0] * 4 + [1] * 6)
    _, equity = backtest.run_backtest(JUMP, FLAT, 1.0, decided_before_jump, 0.0, capital=1000)
    assert equity.iloc[-1] == pytest.approx(1100)


def test_execution_lag_delays_fill():
    """With execution_lag=1 a signal on the bar before the jump is filled one bar later and misses it."""
    decided_before_jump = _series([0] * 4 + [1] * 6)
    _, equity = backtest.run_backtest(JUMP, FLAT, 1.0, decided_before_jump, 0.0, capital=1000, execution_lag=1)
    assert equity.iloc[-1] == pytest.approx(1000)


def test_zero_positions_zero_pnl():
    """If the strategy never trades, equity stays at initial capital and there are no trades."""
    trades, equity = backtest.run_backtest(JUMP, FLAT, 1.0, _series([0] * 10), 0.001, borrow_per_bar=0.001)
    assert (equity == 10000).all() and trades.empty


def test_fees_reduce_pnl():
    """The same positions with fee_per_leg > 0 must end with lower equity than with zero fees."""
    positions = _series([0, 1, 1, 0, -1, -1, 0, 1, 0, 0])
    _, free = backtest.run_backtest(JUMP, FLAT, 1.0, positions, 0.0)
    _, paid = backtest.run_backtest(JUMP, FLAT, 1.0, positions, 0.001)
    assert paid.iloc[-1] < free.iloc[-1]


def test_hand_computed_long_spread_trade():
    """Long a / short b for one bar while a rises 10%: 1000 gain - 3 borrow - 2 x 30 trading costs = 937.

    capital 10000, size 1, fee 10 bps + slippage 5 bps per leg -> 15 per leg -> 30 at entry and at exit;
    borrow 3 bps per bar on the 10000 short leg -> 3.
    """
    a, b = _series([100, 110, 121, 121]), _series([100, 100, 100, 100])
    trades, equity = backtest.run_backtest(
        a, b, 1.0, _series([1, 0, 0, 0]), 0.001, slippage=0.0005, borrow_per_bar=0.0003, capital=10000
    )
    assert equity.iloc[-1] == pytest.approx(10937)
    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["pnl"] == pytest.approx(937) and t["costs"] == pytest.approx(63)
    assert t["direction"] == 1 and t["holding_bars"] == 1
    assert t["entry_time"] == a.index[0] and t["exit_time"] == a.index[1]


def test_short_spread_pays_borrow_on_a_and_gains_when_a_falls():
    """Short a / long b for two bars while a falls 10% then 0%: pnl = 1000 - 2 bars of borrow, no fees."""
    a, b = _series([100, 90, 90, 90]), _series([50, 50, 50, 50])
    trades, equity = backtest.run_backtest(a, b, 1.0, _series([-1, -1, 0, 0]), 0.0, borrow_per_bar=0.001, capital=10000)
    assert equity.iloc[-1] == pytest.approx(10000 + 1000 - 2 * 10)
    assert trades.iloc[0]["direction"] == -1 and trades.iloc[0]["holding_bars"] == 2


def test_flip_is_two_trades_and_open_trade_marked_end():
    """A sign flip closes one trade and opens the next; a trade still open on the last bar has exit_reason end."""
    trades, _ = backtest.run_backtest(JUMP, FLAT, 1.0, _series([1, 1, -1, -1, -1, -1, -1, -1, -1, -1]), 0.001)
    assert trades["direction"].tolist() == [1, -1]
    assert trades["exit_reason"].tolist() == ["signal", "end"]


def test_exit_reason_follows_execution_lag():
    """The exit reason recorded on the signal bar is attached to the trade whose fill happens lag bars later."""
    positions = _series([1, 1, 0, 0, 0, 0])
    reasons = _series([0] * 6).astype(object).replace(0, "")
    reasons.iloc[2] = "stop_z"
    trades, _ = backtest.run_backtest(FLAT.iloc[:6], FLAT.iloc[:6], 1.0, positions, 0.0, execution_lag=1, exit_reasons=reasons)
    assert trades.iloc[0]["exit_reason"] == "stop_z"
    assert trades.iloc[0]["exit_time"] == positions.index[3]   # fill at close of bar 2 + 1


def test_compute_metrics_basic():
    """Metrics on a known equity path: total return, drawdown, win rate and profit factor."""
    equity = _series([100, 110, 99, 121])
    trades = pd.DataFrame({"pnl": [10.0, -5.0, 20.0], "return": [0.1, -0.05, 0.2], "holding_bars": [1, 2, 3]})
    m = backtest.compute_metrics(equity, trades, periods_per_year=365)
    assert m["total_return"] == pytest.approx(0.21)
    assert m["max_drawdown"] == pytest.approx((99 - 110) / 100)
    assert m["win_rate"] == pytest.approx(2 / 3) and m["profit_factor"] == pytest.approx(6.0)
    assert m["n_trades"] == 3 and np.isfinite(m["sharpe"]) and np.isfinite(m["sortino"])


def test_compute_metrics_compounding_for_buy_and_hold():
    """A price that doubles then halves: compounding drawdown is -50% and total return 0."""
    m = backtest.compute_metrics(_series([100, 200, 100]), pd.DataFrame(columns=backtest.TRADE_COLUMNS), 365, compounding=True)
    assert m["max_drawdown"] == pytest.approx(-0.5) and m["total_return"] == pytest.approx(0.0)
    assert m["n_trades"] == 0 and np.isnan(m["win_rate"])
