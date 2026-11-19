"""Sanity tests for the backtest engine on small synthetic data. Run with `python -m pytest` from the repo root."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import run_backtest


def hourly(n: int, start: str = "2024-01-01 00:00") -> pd.DatetimeIndex:
    """n consecutive hourly UTC timestamps."""
    return pd.date_range(start, periods=n, freq="h", tz="UTC")


def flat_prices(index: pd.DatetimeIndex, value: float = 100.0) -> pd.Series:
    return pd.Series(value, index=index)


def test_zero_positions_flat_equity_no_trades():
    """If the strategy never trades, equity stays at initial capital and there are no trades."""
    idx = hourly(50)
    rng = np.random.default_rng(0)
    a = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 50))), index=idx)
    b = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 50))), index=idx)
    funding = pd.Series(0.001, index=idx)

    res = run_backtest(a, b, 1.0, pd.Series(0.0, index=idx), 0.0005, 0.0002,
                       funding_a=funding, funding_b=funding, initial_capital=10_000)

    assert np.allclose(res.equity, 10_000)
    assert res.trades.empty
    assert res.metrics["n_trades"] == 0


def test_no_lookahead():
    """A position decided at the close of bar t earns only the returns from bar t+1 on."""
    idx = hourly(6)
    a = pd.Series([100, 100, 100, 110, 110, 110], index=idx, dtype=float)  # +10% jump in bar 3
    b = flat_prices(idx)

    # Decided at the close of bar 3, i.e. AFTER the jump -> must not profit from it
    late = pd.Series([0, 0, 0, 1, 1, 1], index=idx, dtype=float)
    res_late = run_backtest(a, b, 1.0, late, 0.0, 0.0)
    assert np.allclose(res_late.equity, 10_000)

    # Decided at the close of bar 2, i.e. BEFORE the jump -> earns w_a * 10% = 0.5 * 10% = 5%
    early = pd.Series([0, 0, 1, 1, 1, 1], index=idx, dtype=float)
    res_early = run_backtest(a, b, 1.0, early, 0.0, 0.0)
    assert res_early.equity.iloc[2] == pytest.approx(10_000)
    assert res_early.equity.iloc[3] == pytest.approx(10_500)

    # Changing future prices must not change the past equity curve
    a_future = a.copy()
    a_future.iloc[5] = 200.0
    res_future = run_backtest(a_future, b, 1.0, early, 0.0, 0.0)
    pd.testing.assert_series_equal(res_future.equity.iloc[:5], res_early.equity.iloc[:5])


def test_hand_calculated_five_bars():
    """5-bar example, beta = 1, fee 0.1% per leg, no slippage. Units are fixed at entry.

    bar 0: open at the close: 50% of 10000 in each leg -> +50 units A at 100, -50 units B at 100
           traded notional 5000 + 5000 = 10000 -> fee 10                       -> equity  9990
    bar 1: A 100 -> 110, B 100 -> 105: PnL = 50*10 - 50*5 = +250              -> equity 10240
    bar 2: A 110 -> 99,  B 105 -> 105: PnL = 50*(-11) = -550                   ->         9690
           close at the close: traded 50*99 + 50*105 = 10200 -> fee 10.2      -> equity  9679.8
    bars 3-4: flat                                                             -> equity  9679.8
    """
    idx = hourly(5)
    a = pd.Series([100, 110, 99, 120, 80], index=idx, dtype=float)
    b = pd.Series([100, 105, 105, 90, 130], index=idx, dtype=float)
    positions = pd.Series([1, 1, 0, 0, 0], index=idx, dtype=float)

    res = run_backtest(a, b, 1.0, positions, fee_per_leg=0.001, slippage_per_leg=0.0, initial_capital=10_000)

    assert np.allclose(res.equity.to_numpy(), [9990.0, 10240.0, 9679.8, 9679.8, 9679.8])
    assert res.units["units_a"].tolist() == [50, 50, 0, 0, 0]
    assert res.units["units_b"].tolist() == [-50, -50, 0, 0, 0]
    # the weights drift with prices: after bar 1, A is worth 50 * 110 = 5500 of 10240 equity
    assert res.weights["w_a"].iloc[1] == pytest.approx(5500 / 10240)

    trade = res.trades.iloc[0]
    assert len(res.trades) == 1
    assert trade["pnl_usdt"] == pytest.approx(-320.2)
    assert trade["fees"] == pytest.approx(-20.2)
    assert trade["pnl_pct"] == pytest.approx(-320.2 / 10_000)


def test_fees_and_slippage_reduce_equity_and_round_trip_cost():
    """Costs lower the equity, and one round trip at constant prices costs exactly 2 x 7 bps of the notional."""
    idx = hourly(10)
    a, b = flat_prices(idx), flat_prices(idx)
    positions = pd.Series([0, 1, 1, 1, 0, 0, 0, 0, 0, 0], index=idx, dtype=float)

    free = run_backtest(a, b, 1.0, positions, 0.0, 0.0)
    costly = run_backtest(a, b, 1.0, positions, 0.0005, 0.0002)
    assert costly.equity.iloc[-1] < free.equity.iloc[-1]

    # Entry: trade 10000 of notional at 7 bps = 7 USDT. Exit: the same units at the same prices = 7 USDT.
    assert costly.equity.iloc[-1] == pytest.approx(10_000 - 7 - 7)
    assert costly.metrics["total_fees"] == pytest.approx(-10.0)
    assert costly.metrics["total_slippage"] == pytest.approx(-4.0)


def test_held_position_pays_no_costs_between_entry_and_exit():
    """While the position is unchanged, no costs are charged, even if prices and the rolling beta move."""
    idx = hourly(8)
    rng = np.random.default_rng(3)
    a = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 8))), index=idx)
    b = pd.Series(50 * np.exp(np.cumsum(rng.normal(0, 0.02, 8))), index=idx)
    beta = pd.Series([1.0, 1.0, 1.3, 0.7, 1.5, 0.9, 1.1, 1.2], index=idx)   # moving hedge ratio
    positions = pd.Series([0, 1, 1, 1, 1, 1, 0, 0], index=idx, dtype=float)

    res = run_backtest(a, b, beta, positions, 0.0005, 0.0002)
    costs = res.components["fees"] + res.components["slippage"]

    assert (costs.iloc[2:6] == 0).all()                  # bars 2-5: holding, no trading
    assert costs.iloc[1] < 0 and costs.iloc[6] < 0       # entry and exit cost money
    # units are frozen at entry with beta = 1.0 (the beta of the entry bar)
    assert res.units["units_a"].iloc[1:6].nunique() == 1
    assert res.units["units_b"].iloc[1] == pytest.approx(-res.units["units_a"].iloc[1] * a.iloc[1] / b.iloc[1])


def test_funding_sign_long_pays_short_receives():
    """With a positive funding rate on leg A, long A pays it and short A receives it."""
    idx = hourly(10)
    a, b = flat_prices(idx), flat_prices(idx)
    funding_a = pd.Series(0.0, index=idx)
    funding_a.iloc[5] = 0.01  # one funding event of 1%

    long_spread = pd.Series(1.0, index=idx)     # long A (w_a = +0.5)
    short_spread = pd.Series(-1.0, index=idx)   # short A (w_a = -0.5)

    res_long = run_backtest(a, b, 1.0, long_spread, 0.0, 0.0, funding_a=funding_a)
    res_short = run_backtest(a, b, 1.0, short_spread, 0.0, 0.0, funding_a=funding_a)

    assert res_long.components["funding"].sum() == pytest.approx(-0.005)   # pays 0.5 * 1%
    assert res_short.components["funding"].sum() == pytest.approx(+0.005)  # receives 0.5 * 1%
    assert res_long.equity.iloc[-1] < 10_000 < res_short.equity.iloc[-1]


def test_funding_timing():
    """Funding at 08:00 is paid by the position held during bar 07:00, not by one opened at its close."""
    idx = hourly(8, start="2024-01-01 04:00")   # bars 04:00 ... 11:00
    a, b = flat_prices(idx), flat_prices(idx)
    funding_a = pd.Series(0.0, index=idx)
    funding_a.loc[pd.Timestamp("2024-01-01 08:00", tz="UTC")] = 0.01

    # Decided at the close of bar 06:00 -> held during bar 07:00 only -> pays the 08:00 funding
    held_during_07 = pd.Series(0.0, index=idx)
    held_during_07.loc[pd.Timestamp("2024-01-01 06:00", tz="UTC")] = 1.0
    res = run_backtest(a, b, 1.0, held_during_07, 0.0, 0.0, funding_a=funding_a)
    assert res.components["funding"].sum() == pytest.approx(-0.005)
    assert res.components["funding"].loc[pd.Timestamp("2024-01-01 07:00", tz="UTC")] == pytest.approx(-0.005)

    # Opened at the close of bar 07:00 (= 08:00) and held afterwards -> does not pay it
    opened_at_08 = pd.Series(0.0, index=idx)
    opened_at_08.loc[pd.Timestamp("2024-01-01 07:00", tz="UTC"):] = 1.0
    res = run_backtest(a, b, 1.0, opened_at_08, 0.0, 0.0, funding_a=funding_a)
    assert res.components["funding"].sum() == pytest.approx(0.0)


def test_extract_trades_count_direction_holding():
    """Round trips are cut correctly, including a direct sign flip and a trade open at the end."""
    idx = hourly(12)
    a, b = flat_prices(idx), flat_prices(idx)
    #              0  1  2  3  4  5   6   7  8  9  10 11
    positions = [0, 1, 1, 1, 0, -1, -1, 1, 1, 0, 0, 1]
    positions = pd.Series(positions, index=idx, dtype=float)
    reasons = pd.Series(None, index=idx, dtype=object)
    reasons.iloc[[4, 7, 9]] = ["signal", "z_stop", "time_stop"]

    res = run_backtest(a, b, 1.0, positions, 0.0005, 0.0, exit_reasons=reasons)
    trades = res.trades

    assert len(trades) == 4
    assert trades["direction"].tolist() == ["long", "short", "long", "long"]
    assert trades["holding_hours"].tolist() == [3.0, 2.0, 2.0, 0.0]
    assert trades["exit_reason"].tolist() == ["signal", "z_stop", "time_stop", "end_of_data"]
    # Flat prices: each trade only loses its costs, and the trade PnLs compound to the final equity
    assert (trades["pnl_pct"] < 0).all()
    # trade PnLs add up to the total PnL, and compound to the final equity
    assert trades["pnl_usdt"].sum() == pytest.approx(res.equity.iloc[-1] - 10_000)
    assert np.prod(1 + trades["pnl_pct"]) * 10_000 == pytest.approx(res.equity.iloc[-1])


def test_test_period_warning_only_once(capsys, monkeypatch):
    """The test-period warning is printed the first time only."""
    from src import backtest
    monkeypatch.setattr(backtest, "_test_warning_shown", False)
    backtest._warn_test_period()
    backtest._warn_test_period()
    assert capsys.readouterr().out.count("WARNING: running on the TEST period") == 1
