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
    """5-bar example, beta = 1 (w_a = +0.5, w_b = -0.5 when long), fee 0.1% per leg, no slippage.

    bar 0: open at the close           cost = 0.001 * (0.5 + 0.5) = 0.001  -> 10000 * 0.999 = 9990
    bar 1: r_a = +10%, r_b = +5%       gross = 0.5*0.10 - 0.5*0.05 = 0.025 -> 9990 * 1.025 = 10239.75
    bar 2: r_a = -10%, r_b = 0%, close gross = -0.05, cost = 0.001          -> 10239.75 * 0.949 = 9717.52275
    bars 3-4: flat                                                          -> 9717.52275
    """
    idx = hourly(5)
    a = pd.Series([100, 110, 99, 120, 80], index=idx, dtype=float)
    b = pd.Series([100, 105, 105, 90, 130], index=idx, dtype=float)
    positions = pd.Series([1, 1, 0, 0, 0], index=idx, dtype=float)

    res = run_backtest(a, b, 1.0, positions, fee_per_leg=0.001, slippage_per_leg=0.0, initial_capital=10_000)

    expected = [9990.0, 10239.75, 9717.52275, 9717.52275, 9717.52275]
    assert np.allclose(res.equity.to_numpy(), expected)
    assert len(res.trades) == 1
    assert res.trades["pnl_pct"].iloc[0] == pytest.approx(9717.52275 / 10_000 - 1)


def test_fees_and_slippage_reduce_equity_and_round_trip_cost():
    """Costs lower the equity, and one round trip at constant prices costs exactly 2 x (fee + slippage) x |p|."""
    idx = hourly(10)
    a, b = flat_prices(idx), flat_prices(idx)
    positions = pd.Series([0, 1, 1, 1, 0, 0, 0, 0, 0, 0], index=idx, dtype=float)

    free = run_backtest(a, b, 1.0, positions, 0.0, 0.0)
    costly = run_backtest(a, b, 1.0, positions, 0.0005, 0.0002)
    assert costly.equity.iloc[-1] < free.equity.iloc[-1]

    # Entry: trade |w_a| + |w_b| = 1 of equity at 7 bps -> x 0.9993. Exit: same again.
    assert costly.equity.iloc[-1] == pytest.approx(10_000 * (1 - 0.0007) ** 2)
    assert costly.metrics["total_fees"] + costly.metrics["total_slippage"] == pytest.approx(
        costly.equity.iloc[-1] - 10_000)

    # Hedge-ratio rebalancing while in a trade also costs money
    beta = pd.Series([1, 1, 1, 2, 2, 2, 2, 2, 2, 2], index=idx, dtype=float)
    rebalanced = run_backtest(a, b, beta, positions, 0.0005, 0.0002)
    assert rebalanced.equity.iloc[-1] < costly.equity.iloc[-1]


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
    assert np.prod(1 + trades["pnl_pct"]) * 10_000 == pytest.approx(res.equity.iloc[-1])
