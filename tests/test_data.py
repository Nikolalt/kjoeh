"""Tests for resampling prices and funding to longer bars."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import run_backtest
from src.data import resample_data


def hourly_frame(n_days: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hourly prices for two coins and funding paid every 8h (00:00, 08:00, 16:00)."""
    idx = pd.date_range("2024-01-01", periods=24 * n_days, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    prices = pd.DataFrame({"A": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))),
                           "B": 50 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))}, index=idx)
    funding = pd.DataFrame(0.0, index=idx, columns=["A", "B"])
    funding.loc[idx.hour % 8 == 0, "A"] = 0.001
    return prices, funding


def test_resample_1h_changes_nothing():
    prices, funding = hourly_frame()
    p, f = resample_data(prices, funding, "1h")
    pd.testing.assert_frame_equal(p, prices, check_freq=False)
    pd.testing.assert_frame_equal(f, funding, check_freq=False)


def test_resample_daily_prices_and_funding_sum():
    prices, funding = hourly_frame()
    p, f = resample_data(prices, funding, "1D")
    # the daily close is the close of the 23:00 hourly bar of that day
    jan2 = pd.Timestamp("2024-01-02", tz="UTC")
    assert p.loc[jan2, "A"] == prices.loc[pd.Timestamp("2024-01-02 23:00", tz="UTC"), "A"]
    # label Jan 3 holds the payments in (Jan 2 00:00, Jan 3 00:00]: 08:00, 16:00 of Jan 2 and 00:00 of Jan 3
    assert f.loc[pd.Timestamp("2024-01-03", tz="UTC"), "A"] == pytest.approx(0.003)


def test_daily_funding_is_paid_by_the_day_it_is_held():
    """A position held only during the Jan 2 daily bar pays the 08:00 and 16:00 Jan 2 and 00:00 Jan 3 funding."""
    prices, funding = hourly_frame()
    prices[:] = 100.0                                   # flat prices: only funding moves equity
    p, f = resample_data(prices, funding, "1D")
    positions = pd.Series(0.0, index=p.index)
    positions.loc[pd.Timestamp("2024-01-01", tz="UTC")] = 1.0   # decided at close of Jan 1 -> held during Jan 2

    res = run_backtest(p["A"], p["B"], 1.0, positions, 0.0, 0.0, funding_a=f["A"], funding_b=f["B"])
    # w_a = 0.5 -> 5000 USDT long A pays 3 x 0.1% = 15 USDT, all in the Jan 2 bar
    funding_usdt = res.components["funding"] * res.equity.shift(1).fillna(10_000)
    assert funding_usdt.sum() == pytest.approx(-15.0)
    assert funding_usdt.loc[pd.Timestamp("2024-01-02", tz="UTC")] == pytest.approx(-15.0)
