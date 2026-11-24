"""Small sanity checks of the trading rules and the backtest. Run with `python -m pytest`."""

import numpy as np
import pandas as pd
import pytest

from src.strategy import half_life, positions

CFG = {"entry_z": 2.0, "exit_z": 0.0, "stop_z": 3.5, "time_stop_hours": 100}


def z_series(values):
    return pd.Series(values, index=pd.date_range("2024-01-01", periods=len(values), freq="h", tz="UTC"), dtype=float)


def test_entry_and_take_profit():
    # z falls below -2 -> long the spread, z back to 0 -> close
    pos, reason = positions(z_series([0, -1, -2.5, -1, 0.1, 0]), CFG)
    assert list(pos) == [0, 0, 1, 1, 0, 0]
    assert reason.iloc[4] == "take_profit"


def test_short_entry():
    pos, _ = positions(z_series([0, 2.5, 1, -0.1]), CFG)
    assert list(pos) == [0, -1, -1, 0]


def test_stop_loss_and_wait():
    # stopped at -4, then no new trade until |z| < entry_z again, even though z is still < -2
    pos, reason = positions(z_series([0, -2.5, -4, -3, -2.5, -1, -2.5, 0]), CFG)
    assert reason.iloc[2] == "stop_loss"
    assert list(pos) == [0, 1, 0, 0, 0, 0, 1, 0]


def test_no_entry_beyond_stop():
    # |z| already beyond stop_z -> no entry (we would be stopped out immediately)
    pos, _ = positions(z_series([0, -4, -4, 0]), CFG)
    assert (pos == 0).all()


def test_time_stop():
    pos, reason = positions(z_series([0, -2.5, -2.4, -2.3, -2.2]), dict(CFG, time_stop_hours=2))
    assert list(pos) == [0, 1, 1, 0, 0]
    assert reason.iloc[3] == "time_stop"


def test_half_life_of_ar1():
    # spread(t) = 0.9 * spread(t-1) + noise  ->  half-life = ln 2 / -ln 0.9 ≈ 6.6 hours
    rng = np.random.default_rng(0)
    x = np.zeros(20_000)
    for t in range(1, len(x)):
        x[t] = 0.9 * x[t - 1] + rng.normal()
    assert half_life(pd.Series(x)) == pytest.approx(np.log(2) / -np.log(0.9), rel=0.1)
