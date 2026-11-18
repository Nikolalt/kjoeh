"""Sanity tests for the strategy logic (features, signals, risk). Run with `python -m pytest` from the repo root."""

import numpy as np
import pandas as pd
import pytest

from src import strategy


def _series(values, start: str = "2024-01-01") -> pd.Series:
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="D", tz="UTC"), dtype=float)


# =============================================================================
# FEATURES
# =============================================================================

def test_compute_ratio_known_prices():
    """alt=2, btc=4 gives ratio 0.5 and log ratio log(0.5); log ratio change equals alt return minus btc return."""
    alt, btc = _series([2.0, 3.0]), _series([4.0, 4.0])
    assert strategy.compute_ratio(alt, btc, log=False).iloc[0] == pytest.approx(0.5)
    log_ratio = strategy.compute_ratio(alt, btc)
    assert log_ratio.iloc[0] == pytest.approx(np.log(0.5))
    assert log_ratio.diff().iloc[1] == pytest.approx(np.log(3 / 2) - np.log(4 / 4))


def test_fair_value_inverts_ratio():
    """If the alt sits exactly on its mean ratio, its fair value equals its price."""
    alt, btc = _series([10.0, 12.0]), _series([100.0, 150.0])
    mean = strategy.compute_ratio(alt, btc)
    assert np.allclose(strategy.fair_value(btc, mean), alt)


def test_zscore_nan_until_window_full():
    """The z-score is undefined until `lookback` bars are available, then defined."""
    z = strategy.zscore(_series(np.random.default_rng(0).normal(size=20)), lookback=5)
    assert z.iloc[:4].isna().all() and z.iloc[4:].notna().all()


@pytest.mark.parametrize("method", ["rolling", "expanding"])
def test_zscore_no_lookahead(method):
    """Changing future values must not change any past z-score (statistics use data up to t only)."""
    x = _series(np.random.default_rng(1).normal(size=50))
    shocked = x.copy()
    shocked.iloc[30:] += 100.0
    z, z_shocked = strategy.zscore(x, 10, method), strategy.zscore(shocked, 10, method)
    pd.testing.assert_series_equal(z.iloc[:30], z_shocked.iloc[:30])
    assert not np.isclose(z.iloc[30], z_shocked.iloc[30])


def test_zscore_hand_computed():
    """Last value of [1, 2, 3, 6] with lookback 4: mean 3, sample std sqrt(14/3), z = 3 / sqrt(14/3)."""
    z = strategy.zscore(_series([1.0, 2.0, 3.0, 6.0]), lookback=4)
    assert z.iloc[-1] == pytest.approx(3 / np.sqrt(14 / 3))


def test_estimate_half_life_recovers_ar1():
    """An AR(1) with phi = 0.5**(1/10) has a half-life of 10 bars; the estimate should be close."""
    rng = np.random.default_rng(2)
    rho, x = 0.5 ** (1 / 10), np.zeros(20000)
    for t in range(1, len(x)):
        x[t] = rho * x[t - 1] + rng.normal()
    assert strategy.estimate_half_life(_series(x)) == pytest.approx(10, rel=0.15)


def test_estimate_half_life_random_walk_like_trend():
    """A steadily trending series does not revert, so the half-life is infinite."""
    assert strategy.estimate_half_life(_series(np.arange(100.0) ** 1.5)) == float("inf")


def test_ratio_features_columns_and_deviation():
    """ratio_features returns all indicator columns, and deviation is 0 where the alt sits on its fair value."""
    alt, btc = _series(np.full(10, 5.0)), _series(np.full(10, 50.0))
    f = strategy.ratio_features(alt, btc, lookback=3)
    assert list(f.columns) == ["ratio", "log_ratio", "mean", "std", "z", "fair_value", "deviation"]
    assert np.allclose(f["deviation"].iloc[2:], 0.0)
    assert f["z"].isna().all()  # constant ratio -> std 0 -> z undefined, never a false signal


# =============================================================================
# SIGNALS
# =============================================================================

def test_generate_signals_enter_hold_exit():
    """High z opens a short, it is held until z crosses exit_z, then a low z opens a long."""
    z = _series([0.0, 2.5, 1.5, 0.5, -0.1, -0.5, -2.5, -1.0, 0.2, np.nan])
    expected = [0, -1, -1, -1, 0, 0, 1, 1, 0, 0]
    assert strategy.generate_signals(z, entry_z=2.0, exit_z=0.0).tolist() == expected


def test_generate_signals_flip_on_same_bar():
    """A jump from above +entry to below -entry closes the short and opens a long on the same bar."""
    z = _series([2.5, -2.5])
    assert strategy.generate_signals(z, 2.0, 0.0).tolist() == [-1, 1]


def test_generate_signals_sine_mean_reversion():
    """On a sine wave the strategy is short near peaks and long near troughs, never both."""
    z = _series(3 * np.sin(np.linspace(0, 4 * np.pi, 200)))
    sig = strategy.generate_signals(z, 2.0, 0.0)
    assert (sig[z > 2.0] == -1).all() and (sig[z < -2.0] == 1).all()
    assert (sig[(z > 0) & (sig != 0)] == -1).all()  # only shorts while z is positive


# =============================================================================
# RISK MANAGEMENT
# =============================================================================

def test_apply_risk_z_stop_then_flat_until_fresh_signal():
    """A breakout beyond stop_z closes the trade with reason stop_z, and no re-entry happens while the raw signal is unchanged."""
    z = _series([2.5, 3.0, 4.5, 3.5, 2.5, -0.1, 2.5])
    sig = strategy.generate_signals(z, 2.0, 0.0)            # -1 -1 -1 -1 -1 0 -1
    out = strategy.apply_risk(sig, z, {"stop_z": 4.0, "position_size": 0.5}, return_reasons=True)
    assert out["position"].tolist() == [-0.5, -0.5, 0, 0, 0, 0, -0.5]
    assert out["exit_reason"].tolist() == ["", "", "stop_z", "", "", "", ""]


def test_apply_risk_time_stop():
    """With half_life 2 and 1.5 half-lives the trade is closed after 3 bars, reason time."""
    z = _series([2.5] * 6)
    sig = strategy.generate_signals(z, 2.0, 0.0)
    out = strategy.apply_risk(sig, z, {"half_life": 2, "time_stop_halflives": 1.5}, return_reasons=True)
    assert out["position"].tolist() == [-1, -1, -1, 0, 0, 0]
    assert out["exit_reason"].iloc[3] == "time"


def test_apply_risk_max_holding_days_caps_time_stop():
    """max_holding_days caps the time stop even when the half-life is infinite."""
    params = {"half_life": float("inf"), "time_stop_halflives": 3, "max_holding_days": 2}
    assert strategy.max_holding_bars(params) == 2


def test_apply_risk_mean_exit_and_no_entry_beyond_stop():
    """Reversion gives exit reason mean; an entry signal with |z| above stop_z is not taken."""
    z = _series([2.5, -0.1, 5.0, 3.0])
    sig = strategy.generate_signals(z, 2.0, 0.0)            # -1 0 -1 -1
    out = strategy.apply_risk(sig, z, {"stop_z": 4.0}, return_reasons=True)
    assert out["position"].tolist() == [-1, 0, 0, 0]
    assert out["exit_reason"].tolist() == ["", "mean", "", ""]


def test_apply_risk_without_rules_keeps_signals():
    """With no stops and size 1 the final positions equal the raw signals."""
    z = _series(3 * np.sin(np.linspace(0, 4 * np.pi, 100)))
    sig = strategy.generate_signals(z, 2.0, 0.0)
    pd.testing.assert_series_equal(strategy.apply_risk(sig, z, {}), sig)


def test_equal_risk_sizes():
    """Sizes are inversely proportional to volatility and gross exposure with all pairs open equals the cap."""
    sizes = strategy.equal_risk_sizes(pd.Series({"A": 0.5, "B": 1.0}), max_gross_leverage=2.0)
    assert sizes["A"] == pytest.approx(2 * sizes["B"])
    assert 2 * sizes.sum() == pytest.approx(2.0)
    flat = strategy.equal_risk_sizes(pd.Series({"A": 0.5, "B": 1.0}), 2.0, equal_capital=True)
    assert flat["A"] == flat["B"] == pytest.approx(0.5)


def test_apply_risk_stop_loss_and_take_profit_on_spread():
    """A short spread is stopped when the spread rises more than stop_loss since entry, and a long one
    takes profit when it rises more than take_profit; z alone (below stop_z) would not close either."""
    z = _series([2.5, 2.4, 2.3, 2.2])
    spread = _series([0.0, 0.05, 0.12, 0.20])
    out = strategy.apply_risk(strategy.generate_signals(z, 2.0, 0.0), z, {"stop_loss": 0.1, "stop_z": 4.0},
                              return_reasons=True, spread=spread)
    assert out["position"].tolist() == [-1, -1, 0, 0] and out["exit_reason"].iloc[2] == "stop_loss"

    z_long = -z
    out = strategy.apply_risk(strategy.generate_signals(z_long, 2.0, 0.0), z_long, {"take_profit": 0.1},
                              return_reasons=True, spread=spread)
    assert out["position"].tolist() == [1, 1, 0, 0] and out["exit_reason"].iloc[2] == "take_profit"
