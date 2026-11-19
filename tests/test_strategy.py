"""Quick tests for the signal and risk functions on hand-made data."""

import numpy as np
import pandas as pd
import pytest

from src import strategy  # imported as a module so pytest does not collect strategy.test_cointegration


def hourly_series(values) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=len(values), freq="h", tz="UTC")
    return pd.Series(values, index=idx, dtype=float)


def base_params(**overrides) -> dict:
    params = {"entry_z": 2.0, "exit_z": 0.0, "stop_z": None, "time_stop_hours": None,
              "position_size": 1.0, "use_coint_filter": False, "coint_max_pvalue": 0.1}
    params.update(overrides)
    return params


def test_zscore_uses_no_future_data():
    """Changing the spread after bar t must not change z up to bar t."""
    rng = np.random.default_rng(1)
    spread = hourly_series(rng.normal(size=200).cumsum())
    z = strategy.zscore(spread, lookback=24)

    changed = spread.copy()
    changed.iloc[150:] += 100.0
    z_changed = strategy.zscore(changed, lookback=24)

    pd.testing.assert_series_equal(z.iloc[:150], z_changed.iloc[:150])
    assert z.iloc[:23].isna().all() and z.iloc[23:].notna().all()


def test_generate_signals_hand_made():
    """entry_z = 2, exit_z = 0: enter beyond +-2, exit when z crosses back through 0, flip possible."""
    z = hourly_series([np.nan, 0, -2.5, -1, 0.5, 1, 2.5, 1, -0.1, -3, 3, 1, -1])
    expected = [0, 0, 1, 1, 0, 0, -1, -1, 0, 1, -1, -1, 0]
    #  NaN warm-up; -2.5 long; -1 hold; 0.5 exit; 2.5 short; 1 hold; -0.1 exit;
    #  -3 long; 3 exit long and flip short in the same bar; 1 hold; -1 exit
    signals = strategy.generate_signals(z, entry_z=2.0, exit_z=0.0)
    assert signals.tolist() == expected

    # exit_z = 0.5: the long trade is closed already at z = -0.5 (within 0.5 of the mean)
    z2 = hourly_series([-2.5, -1, -0.5, 0])
    assert strategy.generate_signals(z2, entry_z=2.0, exit_z=0.5).tolist() == [1, 1, 0, 0]


def test_no_rules_positions_equal_scaled_signals():
    """With every risk rule switched off, positions are just signals x position_size."""
    z = hourly_series([0, -2.5, -1, 0.5, 2.5, 1, -0.5, 0])
    signals = strategy.generate_signals(z, 2.0, 0.0)
    positions, reasons, _ = strategy.apply_risk(signals, z, params=base_params(position_size=0.5))
    assert positions.tolist() == (signals * 0.5).tolist()
    assert reasons.dropna().tolist() == ["signal", "signal"]


def test_z_stop_triggers_and_cooldown():
    """Close when |z| > stop_z; no re-entry until |z| < entry_z and a fresh signal appears."""
    z = hourly_series([0, -2.5, -3.0, -4.0, -3.0, -2.5, -1.5, -2.5, 0.5])
    signals = strategy.generate_signals(z, 2.0, 0.0)   # raw: long from bar 1 to bar 7
    positions, reasons, _ = strategy.apply_risk(signals, z, params=base_params(stop_z=3.5))

    #                     0  1  2  3  4  5  6  7  8
    assert positions.tolist() == [0, 1, 1, 0, 0, 0, 0, 1, 0]
    assert reasons.iloc[3] == "z_stop"   # |z| = 4 > 3.5
    # bars 4-5: still |z| > entry_z -> cooldown; bar 6: |z| = 1.5 ends it; bar 7: fresh entry
    assert reasons.iloc[8] == "signal"


def test_no_entry_beyond_stop():
    """A signal that appears with |z| already beyond stop_z is not traded (it would be stopped at once)."""
    z = hourly_series([0, -4.0, -3.0, -1.0])
    signals = strategy.generate_signals(z, 2.0, 0.0)   # raw: long from bar 1
    positions, _, _ = strategy.apply_risk(signals, z, params=base_params(stop_z=3.5))
    assert positions.tolist() == [0, 0, 1, 0]   # bar 2: |z| = 3 is inside (2, 3.5] -> enter


def test_time_stop_triggers():
    """Close after time_stop_hours, even if z has not reverted."""
    z = hourly_series([0, -2.5, -2.2, -2.1, -2.1, -2.1, -2.1, -1.0])
    signals = strategy.generate_signals(z, 2.0, 0.0)
    # time stop 3 h: entered at bar 1 -> closed at bar 4
    positions, reasons, _ = strategy.apply_risk(
        signals, z, params=base_params(time_stop_hours=3))
    assert positions.tolist() == [0, 1, 1, 1, 0, 0, 0, 0]
    assert reasons.iloc[4] == "time_stop"


def test_open_trade_closed_at_end_of_data():
    z = hourly_series([0, -2.5, -2.0, -1.5])
    signals = strategy.generate_signals(z, 2.0, 0.0)
    positions, reasons, _ = strategy.apply_risk(signals, z, params=base_params())
    assert positions.tolist() == [0, 1, 1, 0]
    assert reasons.iloc[-1] == "end_of_data"


def test_frozen_zscore_matches_spread_with_entry_hedge():
    """frozen_zscore(entry, t) equals the plain z-score of the spread built with beta/alpha of the entry bar."""
    rng = np.random.default_rng(7)
    idx = pd.date_range("2024-01-01", periods=300, freq="h", tz="UTC")
    b = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))), index=idx)
    a = pd.Series(50 * np.exp(0.8 * np.log(b / 100) + rng.normal(0, 0.01, 300).cumsum() * 0.3), index=idx)
    beta = strategy.hedge_ratio(a, b, window=48)
    alpha = strategy.hedge_intercept(a, b, beta, window=48)
    z_at = strategy.frozen_zscore(a, b, beta, alpha, lookback=24)

    entry = 150
    expected = strategy.zscore(strategy.compute_spread(a, b, beta.iloc[entry], alpha.iloc[entry]), 24)
    for t in [150, 170, 220, 299]:
        assert z_at(entry, t) == pytest.approx(expected.iloc[t])


def test_trade_exits_on_frozen_z_not_rolling_z():
    """While in a trade, the exit is decided on the frozen-hedge z, even if the rolling z says otherwise."""
    z_rolling = hourly_series([0, -2.5, -2.4, -2.3, -2.2, -2.1])   # rolling z never reverts
    frozen = {2: -1.5, 3: 0.1, 4: 0.5, 5: 0.5}                      # frozen z reverts at bar 3
    signals = strategy.generate_signals(z_rolling, 2.0, 0.0)
    positions, reasons, z_used = strategy.apply_risk(
        signals, z_rolling, params=base_params(),
        trade_z=lambda entry_bar, t: frozen[t])

    assert positions.tolist() == [0, 1, 1, 0, 0, 0]
    assert reasons.iloc[3] == "signal"
    assert z_used.iloc[3] == 0.1
    # cooldown: rolling z is still beyond -2, so no immediate re-entry in the same direction
    assert (positions.iloc[4:] == 0).all()
