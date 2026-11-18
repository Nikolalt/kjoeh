"""Sanity tests for the data layer (offline, on synthetic parquet files). Run with `python -m pytest` from the repo root."""

import numpy as np
import pandas as pd

from src import data


def _cfg(max_ffill_days: int = 2) -> dict:
    """Small config with one numeraire and two alts over 10 days."""
    return {
        "setup": {
            "numeraire": "BTC-USD",
            "alts": ["AAA-USD", "BBB-USD"],
            "timeframe": "1d",
            "download_start": "2024-01-01",
            "end": "2024-01-11",
            "max_ffill_days": max_ffill_days,
        }
    }


def _write(tmp_path, ticker: str, dates: pd.DatetimeIndex) -> None:
    """Cache a synthetic close series 1, 2, 3, ... on the given dates."""
    df = pd.DataFrame({"close": np.arange(1.0, len(dates) + 1)}, index=dates)
    df.to_parquet(data.cache_path(ticker, "1d", tmp_path))


def _days(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="D", tz="UTC")


def test_split_train_test_no_overlap():
    """Train ends strictly before the cutoff and test starts on it, together they cover every row exactly once."""
    prices = pd.DataFrame({"BTC": range(10)}, index=_days("2024-01-01", "2024-01-10"))
    train, test = data.split_train_test(prices, "2024-01-06")
    assert train.index.max() < pd.Timestamp("2024-01-06", tz="UTC") == test.index.min()
    assert len(train) + len(test) == len(prices)


def test_load_daily_prices_aligns_on_common_period(tmp_path):
    """A coin that starts later defines the start of the common period, and columns are named by coin."""
    _write(tmp_path, "BTC-USD", _days("2024-01-01", "2024-01-10"))
    _write(tmp_path, "AAA-USD", _days("2024-01-01", "2024-01-10"))
    _write(tmp_path, "BBB-USD", _days("2024-01-04", "2024-01-10"))
    prices = data.load_daily_prices(_cfg(), tmp_path)
    assert list(prices.columns) == ["BTC", "AAA", "BBB"]
    assert prices.index[0] == pd.Timestamp("2024-01-04", tz="UTC")
    assert prices.notna().all().all()


def test_short_gap_filled_long_gap_dropped(tmp_path):
    """Gaps up to max_ffill_days carry the last price; longer gaps are not filled and those days are dropped."""
    full = _days("2024-01-01", "2024-01-10")
    _write(tmp_path, "BTC-USD", full)
    _write(tmp_path, "AAA-USD", full.delete([2, 3]))           # 2-day gap -> filled
    _write(tmp_path, "BBB-USD", full.delete([5, 6, 7]))        # 3-day gap -> 1 day stays NaN
    prices = data.load_daily_prices(_cfg(max_ffill_days=2), tmp_path)
    assert prices.loc["2024-01-04", "AAA"] == prices.loc["2024-01-02", "AAA"]
    assert pd.Timestamp("2024-01-08", tz="UTC") not in prices.index
    assert len(prices) == 9

    report = data.coverage_report(_cfg(max_ffill_days=2), tmp_path)
    assert report.loc["AAA", "missing_days"] == 2 and report.loc["AAA", "unfilled"] == 0
    assert report.loc["BBB", "longest_gap"] == 3 and report.loc["BBB", "unfilled"] == 1
