"""Data layer: config handling, downloading OHLCV from the exchange, loading clean prices.

Run `python src/data.py` once to download the full universe into data/.
Run `python src/data.py config_ratio.yaml` for the daily Yahoo universe of the alt/BTC ratio strategy.
"""

import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"


def load_config(path: str = "config.yaml") -> dict:
    """Read the project config from YAML (relative paths are resolved against the repo root).

    Input:  path to the YAML file.
    Output: dict with the two sections "setup" and "parameters".
    """
    full_path = Path(path) if Path(path).is_absolute() else REPO_ROOT / path
    with open(full_path) as f:
        return yaml.safe_load(f)


def update_config(section: str, values: dict, path: str = "config.yaml") -> None:
    """Write decided values into one section of the config (used only by notebooks 02 and 03).

    Input:  section name ("parameters"), dict of keys to overwrite, path to the YAML file.
    Output: None, the file is updated on disk.
    """
    # TODO: load, update cfg[section], write back (note: yaml.safe_dump drops comments)
    raise NotImplementedError


def fetch_ohlcv(symbol: str, timeframe: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV candles for one symbol from Binance via ccxt, paginating over the date range.

    Input:  symbol (e.g. "BTC/USDT"), timeframe (e.g. "1h"), start and end dates as "YYYY-MM-DD".
    Output: DataFrame with columns open, high, low, close, volume and a UTC DatetimeIndex.
    """
    # TODO: ccxt.binance().fetch_ohlcv in a loop (limit ~1000 bars per call), respect rate limit
    raise NotImplementedError


def download_universe(cfg: dict) -> None:
    """Fetch every symbol in the universe and cache each one to data/ as a parquet file.

    Input:  config dict (uses setup.universe, timeframe, start, end).
    Output: None, writes one file per symbol, e.g. data/BTC_USDT_1h.parquet.
    """
    # TODO: loop over cfg["setup"]["universe"], call fetch_ohlcv, save with to_parquet
    raise NotImplementedError


def load_prices(cfg: dict) -> pd.DataFrame:
    """Load the cached parquet files and build one aligned, cleaned table of close prices.

    Input:  config dict.
    Output: DataFrame with one column per coin (e.g. "BTC"), hourly UTC index, no NaNs.
    """
    # TODO: read parquet files, keep close, join on index, reindex to full hourly grid,
    #       forward-fill small gaps, drop coins or rows with too much missing data
    raise NotImplementedError


def split_train_test(prices: pd.DataFrame, train_end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split prices at the cutoff date into a train part (before) and a test part (from cutoff on).

    Input:  price DataFrame, cutoff date "YYYY-MM-DD".
    Output: (train, test) DataFrames with no overlapping timestamps.
    """
    cutoff = pd.Timestamp(train_end, tz=prices.index.tz)
    return prices[prices.index < cutoff], prices[prices.index >= cutoff]


# =============================================================================
# DAILY YAHOO DATA (alt/BTC ratio strategy, config_ratio.yaml)
# =============================================================================

def coin_name(ticker: str) -> str:
    """Short coin name used as column label.

    Input:  Yahoo ticker, e.g. "ETH-USD".
    Output: coin name, e.g. "ETH".
    """
    return ticker.split("-")[0]


def cache_path(ticker: str, timeframe: str, data_dir: Path = DATA_DIR) -> Path:
    """Location of the cached parquet file for one ticker.

    Input:  Yahoo ticker, timeframe (e.g. "1d"), data directory.
    Output: path, e.g. data/BTC-USD_1d.parquet.
    """
    return Path(data_dir) / f"{ticker}_{timeframe}.parquet"


def fetch_yahoo_daily(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV for one ticker from Yahoo Finance via yfinance (adjusted prices).

    Input:  Yahoo ticker (e.g. "BTC-USD"), start and end dates as "YYYY-MM-DD" (end exclusive).
    Output: DataFrame with columns open, high, low, close, volume and a UTC DatetimeIndex; empty if Yahoo has nothing.
    """
    import yfinance as yf  # imported here so the rest of the module works without it

    raw = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
    if raw.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    if isinstance(raw.columns, pd.MultiIndex):  # newer yfinance returns (field, ticker) columns
        raw.columns = raw.columns.get_level_values(0)
    df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df.index = pd.DatetimeIndex(df.index).tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    df.index.name = "date"
    return df.dropna(subset=["close"])


def download_yahoo_universe(cfg: dict, data_dir: Path = DATA_DIR) -> None:
    """Fetch the numeraire and every alt from Yahoo and cache each one to data/ as a parquet file.

    Input:  config dict (uses setup.numeraire, alts, timeframe, download_start, end), data directory.
    Output: None, writes one file per ticker, e.g. data/BTC-USD_1d.parquet; tickers without data are reported.
    """
    setup = cfg["setup"]
    for ticker in [setup["numeraire"], *setup["alts"]]:
        df = fetch_yahoo_daily(ticker, setup["download_start"], setup["end"])
        if df.empty:
            print(f"{ticker}: no data returned, not cached")
            continue
        df.to_parquet(cache_path(ticker, setup["timeframe"], data_dir))
        print(f"{ticker}: {len(df)} days, {df.index[0].date()} -> {df.index[-1].date()}")


def _load_raw_closes(cfg: dict, data_dir: Path) -> pd.DataFrame:
    """Read the cached close prices of numeraire and alts, outer-joined, without any cleaning."""
    setup = cfg["setup"]
    closes = {
        coin_name(t): pd.read_parquet(cache_path(t, setup["timeframe"], data_dir))["close"]
        for t in [setup["numeraire"], *setup["alts"]]
    }
    return pd.concat(closes, axis=1).sort_index()


def _calendar(cfg: dict, tz) -> pd.DatetimeIndex:
    """Full daily calendar [download_start, end) of the config (crypto trades every day)."""
    setup = cfg["setup"]
    return pd.date_range(setup["download_start"], setup["end"], freq="D", inclusive="left", tz=tz, name="date")


def coverage_report(cfg: dict, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Check the cached data of every coin against the full daily calendar.

    Input:  config dict, data directory.
    Output: DataFrame, one row per coin: first, last, n_days, missing_days (inside first..end),
            longest_gap (days), ffilled (days filled by load_daily_prices), unfilled (days left NaN).
    """
    raw = _load_raw_closes(cfg, data_dir)
    calendar = _calendar(cfg, raw.index.tz)
    limit = cfg["setup"]["max_ffill_days"]
    rows = {}
    for coin, s in raw.items():
        s = s.dropna()
        on_grid = s.reindex(calendar)
        live = on_grid.loc[s.index[0]:]                     # ignore the period before the coin was listed
        missing = live.isna()
        runs = missing.groupby((~missing).cumsum()).sum()   # length of each run of missing days
        filled = live.ffill(limit=limit)
        rows[coin] = {
            "first": s.index[0].date(),
            "last": s.index[-1].date(),
            "n_days": len(s),
            "missing_days": int(missing.sum()),
            "longest_gap": int(runs.max()) if len(runs) else 0,
            "ffilled": int((missing & filled.notna()).sum()),
            "unfilled": int(filled.isna().sum()),
        }
    return pd.DataFrame(rows).T


def load_daily_prices(cfg: dict, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load the cached Yahoo files and build one aligned, cleaned table of daily adjusted closes.

    Gaps of up to setup.max_ffill_days are forward-filled (the last known price is carried, which means
    zero return on those days); longer gaps are not filled and those days are dropped, see coverage_report.

    Input:  config dict, data directory.
    Output: DataFrame with one column per coin (e.g. "BTC", "ETH"), daily UTC index on the common
            period of all coins, no NaNs.
    """
    raw = _load_raw_closes(cfg, data_dir)
    prices = raw.reindex(_calendar(cfg, raw.index.tz)).ffill(limit=cfg["setup"]["max_ffill_days"])
    return prices.dropna(how="any")


if __name__ == "__main__":
    config = load_config(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
    if config["setup"].get("source") == "yahoo":
        download_yahoo_universe(config)
    else:
        download_universe(config)
