"""Data layer: config handling, downloading OHLCV from the exchange, loading clean prices.

Run `python src/data.py` once to download the full universe into data/.
"""

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
    raise NotImplementedError


if __name__ == "__main__":
    download_universe(load_config())
