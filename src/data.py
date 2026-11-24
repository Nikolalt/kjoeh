"""Download and load the data: hourly prices and funding rates of Binance perpetual futures.

We use perpetual futures (not spot) because we need to SHORT one coin of the pair.
Perpetuals have a funding rate, paid every 8 hours: if it is positive, longs pay shorts.

Run `python -m src.data` once to download everything into data/ (takes a few minutes).
"""

from pathlib import Path

import ccxt
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def load_config() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def download(cfg: dict) -> None:
    """Download hourly candles and funding rates for every coin and save them as parquet files."""
    exchange = ccxt.binanceusdm({"enableRateLimit": True})
    start = exchange.parse8601(cfg["start"] + "T00:00:00Z")
    end = exchange.parse8601(cfg["end"] + "T23:59:59Z")
    DATA_DIR.mkdir(exist_ok=True)

    for coin in cfg["universe"]:
        symbol = f"{coin}/USDT:USDT"   # ccxt name of the USDT perpetual

        # The exchange gives at most 1000 rows per request, so we ask again and again
        candles, since = [], start
        while since < end:
            batch = exchange.fetch_ohlcv(symbol, "1h", since, 1000)
            if not batch:
                break
            candles += batch
            since = batch[-1][0] + 3600 * 1000          # continue one hour after the last candle
        prices = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        prices["timestamp"] = pd.to_datetime(prices["timestamp"], unit="ms", utc=True)
        prices.drop_duplicates("timestamp").set_index("timestamp").to_parquet(DATA_DIR / f"{coin}.parquet")

        rates, since = [], start
        while since < end:
            batch = exchange.fetch_funding_rate_history(symbol, since, 1000)
            if not batch:
                break
            rates += [(r["timestamp"], r["fundingRate"]) for r in batch]
            since = batch[-1]["timestamp"] + 1
        funding = pd.DataFrame(rates, columns=["timestamp", "funding_rate"])
        funding["timestamp"] = pd.to_datetime(funding["timestamp"], unit="ms", utc=True)
        funding.drop_duplicates("timestamp").set_index("timestamp").to_parquet(DATA_DIR / f"{coin}_funding.parquet")
        print(f"{coin}: {len(prices)} candles, {len(funding)} funding payments")


def load_data(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the saved files into two tables with one column per coin and one row per hour.

    prices:  close price of each hourly candle. A candle is labelled with its START time,
             so the row 07:00 holds the price at 08:00.
    funding: funding rate paid at that hour (0 in hours without a payment).
    """
    hours = pd.date_range(cfg["start"], pd.Timestamp(cfg["end"]) + pd.Timedelta("23h"), freq="h", tz="UTC")
    prices, funding = {}, {}
    for coin in cfg["universe"]:
        prices[coin] = pd.read_parquet(DATA_DIR / f"{coin}.parquet")["close"].reindex(hours)
        rate = pd.read_parquet(DATA_DIR / f"{coin}_funding.parquet")["funding_rate"]
        rate.index = rate.index.floor("h")              # Binance sometimes stamps 08:00:00.008
        funding[coin] = rate.groupby(level=0).sum().reindex(hours, fill_value=0.0)
    prices = pd.DataFrame(prices)
    print(f"Loaded {len(hours)} hours x {len(prices.columns)} coins, {prices.isna().sum().sum()} missing prices")
    return prices.ffill(), pd.DataFrame(funding)


def split(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into train (up to and including train_end) and test (after train_end)."""
    cutoff = pd.Timestamp(cfg["train_end"], tz="UTC") + pd.Timedelta(days=1)
    return df[df.index < cutoff], df[df.index >= cutoff]


if __name__ == "__main__":
    download(load_config())
