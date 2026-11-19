"""Data layer: config handling, downloading perpetual prices and funding rates, loading clean data.

We trade crypto *perpetual futures*, because shorting one leg of a pair is only practical with perps.
So for every coin we need two things:
    1. hourly OHLCV candles of the perpetual       -> data/<COIN>.parquet
    2. the funding rate history (paid every ~8h)   -> data/<COIN>_funding.parquet

Run `python src/data.py` once to download the full universe into data/.

Time conventions (important for the backtest):
    - All timestamps are UTC.
    - A candle is labelled with its OPEN time (ccxt convention): the bar at 07:00 covers 07:00-08:00,
      and its close price is the price at 08:00.
    - Funding is labelled with the instant it is paid (e.g. 08:00). It is paid by whoever holds a
      position at that instant, i.e. the position carried from bar 07:00 into bar 08:00.
    - Funding sign: a positive rate means longs pay shorts, a negative rate means shorts pay longs.
    - Config dates are whole days and inclusive: end "2026-10-31" includes the 23:00 bar of that day.
"""

import time
from pathlib import Path

import ccxt
import pandas as pd
import yaml
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedSeq

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

DEFAULT_EXCHANGE = "binanceusdm"   # Binance USDⓈ-M perpetual futures
CANDLES_PER_REQUEST = 1000         # Binance futures returns at most ~1000-1500 candles per call
FUNDING_PER_REQUEST = 1000         # Binance returns at most 1000 funding records per call
MAX_RETRIES = 3                    # retries for temporary network errors before giving up
CACHE_TOLERANCE = pd.Timedelta("1D")  # a cached file may start/end up to this much inside the range


# =============================================================================
# 1. CONFIG
# =============================================================================

def load_config(path: str = "config.yaml") -> dict:
    """Read the project config from YAML (relative paths are resolved against the repo root).

    Resolving against the repo root (not the current working directory) means the same call works
    from the repo root, from notebooks/ and from src/.

    Input:  path to the YAML file.
    Output: dict with the two sections "setup" and "parameters".
    """
    full_path = Path(path) if Path(path).is_absolute() else REPO_ROOT / path
    with open(full_path) as f:
        return yaml.safe_load(f)


def get_params(cfg: dict, **overrides) -> dict:
    """Collect everything a backtest run needs into one flat dict, with optional overrides.

    Takes the cost and capital settings from `setup` and all of `parameters`, then applies the
    overrides, e.g. get_params(cfg, entry_z=2.5, stop_z=None). Unknown override names raise an
    error, so a typo like "entryz" cannot silently be ignored in a sweep.

    Input:  config dict, keyword overrides.
    Output: flat dict of parameters.
    """
    setup = cfg["setup"]
    params = {
        "fee_per_leg": setup["fee_per_leg"],
        "slippage_per_leg": setup["slippage_per_leg"],
        "initial_capital": setup["initial_capital"],
        "train_end": setup["train_end"],
        **cfg["parameters"],
    }
    unknown = set(overrides) - set(params)
    if unknown:
        raise KeyError(f"Unknown parameter(s) {sorted(unknown)}. Valid names: {sorted(params)}")
    params.update(overrides)
    return params


def update_config(updates: dict, path: str = "config.yaml") -> None:
    """Write decided values into the `parameters` section of config.yaml (used in notebooks 02 and 03).

    ruamel.yaml reads and writes the file "round trip", so comments and the layout survive
    (plain PyYAML would drop all comments). Prints exactly which values changed.

    Input:  dict of parameter name -> new value, e.g. {"pair": ["ETH", "BTC"], "entry_z": 2.5};
            path to the YAML file.
    Output: None, the file is updated on disk.
    """
    full_path = Path(path) if Path(path).is_absolute() else REPO_ROOT / path
    yaml_rt = YAML()                     # round-trip mode keeps comments and key order
    yaml_rt.preserve_quotes = True       # keep "2023-01-01" quoted
    yaml_rt.indent(mapping=2, sequence=4, offset=2)  # same list indentation as our file
    # write None as "null" (ruamel's default would leave the value empty)
    yaml_rt.representer.add_representer(
        type(None), lambda rep, _: rep.represent_scalar("tag:yaml.org,2002:null", "null"))
    with open(full_path) as f:
        cfg = yaml_rt.load(f)

    params = cfg["parameters"]
    changed = False
    for key, new_value in updates.items():
        if key not in params:
            raise KeyError(f"'{key}' is not in the parameters section of {full_path.name}.")
        old_value = params[key]
        if isinstance(new_value, (list, tuple)):
            new_value = CommentedSeq(new_value)
            new_value.fa.set_flow_style()  # write lists inline: [ETH, BTC]
        if old_value == new_value:
            print(f"  parameters.{key}: unchanged ({old_value})")
            continue
        params[key] = new_value
        changed = True
        print(f"  parameters.{key}: {old_value} -> {list(new_value) if isinstance(new_value, list) else new_value}")

    if changed:
        with open(full_path, "w") as f:
            yaml_rt.dump(cfg, f)
        print(f"Saved {full_path.name}. Reload with data.load_config() to use the new values.")


def perp_symbol(coin: str) -> str:
    """Map a base coin to its ccxt USDT-margined perpetual symbol, e.g. "BTC" -> "BTC/USDT:USDT".

    The ":USDT" suffix is ccxt's notation for a linear perpetual settled in USDT.
    """
    return f"{coin}/USDT:USDT"


def date_range_utc(start: str, end: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Turn inclusive config dates into a half-open UTC interval [start, end + 1 day).

    Example: ("2023-01-01", "2026-10-31") -> [2023-01-01 00:00, 2026-11-01 00:00).
    """
    start_ts = pd.Timestamp(start, tz="UTC").normalize()
    end_ts = pd.Timestamp(end, tz="UTC").normalize() + pd.Timedelta(days=1)
    return start_ts, end_ts


def hourly_index(cfg: dict) -> pd.DatetimeIndex:
    """The complete hourly UTC grid from setup.start to setup.end (both days inclusive).

    Prices and funding are both put on this grid, so they line up row by row.
    """
    start_ts, end_ts = date_range_utc(cfg["setup"]["start"], cfg["setup"]["end"])
    freq = pd.Timedelta(cfg["setup"]["timeframe"])
    return pd.date_range(start_ts, end_ts, freq=freq, inclusive="left", name="timestamp")


# =============================================================================
# 2. DOWNLOAD FROM THE EXCHANGE
# =============================================================================

def _exchange_error_message(exchange_id: str, err: Exception) -> str:
    """Readable error text for a blocked or failing exchange, with the suggested fix."""
    return (
        f"Download from '{exchange_id}' failed: {type(err).__name__}: {err}\n"
        f"The exchange may be blocked in your region (e.g. HTTP 451/403) or temporarily down.\n"
        f"Fix: set `setup.exchange` in config.yaml to another perpetual-futures venue, "
        f"e.g. 'okx' or 'bybit' (symbols like BTC/USDT:USDT work the same there), "
        f"then rerun with force=True."
    )


def _make_exchange(exchange_id: str = DEFAULT_EXCHANGE) -> ccxt.Exchange:
    """Create a ccxt exchange object with the built-in rate limiter switched on.

    enableRateLimit makes ccxt wait between calls by itself, so we never exceed the API limits.
    defaultType 'swap' makes multi-market exchanges (okx, bybit) use perpetuals, not spot.
    """
    if not hasattr(ccxt, exchange_id):
        raise ValueError(f"Unknown ccxt exchange id '{exchange_id}' in config.yaml (setup.exchange).")
    exchange_class = getattr(ccxt, exchange_id)
    return exchange_class({"enableRateLimit": True, "options": {"defaultType": "swap"}})


def _call_with_retry(exchange: ccxt.Exchange, method: str, *args, **kwargs):
    """Call an exchange method, retrying temporary network errors and failing with a clear message.

    - ccxt.NetworkError (timeouts, dropped connections) is often temporary -> wait and retry.
    - Anything else from ccxt (blocked region, bad symbol, ...) -> raise a RuntimeError with a hint.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return getattr(exchange, method)(*args, **kwargs)
        except ccxt.NetworkError as err:
            # ExchangeNotAvailable (incl. geo-blocks) is a NetworkError subclass, but retrying won't help
            if attempt == MAX_RETRIES or isinstance(err, ccxt.ExchangeNotAvailable):
                raise RuntimeError(_exchange_error_message(exchange.id, err)) from err
            time.sleep(2 * attempt)
        except ccxt.BaseError as err:
            raise RuntimeError(_exchange_error_message(exchange.id, err)) from err


def fetch_ohlcv(
    symbol: str, timeframe: str, start: str, end: str, exchange_id: str = DEFAULT_EXCHANGE
) -> pd.DataFrame:
    """Download OHLCV candles for one perpetual via ccxt, paginating over the date range.

    The exchange returns at most ~1000 candles per request, so we loop: ask for 1000 candles
    starting at `since`, then move `since` to just after the last candle we got, until we pass `end`.

    Input:  symbol (e.g. "BTC/USDT:USDT"), timeframe (e.g. "1h"), start and end dates "YYYY-MM-DD"
            (both inclusive), ccxt exchange id (default binanceusdm).
    Output: DataFrame with columns open, high, low, close, volume and a UTC DatetimeIndex (open time).
    """
    exchange = _make_exchange(exchange_id)
    start_ts, end_ts = date_range_utc(start, end)
    end_ms = int(end_ts.timestamp() * 1000)
    step_ms = exchange.parse_timeframe(timeframe) * 1000  # length of one candle in milliseconds

    rows = []
    since = int(start_ts.timestamp() * 1000)
    while since < end_ms:
        batch = _call_with_retry(exchange, "fetch_ohlcv", symbol, timeframe, since, CANDLES_PER_REQUEST)
        if not batch:  # nothing more available (e.g. reached the present)
            break
        rows.extend(batch)
        since = batch[-1][0] + step_ms  # next request starts one candle after the last one received

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates("timestamp").set_index("timestamp").sort_index()
    return df[df.index < end_ts]  # the last request may overshoot the end date


def fetch_funding_rates(
    symbol: str, start: str, end: str, exchange_id: str = DEFAULT_EXCHANGE
) -> pd.Series:
    """Download the funding rate history of one perpetual via ccxt, paginating over the date range.

    Funding is usually paid every 8h (00:00, 08:00, 16:00 UTC); some coins switch to 4h or 1h in
    volatile periods. Each record is a rate per funding event (e.g. 0.0001 = 0.01% of notional).

    Input:  symbol (e.g. "BTC/USDT:USDT"), start and end dates "YYYY-MM-DD" (inclusive), exchange id.
    Output: Series "funding_rate" with a UTC DatetimeIndex (time the funding was paid).
    """
    exchange = _make_exchange(exchange_id)
    start_ts, end_ts = date_range_utc(start, end)
    end_ms = int(end_ts.timestamp() * 1000)

    records = []
    since = int(start_ts.timestamp() * 1000)
    while since < end_ms:
        batch = _call_with_retry(exchange, "fetch_funding_rate_history", symbol, since, FUNDING_PER_REQUEST)
        if not batch:
            break
        records.extend(batch)
        since = batch[-1]["timestamp"] + 1  # continue just after the last record received

    if not records:
        return pd.Series(dtype=float, name="funding_rate")
    df = pd.DataFrame([(r["timestamp"], r["fundingRate"]) for r in records], columns=["timestamp", "funding_rate"])
    # Binance stamps some events a few ms late (e.g. 08:00:00.008), so round to the full second
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.round("s")
    s = df.drop_duplicates("timestamp").set_index("timestamp")["funding_rate"].sort_index()
    return s[s.index < end_ts]


def _price_path(coin: str) -> Path:
    return DATA_DIR / f"{coin}.parquet"


def _funding_path(coin: str) -> Path:
    return DATA_DIR / f"{coin}_funding.parquet"


def cache_covers_range(path: Path, start: str, end: str, tolerance: pd.Timedelta = CACHE_TOLERANCE) -> bool:
    """Check whether a cached parquet file exists and covers the configured date range.

    Without this check, changing setup.start or setup.end in config.yaml would silently keep using
    an old file with a shorter history. The tolerance allows for small gaps at the edges, e.g. funding
    is only paid every 8h, so the first event can be a few hours after the start.

    Input:  path to a cached price or funding parquet file, inclusive config dates, tolerance.
    Output: True if the file exists and its first/last timestamps are within `tolerance` of the range.
    """
    if not path.exists():
        return False
    index = pd.read_parquet(path).index
    if len(index) == 0:
        return False
    start_ts, end_ts = date_range_utc(start, end)
    last_bar = end_ts - pd.Timedelta(hours=1)  # the last hourly bar of the range, e.g. 23:00 on the end day
    return index.min() <= start_ts + tolerance and index.max() >= last_bar - tolerance


def download_universe(cfg: dict, force: bool = False) -> None:
    """Fetch prices and funding rates for every coin in the universe and cache them as parquet.

    Files that already exist AND cover the configured date range are skipped, so rerunning is cheap.
    A file that is missing or too short (e.g. after extending setup.end) is re-downloaded automatically.
    Use force=True to re-download everything, e.g. after changing the exchange in config.yaml.

    Input:  config dict (uses setup.exchange, universe, timeframe, start, end), force flag.
    Output: None, writes data/<COIN>.parquet and data/<COIN>_funding.parquet.
    """
    setup = cfg["setup"]
    exchange_id = setup.get("exchange", DEFAULT_EXCHANGE)
    start, end = setup["start"], setup["end"]
    DATA_DIR.mkdir(exist_ok=True)
    coins = setup["universe"]
    print(f"Downloading {len(coins)} coins from {exchange_id} ({start} to {end}, {setup['timeframe']})")

    for i, coin in enumerate(coins, start=1):
        symbol = perp_symbol(coin)
        prefix = f"[{i:>2}/{len(coins)}] {coin:<5}"

        price_path = _price_path(coin)
        if not force and cache_covers_range(price_path, start, end):
            print(f"{prefix} prices:  cached, skipped")
        else:
            if price_path.exists() and not force:
                print(f"{prefix} prices:  cached file does not cover {start} to {end}, re-downloading")
            prices = fetch_ohlcv(symbol, setup["timeframe"], start, end, exchange_id)
            prices.to_parquet(price_path)
            print(f"{prefix} prices:  {len(prices):>6} bars saved")

        funding_path = _funding_path(coin)
        if not force and cache_covers_range(funding_path, start, end):
            print(f"{prefix} funding: cached, skipped")
        else:
            if funding_path.exists() and not force:
                print(f"{prefix} funding: cached file does not cover {start} to {end}, re-downloading")
            funding = fetch_funding_rates(symbol, start, end, exchange_id)
            funding.to_frame().to_parquet(funding_path)
            print(f"{prefix} funding: {len(funding):>6} events saved")


# =============================================================================
# 3. LOAD CLEAN, ALIGNED DATA
# =============================================================================

def load_raw_closes(cfg: dict) -> pd.DataFrame:
    """Read the cached close prices of all coins onto the complete hourly grid, WITHOUT any cleaning.

    Missing candles show up as NaN, which is what the quality report counts.

    Input:  config dict.
    Output: DataFrame of close prices, one column per coin (e.g. "BTC"), full hourly UTC index.
    """
    index = hourly_index(cfg)
    closes = {}
    for coin in cfg["setup"]["universe"]:
        path = _price_path(coin)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run `python src/data.py` first to download the data.")
        closes[coin] = pd.read_parquet(path)["close"].reindex(index)
    return pd.DataFrame(closes, index=index)


def data_quality(raw: pd.DataFrame) -> pd.DataFrame:
    """Per-coin data-quality summary of raw (uncleaned) prices on the full hourly grid.

    Input:  output of load_raw_closes.
    Output: DataFrame with one row per coin: bars, missing_bars, missing_pct, longest_gap (bars),
            first and last timestamp with data.
    """
    rows = {}
    for coin, s in raw.items():
        missing = s.isna()
        # length of each run of consecutive NaNs -> the longest one is the worst gap
        run_id = (~missing).cumsum()
        longest_gap = int(missing.groupby(run_id).sum().max()) if missing.any() else 0
        rows[coin] = {
            "bars": int(s.notna().sum()),
            "missing_bars": int(missing.sum()),
            "missing_pct": missing.mean(),
            "longest_gap": longest_gap,
            "first": s.first_valid_index(),
            "last": s.last_valid_index(),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def fill_short_gaps(s: pd.Series, max_gap: int) -> pd.Series:
    """Forward-fill gaps of at most `max_gap` consecutive NaNs; longer gaps stay completely NaN.

    Note: pandas' ffill(limit=n) would also fill the first n bars of a long gap, which would hide
    an outage. Here a gap is either filled as a whole or not at all.
    """
    missing = s.isna()
    run_id = (~missing).cumsum()                         # same id for a value and the NaNs after it
    gap_len = missing.groupby(run_id).transform("sum")   # length of the gap each NaN belongs to
    fill_mask = missing & (gap_len <= max_gap)
    return s.where(~fill_mask, s.ffill())


def load_prices(cfg: dict, verbose: bool = True) -> pd.DataFrame:
    """Load cached prices into one aligned, cleaned table of close prices.

    Steps: put every coin on the complete hourly UTC grid -> report data quality -> forward-fill
    short gaps (<= setup.max_ffill_bars) -> drop coins with more than setup.max_missing_pct missing.

    Input:  config dict, verbose flag (print the quality report).
    Output: DataFrame with one column per remaining coin (e.g. "BTC"), hourly UTC index.
            Longer gaps remain NaN, so downstream code must handle (or drop) them explicitly.
    """
    setup = cfg["setup"]
    max_gap = setup.get("max_ffill_bars", 3)
    max_missing = setup.get("max_missing_pct", 0.05)

    raw = load_raw_closes(cfg)
    report = data_quality(raw)
    if verbose:
        print("Data quality (raw, before cleaning):")
        print(report.to_string(formatters={"missing_pct": "{:.2%}".format}))

    prices = raw.apply(fill_short_gaps, max_gap=max_gap)

    too_sparse = report.index[report["missing_pct"] > max_missing].tolist()
    if too_sparse:
        print(f"WARNING: dropping {too_sparse}: more than {max_missing:.0%} of hourly bars missing.")
        prices = prices.drop(columns=too_sparse)
    if verbose:
        print(f"Kept {prices.shape[1]} coins, {prices.isna().sum().sum()} NaNs left after filling gaps <= {max_gap} bars.")
    return prices


def load_funding(cfg: dict, coins: list[str] | None = None) -> pd.DataFrame:
    """Load cached funding rates onto the same hourly UTC grid as the prices.

    Each cell holds the funding rate paid at that hour, and 0 in hours without a funding event.
    That way the backtest can simply sum funding per bar: cost of a leg = -position * rate.

    Input:  config dict, optional list of coins (default: the whole universe).
    Output: DataFrame of funding rates, one column per coin, full hourly UTC index.
    """
    index = hourly_index(cfg)
    coins = coins if coins is not None else cfg["setup"]["universe"]
    funding = {}
    for coin in coins:
        path = _funding_path(coin)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run `python src/data.py` first to download the data.")
        s = pd.read_parquet(path)["funding_rate"]
        # snap every event to its hour (events are on the hour anyway) and add up if two land in one hour
        s = s.groupby(s.index.floor("h")).sum()
        funding[coin] = s.reindex(index, fill_value=0.0)
    return pd.DataFrame(funding, index=index)


def split_train_test(df: pd.DataFrame, train_end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split any time-indexed DataFrame into train (up to and including train_end) and test (after it).

    train_end is a whole day: "2025-12-31" puts the 23:00 bar of Dec 31 in train, and the test
    period starts strictly after it, at 2026-01-01 00:00.

    Input:  DataFrame with a UTC DatetimeIndex, cutoff date "YYYY-MM-DD".
    Output: (train, test) DataFrames with no overlapping timestamps.
    """
    cutoff = pd.Timestamp(train_end, tz="UTC").normalize() + pd.Timedelta(days=1)
    return df[df.index < cutoff], df[df.index >= cutoff]


if __name__ == "__main__":
    cfg = load_config()
    download_universe(cfg)
    print()
    prices = load_prices(cfg)
    funding = load_funding(cfg, coins=list(prices.columns))
    train, test = split_train_test(prices, cfg["setup"]["train_end"])
    print(f"\nPrices:  {prices.shape[0]} bars x {prices.shape[1]} coins, {prices.index[0]} to {prices.index[-1]}")
    print(f"Train:   {len(train)} bars ({train.index[0]} to {train.index[-1]})")
    print(f"Test:    {len(test)} bars ({test.index[0]} to {test.index[-1]})")
    # mean funding per hour x hours per year = average annualized funding a long position pays
    annual_funding = funding.mean() * 24 * 365
    print("Average annualized funding (paid by longs):")
    print(annual_funding.map("{:.2%}".format).to_string())
