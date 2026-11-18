# Crypto Pairs Trading: AMA4370 Applied Algorithmic Trading Strategies

Group project, PolyU. Statistical arbitrage (pairs trading) on liquid Binance USDT pairs, hourly data.

## Strategy in 5 lines

1. TODO: universe and data (which coins, which period, why)
2. TODO: pair selection (correlation screen, then cointegration test on train data)
3. TODO: signal (rolling hedge ratio, z-score of spread, enter at ±entry_z, exit at exit_z)
4. TODO: risk (z-stop, time stop in half-lives, position sizing)
5. TODO: evaluation (train for decisions, test looked at once, fees included)

## Repo structure

| Path | Content |
|---|---|
| `config.yaml` | All settings: `setup` (fixed) and `parameters` (decided in notebooks 02/03) |
| `src/data.py` | Load config, download OHLCV via ccxt, load clean aligned prices, train/test split |
| `src/strategy.py` | Pair selection, signal generation, risk management |
| `src/backtest.py` | Two-leg backtest, metrics, `run_strategy` pipeline |
| `src/plots.py` | All figures, one consistent style |
| `notebooks/01_data.ipynb` | Data download, quality checks, overview |
| `notebooks/02_pair_selection.ipynb` | Choose the pair (train data only) |
| `notebooks/03_calibration.ipynb` | Choose signal and risk parameters (train data only) |
| `notebooks/04_results.ipynb` | In-sample and out-of-sample results, robustness |
| `tests/` | Sanity tests for the backtest engine |
| `data/` | Cached price data (gitignored) |
| `results/` | Saved figures and tables |

## How to run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
nbstripout --install          # strip notebook outputs on commit

python src/data.py            # download the universe into data/
jupyter notebook              # then run notebooks 01 -> 04 in order

python -m pytest              # run tests
```

## Workflow rules

- **Decisions only in 02 and 03, on train data only.** The chosen pair and parameters are written to `config.yaml`.
- **The test period is looked at once, in 04.** Notebook 04 reads `config.yaml` and decides nothing.
- **All parameters live in `config.yaml`.** No magic numbers in notebooks or `src/`.
- **Logic lives in `src/`.** Notebooks only import from `src/`, explain, and plot.

## Results

TBD
