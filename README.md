# Crypto Pairs Trading: AMA4370 Applied Algorithmic Trading Strategies

Group project, PolyU. Statistical arbitrage (pairs trading) on liquid Binance USDT pairs, hourly data.
Second track: [alt/BTC ratio mean reversion](#second-track-altbtc-ratio-mean-reversion-notebook-05) on daily Yahoo data (notebook 05).

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
| `config_ratio.yaml` | Same layout for the alt/BTC ratio track (notebook 05) |
| `src/data.py` | Load config, download OHLCV via ccxt or daily closes via Yahoo, load clean aligned prices, coverage report, train/test split |
| `src/strategy.py` | Pair selection, signal generation, risk management, ratio vs BTC features and diagnostics |
| `src/backtest.py` | Two-leg backtest, metrics, `run_strategy` pipeline, ratio portfolio pipeline, benchmarks, sweeps |
| `src/plots.py` | All figures, one consistent style |
| `notebooks/01_data.ipynb` | Data download, quality checks, overview |
| `notebooks/02_pair_selection.ipynb` | Choose the pair (train data only) |
| `notebooks/03_calibration.ipynb` | Choose signal and risk parameters (train data only) |
| `notebooks/04_results.ipynb` | In-sample and out-of-sample results, robustness |
| `notebooks/05_alt_btc_ratio.ipynb` | Walkthrough of the alt/BTC ratio strategy (decides nothing) |
| `tests/` | Sanity tests for data layer, strategy logic and backtest engine |
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

python src/data.py config_ratio.yaml   # daily Yahoo data for the ratio track, then run notebook 05

python -m pytest              # run tests
```

## Workflow rules

- **Decisions only in 02 and 03, on train data only.** The chosen pair and parameters are written to `config.yaml`.
- **The test period is looked at once, in 04.** Notebook 04 reads `config.yaml` and decides nothing.
- **All parameters live in `config.yaml`.** No magic numbers in notebooks or `src/`.
- **Logic lives in `src/`.** Notebooks only import from `src/`, explain, and plot.

## Second track: alt/BTC ratio mean reversion (notebook 05)

1. Universe and data: BTC as numeraire plus ETH, BNB, XRP, ADA, DOGE, LTC, LINK, BCH, XLM, TRX (large, liquid, complete daily Yahoo history since Nov 2017; SOL, AVAX, DOT dropped because their history starts in 2020)
2. Feature: log ratio alt/BTC, 90-day rolling mean and std, z-score; fair value = BTC x mean ratio
3. Signal: short alt / long BTC at z > +2, long alt / short BTC at z < -2, exit at z = 0; signal at close t, fill at close t+1
4. Risk: dollar-neutral pairs, equal risk per pair (1 / ratio vol), gross exposure <= 2x; z-stop 4, stop-loss 0.10, 60-day time stop
5. Evaluation: train 2018-2022 for all decisions (36-cell sweep), test 2023-2026 once; fee 10 bps + slippage 5 bps per leg, borrow 3 bps/day on the short leg

Same workflow rules as above, with `config_ratio.yaml` in place of `config.yaml`: the parameters were fixed on train with `backtest.parameter_sweep`, and notebook 05 only reads them.

## Results

Pairs track (notebooks 01-04): TBD

Ratio track (notebook 05), test period 2023-01-01 to 2026-10-31:

| | Ann. return | Sharpe | Max DD | Trades | Win rate |
|---|---|---|---|---|---|
| Strategy | -4.6% | -0.44 | -31% | 138 | 42% |
| Strategy before costs | +0.2% | 0.02 | -20% | 138 | |
| Buy-and-hold BTC | +49.4% | 1.10 | -53% | | |
| Equal-weight alts | +25.6% | 0.69 | -60% | | |

No edge: every parameter combination also lost on train. Alt/BTC ratios trended more than they reverted at a 3-to-13-week horizon. Details and limitations are in notebook 05, sections 8 and 10.
