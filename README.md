# Crypto Pairs Trading: AMA4370 Applied Algorithmic Trading Strategies

Group project, PolyU. Statistical arbitrage (pairs trading) on liquid Binance USDT pairs, hourly data.

## Strategy in 5 lines

1. Universe and data: 12 liquid Binance USDⓈ-M perpetuals, hourly candles and funding rates, 2023-01-01 to 2026-10-31 (train until 2025-12-31).
2. Pair selection: correlation screen, then Engle-Granger cointegration, stability and half-life of every pair on train data.
3. Signal: (rolling) hedge ratio, z-score of the spread, enter at ±entry_z, exit when z is back within exit_z.
4. Risk: z-stop with cooldown, time stop in half-lives, optional cointegration filter, position sizing.
5. Evaluation: decisions on train only, test looked at once in notebook 04, fees, slippage and funding included.

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
| `tests/` | Tests for the backtest engine and the strategy rules |
| `data/` | Cached price data (gitignored) |
| `results/` | Saved figures and tables |

## How to run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
nbstripout --install          # strip notebook outputs on commit

python src/data.py            # download the universe into data/ (first run takes about 6 minutes)
jupyter notebook              # then run notebooks 01 -> 04 in order

python -m pytest              # run tests
```

## Workflow rules

- **Decisions only in 02 and 03, on train data only.** The chosen pair and parameters are written to `config.yaml`.
- **The test period is looked at once, in 04.** Notebook 04 reads `config.yaml` and decides nothing.
- **All parameters live in `config.yaml`.** No magic numbers in notebooks or `src/`.
- **Logic lives in `src/`.** Notebooks only import from `src/`, explain, and plot.

## Guide for teammates

### 1. Setup (once)

```bash
git clone <repo> && cd <repo>
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
nbstripout --install          # strips notebook outputs on commit, keeps diffs small
python src/data.py            # downloads prices + funding into data/ (first run about 6 minutes)
python -m pytest              # all tests should pass
```

**Every teammate: run `nbstripout --install` once after cloning** (it is in the block above). It sets up a git
filter in your local clone that removes notebook outputs on commit, so diffs stay small and no test-period results
end up in the repository. Check with `git config --get filter.nbstripout.clean` (prints a command if installed).

In VS Code / Jupyter, select the `.venv` kernel. If Binance is blocked in your region, set `setup.exchange`
to `okx` or `bybit` in `config.yaml` and run `python src/data.py` again.

### 2. Notebook order and where each decision is made

| Notebook | What you do | Decision (🟡 cells) | Saved to `config.yaml` |
|---|---|---|---|
| `01_data` | check data quality, prices, correlations, funding | none | – |
| `02_pair_selection` | ranking, multiple testing, stability, spreads | which pair | `pair` |
| `03_calibration` (0) | 1h vs 4h vs 1D bars | bar length | `resample` |
| `03_calibration` (a) | static vs rolling hedge ratio | hedge ratio method | `hedge_window` |
| `03_calibration` (b) | sweep entry_z × lookback × exit_z, heatmaps | signal parameters | `entry_z`, `zscore_lookback`, `exit_z` |
| `03_calibration` (c) | stops on/off, coint filter, stop_z sweep | risk rules | `stop_z`, `time_stop_hours`, `use_coint_filter` |
| `03_calibration` (d) | PnL breakdown: gross vs fees vs slippage vs funding | none | – |
| `04_results` | in-sample, trade walkthrough, out-of-sample, robustness | **none** | – |

Every decision cell is marked **🟡 TEAM DECISION** and explains what to look at. Below it is a commented-out
`data.update_config({...})` call: edit the values, uncomment, run it (it prints exactly what changed and keeps
the comments in `config.yaml`), then rerun the notebook from the top. Values in `config.yaml` that are not decided
yet are placeholders (ETH/BTC and default parameters).

To try something quickly without touching the config, override parameters in code:
`backtest.run_strategy(prices, funding, ("ETH", "BTC"), data.get_params(cfg, entry_z=2.5))`.

### 3. The test-period rule

- Notebooks 01–03 use the **train** period only. All `run_strategy` and `sweep` calls there use the default
  `period="train"`.
- `period="test"` is used **only in notebook 04**, after all decisions are in `config.yaml`. `run_strategy` prints
  a warning whenever the test period is used.
- If you change a parameter after seeing the test results, the test result is no longer out-of-sample. Don't do it,
  or if it happens, say so in the report.

### 4. Working on your own branch

```bash
git checkout main && git pull
git checkout -b <yourname>/<topic>     # e.g. anna/pair-selection
# ... work, run notebooks ...
git add -p && git commit -m "Choose pair LTC/XRP"
git push -u origin <yourname>/<topic>  # then open a pull request on GitHub
```

Change `config.yaml` only through the decision cells, and only for the decision your branch is about, so merges
stay simple. Logic changes go into `src/` (with a test in `tests/` if they touch the backtest or the risk rules),
never into a notebook.

## Results

TBD
