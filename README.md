# Crypto Pairs Trading (AMA4370 Applied Algorithmic Trading Strategies, PolyU)

Statistical arbitrage on two Binance perpetual futures, hourly data, 2023-01 to 2026-10.

**Strategy:** ETH and BNB move together. We compute the spread log(ETH) − beta · log(BNB) with a rolling
30-day hedge ratio and turn it into a z-score (2-week rolling mean and std).
- z < −2.5: buy ETH, short BNB. z > +2.5: the opposite.
- Close when z is back at 0.
- Stop loss at |z| > 3.5, time stop after 7 days.
- Costs: 0.05% fee + 0.02% slippage per leg, funding included.

| | Train (2023–2025) | Test (Jan–Oct 2026) |
|---|---|---|
| Sharpe | 0.82 | 0.33 |
| Total return | +44% | +2.6% |
| Max drawdown | −17% | −9% |
| Trades | 101 | 26 |

## Files

| File | What it does |
|---|---|
| `config.yaml` | All settings (data period, costs, strategy parameters) |
| `src/data.py` | Download and load hourly prices + funding rates |
| `src/strategy.py` | Pair selection, spread/z-score, trading rules, backtest, metrics |
| `notebooks/pairs_trading.ipynb` | The whole project top to bottom: data → pair → calibration → train → test |
| `tests/test_strategy.py` | Checks of the trading rules |

## How to run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.data          # download data into data/ (only needed once)
jupyter notebook notebooks/pairs_trading.ipynb
python -m pytest            # run the tests
```

Parameters were chosen on train data only; the test period was run once with the final parameters.
