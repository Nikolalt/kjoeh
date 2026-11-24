# The whole project, explained from zero

This document assumes you know nothing about markets, crypto or statistics beyond school maths. It goes in the
same order as the notebook `notebooks/pairs_trading.ipynb`. Each part explains the idea first, then shows where
it happens in the code.

**Contents**
0. **Interview answers: the 11 questions we may be asked, answered briefly** (read this last, as a summary)
1. Markets basics: prices, returns, long, short
2. Crypto and perpetual futures (and why we need funding)
3. The idea of pairs trading
4. Statistics you need: regression, stationarity, ADF test, p-values
5. Choosing the pair: Engle-Granger, multiple testing, half-life
6. The spread and the hedge ratio
7. The z-score: why it is calculated like that
8. Trading rules: entry, exit, stops, trading frequency
9. The backtest: how we compute profit and loss
10. Costs: fees, slippage, funding
11. Metrics: Sharpe, drawdown and friends
12. Train / test split and overfitting
13. Our results, and what they mean
14. Going through the trades (with real numbers)
15. Challenges
16. What we learned
17. Glossary

**Which code to look at before the Q&A** (in `src/strategy.py`, about 230 lines in total):
- `rank_pairs`: pair selection (section 5)
- `spread_and_zscore`: 15 lines, the indicators (sections 6–7)
- `positions`: the trading rules, one loop (section 8)
- `backtest`: profit and loss, about 40 lines (section 9)

The notebook `notebooks/pairs_trading.ipynb` shows all tables and charts mentioned here.

---

## 0. Interview answers

These are the questions from the interview brief, each with a short answer and where to read more. The grading
gives 10% each to strategy design (the idea, **not** the PnL), risk management, documentation and our individual
answers. A weak PnL explained honestly is fine; an idea we can't explain is not.

**1. Objective, and how the strategy makes money**
Market-neutral statistical arbitrage. ETH and BNB usually move together. When the gap between them (the spread) is
unusually large, we buy the relatively cheap coin and short the relatively expensive one, and profit when the gap
closes, whether the market goes up or down. → sections 3, 6, 7

**2. Which products, and why**
Binance USDT-margined perpetual futures on ETH and BNB.
- Perpetuals, because we must short one leg, and that is easy and cheap with perps but awkward on spot.
- Binance, because it is the most liquid exchange: low fees, tight prices, long clean history.
- ETH/BNB: the native coins of the two largest smart-contract platforms, so they have an economic link. Return
  correlation 0.71, neither coin is stationary on its own, and the best region in our train calibration with lower
  drawdowns than the alternative.
→ sections 2, 5

**3. Trading frequency and time horizon**
Hourly data and decisions. About 3 trades per month (101 trades in 3 years of train, 26 in the 10 test months). A
trade lasts 2–3 days on average (median 58 h on train), at most 7 days because of the time stop. We are in a trade
about 25% of the time. It's a short-term swing strategy, not high-frequency. → section 8

**4. Data features and indicators**
- Data: hourly close prices and funding rates for 12 coins, 2023-01 to 2026-10 (33,600 hours, no gaps).
- Indicators, all rolling and backward-looking:
  - hedge ratio beta over 30 days
  - spread = log(ETH) − beta·log(BNB) − alpha
  - z-score of the spread over 2 weeks
- For pair selection: return correlation, Engle-Granger p-value, ADF p-value of each coin alone, half-life.
→ sections 5, 6, 7

**5. How entry and exit signals are generated**
- Entry: z < −2.5 → long the spread (buy ETH, short BNB); z > +2.5 → short the spread. No entry if |z| > 3.5.
- Exit:
  - take profit when z is back at 0
  - stop loss at |z| > 3.5
  - time stop after 168 h
- After a stop we wait until |z| < 2.5 before trading again.
- Decisions happen at each hourly close, and we trade at that close price.
→ section 8, code `positions()`

**6. Position sizing and management**
All capital, no leverage, split in the ratio 1 : beta. With beta = 0.8: 56% long ETH and 44% short BNB. This makes
the position neutral to common moves of both coins. Beta is frozen at entry for the whole trade. Only one trade at a
time; we never add to a position. → section 9

**7. Risk management**
- **Hedging:** long one coin, short the other, so market-wide moves mostly cancel (market neutral).
- **Stop loss** at |z| > 3.5: limits losses when the relationship breaks. On train it cut the max drawdown from −27% to −16%.
- **Time stop** after 7 days: capital is never stuck in a dead trade. Honest note: on train it lowered the Sharpe.
- **Take profit** at z = 0: we take the profit as soon as the gap is closed instead of hoping for more.
- **Cooldown after a stop:** no immediate re-entry into the same runaway move.
- **No entry beyond the stop level**, no leverage, one trade at a time.
- **Realistic costs:** fees, slippage and funding are included, so the risk isn't hidden by optimistic assumptions.
→ sections 8, 12

**8. Back-testing approach**
- Bar-by-bar simulation on hourly data.
- No lookahead: the position decided at the close of hour t earns only the return of hour t+1, and all indicators
  use only past data.
- Fees 0.05%, slippage 0.02% per leg, and real funding payments.
- Train 2023–2025 for every decision, test Jan–Oct 2026 run once at the end.
- Robustness checks: double fees, no funding.
- Unit tests check the trading rules.
→ sections 9, 10, 12

**9. The PnL results**
- Train: +44% in 3 years, Sharpe 0.82, max drawdown −17%.
- Test: +2.6% in 10 months, Sharpe 0.33, max drawdown −9%.
- The test equity peaked at about +9% in June 2026, then gave most of it back through stop losses.
- Gross profit is reduced a lot by costs (a quarter on train, half on test). Funding is negligible.
- The drop from train to test was expected: the chosen parameters were an isolated peak (overfitting risk).
→ sections 13, 14

**10. Go through the trade transactions**
Take-profit trades make the money (+2.0% on average, 92% win rate on train); stop losses cost −1.8% on average.
Know the best trade (+7.0%, Jan 2024) and the worst trade (−7.2%, Sep 2025). In both, beta was only about 0.1, so
they were mostly ETH bets. Know the example trade (+3.1%, Mar 2023) as well. → section 14 and the notebook charts

**11. What we learned**
→ section 16. In short:
- correlation is not cointegration
- most "significant" pairs are luck or fake
- costs decide whether an edge survives
- a good backtest on train is not proof
- a rolling hedge ratio needs safeguards
- honest out-of-sample testing matters more than a high Sharpe

**What makes the design ours (strategy design and creativity marks):**
- an intercept in the rolling spread, so beta changes don't make the spread jump
- beta frozen at entry
- realistic funding modelling for perps, with correct timing
- the single-leg stationarity check that exposed the fake LTC pairs
- the Bonferroni correction for testing 66 pairs
- a cooldown after stops
- strict train/test discipline

---

## 1. Markets basics

### Price and return
A **price** is what you pay for one unit of something, e.g. 1 ETH = 1,400 USDT.
A **return** is how much the price changed, in percent:

```
return = new price / old price − 1
1,400 → 1,456  gives a return of 1,456 / 1,400 − 1 = +4%
```

We care about returns, not price changes in dollars. Gaining $10 on a $100 coin is a lot; gaining $10 on a $50,000 coin is nothing.

### Long and short
- **Long** = you buy. You profit when the price goes **up**.
- **Short** = you borrow the coin and sell it now, then buy it back later to return it. You profit when the price goes **down**.
  Sold at 100, bought back at 90 → +10.

Pairs trading always does both at the same time: long one coin, short another.

### Log prices
We often use the natural logarithm of the price, `log(price)`, for three reasons:
1. A difference of log prices is (almost exactly) a percentage return: `log(1,456) − log(1,400) ≈ 0.039 ≈ 3.9%`.
2. Equal percentage moves look equal. On a log chart, 100 → 200 and 1,000 → 2,000 are the same height.
3. A regression on log prices gives a hedge ratio in percent terms (section 6).

Typical values in our data: log(ETH price) ≈ 7.1 to 8.5, log(BNB price) ≈ 5.3 to 7.2.

### Hourly candles
The exchange summarises trading in **candles**: for each hour, the open, high, low and close price and the volume.
We only use the **close**. A candle is labelled by its start time: the candle "07:00" covers 07:00–08:00, and its
close is the price at 08:00. That matters for the timing in the backtest (section 9).

---

## 2. Crypto and perpetual futures

### The coins
We use 12 large, liquid coins (BTC, ETH, SOL, BNB, XRP, ADA, AVAX, LINK, DOT, LTC, DOGE, TRX), all priced in
**USDT**. USDT is a "stablecoin" worth about 1 US dollar. "Liquid" means lots of trading, so we can buy and sell
quickly without moving the price.

Our pair:
- **ETH** (Ether): the coin of Ethereum, the biggest "smart-contract" blockchain (a platform for running programs and apps).
- **BNB**: the coin of BNB Chain, a competing smart-contract blockchain run by the Binance ecosystem.

Both are bets on the same industry, so their prices tend to move together. That is the economic reason we
expect a relationship between them.

### Why perpetual futures and not the coins themselves?
To short a coin you must borrow it, which is awkward and expensive on normal ("spot") crypto markets. Exchanges
offer **perpetual futures** ("perps") instead. A perp is a contract that tracks the coin's price, and you can go
long or short with one click. Profit and loss is settled in USDT. That is why we trade Binance USDⓈ-M perpetuals
(USDⓈ-M = margined in USDT).

### Funding: why it exists
A normal futures contract has an expiry date, and on that date it is settled at the real coin price, so it can't
drift away from it. A **perpetual never expires**. So what stops its price from drifting away from the real coin
price? The **funding rate**:

- Every 8 hours (00:00, 08:00, 16:00 UTC) the two sides of the contract pay each other a small amount.
- If the perp trades **above** the spot price (too many buyers), the rate is **positive**: **longs pay shorts**.
  That makes being long more expensive and pulls the price back down.
- If it trades **below**, the rate is **negative**: **shorts pay longs**.
- Typical size: 0.01% per 8 hours. Small, but about 11% per year if it stays like that.

### Why we need funding in the backtest
Because we really hold perps, funding is a real cash flow that hits our account. Leaving it out would make the
results too optimistic (or too pessimistic). Over our train period the average funding was:
- ETH: **+8.7% per year**, so being long ETH costs money.
- BNB: **−4.6% per year** (negative), so being short BNB *also* costs money, because shorts pay longs.

So a "long the spread" trade (long ETH, short BNB) pays funding on both legs. A "short the spread" trade (short ETH,
long BNB) receives on both. On train we had 51 long-spread and 50 short-spread trades: the long ones paid 3.5% in
funding and the short ones received 3.3%. In total funding came out at about −0.1%, which is small, but we only
know that because we included it.

Code: funding is loaded in `data.load_data` and applied in `strategy.backtest` (the `fund = ...` line).

---

## 3. The idea of pairs trading

Imagine two coins that usually move together, like two ships tied with an elastic rope. Sometimes one ship drifts
ahead, but the rope pulls them back together.

- When ETH is **unusually expensive** compared with BNB: **short ETH, buy BNB**. When the gap closes, we earn money
  whether both rise or both fall.
- When ETH is **unusually cheap** compared with BNB: **buy ETH, short BNB**.

This is called **statistical arbitrage**, or "stat arb". It is **market neutral**: if the whole crypto market
crashes 20%, both coins fall, and our long loss is roughly offset by our short gain. We only bet on the
*relative* price.

To turn this into a strategy we have to answer four questions:
1. Which two coins? → **pair selection** (sections 4–5)
2. How do we measure "the gap"? → **spread with a hedge ratio** (section 6)
3. When is the gap "unusually" large? → **z-score** (section 7)
4. When do we get out, including when it goes wrong? → **trading rules** (section 8)

---

## 4. The statistics you need

### Linear regression (OLS)
You have points (x, y) and you fit the best straight line `y = alpha + beta · x`. "Best" means the sum of squared
vertical distances from the points to the line is as small as possible. That is called Ordinary Least Squares (OLS).
- **beta** = slope: if x goes up by 1, y goes up by beta on average.
- **alpha** = intercept: where the line crosses x = 0.
- **residual** = actual y − value on the line, i.e. how far each point is from the line.

A useful formula: `beta = cov(x, y) / var(x)`. Covariance measures how much x and y move together; variance
measures how much x moves on its own. Our code uses this formula for the rolling beta.

### Stationary vs non-stationary
This is the most important concept in the project.
- A **stationary** series keeps coming back to the same average and has a stable spread around it. Example: the
  temperature in January in Hong Kong. It wiggles, but it doesn't wander off to 50 °C.
- A **non-stationary** series can wander anywhere and has no average it returns to. Example: a coin price. It went
  from 1,200 to 4,800 and nothing pulls it back. The classic model is a **random walk**: each step is random, so
  the best guess for tomorrow is today's value.

Prices are non-stationary. You can't trade "the price will return to its mean" on one price, because there is no
mean to return to. **But** a combination of two prices can be stationary, and that is what we look for.

The jargon: a non-stationary price whose changes (returns) are stationary is called **I(1)**, "integrated of order 1".
A stationary series is **I(0)**.

### Hypothesis tests and p-values
A statistical test starts from a **null hypothesis** (the "boring" assumption) and asks: *if the boring assumption
were true, how likely would data this extreme be?* That probability is the **p-value**.
- Small p-value (e.g. < 0.05): the data would be very unusual under the boring assumption, so we **reject** it.
- Large p-value: the data is compatible with the boring assumption, and we can't reject it.

A p-value of 0.05 means: even if nothing is going on, you would see a result like this 5% of the time by luck.

### The ADF test (Augmented Dickey-Fuller)
It tests whether a series is stationary.
- Null hypothesis: the series is a random walk (**non**-stationary).
- Idea: regress the change of the series on its previous level: `change(t) = c + phi · level(t−1) + ...`.
  If phi is clearly negative, then whenever the level is high the next change tends to be down, and vice versa.
  So the series is pulled back to a mean, i.e. it is stationary.
- "Augmented" means it also includes past changes in the regression, to soak up short-term patterns. We allow up
  to 24 of them (`maxlag=24`, one day of hourly data).
- Low p-value → reject "random walk" → stationary.

---

## 5. Choosing the pair (`strategy.rank_pairs`)

### Correlation is not enough
**Correlation** (−1 to +1) measures whether two series move together. We compute it on **hourly returns**. Not on
prices: any two prices that both trended upward look highly "correlated" even if they are unrelated. That is called
spurious correlation.

But even a high return correlation doesn't mean the prices stay together. Two coins can move in the same direction
every hour, with one moving a bit more each time, and drift apart forever. Correlation is about *short-term
co-movement*; we need a *long-term tie*. That tie is **cointegration**.

### Cointegration
Two non-stationary prices A and B are **cointegrated** if some combination `log(A) − beta · log(B)` **is stationary**.
In plain words: each price wanders around freely, but they wander *together*, and the gap between them keeps coming
back. That gap is exactly what we want to trade.

### The Engle-Granger test (the method we use)
A two-step recipe, named after the two economists who got a Nobel prize partly for it:
1. **Regression:** `log(A) = alpha + beta · log(B) + residual`. The residual is the spread.
2. **ADF test on the residual:** is the spread stationary? A low p-value means cointegrated.

One subtlety: in step 1, OLS chooses beta to make the residual as small and as stationary-looking as possible. So a
normal ADF test would be too optimistic. Engle-Granger therefore uses stricter critical values (MacKinnon's), so
the p-value is fair. `statsmodels.coint()` does all of this in one call.

### The checks we add
In `rank_pairs` every pair gets these columns:

| Column | Meaning | What we want |
|---|---|---|
| `correlation` | correlation of hourly returns | high (> 0.6) |
| `p_value` | Engle-Granger p-value | low |
| `p_bonferroni` | p-value × 66 (see below) | < 0.05 would be convincing |
| `beta` | hedge ratio from the regression | positive, sensible |
| `half_life_h` | hours for the spread to close half the gap | not too long |
| `adf_p_a`, `adf_p_b` | ADF p-value of each coin **alone** | **high** (each coin should be a random walk) |

**Multiple testing (Bonferroni).** 12 coins give 12·11/2 = 66 pairs, so we run 66 tests. With a 5% threshold, about
66 × 0.05 ≈ 3 pairs would look "significant" **by pure luck**, even if no pair were really cointegrated. It's like
flipping 66 coins: some will land heads five times in a row. The Bonferroni correction multiplies each p-value by
the number of tests. It is crude and conservative, but it makes the point: our best p-value of 0.014 becomes 0.91,
i.e. not significant at all.

**Single-leg stationarity check.** Engle-Granger assumes both coins are random walks on their own. If one coin
alone is already stationary (e.g. it moved sideways in a range the whole time), the regression finds a tiny beta,
the "spread" is basically that one coin, and the test says "cointegrated" although the other coin adds nothing.
In our data, **LTC** alone has an ADF p-value of 0.01, so it is stationary on its own, and the top two pairs
(LTC/DOGE, LTC/TRX) are fake for this reason.

**Half-life** (`strategy.half_life`). It answers: if the spread is away from its mean, how long until half the gap
is gone? We fit `change(t) = c + phi · spread(t−1)` (the same regression as in ADF). Each hour, a share phi of the
gap disappears, so after n hours `(1 + phi)^n` of it is left. Setting that to ½ gives

```
half-life = ln(2) / −ln(1 + phi)
```

Example: phi = −0.1 → 10% of the gap closes per hour → half-life ≈ 6.6 hours. If phi ≥ 0 there is no pull back, so
the half-life is infinite. Our pairs have half-lives of weeks over the full train period, which tells us the
relationship is slow and loose.

### What we found, and how we chose
- No pair is significantly cointegrated after the correction.
- So the statistics alone don't pick a winner. We then **backtested a few liquid, economically related candidates**
  on train data (section 12) and picked **ETH/BNB**: same industry, return correlation 0.71, neither leg stationary
  alone, the best region of train results, and smaller drawdowns than the alternative ADA/DOGE.

---

## 6. The spread and the hedge ratio (`strategy.spread_and_zscore`)

### Why not just log(ETH) − log(BNB)?
Because the two coins don't move one-for-one. If BNB typically moves 1% and ETH moves 0.8% at the same time, you
need a ratio. That ratio is the **hedge ratio beta**, the slope of the regression of log(ETH) on log(BNB):

```
log(ETH) ≈ alpha + beta · log(BNB)
spread   = log(ETH) − beta · log(BNB) − alpha
```

Because we use logs, beta means: "BNB +1% → ETH +beta% on average".

### Why rolling?
The relationship isn't constant: over three years the full-period regression doesn't even pass the cointegration
test. So we **re-estimate beta every hour using only the last 720 hours (30 days)**. This is a **rolling window**:
at each hour, look back 30 days, compute, move on. It uses only past data, so there is no peeking into the future.

```python
beta  = log_a.rolling(720).cov(log_b) / log_b.rolling(720).var()       # slope = cov / var
alpha = log_a.rolling(720).mean() - beta * log_b.rolling(720).mean()   # intercept
```

(These are exactly the OLS formulas, computed for every 30-day window at once.)

Our rolling beta is about 1 on average, but it moves a lot (25% of the time below 0.62, 25% above 1.37). That is a
weakness worth knowing: the relationship is unstable.

### Why the alpha (intercept)?
log(BNB) is about 6. If beta changes from 1.00 to 1.01 between two hours, `beta · log(BNB)` changes by 0.06. That's
a 6% jump in the spread from a tiny change in the estimate, not from the market. Subtracting the matching rolling
alpha keeps the spread centred around 0, so it only moves when the prices really move relative to each other.

---

## 7. The z-score (`strategy.spread_and_zscore`)

```python
z = (spread − rolling mean of spread) / rolling standard deviation of spread    # over the last 336 hours
```

### Why do we need it at all?
The spread is a number like 0.03 or −0.05. Is 0.03 "a lot"? That depends on how much the spread normally moves.
In a calm month 0.03 could be extreme; in a wild month it could be nothing. The z-score translates the spread into
**"how many typical deviations are we away from normal?"**:
- z = 0: the spread is at its normal level.
- z = +2.5: the spread is 2.5 standard deviations **above** normal, so ETH is unusually expensive compared with BNB.
- z = −2.5: ETH is unusually cheap.

That gives us one rule that works in calm and wild periods alike: "trade when |z| > 2.5".

For a rough feel: if the spread were normally distributed, |z| > 2.5 would happen only about 1.2% of the time.
Crypto has fat tails (extreme moves happen more often than a normal distribution says), so in practice it is more often.

### Why subtract a rolling mean, and why 336 hours?
- **Rolling mean:** the "normal level" of the spread drifts over time, so we compare it with its own recent past
  (the last 2 weeks), not with a fixed number.
- **Rolling standard deviation:** volatility changes too, so "a typical deviation" is measured over the same 2 weeks.
- **Why 2 weeks and not 1 day?** The window must be **long compared with how long a deviation takes to revert**.
  If it's too short, the mean simply follows the spread: when the spread moves away, the mean follows it within
  hours, z never gets large, and we trade noise and pay costs. If it's too long, the mean is stale and z stays on one
  side for weeks. In calibration, short windows (168 h) lost money for almost every pair; 336 h worked best.

### No lookahead
Every rolling value at hour t uses only the hours up to and including t. The z-score at 14:00 is known at 14:00.

---

## 8. Trading rules (`strategy.positions`)

The function walks through the z-score **hour by hour** and keeps a state: flat (0), long the spread (+1) or short
the spread (−1). At the close of each hour it decides:

**When flat (entry):**
- z < −2.5 → **+1, long the spread** = buy ETH, short BNB (ETH is cheap, so we expect the spread to rise back to 0).
- z > +2.5 → **−1, short the spread** = short ETH, buy BNB.
- But not if |z| > 3.5 already, because we would be stopped out at once and only pay fees.

**When in a trade (exit), checked in this order:**
1. **Stop loss**, |z| > 3.5: the gap keeps widening. Maybe the relationship broke (e.g. news about one coin).
   We cut the loss instead of hoping.
2. **Time stop**, open for 168 hours (one week): the gap isn't closing in a reasonable time. Capital stuck in a
   dead trade can't be used elsewhere.
3. **Take profit**, z is back at the mean (exit_z = 0): long trades close when z ≥ 0, short trades when z ≤ 0.
   This is the normal, winning exit.

**Cooldown after a stop.** After a stop or time stop we don't trade again until |z| < 2.5. Otherwise we would
re-enter the same runaway move on the next hour and lose again.

At the very end of the data, any open trade is closed (`end_of_data`) so its costs are counted.

Why these numbers (2.5, 0, 3.5, 168)? They were chosen on train data in the calibration (section 12). The general
trade-off: a lower entry threshold gives more trades, each with less profit and the same costs; a higher one gives
fewer but better trades, and too few to judge.

### Trading frequency and time horizon

| | Train (36 months) | Test (10 months) |
|---|---|---|
| Trades | 101 | 26 |
| Trades per month | 2.8 | 2.6 |
| Median time in a trade | 58 h (~2.5 days) | 46 h (~2 days) |
| Longest trade | 168 h (time stop) | 168 h (time stop) |
| Share of time in a trade | 27% | 24% |

So it's a **short-term swing strategy**: we look at the market every hour, but we trade only a few times a month
and hold for days. Why not more often? Each round trip costs ~0.14%, so frequent small trades would be eaten by
costs. Why not longer? The spread's relationship isn't stable over months, and funding costs pile up.

---

## 9. The backtest (`strategy.backtest`)

A **backtest** simulates "what would have happened if we had traded these rules in the past".

### How much of each coin? (position sizing)
We always use all our capital, split between the two legs in the ratio 1 : beta:

```
weight ETH =  1    / (1 + |beta|)      (long, for position +1)
weight BNB = −beta / (1 + |beta|)      (short)
```

With beta = 0.8: 56% of capital long ETH and 44% short BNB. Why 1 : beta? If BNB moves 1%, ETH moves about beta%.
So $1 of ETH changes by $beta·1%, and $beta of BNB also changes by $beta·1%, which cancels exactly. That is what
"hedged" means: the common movement cancels and only the *relative* movement (the spread) is left.

We **freeze beta at entry** for the whole trade. We bought fixed amounts at the start, and the hedge doesn't
change just because the rolling estimate changes.

### Timing: no lookahead
The most common way to cheat by accident in a backtest is to use information you didn't have yet. Our rule:
**the position decided at the close of hour t earns the return of hour t+1.** In code:

```python
held = pos.shift(1)    # the position we actually hold DURING each hour = decided one hour earlier
```

Example: z crosses −2.5 at the 14:00 close. We buy at that close price, and our first profit or loss comes from the
next hour's price move. We never earn the move that caused the signal.

### Profit and loss per hour
For each hour t (as fractions of our capital):

```
gross   = held × (weight_ETH × return_ETH + weight_BNB × return_BNB)
costs   = (fee + slippage) × |change of position|           only in hours where we trade
funding = −held × (weight_ETH × rate_ETH + weight_BNB × rate_BNB)
net     = gross − costs + funding
equity  = (1 + net_1) × (1 + net_2) × ...                 start = 1, i.e. 100% of our capital
```

The funding sign: if the rate is positive and we are long (positive weight), we pay, so there is a minus sign. If we
are short (negative weight), the minus signs cancel and we receive.

Funding timing: the payment stamped 08:00 is paid by whoever holds a position at 08:00, i.e. the position held during
the 07:00–08:00 candle. That's why the code uses `funding.shift(-1)`.

Equity **compounds**: profits are reinvested, so +1% then +1% gives +2.01%.

### Where the z-score is computed
The z-score is computed on the **full history** first, then cut to the train or test period. That isn't cheating:
rolling values only look backwards. It means the test period (starting 1 Jan 2026) begins with a z-score that is
already "warmed up" from December 2025 data, as it would be in real life.

### Trades table (`strategy.list_trades`)
It groups the hours into trades: entry time, exit time, direction, z at entry and exit, return of the trade
(compounded net, including both trading costs), and why it was closed.

---

## 10. Costs

Every trade costs money. Ignoring this is the most common reason backtests look great and real trading doesn't.

- **Fee: 0.05% per leg.** Binance charges this on the traded amount when you take an existing offer (a "taker" order).
- **Slippage: 0.02% per leg.** You don't get exactly the last price: buying pushes the price up a bit and there is a
  gap between the buy and sell price (the "spread" of the order book, not our spread). We assume a fixed 0.02%.
- One trade = open both legs, close both legs. Because we trade all our capital, opening costs
  (0.05% + 0.02%) × 100% of capital = 0.07%, and closing costs the same: **about 0.14% per round trip.**

So a trade must earn more than 0.14% just to break even. On train, costs ate about a quarter of the gross profit; on
test about half.

---

## 11. Metrics (`strategy.metrics`)

| Metric | What it means |
|---|---|
| **total_return** | equity at the end − 1. 0.44 = +44% |
| **annual_return** | the return per year that would compound to the total: `equity^(1/years) − 1` |
| **sharpe** | return per unit of risk: `mean(hourly return) / std(hourly return) × √8760`. Rough guide: < 0.5 weak, ~1 decent, > 2 very good (and suspicious in a backtest) |
| **max_drawdown** | worst fall from a previous peak. −0.17 = at some point we were 17% below our best level so far. It measures the pain |
| **n_trades** | number of round trips. Fewer than ~30 is too few to trust the other numbers |
| **win_rate** | share of trades that made money |
| **avg_trade_return** | average return per trade, after costs |
| **avg_hours_in_trade** | how long a trade lasts on average |
| **time_in_market** | share of hours with an open position |
| **sum_gross / sum_costs / sum_funding** | where the profit came from and went to |

Why × √8760 in the Sharpe? Crypto trades 24/7, so a year has 24 × 365 = 8,760 hours. Risk (standard deviation)
grows with the square root of time, return grows linearly, so the ratio scales with √(hours per year).

---

## 12. Train / test split and overfitting

### The split
- **Train**: 2023-01-01 to 2025-12-31 (3 years). Used for every decision: pair, parameters, rules.
- **Test**: 2026-01-01 to 2026-10-31 (10 months). Used **once**, at the very end, with everything fixed.

Why? If you try many parameter combinations on the same data and keep the best, the "best" is partly luck: it fits
the noise of that particular period. This is called **overfitting**. The test period is data the strategy has never
seen, so it shows what you could actually expect. If you change parameters after looking at the test, it is no
longer a fair test.

### Calibration (notebook section 4)
We backtested a grid on train: 6 candidate pairs × 3 hedge windows × 3 z-score lookbacks × 3 entry thresholds = 162
backtests. What we look for is a **plateau, not a peak**: a region where a cell *and its neighbours* do well. A
real effect shouldn't vanish when you change a window from 720 to 1,080 hours.

What we found: most combinations lose money. ETH/BNB at 720 h / 336 h / 2.5 has a Sharpe of 0.82, but its
neighbours are much worse. That's a peak, a warning sign for overfitting, and it predicted that the test would be
weaker.

### Risk rules
- Stop loss: max drawdown −27% → −16%, and Sharpe goes up. Clearly useful.
- Time stop: on train, the Sharpe is *lower* with it (0.82 vs 1.10 without). We kept it as a safety rule. A strict
  train-only decision would have dropped it. We did not change it after seeing the test.

---

## 13. Our results, and what they mean

| | Train (3 years) | Test (10 months) |
|---|---|---|
| Sharpe | 0.82 | 0.33 |
| Total return | +44% | +2.6% |
| Max drawdown | −17% | −9% |
| Trades | 101 | 26 |
| Win rate | 57% | 50% |

How to read it:
- The strategy **stays slightly positive out of sample**, but much weaker than in sample, as the overfitting warning predicted.
- With 26 trades, a Sharpe of 0.33 is **not statistically distinguishable from zero**. We can't claim a proven edge.
- **Costs matter:** with double fees the test return is 0.
- **Funding barely matters** (about −0.1%), because long-spread and short-spread trades pay and receive in opposite directions and roughly cancel.
- **Big picture:** crypto pairs are not reliably cointegrated, so classic pairs trading on one pair has a small, fragile edge.

What would improve it: trade many pairs at once (more trades, more diversification, more reliable statistics),
re-calibrate regularly on recent data (walk-forward), use cheaper maker orders, and stop trading when the
relationship breaks down.

### Simplifications we made (know these for the Q&A)
- Leg weights are held constant during a trade. In reality we hold fixed coin amounts, so the weights drift slightly.
- Fixed slippage, taker fees, no leverage or margin calls.
- Only one pair.
- Exits use the rolling z-score, not a z-score recomputed with the frozen entry beta.

---

## 14. Going through the trades

### 14a. A complete example trade (real numbers, train period)

| Time (UTC, candle label) | ETH price | BNB price | beta | z | Position |
|---|---|---|---|---|---|
| 2023-03-10 13:00 | 1,423.83 | 274.11 | 0.788 | −2.48 | 0 (just above −2.5, no entry) |
| 2023-03-10 14:00 | 1,400.76 | 271.66 | 0.794 | **−2.86** | **+1 → enter** |
| 2023-03-10 15:00 | 1,416.42 | 274.35 | 0.800 | −2.64 | +1 (holding) |
| ... 51 hours later ... | | | | | |
| 2023-03-12 17:00 | | | | **+0.59** | **0 → take profit** |

What happened:
1. At the 14:00 close, z fell to −2.86, below −2.5: ETH was unusually cheap compared with BNB.
2. We went **long the spread** with frozen beta 0.794: 1/1.794 = **56% of capital long ETH**, 0.794/1.794 =
   **44% short BNB**. We paid 0.07% in costs.
3. Over the next 51 hours ETH recovered relative to BNB, and the spread moved back up.
4. At the 17:00 close on 12 March, z had crossed 0 (it was +0.59), so the take-profit rule closed the trade. We paid
   another 0.07%.
5. Result: **+3.06%** on our capital, after costs and funding.

Not every trade goes like this. On 2023-03-03 a long trade was entered at z = −2.73, the gap kept widening, and the
stop loss closed it at z = −4.40 for −4.1%.

### 14b. How the trades ended (train)

| Exit reason | Trades | Avg return per trade | Win rate |
|---|---|---|---|
| take profit | 53 | +2.0% | 92% |
| stop loss | 31 | −1.8% | 0% |
| time stop | 16 | −0.5% | 50% |
| end of data | 1 | +0.3% | – |

The pattern of every mean-reversion strategy: about half the trades revert as hoped and make the money. About a
third go wrong and are cut by the stop loss. The rest are cut by the time stop at about break-even. The average win
(+2.0%) is only slightly bigger than the average loss (−1.8%), so the 57% win rate is what makes it profitable.
Long-spread and short-spread trades are balanced: 51 and 50.

### 14c. Best and worst trade (train) — the charts are in notebook section 5

**Best trade: +7.0%** (2 Jan 2024, 28 hours, short spread)
- z jumped to +3.18: ETH was very expensive compared with BNB.
- We shorted the spread: short ETH, long BNB.
- Over the next 28 hours ETH fell 7.4%, BNB only 1.0%. z fell back through 0 → take profit.

**Worst trade: −7.2%** (22 Sep 2025, 84 hours, long spread)
- z fell to −2.67: ETH cheap compared with BNB. We went long the spread: long ETH, short BNB.
- The spread recovered only halfway (z about −1.2) and never reached 0.
- On 25 Sep ETH fell again, z passed −3.5 → stop loss. Over the trade ETH fell 9.0%, BNB 6.2%.

**What both have in common (important):** the rolling beta at entry was very low, 0.10 and 0.16. With beta = 0.1 the
split is 1/1.1 = **91% in ETH and only 9% in BNB**. Both trades were therefore mostly **bets on ETH's direction**,
not hedged spread trades. That's why they were the biggest winner and the biggest loser. About 1 in 5 trades is
entered with beta < 0.5. Those trades are much riskier (std of trade return 3.5% vs 2.0%), though on average they
happened to make money. A beta filter (e.g. only trade when 0.5 < beta < 2) would make the strategy cleaner. We
didn't add it, because we would have been choosing it after looking at the results.

### 14d. The test trades (Jan–Oct 2026)

| Exit reason | Trades | Avg return per trade | Win rate |
|---|---|---|---|
| take profit | 13 | +1.2% | 85% |
| stop loss | 9 | −1.3% | 0% |
| time stop | 4 | −0.1% | 50% |

The story of the test period:
- **January to early June:** mostly take-profits. The equity rose to about **+9.3% on 1 June**. Best test trade
  (31 May, short spread): +4.2% in 29 hours.
- **June to October:** 8 of the last 10 trades were stop losses or time stops, and the equity fell back to +2.6%.
  The ETH/BNB relationship was less stable in that period.
- **Worst test trade** (5 Feb): entered at z = +3.48, just below the stop level of 3.5, and stopped out 15 hours
  later at z = 4.08 for −2.9%.
- **Stop losses come fast:** all 9 test stop losses came within 15 hours of entry, 5 of them within 3 hours, and
  mostly from entries just above 2.5. z often jumps by more than 1 within a few hours, so one strong move goes
  straight through the narrow band between entry (2.5) and stop (3.5).
- **Two take-profit trades lost money** (1 Apr −0.7%, 8 May −2.7%). How can that happen? The z-score is measured
  against a *rolling* mean. If the mean moves toward the spread, z returns to 0 although the spread itself
  hasn't come back. On top of that, the PnL uses the beta frozen at entry, while z uses today's beta.

All 26 test trades are listed in the notebook (section 6).

---

## 15. Challenges

1. **No pair is truly cointegrated.** None of 66 pairs survives the multiple-testing correction. The core
   assumption of pairs trading only holds loosely and for limited periods. Our answer: rolling beta and z-score.
2. **Fake pairs.** The two "best" pairs contained LTC, which is stationary on its own. We only found that by also
   testing each coin alone.
3. **Overfitting.** We tried 162 combinations on train. The chosen one is an isolated peak, not a plateau, and the
   test Sharpe dropped from 0.82 to 0.33.
4. **Costs.** A round trip costs ~0.14% of capital. Costs ate a quarter of the gross profit on train and half on
   test; with double fees the test return is zero.
5. **Unstable hedge ratio.** The rolling beta ranged from −0.5 to 4.1. When it is near 0, a "pair trade" turns into
   a directional ETH bet (see 14c).
6. **The z-score is a moving target.** It can return to 0 because the rolling mean moved, not the spread, so a
   take-profit can still lose money.
7. **Narrow band between entry and stop.** With entry at 2.5 and stop at 3.5, fast moves trigger a stop within hours.
8. **Timing without lookahead.** Candles are labelled by their start time but we trade at their close, and funding
   is stamped at the moment it is paid. Both had to be aligned carefully, so that a position never earns a move or
   pays funding from before it was opened.
9. **Few trades.** One pair gives only 26 test trades, too few to tell a small edge from luck.

---

## 16. What we learned

1. **Correlation is not cointegration.** Coins that move together hour by hour can still drift apart for months.
2. **Statistical tests need care.** With 66 tests some look significant by luck (multiple testing), and a test can
   be fooled when one coin is stationary on its own (the LTC pairs).
3. **Costs decide whether an edge survives.** A strategy that's profitable before costs can be worthless after them.
   Every assumption (fees, slippage, funding) matters.
4. **In-sample results are optimistic.** Picking the best of many combinations always looks good on train. Look for
   plateaus, not peaks, and keep a test period you only use once.
5. **Risk management is about the losses, not the average.** The stop loss didn't need to raise returns to be
   useful: it cut the max drawdown from −27% to −16%.
6. **Check what you are really trading.** A low beta silently turned some pair trades into directional bets.
   Looking at individual trades, not just summary numbers, revealed this.
7. **Honesty over a high Sharpe.** Our final result is modest: positive but weak out of sample. We can explain why,
   and what we would change.

**What we would do next:**
- trade many pairs as a portfolio (more trades, more diversification, more reliable statistics)
- re-calibrate regularly on recent data (walk-forward testing)
- a beta filter or a Kalman-filter hedge ratio
- a wider band between entry and stop
- maker orders to cut fees
- pause trading when the rolling cointegration breaks down

---

## 17. Glossary

| Term | Meaning |
|---|---|
| ADF test | Augmented Dickey-Fuller: tests whether a series is stationary |
| alpha | intercept of the regression; centres the spread |
| beta / hedge ratio | how many % ETH moves per 1% move in BNB; sets the size of the two legs |
| backtest | simulation of the strategy on past data |
| Bonferroni | multiply p-values by the number of tests, to correct for testing many pairs |
| candle | open/high/low/close/volume summary of one hour |
| cointegration | two wandering prices whose combination (the spread) does not wander |
| correlation | how much two series move together, −1 to +1 |
| drawdown | fall from the previous peak of the equity curve |
| Engle-Granger | regression + ADF test on the residual = cointegration test |
| equity | value of our account over time (start = 1) |
| fat tails | extreme moves happen more often than a normal distribution predicts |
| funding rate | payment between longs and shorts on a perpetual, every 8 h |
| half-life | time for the spread to close half its gap to the mean |
| I(0) / I(1) | stationary / random-walk-like (stationary after taking differences) |
| lookahead bias | accidentally using future information in a backtest |
| long / short | bet on price up / bet on price down |
| market neutral | profit doesn't depend on the overall market direction |
| OLS | ordinary least squares, the standard way to fit a regression line |
| overfitting | choosing parameters that fit the noise of past data |
| perpetual future (perp) | contract tracking a coin's price, no expiry, allows shorting |
| p-value | probability of data this extreme if the null hypothesis were true |
| rolling window | statistic computed over the last n hours, recomputed every hour |
| Sharpe ratio | return per unit of volatility, annualised |
| slippage | getting a worse price than expected when trading |
| spread | log(ETH) − beta · log(BNB) − alpha: the gap we trade |
| stationary | keeps returning to a stable mean with a stable spread around it |
| stop loss | closing a losing trade at a predefined level |
| taker fee | fee for an order that trades immediately against existing orders |
| train / test | data for decisions / data for the final honest check |
| USDT | stablecoin worth about 1 USD |
| z-score | how many standard deviations the spread is from its recent mean |
