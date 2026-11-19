"""All figures for the notebooks, in one consistent style.

Rules every function follows:
    - returns a matplotlib Figure (and closes it, so Jupyter shows it exactly once when it is the
      last line of a cell; use display(fig) to show several figures from one cell)
    - has a title and axis labels
    - save="name.png" also writes the figure to results/name.png
    - colours come from the palette below; it is colour-blind safe (validated categorical order),
      and markers differ by shape too, so colour is never the only cue
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from .backtest import Result

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# Categorical colours in a fixed order (colour-blind-safe sequence; use them in this order).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK = "#0b0b0b"          # titles, text
MUTED = "#898781"        # axes, secondary lines, "other" series
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

LEG_COLORS = {"a": PALETTE[0], "b": PALETTE[1]}
DIRECTION_STYLE = {"long": (PALETTE[0], "^"), "short": (PALETTE[1], "v")}
EXIT_STYLE = {                     # exit reason -> (colour, marker)
    "signal": (PALETTE[2], "o"),
    "z_stop": (PALETTE[7], "X"),
    "time_stop": (PALETTE[3], "s"),
    "coint_break": (PALETTE[6], "D"),
    "end_of_data": (MUTED, "P"),
}
COMPONENT_COLORS = {"gross": PALETTE[0], "fees": PALETTE[7], "slippage": PALETTE[1],
                    "funding": PALETTE[6], "net": INK}

# Diverging map for signed values (e.g. Sharpe): red = negative, grey = 0, blue = positive.
DIVERGING = LinearSegmentedColormap.from_list("red_grey_blue", ["#b52f2f", "#f0efec", "#1c5cab"])
SEQUENTIAL = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#2a78d6", "#0d366b"])

FIGSIZE = (12, 5)


def apply_style() -> None:
    """Set the shared matplotlib style; call once in each notebook's import cell."""
    plt.rcParams.update({
        "figure.figsize": FIGSIZE,
        "figure.dpi": 100,
        "savefig.dpi": 150,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.frameon": False,
        "lines.linewidth": 1.5,
        "font.size": 10,
    })


def _finish(fig: plt.Figure, save: str | None) -> plt.Figure:
    """Tidy the layout, optionally save to results/, close (so Jupyter shows it once) and return."""
    fig.tight_layout()
    if save:
        RESULTS_DIR.mkdir(exist_ok=True)
        fig.savefig(RESULTS_DIR / save, bbox_inches="tight")
    plt.close(fig)
    return fig


def _pair_label(result: Result) -> str:
    pair = result.params.get("pair")
    return f"{pair[0]}/{pair[1]}" if pair else "pair"


# =============================================================================
# DATA AND PAIR SELECTION
# =============================================================================

def plot_prices(
    prices: pd.DataFrame,
    highlight: list[str] | None = None,
    train_end: str | None = None,
    save: str | None = None,
) -> plt.Figure:
    """Prices rebased to 100 at the start, log scale. Up to 6 coins in colour, the rest in grey.

    Input:  price DataFrame, coins to highlight (default: the first 6 columns), optional train_end
            date (dashed line), save file name.
    Output: Figure.
    """
    highlight = list(highlight or prices.columns[:6])[:6]
    rebased = 100 * prices / prices.bfill().iloc[0]

    fig, ax = plt.subplots(figsize=(12, 6))
    for coin in rebased.columns:
        if coin not in highlight:
            ax.plot(rebased.index, rebased[coin], color=MUTED, lw=0.7, alpha=0.5)
    for i, coin in enumerate(highlight):
        ax.plot(rebased.index, rebased[coin], color=PALETTE[i], lw=1.3, label=coin)
    ax.plot([], [], color=MUTED, lw=0.7, label="other coins")
    if train_end:
        ax.axvline(pd.Timestamp(train_end, tz="UTC"), color=INK, ls="--", lw=1, label="train end")
    ax.set_yscale("log")
    ax.set_title("Perpetual prices, rebased to 100 (log scale)")
    ax.set_xlabel("Date (UTC)")
    ax.set_ylabel("Index (start = 100)")
    ax.legend(ncol=4, loc="upper left")
    return _finish(fig, save)


def plot_correlation_matrix(corr: pd.DataFrame, title: str | None = None, save: str | None = None) -> plt.Figure:
    """Correlation matrix as an annotated heatmap (single-hue scale: darker = more correlated).

    Input:  square correlation DataFrame, optional title, save file name.
    Output: Figure.
    """
    n = len(corr)
    fig, ax = plt.subplots(figsize=(0.65 * n + 2, 0.6 * n + 1))
    im = ax.imshow(corr.to_numpy(), cmap=SEQUENTIAL, vmin=max(0, corr.min().min()), vmax=1)
    for i in range(n):
        for j in range(n):
            value = corr.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if value > 0.75 else INK)
    ax.set_xticks(range(n), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(n), corr.index)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Correlation", shrink=0.8)
    ax.set_title(title or "Correlation of hourly log returns")
    ax.set_xlabel("Coin")
    ax.set_ylabel("Coin")
    return _finish(fig, save)


def plot_pair_ranking(
    ranking: pd.DataFrame, top_n: int = 10, n_tests: int | None = None, save: str | None = None
) -> plt.Figure:
    """Top pairs from strategy.rank_pairs: cointegration p-value (log scale) and stability side by side.

    Input:  output of rank_pairs, number of pairs to show, number of tests for the Bonferroni line
            (default: 2 x number of rows, as in rank_pairs), save file name.
    Output: Figure with two panels.
    """
    top = ranking.head(top_n).iloc[::-1]   # reverse so the best pair is at the top
    labels = top["a"] + "/" + top["b"]
    n_tests = n_tests or 2 * len(ranking)

    fig, (ax_p, ax_s) = plt.subplots(1, 2, figsize=(12, 0.45 * top_n + 1.5), sharey=True)
    ax_p.barh(labels, top["p_value"], color=PALETTE[0], height=0.6)
    ax_p.set_xscale("log")
    ax_p.axvline(0.05, color=INK, ls="--", lw=1, label="p = 0.05")
    ax_p.axvline(0.05 / n_tests, color=PALETTE[7], ls=":", lw=1.5, label=f"Bonferroni 0.05/{n_tests}")
    ax_p.set_title("Engle-Granger p-value (smaller = stronger)")
    ax_p.set_xlabel("p-value (log scale)")
    ax_p.set_ylabel("Pair (A/B)")
    ax_p.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18 - 1.5 / top_n), ncol=2)  # below the axis

    ax_s.barh(labels, top["stability"], color=PALETTE[2], height=0.6)
    for y, (stab, hl) in enumerate(zip(top["stability"], top["half_life"])):
        ax_s.text(stab + 0.01, y, f"half-life {hl / 24:.0f} d", va="center", fontsize=8, color=INK)
    ax_s.set_xlim(0, 1.15)
    ax_s.set_title("Stability: share of rolling windows with p < 0.05")
    ax_s.set_xlabel("Share of windows")
    return _finish(fig, save)


# =============================================================================
# SIGNALS AND TRADES
# =============================================================================

def _threshold_lines(ax, params: dict) -> None:
    """Horizontal lines at +-entry_z, +-exit_z and +-stop_z on a z-score axis."""
    ax.axhline(0, color=MUTED, lw=0.8)
    for level, style, label in [(params.get("entry_z"), "--", "entry"),
                                (params.get("exit_z"), ":", "exit"),
                                (params.get("stop_z"), "-.", "stop")]:
        if level is None or (label == "exit" and level == 0):
            continue
        ax.axhline(level, color=INK, ls=style, lw=0.9, label=f"±{label} ({level})")
        ax.axhline(-level, color=INK, ls=style, lw=0.9)


def plot_spread_zscore(result: Result, title: str | None = None, save: str | None = None) -> plt.Figure:
    """Spread (top) and z-score with entry/exit/stop thresholds (bottom), with trade markers.

    Entries are marked by direction (long ▲ / short ▼), exits by exit reason (colour and shape).

    Input:  Result from backtest.run_strategy, optional title, save file name.
    Output: Figure with two stacked panels sharing the time axis.
    """
    fig, (ax_s, ax_z) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, height_ratios=[1, 1.3])
    ax_s.plot(result.spread.index, result.spread, color=PALETTE[0], lw=0.8)
    ax_s.set_title(title or f"{_pair_label(result)}: spread and z-score")
    ax_s.set_ylabel("Spread (log)")

    z = result.z
    ax_z.plot(z.index, z, color=MUTED, lw=0.6, label="z-score")
    _threshold_lines(ax_z, result.params)
    trades = result.trades
    for direction, (color, marker) in DIRECTION_STYLE.items():
        t = trades[trades["direction"] == direction]
        ax_z.scatter(t["entry_time"], t["entry_z"], color=color, marker=marker, s=40, zorder=3,
                     edgecolor=SURFACE, linewidth=0.8, label=f"entry {direction}")
    for reason, (color, marker) in EXIT_STYLE.items():
        t = trades[trades["exit_reason"] == reason]
        if len(t):
            ax_z.scatter(t["exit_time"], t["exit_z"], color=color, marker=marker, s=40, zorder=3,
                         edgecolor=SURFACE, linewidth=0.8, label=f"exit: {reason}")
    ax_z.set_ylabel("z-score")
    ax_z.set_xlabel("Date (UTC)")
    ax_z.legend(ncol=4, fontsize=8, loc="upper left")
    return _finish(fig, save)


def plot_trade(
    result: Result, trade: int | pd.Series, padding: int = 48, title: str | None = None, save: str | None = None
) -> plt.Figure:
    """Zoom into one trade: both legs (rebased to 100 at entry) and the z-score, with entry and exit.

    Input:  Result from run_strategy, a row of result.trades or its integer position, bars of
            context before entry and after exit, optional title, save file name.
    Output: Figure with two stacked panels.
    """
    if not isinstance(trade, pd.Series):
        trade = result.trades.iloc[trade]
    index = result.z.index
    i0 = max(index.get_loc(trade["entry_time"]) - padding, 0)
    i1 = min(index.get_loc(trade["exit_time"]) + padding, len(index) - 1)
    window = index[i0:i1 + 1]

    fig, (ax_p, ax_z) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    prices = result.prices.loc[window]
    for (coin, series), leg in zip(prices.items(), ["a", "b"]):
        ax_p.plot(window, 100 * series / series.loc[trade["entry_time"]], color=LEG_COLORS[leg],
                  label=f"{coin} ({'A' if leg == 'a' else 'B'})")
    ax_z.plot(window, result.z.loc[window], color=MUTED, lw=1, label="z-score")
    _threshold_lines(ax_z, result.params)

    exit_color, exit_marker = EXIT_STYLE.get(trade["exit_reason"], (MUTED, "o"))
    entry_color, entry_marker = DIRECTION_STYLE[trade["direction"]]
    for ax in (ax_p, ax_z):
        ax.axvspan(trade["entry_time"], trade["exit_time"], color=PALETTE[0], alpha=0.08)
    ax_z.scatter([trade["entry_time"]], [trade["entry_z"]], color=entry_color, marker=entry_marker,
                 s=80, zorder=3, label=f"entry ({trade['direction']} spread)")
    ax_z.scatter([trade["exit_time"]], [trade["exit_z"]], color=exit_color, marker=exit_marker,
                 s=80, zorder=3, label=f"exit ({trade['exit_reason']})")

    ax_p.set_title(title or (
        f"{_pair_label(result)} trade {trade['entry_time']:%Y-%m-%d %H:%M}: {trade['direction']} spread, "
        f"PnL {trade['pnl_pct']:+.2%}, held {trade['holding_hours']:.0f} h"))
    ax_p.set_ylabel("Price (entry = 100)")
    ax_p.legend(loc="upper left")
    ax_z.set_ylabel("z-score")
    ax_z.set_xlabel("Date (UTC)")
    ax_z.legend(ncol=3, fontsize=8, loc="best")
    return _finish(fig, save)


def plot_trade_distribution(result: Result, title: str | None = None, save: str | None = None) -> plt.Figure:
    """Histograms of trade PnL and holding time.

    Input:  Result, optional title, save file name.
    Output: Figure with two panels.
    """
    trades = result.trades
    fig, (ax_pnl, ax_hold) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax_pnl.hist(trades["pnl_pct"] * 100, bins=40, color=PALETTE[0], edgecolor=SURFACE)
    ax_pnl.axvline(0, color=INK, lw=1)
    ax_pnl.set_title(title or f"{_pair_label(result)}: trade PnL ({len(trades)} trades)")
    ax_pnl.set_xlabel("PnL per trade (% of equity)")
    ax_pnl.set_ylabel("Number of trades")
    ax_hold.hist(trades["holding_hours"], bins=40, color=PALETTE[2], edgecolor=SURFACE)
    ax_hold.set_title("Holding time")
    ax_hold.set_xlabel("Hours in trade")
    ax_hold.set_ylabel("Number of trades")
    return _finish(fig, save)


# =============================================================================
# PERFORMANCE
# =============================================================================

def _as_dict(results) -> dict:
    """Accept a single Result/Series or a dict of label -> Result/Series."""
    return results if isinstance(results, dict) else {"strategy": results}


def _equity(x) -> pd.Series:
    return x.equity if isinstance(x, Result) else x


def plot_equity_curve(results, title: str | None = None, save: str | None = None) -> plt.Figure:
    """One or more equity curves overlaid, e.g. with vs without stops.

    Input:  Result, equity Series, or dict of label -> Result/Series; optional title, save file name.
    Output: Figure.
    """
    results = _as_dict(results)
    fig, ax = plt.subplots()
    for i, (label, res) in enumerate(results.items()):
        ax.plot(_equity(res).index, _equity(res), color=PALETTE[i % len(PALETTE)], label=label)
    first = next(iter(results.values()))
    initial = first.params["initial_capital"] if isinstance(first, Result) else first.iloc[0]
    ax.axhline(initial, color=MUTED, lw=0.8, ls="--")   # starting capital as reference
    ax.set_title(title or "Equity curve")
    ax.set_xlabel("Date (UTC)")
    ax.set_ylabel("Equity (USDT)")
    if len(results) > 1:
        ax.legend(loc="upper left")
    return _finish(fig, save)


def plot_drawdown(results, title: str | None = None, save: str | None = None) -> plt.Figure:
    """Drawdown from the running equity peak (filled for one curve, lines for several).

    Input:  Result, equity Series, or dict of label -> Result/Series; optional title, save file name.
    Output: Figure.
    """
    results = _as_dict(results)
    fig, ax = plt.subplots(figsize=(12, 3.5))
    for i, (label, res) in enumerate(results.items()):
        equity = _equity(res)
        drawdown = 100 * (equity / equity.cummax() - 1)
        color = PALETTE[i % len(PALETTE)]
        if len(results) == 1:
            ax.fill_between(drawdown.index, drawdown, 0, color=color, alpha=0.3, lw=0)
        ax.plot(drawdown.index, drawdown, color=color, lw=1, label=label)
    ax.set_title(title or "Drawdown from running peak")
    ax.set_xlabel("Date (UTC)")
    ax.set_ylabel("Drawdown (%)")
    if len(results) > 1:
        ax.legend(loc="lower left")
    return _finish(fig, save)


def plot_heatmap(
    sweep_results: pd.DataFrame,
    x: str,
    y: str,
    values: str | list[str] = "sharpe",
    chosen: dict | None = None,
    title: str | None = None,
    save: str | None = None,
) -> plt.Figure:
    """Parameter-sweep results as heatmaps, one panel per metric, e.g. Sharpe and n_trades side by side.

    Signed metrics (sharpe, returns, calmar) use a red-grey-blue scale centred at 0; others use blue.
    If the sweep varied more parameters than x and y, filter the DataFrame first.

    Input:  output of backtest.sweep, column for the x axis, column for the y axis, metric name(s),
            chosen cell as {x: value, y: value} (drawn with a black frame), optional title, save name.
    Output: Figure.
    """
    values = [values] if isinstance(values, str) else list(values)
    if sweep_results.duplicated([x, y]).any():
        raise ValueError(f"Several rows per ({x}, {y}) cell: filter the other swept parameters first.")

    fig, axes = plt.subplots(1, len(values), figsize=(6 * len(values), 4.8), squeeze=False)
    for ax, value in zip(axes[0], values):
        grid = sweep_results.pivot(index=y, columns=x, values=value).sort_index(ascending=False)
        data = grid.to_numpy(dtype=float)
        signed = value in ("sharpe", "total_return", "annual_return", "calmar", "avg_trade_pnl")
        if signed:
            limit = np.nanmax(np.abs(data)) or 1.0
            norm, cmap = TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit), DIVERGING
        else:
            norm, cmap = None, SEQUENTIAL
        im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                text = f"{data[i, j]:.0f}" if value.startswith("n_") else f"{data[i, j]:.2f}"
                rgba = im.cmap(im.norm(data[i, j]))
                dark = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2] < 0.5
                ax.text(j, i, text, ha="center", va="center", fontsize=9, color="white" if dark else INK)
        if chosen is not None:
            i = list(grid.index).index(chosen[y])
            j = list(grid.columns).index(chosen[x])
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK, lw=3))
        ax.set_xticks(range(len(grid.columns)), grid.columns)
        ax.set_yticks(range(len(grid.index)), grid.index)
        ax.grid(False)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(value)
        fig.colorbar(im, ax=ax, shrink=0.85)
    if title:
        fig.suptitle(title, fontweight="bold")
    return _finish(fig, save)


def plot_pnl_breakdown(results, title: str | None = None, save: str | None = None) -> plt.Figure:
    """Where the money came from: gross price PnL vs fees vs slippage vs funding, and the net result (USDT).

    Input:  Result or dict of label -> Result (grouped bars), optional title, save file name.
    Output: Figure.
    """
    results = _as_dict(results)
    parts = ["gross", "fees", "slippage", "funding"]
    table = pd.DataFrame({
        label: [res.metrics[f"total_{p}"] for p in parts] + [res.equity.iloc[-1] - res.params["initial_capital"]]
        for label, res in results.items()
    }, index=parts + ["net"])

    fig, ax = plt.subplots(figsize=(10, 4.5))
    n = len(results)
    width = 0.8 / n
    x = np.arange(len(table.index))
    for k, label in enumerate(table.columns):
        offset = (k - (n - 1) / 2) * width
        colors = [COMPONENT_COLORS[p] for p in table.index] if n == 1 else PALETTE[k % len(PALETTE)]
        bars = ax.bar(x + offset, table[label], width=width * 0.9, color=colors, label=label)
        ax.bar_label(bars, fmt="{:,.0f}", fontsize=8, padding=2, color=INK)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(x, ["gross PnL", "fees", "slippage", "funding", "net"])
    ax.set_title(title or "PnL breakdown")
    ax.set_xlabel("Component")
    ax.set_ylabel("USDT")
    if n > 1:
        ax.legend()
    return _finish(fig, save)
