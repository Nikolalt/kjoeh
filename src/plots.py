"""All figures for the notebooks. Every function returns a matplotlib Figure, in one shared style."""

import math

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# One palette for the whole project, so every figure looks the same.
COLORS = {
    "a": "#1f77b4",       # leg a / main series
    "b": "#ff7f0e",       # leg b
    "long": "#2ca02c",    # long-spread entries
    "short": "#d62728",   # short-spread entries
    "exit": "#7f7f7f",    # exits
    "neutral": "#333333",
}
# Markers are a second encoding next to colour, so long/short/exit stay readable for colour-blind readers.
MARKERS = {"long": "^", "short": "v", "exit": "X"}
FIGSIZE = (12, 5)


def apply_style() -> None:
    """Set the shared matplotlib style; call once in each notebook's import cell.

    Input:  none.
    Output: None (updates matplotlib rcParams).
    """
    plt.rcParams.update({
        "figure.dpi": 100,
        "figure.figsize": FIGSIZE,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.6,
        "lines.linewidth": 1.3,
        "legend.frameon": False,
        "axes.prop_cycle": plt.cycler(color=[COLORS["a"], COLORS["b"], COLORS["neutral"]]),
    })


def _entries_exits(positions: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Positions at entry bars and the previous position at exit bars (both nonzero, sign kept)."""
    prev = positions.shift(1, fill_value=0)
    changed = positions != prev
    return positions[changed & (positions != 0)], prev[changed & (prev != 0)]


def _mark_trades(ax: plt.Axes, series: pd.Series, positions: pd.Series) -> None:
    """Draw entry (long ^ / short v) and exit (X) markers on top of a series."""
    entries, exits = _entries_exits(positions)
    for side, mask in (("long", entries > 0), ("short", entries < 0)):
        idx = entries.index[mask]
        ax.scatter(idx, series.reindex(idx), marker=MARKERS[side], s=64, color=COLORS[side],
                   edgecolor="white", linewidth=0.8, zorder=3, label=f"{side} entry")
    ax.scatter(exits.index, series.reindex(exits.index), marker=MARKERS["exit"], s=48, color=COLORS["exit"],
               edgecolor="white", linewidth=0.8, zorder=3, label="exit")


def plot_prices(prices: pd.DataFrame, normalise: bool = True) -> plt.Figure:
    """Plot price paths of several coins, optionally rebased to 1 at the start for comparison.

    More than three coins are drawn as small multiples (one panel each, log scale) instead of one
    crowded chart.

    Input:  price DataFrame, normalise flag.
    Output: Figure.
    """
    data = prices / prices.iloc[0] if normalise else prices
    if data.shape[1] <= 3:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        data.plot(ax=ax, logy=True)
        ax.set_ylabel("rebased price (log)" if normalise else "price (log)")
        return fig
    ncols = 4
    nrows = math.ceil(data.shape[1] / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(FIGSIZE[0], 2.4 * nrows), sharex=True)
    for ax, coin in zip(axes.flat, data.columns):
        ax.plot(data.index, data[coin], color=COLORS["a"], linewidth=1)
        ax.set_yscale("log")
        ax.set_title(coin, loc="left")
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for ax in axes.flat[data.shape[1]:]:
        ax.set_visible(False)
    fig.suptitle("Price rebased to 1 at the start (log scale)" if normalise else "Price (log scale)")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_correlation_matrix(corr: pd.DataFrame) -> plt.Figure:
    """Show a correlation matrix as an annotated heatmap.

    Input:  square correlation DataFrame.
    Output: Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, square=True,
                linewidths=2, linecolor="white", cbar_kws={"label": "correlation"}, ax=ax)
    ax.grid(False)
    ax.set_title("Correlation of daily log returns")
    return fig


def plot_spread_zscore(
    spread: pd.Series,
    z: pd.Series,
    positions: pd.Series,
    entry_z: float,
    exit_z: float,
) -> plt.Figure:
    """Plot the spread (top) and z-score with threshold lines (bottom), marking entries and exits.

    Input:  spread, z-score, positions (to locate entries/exits), entry and exit thresholds.
    Output: Figure with two stacked axes.
    """
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(FIGSIZE[0], 7), sharex=True, height_ratios=[3, 2])
    top.plot(spread.index, spread, color=COLORS["a"], label=spread.name or "spread")
    _mark_trades(top, spread, positions)
    top.set_ylabel(spread.name or "spread")
    top.legend(loc="upper left", ncols=4)

    bottom.plot(z.index, z, color=COLORS["neutral"], linewidth=1, label="z-score")
    for level, style in ((entry_z, "--"), (-entry_z, "--"), (exit_z, ":")):
        bottom.axhline(level, color=COLORS["exit"], linestyle=style, linewidth=1)
    _mark_trades(bottom, z, positions)
    bottom.set_ylabel("z-score")
    label_x = z.index[-1] + (z.index[-1] - z.index[0]) * 0.01
    bottom.text(label_x, entry_z, f"+{entry_z:g}\nshort entry", va="center", color=COLORS["neutral"])
    bottom.text(label_x, -entry_z, f"-{entry_z:g}\nlong entry", va="center", color=COLORS["neutral"])
    fig.tight_layout()
    return fig


def plot_ratio_bands(
    ratio: pd.Series,
    mean: pd.Series,
    std: pd.Series,
    entry_z: float,
    positions: pd.Series | None = None,
    title: str = "",
) -> plt.Figure:
    """Plot the alt/BTC ratio with its fair ratio (rolling mean) and the entry bands, log scale.

    mean and std are statistics of the log ratio, so the band is exp(mean +- entry_z * std).

    Input:  plain ratio, mean and std of the log ratio, entry threshold, optional positions to mark
            entries and exits, title.
    Output: Figure.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.fill_between(ratio.index, np.exp(mean - entry_z * std), np.exp(mean + entry_z * std),
                    color=COLORS["a"], alpha=0.12, linewidth=0, label=f"fair ratio +- {entry_z:g} std")
    ax.plot(ratio.index, ratio, color=COLORS["a"], label="ratio alt / BTC")
    ax.plot(mean.index, np.exp(mean), color=COLORS["b"], linestyle="--", label="fair ratio (rolling mean)")
    if positions is not None:
        _mark_trades(ax, ratio, positions)
    ax.set_yscale("log")
    ax.set_ylabel("price in BTC (log)")
    ax.set_title(title, loc="left")
    ax.legend(loc="upper left", ncols=3)
    fig.tight_layout()
    return fig


def plot_mean_comparison(ratio: pd.Series, means: dict[str, pd.Series], title: str = "") -> plt.Figure:
    """Compare definitions of the 'average ratio' (e.g. rolling vs expanding mean) on one pair.

    Input:  plain ratio, dict label -> mean of the log ratio, title.
    Output: Figure (log scale).
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(ratio.index, ratio, color=COLORS["a"], label="ratio alt / BTC")
    for (label, mean), color, style in zip(means.items(), [COLORS["b"], COLORS["neutral"]], ["--", ":"]):
        ax.plot(mean.index, np.exp(mean), color=color, linestyle=style, linewidth=1.6, label=label)
    ax.set_yscale("log")
    ax.set_ylabel("price in BTC (log)")
    ax.set_title(title, loc="left")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def plot_equity_curve(equity: dict[str, pd.Series]) -> plt.Figure:
    """Plot one or more equity curves, e.g. with vs without stops, or train vs test.

    Input:  dict of label -> equity Series.
    Output: Figure.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for (label, series), color in zip(equity.items(), [COLORS["a"], COLORS["b"], COLORS["neutral"], COLORS["exit"]]):
        ax.plot(series.index, series, color=color, label=label)
        ax.annotate(f"{series.iloc[-1]:,.0f}", (series.index[-1], series.iloc[-1]), xytext=(4, 0),
                    textcoords="offset points", va="center", color=COLORS["neutral"])
    ax.set_ylabel("equity")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_drawdown(equity: pd.Series, compounding: bool = False) -> plt.Figure:
    """Plot the drawdown from the running peak as a filled area.

    Input:  equity Series, compounding flag (False: fall from peak as a fraction of the initial capital,
            matching backtest.compute_metrics; True: equity / peak - 1).
    Output: Figure.
    """
    peak = equity.cummax()
    dd = equity / peak - 1 if compounding else (equity - peak) / equity.iloc[0]
    fig, ax = plt.subplots(figsize=(FIGSIZE[0], 3))
    ax.fill_between(dd.index, dd, 0, color=COLORS["short"], alpha=0.35, linewidth=0)
    ax.plot(dd.index, dd, color=COLORS["short"], linewidth=1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylabel("drawdown")
    ax.set_title(f"Max drawdown {dd.min():.1%} on {dd.idxmin():%Y-%m-%d}", loc="left")
    fig.tight_layout()
    return fig


def plot_heatmap(
    results: pd.DataFrame, x: str, y: str, value: str, facet: str | None = None
) -> plt.Figure:
    """Show parameter sweep results as a heatmap, e.g. Sharpe over entry_z x zscore_lookback.

    Diverging colours centred on 0, so losing cells are red and winning cells are blue.

    Input:  long-format DataFrame (one row per run), column names for x axis, y axis and colour value,
            optional facet column (one panel per value, e.g. stop_loss).
    Output: Figure.
    """
    groups = list(results.groupby(facet, dropna=False)) if facet else [(None, results)]
    limit = np.nanmax(np.abs(results[value].to_numpy(dtype=float)))
    fig, axes = plt.subplots(1, len(groups), figsize=(4.2 * len(groups) + 1, 4), squeeze=False)
    for ax, (key, group) in zip(axes.flat, groups):
        table = group.pivot_table(index=y, columns=x, values=value)
        sns.heatmap(table, annot=True, fmt=".2f", cmap="RdBu", center=0, vmin=-limit, vmax=limit,
                    linewidths=2, linecolor="white", cbar=False, ax=ax)
        ax.grid(False)
        label = "off" if key is None or (isinstance(key, float) and np.isnan(key)) else key
        ax.set_title(f"{facet} = {label}" if facet else value, loc="left")
    fig.suptitle(f"{value} per parameter combination")
    fig.tight_layout()
    return fig


def plot_trade(
    price_a: pd.Series,
    price_b: pd.Series,
    z: pd.Series,
    trade: pd.Series,
    padding: int = 48,
) -> plt.Figure:
    """Zoom into one trade: both legs and the z-score around entry and exit.

    Top: both legs rebased to 1 at the entry fill. Bottom: z-score with the signal, entry fill and exit
    fill marked. The title gives direction, PnL and exit reason.

    Input:  prices of both legs, z-score, one row of the trades DataFrame, bars of context on each side.
    Output: Figure.
    """
    index = price_a.index
    lo = max(index.get_loc(trade["entry_time"]) - padding, 0)
    hi = min(index.get_loc(trade["exit_time"]) + padding, len(index) - 1)
    window = index[lo:hi + 1]
    a, b, zw = price_a.loc[window], price_b.loc[window], z.loc[window]
    names = str(trade.get("pair", f"{price_a.name}/{price_b.name}")).split("/")

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(FIGSIZE[0], 6.5), sharex=True, height_ratios=[3, 2])
    top.plot(window, a / price_a.loc[trade["entry_time"]], color=COLORS["a"], label=names[0])
    top.plot(window, b / price_b.loc[trade["entry_time"]], color=COLORS["b"], label=names[-1])
    top.set_ylabel("rebased to 1 at entry")
    top.legend(loc="upper left")
    bottom.plot(window, zw, color=COLORS["neutral"], linewidth=1, label="z-score")
    bottom.axhline(0, color=COLORS["exit"], linestyle=":", linewidth=1)
    side = "long" if trade["direction"] > 0 else "short"
    for ax in (top, bottom):
        ax.axvspan(trade["entry_time"], trade["exit_time"], color=COLORS[side], alpha=0.08, linewidth=0)
    if "signal_time" in trade:
        bottom.scatter([trade["signal_time"]], [z.loc[trade["signal_time"]]], marker="o", s=48,
                       facecolor="white", edgecolor=COLORS["neutral"], zorder=3, label="signal")
    bottom.scatter([trade["entry_time"]], [z.loc[trade["entry_time"]]], marker=MARKERS[side], s=72,
                   color=COLORS[side], edgecolor="white", zorder=3, label=f"{side} entry fill")
    bottom.scatter([trade["exit_time"]], [z.loc[trade["exit_time"]]], marker=MARKERS["exit"], s=60,
                   color=COLORS["exit"], edgecolor="white", zorder=3, label=f"exit fill ({trade['exit_reason']})")
    bottom.set_ylabel("z-score")
    bottom.legend(loc="upper left", ncols=3)
    leg = f"{side.upper()} {names[0]} / {'SHORT' if side == 'long' else 'LONG'} {names[-1]}"
    top.set_title(f"{leg}: {trade['entry_time']:%Y-%m-%d} -> {trade['exit_time']:%Y-%m-%d}, "
                  f"{trade['holding_bars']} days, PnL {trade['pnl']:,.0f} ({trade['return']:+.1%} of notional)",
                  loc="left")
    fig.tight_layout()
    return fig


def plot_pnl_by_pair(pnl: pd.Series) -> plt.Figure:
    """Horizontal bars of total PnL per pair, sorted, with the value printed at each bar end.

    Input:  Series pair -> PnL.
    Output: Figure.
    """
    pnl = pnl.sort_values()
    fig, ax = plt.subplots(figsize=(FIGSIZE[0] * 0.7, 0.45 * len(pnl) + 1))
    colors = [COLORS["long"] if v > 0 else COLORS["short"] for v in pnl]
    ax.barh(pnl.index, pnl, color=colors, height=0.6)
    ax.axvline(0, color=COLORS["neutral"], linewidth=0.8)
    for name, value in pnl.items():
        ax.annotate(f"{value:,.0f}", (value, name), xytext=(4 if value >= 0 else -4, 0), textcoords="offset points",
                    ha="left" if value >= 0 else "right", va="center", color=COLORS["neutral"])
    ax.set_xlabel("PnL")
    ax.margins(x=0.18)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def plot_exposure(gross: pd.Series, cap: float) -> plt.Figure:
    """Gross exposure of the portfolio over time against its cap.

    Input:  gross exposure Series (fraction of capital), leverage cap.
    Output: Figure.
    """
    fig, ax = plt.subplots(figsize=(FIGSIZE[0], 3))
    ax.fill_between(gross.index, gross, 0, color=COLORS["a"], alpha=0.25, linewidth=0)
    ax.plot(gross.index, gross, color=COLORS["a"], linewidth=1, label="gross exposure")
    ax.axhline(cap, color=COLORS["short"], linestyle="--", linewidth=1, label=f"cap {cap:g}x")
    ax.set_ylabel("x capital")
    ax.set_title(f"Average gross exposure {gross.mean():.2f}x", loc="left")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig
