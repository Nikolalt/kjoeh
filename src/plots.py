"""All figures for the notebooks. Every function returns a matplotlib Figure, in one shared style."""

import matplotlib.pyplot as plt
import pandas as pd

# One palette for the whole project, so every figure looks the same.
COLORS = {
    "a": "#1f77b4",       # leg a / main series
    "b": "#ff7f0e",       # leg b
    "long": "#2ca02c",    # long-spread entries
    "short": "#d62728",   # short-spread entries
    "exit": "#7f7f7f",    # exits
    "neutral": "#333333",
}
FIGSIZE = (12, 5)


def apply_style() -> None:
    """Set the shared matplotlib style; call once in each notebook's import cell.

    Input:  none.
    Output: None (updates matplotlib rcParams).
    """
    # TODO: plt.rcParams.update(...) for font size, grid, spines, dpi
    raise NotImplementedError


def plot_prices(prices: pd.DataFrame, normalise: bool = True) -> plt.Figure:
    """Plot price paths of several coins, optionally rebased to 1 at the start for comparison.

    Input:  price DataFrame, normalise flag.
    Output: Figure.
    """
    raise NotImplementedError


def plot_correlation_matrix(corr: pd.DataFrame) -> plt.Figure:
    """Show a correlation matrix as an annotated heatmap.

    Input:  square correlation DataFrame.
    Output: Figure.
    """
    raise NotImplementedError


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
    raise NotImplementedError


def plot_equity_curve(equity: dict[str, pd.Series]) -> plt.Figure:
    """Plot one or more equity curves, e.g. with vs without stops, or train vs test.

    Input:  dict of label -> equity Series.
    Output: Figure.
    """
    raise NotImplementedError


def plot_drawdown(equity: pd.Series) -> plt.Figure:
    """Plot the drawdown from the running peak as a filled area.

    Input:  equity Series.
    Output: Figure.
    """
    raise NotImplementedError


def plot_heatmap(results: pd.DataFrame, x: str, y: str, value: str) -> plt.Figure:
    """Show parameter sweep results as a heatmap, e.g. Sharpe over entry_z x zscore_lookback.

    Input:  long-format DataFrame (one row per run), column names for x axis, y axis and colour value.
    Output: Figure.
    """
    raise NotImplementedError


def plot_trade(
    price_a: pd.Series,
    price_b: pd.Series,
    z: pd.Series,
    trade: pd.Series,
    padding: int = 48,
) -> plt.Figure:
    """Zoom into one trade: both legs and the z-score around entry and exit.

    Input:  prices of both legs, z-score, one row of the trades DataFrame, bars of context on each side.
    Output: Figure.
    """
    raise NotImplementedError
