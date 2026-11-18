"""Sanity tests for the backtest engine. Run with `python -m pytest` from the repo root."""

import pytest


@pytest.mark.skip(reason="run_backtest not implemented yet")
def test_no_lookahead():
    """A position decided at bar t must only earn the return from bar t to t+1 onwards (positions shifted by one bar)."""
    # TODO: one-bar signal on a known price jump; PnL must appear only from the next bar
    raise NotImplementedError


@pytest.mark.skip(reason="run_backtest not implemented yet")
def test_zero_positions_zero_pnl():
    """If the strategy never trades, equity stays at initial capital and there are no trades."""
    raise NotImplementedError


@pytest.mark.skip(reason="run_backtest not implemented yet")
def test_fees_reduce_pnl():
    """The same positions with fee_per_leg > 0 must end with lower equity than with zero fees."""
    raise NotImplementedError
