"""
test_pipeline_oos.py -- Tests for the out-of-sample reporting pipeline.

These tests exercise ``Pipeline.run_oos`` / ``print_oos_report`` and the
``oos=True`` route of ``run_and_summarise``.  ``DataLoader.fetch`` is patched so
no network calls are made, and nothing here imports torch (the classical signal
generator is used throughout).
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd

from wsq_trading.constants import Regime
from wsq_trading.pipeline import OOSReport, Pipeline


def _make_ohlcv(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Minimal OHLCV DataFrame on a business-day index."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2017-01-02", periods=n)
    close = 100.0 * np.cumprod(1 + rng.normal(0.0003, 0.012, n))
    df = pd.DataFrame(
        {
            "Open": close * rng.uniform(0.99, 1.00, n),
            "High": close * rng.uniform(1.00, 1.01, n),
            "Low": close * rng.uniform(0.99, 1.00, n),
            "Close": close,
            "Volume": rng.integers(1_000, 100_000, n).astype(float),
        },
        index=idx,
    )
    df.index.name = "Date"
    return df


def _patch_fetch_varied(n: int = 600):
    """Patch DataLoader.fetch with per-ticker varied OHLCV (deterministic)."""
    def _fetch(ticker, **kwargs):
        return _make_ohlcv(n=n, seed=abs(hash(ticker)) % 1000)
    return patch("wsq_trading.pipeline.DataLoader.fetch", side_effect=_fetch)


class TestPipelineRunOOS:
    def _report(self) -> tuple[Pipeline, OOSReport]:
        with _patch_fetch_varied():
            p = Pipeline(mode="classical")
            rep = p.run_oos(tickers=["ES=F", "NQ=F", "CL=F"])
        return p, rep

    def test_returns_oos_report(self) -> None:
        _, rep = self._report()
        assert isinstance(rep, OOSReport)

    def test_test_window_after_train_end(self) -> None:
        """No look-ahead: the test window must start after the train cutoff."""
        _, rep = self._report()
        assert rep.test_start > rep.train_end
        assert rep.test_end >= rep.test_start

    def test_cost_sensitivity_default_grid(self) -> None:
        _, rep = self._report()
        assert list(rep.cost_sensitivity.index) == [0, 2, 5, 10, 20, 30]

    def test_cost_sensitivity_columns(self) -> None:
        _, rep = self._report()
        for col in ("portfolio_sharpe", "portfolio_ann_return",
                    "portfolio_max_drawdown"):
            assert col in rep.cost_sensitivity.columns

    def test_regime_attribution_has_all_regimes(self) -> None:
        _, rep = self._report()
        regimes = set(rep.regime_attribution.index.get_level_values("regime"))
        for r in Regime:
            assert r.name in regimes

    def test_portfolio_has_expected_keys(self) -> None:
        _, rep = self._report()
        for key in ("sharpe", "annualized_return", "annualized_vol", "alpha",
                    "beta", "information_ratio", "max_drawdown", "n_obs"):
            assert key in rep.portfolio

    def test_weights_index_covers_tickers(self) -> None:
        _, rep = self._report()
        assert set(rep.hrp_weights.index) == {"ES=F", "NQ=F", "CL=F"}

    def test_per_instrument_index_covers_tickers(self) -> None:
        _, rep = self._report()
        assert set(rep.per_instrument.index) == {"ES=F", "NQ=F", "CL=F"}

    def test_print_oos_report_no_exception(self) -> None:
        p, rep = self._report()
        p.print_oos_report(rep)  # must not raise


class TestRunAndSummariseOOS:
    def test_oos_true_routes_to_report(self) -> None:
        with _patch_fetch_varied():
            p = Pipeline(mode="classical")
            out = p.run_and_summarise(tickers=["ES=F", "NQ=F", "CL=F"], oos=True)
        assert isinstance(out, OOSReport)

    def test_oos_false_returns_tuple(self) -> None:
        with _patch_fetch_varied():
            p = Pipeline(mode="classical")
            out = p.run_and_summarise(tickers=["ES=F", "NQ=F"], oos=False)
        assert isinstance(out, tuple) and len(out) == 3
