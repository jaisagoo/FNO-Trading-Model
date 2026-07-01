"""
test_pipeline.py -- Unit tests for wsq_trading.pipeline.

DataLoader.fetch is patched so no network calls are made.
FNO tests skip gracefully if PyTorch is not installed.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from wsq_trading.backtest import BacktestResult
from wsq_trading.pipeline import Pipeline, _FNOSignalGenerator
from wsq_trading.portfolio import HRPAllocator

# Helpers

def _make_ohlcv(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """Minimal OHLCV DataFrame on a business-day index."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-02", periods=n)
    close = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, n))
    df = pd.DataFrame(
        {
            "Open":   close * rng.uniform(0.99, 1.00, n),
            "High":   close * rng.uniform(1.00, 1.01, n),
            "Low":    close * rng.uniform(0.99, 1.00, n),
            "Close":  close,
            "Volume": rng.integers(1_000, 100_000, n).astype(float),
        },
        index=idx,
    )
    df.index.name = "Date"
    return df


def _patch_fetch(n: int = 500):
    """Return a context manager that patches DataLoader.fetch."""
    return patch(
        "wsq_trading.pipeline.DataLoader.fetch",
        return_value=_make_ohlcv(n),
    )


# Pipeline.__init__

class TestPipelineInit:
    def test_default_mode_is_classical(self):
        p = Pipeline()
        assert p.mode == "classical"

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be"):
            Pipeline(mode="invalid")

    def test_cost_bps_stored(self):
        p = Pipeline(cost_bps=5.0)
        assert p.cost_bps == 5.0

    def test_default_allocator_created(self):
        p = Pipeline()
        assert isinstance(p.allocator, HRPAllocator)

    def test_custom_allocator_used(self):
        alloc = HRPAllocator(max_weight=0.25)
        p = Pipeline(allocator=alloc)
        assert p.allocator is alloc

    def test_fno_mode_without_torch_raises(self):
        """If PyTorch is absent, mode='fno' should raise ImportError."""
        import wsq_trading.pipeline as pl_mod
        original = pl_mod._FNO_AVAILABLE
        pl_mod._FNO_AVAILABLE = False
        try:
            with pytest.raises(ImportError, match="PyTorch"):
                Pipeline(mode="fno")
        finally:
            pl_mod._FNO_AVAILABLE = original


# Pipeline.run

class TestPipelineRun:
    def test_run_returns_dict(self):
        with _patch_fetch():
            p = Pipeline()
            results = p.run(tickers=["SPY"], start="2018-01-01", end="2020-01-01")
        assert isinstance(results, dict)

    def test_run_keys_are_tickers(self):
        with _patch_fetch():
            p = Pipeline()
            results = p.run(tickers=["SPY", "QQQ"])
        assert set(results.keys()) == {"SPY", "QQQ"}

    def test_run_values_are_backtest_results(self):
        with _patch_fetch():
            p = Pipeline()
            results = p.run(tickers=["SPY"])
        for v in results.values():
            assert isinstance(v, BacktestResult)

    def test_run_empty_on_all_fetch_failure(self):
        with patch(
            "wsq_trading.pipeline.DataLoader.fetch",
            side_effect=ValueError("no data"),
        ):
            p = Pipeline()
            results = p.run(tickers=["BAD"])
        assert results == {}

    def test_run_skips_failed_ticker(self):
        call_count = 0

        def _fetch_side_effect(ticker, **kwargs):
            nonlocal call_count
            call_count += 1
            if ticker == "BAD":
                raise ValueError("bad ticker")
            return _make_ohlcv()

        with patch("wsq_trading.pipeline.DataLoader.fetch", side_effect=_fetch_side_effect):
            p = Pipeline()
            results = p.run(tickers=["SPY", "BAD"])
        assert "SPY" in results
        assert "BAD" not in results

    def test_run_uses_default_tickers_when_none(self):
        from wsq_trading import config
        fetched = []

        def _capture(ticker, **kwargs):
            fetched.append(ticker)
            return _make_ohlcv()

        with patch("wsq_trading.pipeline.DataLoader.fetch", side_effect=_capture):
            p = Pipeline()
            p.run()
        assert set(fetched) == set(config.TICKERS)


# Pipeline.allocate

class TestPipelineAllocate:
    def _run_results(self):
        with _patch_fetch():
            p = Pipeline()
            return p, p.run(tickers=["SPY", "QQQ"])

    def test_allocate_returns_series(self):
        p, results = self._run_results()
        weights = p.allocate(results)
        assert isinstance(weights, pd.Series)

    def test_allocate_weights_sum_to_one(self):
        p, results = self._run_results()
        weights = p.allocate(results)
        assert abs(weights.sum() - 1.0) < 1e-6

    def test_allocate_weights_non_negative(self):
        p, results = self._run_results()
        weights = p.allocate(results)
        assert (weights >= 0).all()

    def test_allocate_index_matches_tickers(self):
        p, results = self._run_results()
        weights = p.allocate(results)
        assert set(weights.index) == set(results.keys())


# Pipeline.summary

class TestPipelineSummary:
    def _run(self, tickers=("SPY", "QQQ")):
        with _patch_fetch():
            p = Pipeline()
            results = p.run(tickers=list(tickers))
        return p, results

    def test_summary_returns_dataframe(self):
        p, results = self._run()
        df = p.summary(results)
        assert isinstance(df, pd.DataFrame)

    def test_summary_index_matches_tickers(self):
        p, results = self._run()
        df = p.summary(results)
        assert set(df.index) == set(results.keys())

    def test_summary_with_weights_adds_column(self):
        p, results = self._run()
        weights = p.allocate(results)
        df = p.summary(results, weights=weights)
        assert "hrp_weight" in df.columns

    def test_summary_without_weights_no_hrp_column(self):
        p, results = self._run()
        df = p.summary(results)
        assert "hrp_weight" not in df.columns


# Pipeline.run_and_summarise

class TestRunAndSummarise:
    def test_returns_three_tuple(self):
        with _patch_fetch():
            p = Pipeline()
            out = p.run_and_summarise(tickers=["SPY", "QQQ"])
        assert len(out) == 3

    def test_results_weights_summary_types(self):
        with _patch_fetch():
            p = Pipeline()
            results, weights, summary = p.run_and_summarise(tickers=["SPY", "QQQ"])
        assert isinstance(results, dict)
        assert isinstance(weights, pd.Series)
        assert isinstance(summary, pd.DataFrame)

    def test_empty_results_returns_empty(self):
        with patch(
            "wsq_trading.pipeline.DataLoader.fetch",
            side_effect=ValueError("no data"),
        ):
            p = Pipeline()
            results, weights, summary = p.run_and_summarise(tickers=["BAD"])
        assert results == {}
        assert weights.empty
        assert summary.empty

# Pipeline FNO mode  (skips if PyTorch not available)

torch = pytest.importorskip("torch", reason="PyTorch not installed")

from wsq_trading.fno import FNO1d, FNOTrainer


class TestPipelineFNOMode:
    def _fno_pipeline(self, tmp_path: Path) -> Pipeline:
        model = FNO1d(in_channels=2, width=8, n_modes=4, n_layers=1, mlp_hidden=16)
        trainer = FNOTrainer(
            model=model,
            window_size=63,
            stride=10,
            batch_size=8,
            num_epochs=1,
            patience=5,
            checkpoint_dir=tmp_path,
        )
        # Save a dummy checkpoint so load_checkpoint won't fail
        ckpt = tmp_path / "fno_best.pt"
        torch.save({"epoch": 0, "val_loss": 1.0, "model_state": model.state_dict()}, ckpt)

        p = Pipeline.__new__(Pipeline)
        p.mode = "fno"
        p.cost_bps = 2.0
        p.signal_kwargs = {}
        p.fno_checkpoint = ckpt
        p.allocator = HRPAllocator()

        from wsq_trading.backtest import WalkForwardEngine
        sig_gen = _FNOSignalGenerator(trainer)
        p._sig_gen = sig_gen
        p._engine = WalkForwardEngine(signal_generator=sig_gen, cost_bps=p.cost_bps)
        return p

    def test_fno_pipeline_run_returns_dict(self, tmp_path):
        p = self._fno_pipeline(tmp_path)
        with _patch_fetch():
            results = p.run(tickers=["SPY"])
        assert isinstance(results, dict)

    def test_fno_signal_generator_is_fno_type(self, tmp_path):
        p = self._fno_pipeline(tmp_path)
        assert isinstance(p._sig_gen, _FNOSignalGenerator)

    def test_fno_mode_init_with_checkpoint(self, tmp_path):
        model = FNO1d(in_channels=2, width=8, n_modes=4, n_layers=1, mlp_hidden=16)
        ckpt = tmp_path / "fno_best.pt"
        torch.save({"epoch": 0, "val_loss": 1.0, "model_state": model.state_dict()}, ckpt)
        # Pipeline.__init__ with mode='fno' and a valid checkpoint
        mock_trainer = MagicMock(spec=FNOTrainer)
        mock_trainer.window_size = 63
        with patch("wsq_trading.pipeline.FNO1d", return_value=MagicMock()), \
             patch("wsq_trading.pipeline.FNOTrainer", return_value=mock_trainer):
            p = Pipeline(mode="fno", fno_checkpoint=ckpt)
        assert p.mode == "fno"
