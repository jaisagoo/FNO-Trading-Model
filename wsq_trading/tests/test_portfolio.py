"""
Tests for the HRP portfolio allocator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wsq_trading.portfolio import HRPAllocator


def _make_returns(
    n_assets: int = 5,
    n_bars: int = 252,
    seed: int = 0,
    corr_block: bool = False,
) -> pd.DataFrame:
    """Gaussian return matrix, optionally with two correlated blocks."""
    rng = np.random.default_rng(seed)
    tickers = [f"A{i}" for i in range(n_assets)]

    if corr_block and n_assets >= 4:
        # Two correlated blocks: (A0, A1) and (A2, A3)
        common1 = rng.normal(0, 0.01, n_bars)
        common2 = rng.normal(0, 0.01, n_bars)
        cols = {}
        for i, t in enumerate(tickers):
            if i < n_assets // 2:
                cols[t] = 0.7 * common1 + 0.3 * rng.normal(0, 0.01, n_bars)
            else:
                cols[t] = 0.7 * common2 + 0.3 * rng.normal(0, 0.01, n_bars)
        return pd.DataFrame(cols)

    data = rng.normal(0.0, 0.01, (n_bars, n_assets))
    return pd.DataFrame(data, columns=tickers)


# Basic contract

class TestAllocateContract:
    def test_returns_series(self) -> None:
        alloc = HRPAllocator()
        w = alloc.allocate(_make_returns())
        assert isinstance(w, pd.Series)

    def test_weights_sum_to_one(self) -> None:
        alloc = HRPAllocator()
        w = alloc.allocate(_make_returns())
        assert abs(w.sum() - 1.0) < 1e-8

    def test_all_weights_positive(self) -> None:
        alloc = HRPAllocator()
        w = alloc.allocate(_make_returns())
        assert (w >= 0).all()

    def test_index_matches_tickers(self) -> None:
        returns = _make_returns(n_assets=4)
        alloc = HRPAllocator()
        w = alloc.allocate(returns)
        assert set(w.index) == set(returns.columns)

    def test_two_assets(self) -> None:
        """Minimum valid case: two assets."""
        alloc = HRPAllocator()
        w = alloc.allocate(_make_returns(n_assets=2))
        assert len(w) == 2
        assert abs(w.sum() - 1.0) < 1e-8

    def test_single_asset_raises(self) -> None:
        alloc = HRPAllocator()
        with pytest.raises(ValueError, match="at least 2"):
            alloc.allocate(_make_returns(n_assets=1))

    def test_all_nan_column_dropped(self) -> None:
        returns = _make_returns(n_assets=4)
        returns["NAN_ASSET"] = np.nan
        alloc = HRPAllocator()
        w = alloc.allocate(returns)
        assert "NAN_ASSET" not in w.index
        assert abs(w.sum() - 1.0) < 1e-8


# Max weight cap

class TestMaxWeightCap:
    def test_no_weight_exceeds_cap(self) -> None:
        cap = 0.35
        alloc = HRPAllocator(max_weight=cap)
        w = alloc.allocate(_make_returns(n_assets=5))
        assert (w <= cap + 1e-9).all()

    def test_weights_still_sum_to_one_after_cap(self) -> None:
        alloc = HRPAllocator(max_weight=0.25)
        w = alloc.allocate(_make_returns(n_assets=5))
        assert abs(w.sum() - 1.0) < 1e-8

    def test_tight_cap_forces_near_equal_weights(self) -> None:
        """With cap = 1/n, all weights should equal 1/n."""
        n = 5
        alloc = HRPAllocator(max_weight=1.0 / n)
        w = alloc.allocate(_make_returns(n_assets=n))
        for wi in w:
            assert abs(wi - 1.0 / n) < 1e-6


# Diversification: correlated blocks should get lower combined weight

class TestDiversification:
    def test_correlated_block_less_than_half(self) -> None:
        """Two correlated assets should not dominate the portfolio."""
        returns = _make_returns(n_assets=4, corr_block=True, n_bars=500)
        alloc = HRPAllocator(max_weight=1.0)  # no cap
        w = alloc.allocate(returns)
        # Each block of 2 correlated assets should get roughly half
        block1 = w[["A0", "A1"]].sum()
        block2 = w[["A2", "A3"]].sum()
        # Neither block should get more than 80%
        assert block1 < 0.80
        assert block2 < 0.80

    def test_uncorrelated_assets_near_equal_weight(self) -> None:
        """When all assets are iid, HRP should produce roughly equal weights."""
        n = 6
        returns = _make_returns(n_assets=n, n_bars=1000, seed=7)
        alloc = HRPAllocator(max_weight=1.0)
        w = alloc.allocate(returns)
        expected = 1.0 / n
        for wi in w:
            assert abs(wi - expected) < 0.15, f"Weight {wi:.3f} far from {expected:.3f}"


# Internal helpers

class TestCorrToDistance:
    def test_diagonal_is_zero(self) -> None:
        corr = pd.DataFrame(np.eye(3))
        d = HRPAllocator._corr_to_distance(corr)
        assert (np.diag(d) == 0.0).all()

    def test_distance_in_zero_one(self) -> None:
        rng = np.random.default_rng(0)
        corr = pd.DataFrame(np.corrcoef(rng.normal(size=(4, 200))))
        d = HRPAllocator._corr_to_distance(corr)
        assert (d >= 0).all() and (d <= 1).all()

    def test_perfectly_correlated_gives_zero_distance(self) -> None:
        corr = pd.DataFrame([[1.0, 1.0], [1.0, 1.0]])
        d = HRPAllocator._corr_to_distance(corr)
        assert d[0, 1] == pytest.approx(0.0, abs=1e-9)

    def test_perfectly_anticorrelated_gives_one_distance(self) -> None:
        corr = pd.DataFrame([[1.0, -1.0], [-1.0, 1.0]])
        d = HRPAllocator._corr_to_distance(corr)
        assert d[0, 1] == pytest.approx(1.0, abs=1e-9)


class TestQuasiDiag:
    def test_returns_correct_length(self) -> None:
        from scipy.cluster.hierarchy import linkage
        from scipy.spatial.distance import squareform
        rng = np.random.default_rng(0)
        dist = np.abs(rng.normal(0, 0.1, (5, 5)))
        dist = (dist + dist.T) / 2
        np.fill_diagonal(dist, 0.0)
        link = linkage(squareform(dist), method="single")
        order = HRPAllocator._quasi_diag(link, 5)
        assert len(order) == 5
        assert sorted(order) == list(range(5))

    def test_all_indices_present(self) -> None:
        from scipy.cluster.hierarchy import linkage
        from scipy.spatial.distance import squareform
        rng = np.random.default_rng(1)
        n = 8
        dist = np.abs(rng.normal(0, 0.1, (n, n)))
        dist = (dist + dist.T) / 2
        np.fill_diagonal(dist, 0.0)
        link = linkage(squareform(dist), method="single")
        order = HRPAllocator._quasi_diag(link, n)
        assert set(order) == set(range(n))


class TestClusterVar:
    def test_scalar_output(self) -> None:
        cov = pd.DataFrame([[0.01, 0.0], [0.0, 0.01]], index=["A", "B"], columns=["A", "B"])
        v = HRPAllocator._cluster_var(cov, ["A", "B"])
        assert isinstance(v, float)

    def test_single_asset_var(self) -> None:
        cov = pd.DataFrame([[0.04]], index=["A"], columns=["A"])
        v = HRPAllocator._cluster_var(cov, ["A"])
        assert abs(v - 0.04) < 1e-10

    def test_equal_weight_two_assets(self) -> None:
        cov = pd.DataFrame(
            [[0.04, 0.02], [0.02, 0.04]], index=["A", "B"], columns=["A", "B"]
        )
        v = HRPAllocator._cluster_var(cov, ["A", "B"])
        # w=[0.5, 0.5]: var = 0.25*0.04 + 2*0.25*0.02 + 0.25*0.04 = 0.03
        assert abs(v - 0.03) < 1e-10


# allocate_backtest convenience

class TestAllocateBacktest:
    def test_works_with_backtest_results(self) -> None:
        """allocate_backtest should extract strategy_returns and call allocate."""
        from wsq_trading.backtest import WalkForwardEngine

        engine = WalkForwardEngine()
        rng = np.random.default_rng(5)
        data = {}
        for t in ["AAPL", "MSFT", "GOOG"]:
            idx = pd.bdate_range("2020-01-01", periods=400)
            data[t] = pd.DataFrame(
                {"log_return": rng.normal(0.0002, 0.01, 400)}, index=idx
            )
        results = engine.run(data)

        alloc = HRPAllocator()
        w = alloc.allocate_backtest(results)
        assert set(w.index) == set(results.keys())
        assert abs(w.sum() - 1.0) < 1e-8



# Quality-gate look-ahead fix

class TestQualityGateLookahead:
    @staticmethod
    def _stub(ticker, ret):
        from wsq_trading.backtest import BacktestResult
        return BacktestResult(
            ticker=ticker,
            strategy_returns=ret,
            positions=pd.Series(np.sign(ret), index=ret.index),
            H_estimates=pd.Series(0.5, index=ret.index),
            regimes=pd.Series([None] * len(ret), index=ret.index),
            gross_returns=ret,
        )

    def _results(self):
        rng = np.random.default_rng(42)
        idx = pd.date_range("2015-01-01", periods=500)
        g1 = pd.Series(rng.normal(0.001, 0.01, 500), index=idx)
        g2 = pd.Series(rng.normal(0.001, 0.01, 500), index=idx)
        # 'bad' is unprofitable on the train window but gets a large boost in
        # the final (test) 100 days, lifting its FULL-sample Sharpe above 0.30.
        bad = pd.Series(rng.normal(-0.002, 0.01, 500), index=idx)
        bad.iloc[-100:] += 0.02
        results = {
            "g1": self._stub("g1", g1),
            "g2": self._stub("g2", g2),
            "bad": self._stub("bad", bad),
        }
        return results, idx

    def test_train_cutoff_excludes_late_booster(self):
        """With a train cutoff, 'bad' is gated out using train-only Sharpe."""
        results, idx = self._results()
        alloc = HRPAllocator(quality_sharpe_threshold=0.30)
        w = alloc.allocate_backtest(results, train_end_date=idx[399])
        assert w["bad"] == 0.0
        assert w["g1"] > 0.0 and w["g2"] > 0.0

    def test_full_sample_would_admit_late_booster(self):
        """Without a cutoff, the full-sample Sharpe lets 'bad' through.

        This is exactly the look-ahead the train cutoff removes.
        """
        results, _ = self._results()
        alloc = HRPAllocator(quality_sharpe_threshold=0.30)
        w = alloc.allocate_backtest(results)
        assert w["bad"] > 0.0

    def test_default_none_matches_legacy_full_sample(self):
        """train_end_date=None must reproduce the original full-sample gate."""
        results, _ = self._results()
        alloc = HRPAllocator(quality_sharpe_threshold=0.30)
        w_default = alloc.allocate_backtest(results)
        w_none = alloc.allocate_backtest(results, train_end_date=None)
        pd.testing.assert_series_equal(w_default, w_none)
