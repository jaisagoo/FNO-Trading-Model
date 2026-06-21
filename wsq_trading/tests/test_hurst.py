"""
Tests for the Hurst estimators: RSAnalysis, DFA and WhittleMLE.

Structure
---------
TestRSAnalysis      : unit tests for the R/S estimator
TestDFA             : unit tests for DFA
TestWhittleMLE      : unit tests for the Whittle estimator
TestHRecovery       : integration test — do all three recover the right H
                      from SPDE-generated rough Heston paths?
TestHurstResult     : dataclass contract tests
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wsq_trading.hurst import (
    DFA,
    HurstResult,
    RSAnalysis,
    WhittleMLE,
    WhittleResult,
)
from wsq_trading.spde import _fbm_increments_cholesky


def _fbm_returns(H: float, n: int = 512, seed: int = 0) -> np.ndarray:
    """Generate n fractional Gaussian noise increments with known H."""
    rng = np.random.default_rng(seed)
    return _fbm_increments_cholesky(H, n, 1 / 252, rng)


def _iid_returns(n: int = 512, seed: int = 42) -> np.ndarray:
    """i.i.d. Gaussian returns — should give H ≈ 0.5."""
    return np.random.default_rng(seed).standard_normal(n) * 0.01


@pytest.fixture(scope="module")
def rs() -> RSAnalysis:
    return RSAnalysis(min_window=10, n_scales=20, apply_bias_correction=False)


@pytest.fixture(scope="module")
def dfa() -> DFA:
    return DFA(order=1, min_window=10, n_scales=20)


@pytest.fixture(scope="module")
def whi() -> WhittleMLE:
    return WhittleMLE(bandwidth_exp=0.65)


# RSAnalysis

class TestRSAnalysis:
    def test_fit_returns_float(self, rs: RSAnalysis) -> None:
        H = rs.fit(_iid_returns())
        assert isinstance(H, float)

    def test_fit_iid_near_half(self, rs: RSAnalysis) -> None:
        """i.i.d. process should give H close to 0.5."""
        H = rs.fit(_iid_returns(n=2000, seed=7))
        assert 0.3 < H < 0.7, f"i.i.d. H expected near 0.5, got {H:.3f}"

    def test_fit_bounded(self, rs: RSAnalysis) -> None:
        """H must always be within [HURST_MIN, HURST_MAX]."""
        for seed in range(5):
            H = rs.fit(_iid_returns(n=300, seed=seed))
            assert 0.0 < H < 1.0, f"H out of bounds: {H}"

    def test_fit_with_ci_returns_result(self, rs: RSAnalysis) -> None:
        result = rs.fit_with_ci(_iid_returns())
        assert isinstance(result, HurstResult)
        assert result.method == "R/S"

    def test_ci_contains_estimate(self, rs: RSAnalysis) -> None:
        result = rs.fit_with_ci(_iid_returns())
        assert result.ci_lower <= result.H <= result.ci_upper

    def test_ci_lower_less_than_upper(self, rs: RSAnalysis) -> None:
        result = rs.fit_with_ci(_iid_returns())
        assert result.ci_lower < result.ci_upper

    def test_r_squared_in_range(self, rs: RSAnalysis) -> None:
        result = rs.fit_with_ci(_iid_returns())
        assert 0.0 <= result.r_squared <= 1.0

    def test_fit_rolling_returns_series(self, rs: RSAnalysis) -> None:
        s = pd.Series(_iid_returns(n=300))
        out = rs.fit_rolling(s, window=100, step=10)
        assert isinstance(out, pd.Series)
        assert len(out) == len(s)

    def test_fit_rolling_has_values(self, rs: RSAnalysis) -> None:
        s = pd.Series(_iid_returns(n=300))
        out = rs.fit_rolling(s, window=100, step=10)
        assert out.dropna().shape[0] > 0

    def test_rs_curve_returns_arrays(self, rs: RSAnalysis) -> None:
        scales, rs_vals = rs.rs_curve(_iid_returns())
        assert len(scales) == len(rs_vals)
        assert len(scales) > 0

    def test_rs_curve_monotone_scales(self, rs: RSAnalysis) -> None:
        scales, _ = rs.rs_curve(_iid_returns())
        assert np.all(np.diff(scales) > 0), "Scales should be strictly increasing"

    def test_short_series_raises(self, rs: RSAnalysis) -> None:
        with pytest.raises(ValueError, match="too short"):
            rs.fit(np.array([0.01, -0.01, 0.02]))

    def test_nan_series_handled(self, rs: RSAnalysis) -> None:
        arr = _iid_returns(n=200)
        arr_nan = np.concatenate([[np.nan, np.nan], arr])
        H = rs.fit(arr_nan)   # Should drop NaNs and still work
        assert 0.0 < H < 1.0

    def test_pandas_series_input(self, rs: RSAnalysis) -> None:
        s = pd.Series(_iid_returns())
        H = rs.fit(s)
        assert isinstance(H, float)


# DFA

class TestDFA:
    def test_fit_returns_float(self, dfa: DFA) -> None:
        H = dfa.fit(_iid_returns())
        assert isinstance(H, float)

    def test_fit_iid_near_half(self, dfa: DFA) -> None:
        H = dfa.fit(_iid_returns(n=2000, seed=99))
        assert 0.3 < H < 0.7, f"i.i.d. H expected near 0.5, got {H:.3f}"

    def test_fit_bounded(self, dfa: DFA) -> None:
        for seed in range(5):
            H = dfa.fit(_iid_returns(n=300, seed=seed))
            assert 0.0 < H < 1.0

    def test_order2_also_works(self) -> None:
        dfa2 = DFA(order=2, min_window=15, n_scales=15)
        H = dfa2.fit(_iid_returns(n=500))
        assert 0.0 < H < 1.0

    def test_fit_with_ci_returns_result(self, dfa: DFA) -> None:
        result = dfa.fit_with_ci(_iid_returns())
        assert isinstance(result, HurstResult)
        assert "DFA" in result.method

    def test_ci_order(self, dfa: DFA) -> None:
        result = dfa.fit_with_ci(_iid_returns())
        assert result.ci_lower <= result.H <= result.ci_upper

    def test_fluctuation_curve_returns_arrays(self, dfa: DFA) -> None:
        scales, F = dfa.fluctuation_curve(_iid_returns())
        assert len(scales) == len(F)
        assert len(scales) > 0
        assert np.all(F > 0)

    def test_fluctuation_monotone_scales(self, dfa: DFA) -> None:
        scales, _ = dfa.fluctuation_curve(_iid_returns())
        assert np.all(np.diff(scales) > 0)

    def test_fit_rolling_returns_series(self, dfa: DFA) -> None:
        s = pd.Series(_iid_returns(n=300))
        out = dfa.fit_rolling(s, window=100, step=10)
        assert isinstance(out, pd.Series)
        assert len(out) == len(s)

    def test_fit_rolling_values_in_range(self, dfa: DFA) -> None:
        s = pd.Series(_iid_returns(n=300))
        out = dfa.fit_rolling(s, window=100, step=10).dropna()
        assert ((out > 0) & (out < 1)).all()

    def test_order_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            DFA(order=0)


# WhittleMLE

class TestWhittleMLE:
    def test_fit_returns_result(self, whi: WhittleMLE) -> None:
        res = whi.fit(_iid_returns())
        assert isinstance(res, WhittleResult)

    def test_fit_iid_near_half(self, whi: WhittleMLE) -> None:
        res = whi.fit(_iid_returns(n=2000, seed=13))
        assert 0.3 < res.H < 0.7, f"i.i.d. Whittle H expected near 0.5, got {res.H:.3f}"

    def test_h_bounded(self, whi: WhittleMLE) -> None:
        for seed in range(5):
            res = whi.fit(_iid_returns(n=500, seed=seed))
            assert 0.0 < res.H < 1.0

    def test_se_positive(self, whi: WhittleMLE) -> None:
        res = whi.fit(_iid_returns())
        assert res.se > 0

    def test_ci_order(self, whi: WhittleMLE) -> None:
        res = whi.fit(_iid_returns())
        assert res.ci_lower < res.H < res.ci_upper

    def test_n_freqs_positive(self, whi: WhittleMLE) -> None:
        res = whi.fit(_iid_returns())
        assert res.n_freqs > 0

    def test_t_statistic_iid_small(self, whi: WhittleMLE) -> None:
        """For i.i.d. data, |t| should usually be small."""
        res = whi.fit(_iid_returns(n=2000, seed=5))
        assert abs(res.t_statistic) < 5.0

    def test_periodogram_returns_arrays(self, whi: WhittleMLE) -> None:
        freqs, I = whi.periodogram(_iid_returns())
        assert len(freqs) == len(I)
        assert np.all(freqs > 0)
        assert np.all(I >= 0)

    def test_fit_rolling_returns_dataframe(self, whi: WhittleMLE) -> None:
        s = pd.Series(_iid_returns(n=400))
        out = whi.fit_rolling(s, window=150, step=20)
        assert isinstance(out, pd.DataFrame)
        assert "H" in out.columns
        assert "se" in out.columns

    def test_fit_rolling_h_in_range(self, whi: WhittleMLE) -> None:
        s = pd.Series(_iid_returns(n=400))
        out = whi.fit_rolling(s, window=150, step=20)["H"].dropna()
        assert ((out > 0) & (out < 1)).all()

    def test_invalid_bandwidth_exp(self) -> None:
        with pytest.raises(ValueError, match="bandwidth_exp"):
            WhittleMLE(bandwidth_exp=0.3)


# H Recovery Integration Test

class TestHRecovery:
    """
    Verify all three estimators recover approximately the correct H from
    synthetic fBm increments.

    Tolerance is deliberately generous (±0.20) because:
    - Series length is only 512 (limited by Cholesky cost in CI).
    - All classical estimators have O(1/sqrt(n)) error.
    - We only need to confirm the estimators are *directionally* correct,
      not sub-percent accurate — the FNO will handle that.
    """

    TOL = 0.20   # maximum acceptable |H_hat - H_true|

    @pytest.mark.parametrize("H_true", [0.10, 0.20, 0.35])
    def test_rs_recovery(self, H_true: float) -> None:
        x = _fbm_returns(H_true, n=512, seed=int(H_true * 100))
        rs = RSAnalysis(n_scales=20)
        H_hat = rs.fit(x)
        err = abs(H_hat - H_true)
        assert err < self.TOL, (
            f"R/S: H_true={H_true}, H_hat={H_hat:.3f}, err={err:.3f} > tol={self.TOL}"
        )

    @pytest.mark.parametrize("H_true", [0.10, 0.20, 0.35])
    def test_dfa_recovery(self, H_true: float) -> None:
        x = _fbm_returns(H_true, n=512, seed=int(H_true * 100) + 1)
        dfa = DFA(order=1, n_scales=20)
        H_hat = dfa.fit(x)
        err = abs(H_hat - H_true)
        assert err < self.TOL, (
            f"DFA: H_true={H_true}, H_hat={H_hat:.3f}, err={err:.3f} > tol={self.TOL}"
        )

    @pytest.mark.parametrize("H_true", [0.10, 0.20, 0.35])
    def test_whittle_recovery(self, H_true: float) -> None:
        x = _fbm_returns(H_true, n=512, seed=int(H_true * 100) + 2)
        whi = WhittleMLE()
        res = whi.fit(x)
        err = abs(res.H - H_true)
        assert err < self.TOL, (
            f"Whittle: H_true={H_true}, H_hat={res.H:.3f}, err={err:.3f} > tol={self.TOL}"
        )

    def test_rough_lower_than_persistent(self) -> None:
        """H=0.10 estimate should be lower than H=0.40 estimate for all methods."""
        x_rough = _fbm_returns(0.10, n=512, seed=1)
        x_persist = _fbm_returns(0.40, n=512, seed=2)
        for estimator, name in [
            (RSAnalysis(), "R/S"),
            (DFA(), "DFA"),
        ]:
            H_r = estimator.fit(x_rough)
            H_p = estimator.fit(x_persist)
            assert H_r < H_p, (
                f"{name}: H(rough={H_r:.3f}) should be < H(persistent={H_p:.3f})"
            )

        whi = WhittleMLE()
        H_r = whi.fit(x_rough).H
        H_p = whi.fit(x_persist).H
        assert H_r < H_p, f"Whittle: H(rough={H_r:.3f}) should be < H(persistent={H_p:.3f})"


# Dataclass contract

class TestHurstResult:
    def test_bool_true_if_passed(self) -> None:
        r = HurstResult(H=0.3, ci_lower=0.2, ci_upper=0.4,
                        r_squared=0.95, n_scales=15, method="R/S")
        assert repr(r)   # just check it doesn't crash

class TestWhittleResultContract:
    def test_t_statistic(self) -> None:
        r = WhittleResult(H=0.3, se=0.05, ci_lower=0.2, ci_upper=0.4,
                          objective=-1.0, n_freqs=30, n_obs=512)
        expected_t = (0.3 - 0.5) / 0.05
        assert abs(r.t_statistic - expected_t) < 1e-10

    def test_repr(self) -> None:
        r = WhittleResult(H=0.2, se=0.03, ci_lower=0.14, ci_upper=0.26,
                          objective=0.5, n_freqs=18, n_obs=300)
        assert "WhittleResult" in repr(r)
        assert "0.2" in repr(r)

