"""Tests for the rough-Heston SPDE simulator and its fractional-noise properties."""

from __future__ import annotations

import numpy as np
import pytest

from wsq_trading.hurst import WhittleMLE
from wsq_trading.spde import (
    SPDESimulator,
    _fbm_covariance,
    _fbm_increments_cholesky,
)


@pytest.fixture()
def sim() -> SPDESimulator:
    """Small simulator for fast unit tests (T=63, 20 paths per H)."""
    return SPDESimulator(
        T=63,
        dt=1 / 252,
        n_paths_per_hurst=20,
        hurst_grid=[0.10, 0.30],
        num_workers=1,   # single-process for test reliability
    )


class TestSimulatePath:
    def test_returns_dict(self, sim: SPDESimulator) -> None:
        result = sim.simulate_path(H=0.20, seed=0)
        assert isinstance(result, dict)

    def test_correct_keys(self, sim: SPDESimulator) -> None:
        result = sim.simulate_path(H=0.20, seed=0)
        assert "H" in result
        assert "log_returns" in result
        assert "variance" in result

    def test_output_shape(self, sim: SPDESimulator) -> None:
        result = sim.simulate_path(H=0.20, seed=0)
        assert result["log_returns"].shape == (sim.T,)
        assert result["variance"].shape == (sim.T,)

    def test_variance_nonneg(self, sim: SPDESimulator) -> None:
        result = sim.simulate_path(H=0.20, seed=0)
        assert np.all(result["variance"] >= 0)

    def test_h_label_correct(self, sim: SPDESimulator) -> None:
        result = sim.simulate_path(H=0.35, seed=0)
        assert result["H"] == pytest.approx(0.35)

    def test_return_dataframe_flag(self, sim: SPDESimulator) -> None:
        import pandas as pd
        df = sim.simulate_path(H=0.20, seed=0, return_dataframe=True)
        assert isinstance(df, pd.DataFrame)
        assert "log_return" in df.columns
        assert "H" in df.columns

    def test_reproducible_with_same_seed(self, sim: SPDESimulator) -> None:
        r1 = sim.simulate_path(H=0.15, seed=42)
        r2 = sim.simulate_path(H=0.15, seed=42)
        np.testing.assert_array_equal(r1["log_returns"], r2["log_returns"])

    def test_different_seeds_differ(self, sim: SPDESimulator) -> None:
        r1 = sim.simulate_path(H=0.15, seed=1)
        r2 = sim.simulate_path(H=0.15, seed=2)
        assert not np.allclose(r1["log_returns"], r2["log_returns"])


class TestSimulateBatch:
    def test_returns_dataframe(self, sim: SPDESimulator) -> None:
        import pandas as pd
        df = sim.simulate_batch(H=0.20, n_paths=5)
        assert isinstance(df, pd.DataFrame)

    def test_columns(self, sim: SPDESimulator) -> None:
        df = sim.simulate_batch(H=0.20, n_paths=5)
        for col in ("path_id", "step", "log_return", "variance", "H"):
            assert col in df.columns, f"Missing column: {col}"

    def test_row_count(self, sim: SPDESimulator) -> None:
        n_paths = 5
        df = sim.simulate_batch(H=0.20, n_paths=n_paths)
        assert len(df) == n_paths * sim.T

    def test_h_column_constant(self, sim: SPDESimulator) -> None:
        df = sim.simulate_batch(H=0.30, n_paths=5)
        assert np.allclose(df["H"].to_numpy(), 0.30)

    def test_variance_nonneg(self, sim: SPDESimulator) -> None:
        df = sim.simulate_batch(H=0.10, n_paths=10)
        assert (df["variance"] >= 0).all()


class TestGenerateDataset:
    def test_returns_dataframe(self, sim: SPDESimulator, tmp_path) -> None:
        sim.output_file = tmp_path / "test_paths.parquet"
        df = sim.generate_dataset(save=False)
        import pandas as pd
        assert isinstance(df, pd.DataFrame)

    def test_correct_h_values(self, sim: SPDESimulator, tmp_path) -> None:
        sim.output_file = tmp_path / "test_paths.parquet"
        df = sim.generate_dataset(save=False)
        assert set(df["H"].unique()) == set(sim.hurst_grid)

    def test_total_rows(self, sim: SPDESimulator, tmp_path) -> None:
        sim.output_file = tmp_path / "test_paths.parquet"
        df = sim.generate_dataset(save=False)
        expected = len(sim.hurst_grid) * sim.n_paths_per_hurst * sim.T
        assert len(df) == expected

    def test_save_creates_parquet(self, sim: SPDESimulator, tmp_path) -> None:
        out_file = tmp_path / "paths.parquet"
        sim.output_file = out_file
        sim.generate_dataset(save=True)
        assert out_file.exists()

    def test_load_dataset(self, sim: SPDESimulator, tmp_path) -> None:
        out_file = tmp_path / "paths.parquet"
        sim.output_file = out_file
        df_gen = sim.generate_dataset(save=True)
        df_load = sim.load_dataset()
        assert len(df_gen) == len(df_load)

    def test_no_regenerate_if_exists(self, sim: SPDESimulator, tmp_path) -> None:
        out_file = tmp_path / "paths.parquet"
        sim.output_file = out_file
        sim.generate_dataset(save=True)
        # Second call should just load existing file (fast path)
        df2 = sim.generate_dataset(save=True, force_regenerate=False)
        assert not df2.empty


class TestAsWindows:
    def test_output_shapes(self, sim: SPDESimulator, tmp_path) -> None:
        sim.output_file = tmp_path / "paths.parquet"
        df = sim.generate_dataset(save=False)
        X, y = sim.as_windows(df=df, window_size=21)
        assert X.ndim == 3
        assert X.shape[2] == 1         # single feature channel
        assert X.shape[1] == 21        # window length
        assert y.ndim == 1
        assert len(X) == len(y)

    def test_labels_in_hurst_grid(self, sim: SPDESimulator, tmp_path) -> None:
        sim.output_file = tmp_path / "paths.parquet"
        df = sim.generate_dataset(save=False)
        _, y = sim.as_windows(df=df, window_size=21)
        for label in np.unique(y):
            assert any(abs(label - H) < 1e-6 for H in sim.hurst_grid)


_DT: float = 1 / 252        # daily time step
_T: int = 252               # path length (one trading year)
_N_PATHS: int = 50          # paths per parametrized test
_WHITTLE = WhittleMLE(bandwidth_exp=0.75)   # slightly wider band for T=252


# Shared helpers

def _fgn(H: float, n: int = _T, dt: float = _DT, seed: int = 0) -> np.ndarray:
    """Generate one fGn path of length *n* via the Cholesky method."""
    return _fbm_increments_cholesky(H, n, dt, np.random.default_rng(seed))


def _many_fgn(H: float, n_paths: int = _N_PATHS) -> np.ndarray:
    """Return shape (n_paths, T) array of independent fGn paths."""
    return np.stack([_fgn(H, seed=i) for i in range(n_paths)])


# 1. Covariance matrix structure

class TestFGnCovarianceMatrix:
    """
    The fGn covariance matrix is the only non-trivial mathematical object in
    the Cholesky method.  These tests verify its exact analytical properties
    before any Monte Carlo noise is introduced.

    Theoretical autocovariance of fGn at lag k:
        gamma(k) = 0.5 * dt^(2H) * (|k+1|^(2H) - 2|k|^(2H) + |k-1|^(2H))

    At k=0 (variance): gamma(0) = dt^(2H)
    At k=1:            gamma(1) = 0.5 * dt^(2H) * (2^(2H) - 2)
    """

    @pytest.mark.parametrize("H", [0.10, 0.25, 0.40])
    def test_diagonal_equals_theoretical_variance(self, H: float) -> None:
        """Cov[k, k] = dt^(2H) for every k."""
        n = 40
        cov = _fbm_covariance(H, n, _DT)
        expected = np.full(n, _DT ** (2 * H))
        np.testing.assert_allclose(
            np.diag(cov), expected, rtol=1e-9,
            err_msg=f"Diagonal variance wrong for H={H}",
        )

    @pytest.mark.parametrize("H", [0.10, 0.25, 0.40])
    def test_lag1_covariance_matches_theory(self, H: float) -> None:
        """Cov[k, k+1] = 0.5 * dt^(2H) * (2^(2H) - 2)."""
        n = 40
        cov = _fbm_covariance(H, n, _DT)
        empirical = cov[0, 1]
        theoretical = 0.5 * (_DT ** (2 * H)) * (2 ** (2 * H) - 2)
        np.testing.assert_allclose(
            empirical, theoretical, rtol=1e-9,
            err_msg=f"Lag-1 covariance wrong for H={H}",
        )

    @pytest.mark.parametrize("H", [0.10, 0.25, 0.40])
    def test_matrix_is_symmetric(self, H: float) -> None:
        """Covariance matrices must be symmetric."""
        cov = _fbm_covariance(H, 30, _DT)
        np.testing.assert_array_equal(
            cov, cov.T, err_msg=f"Covariance matrix not symmetric for H={H}",
        )

    @pytest.mark.parametrize("H", [0.10, 0.25, 0.40])
    def test_matrix_is_positive_semidefinite(self, H: float) -> None:
        """All eigenvalues >= -1e-8 (the nugget keeps it numerically PD)."""
        cov = _fbm_covariance(H, 40, _DT)
        min_eig = float(np.linalg.eigvalsh(cov).min())
        assert min_eig >= -1e-8, (
            f"H={H}: smallest eigenvalue {min_eig:.2e} < -1e-8; matrix is not PSD."
        )

    @pytest.mark.parametrize("H_low,H_high", [(0.10, 0.30), (0.20, 0.40)])
    def test_lag1_covariance_magnitude_larger_for_rougher_H(
        self, H_low: float, H_high: float
    ) -> None:
        """
        Both H < 0.5 give negative lag-1 covariance.  The magnitude should be
        larger for the smaller H (further from 0.5 => stronger anti-persistence).
        """
        cov_low = _fbm_covariance(H_low, 30, _DT)
        cov_high = _fbm_covariance(H_high, 30, _DT)
        lag1_low = cov_low[0, 1]
        lag1_high = cov_high[0, 1]
        assert lag1_low < 0, (
            f"H={H_low}: lag-1 cov should be negative, got {lag1_low:.6f}"
        )
        assert lag1_high < 0, (
            f"H={H_high}: lag-1 cov should be negative, got {lag1_high:.6f}"
        )
        assert abs(lag1_low) > abs(lag1_high), (
            f"|gamma(1)| for H={H_low} ({abs(lag1_low):.6f}) should exceed "
            f"|gamma(1)| for H={H_high} ({abs(lag1_high):.6f})"
        )


# 2. Variance scaling

class TestFGnVarianceScaling:
    """
    The variance of each fGn increment is dt^(2H).  For dt = 1/252 < 1,
    this is a *decreasing* function of H:

        H=0.10 -> dt^(0.20) ~= 0.369
        H=0.25 -> dt^(0.50) ~= 0.063
        H=0.40 -> dt^(0.80) ~= 0.011

    We check the empirical mean squared increment against these predictions,
    and verify the ordering across H values.
    """

    @pytest.mark.parametrize("H", [0.10, 0.25, 0.40])
    def test_empirical_variance_within_20_percent_of_theory(self, H: float) -> None:
        """
        Mean squared increment across 50 paths x 252 steps should match
        dt^(2H) within 20% (generous relative to the ~1.3% SE of the estimator).
        """
        paths = _many_fgn(H)
        empirical = float(np.mean(paths ** 2))
        theoretical = _DT ** (2 * H)
        rel_err = abs(empirical - theoretical) / theoretical
        assert rel_err < 0.20, (
            f"H={H}: relative error {rel_err:.3f} > 20%. "
            f"Empirical={empirical:.6f}, theoretical={theoretical:.6f}."
        )

    def test_variance_decreases_as_H_increases_for_dt_lt_1(self) -> None:
        """
        For dt < 1: dt^(2H) is strictly decreasing in H, so
        var(H=0.10) > var(H=0.20) > var(H=0.30) > var(H=0.40).
        """
        H_seq = [0.10, 0.20, 0.30, 0.40]
        variances = [float(np.mean(_many_fgn(H) ** 2)) for H in H_seq]
        for i in range(len(variances) - 1):
            assert variances[i] > variances[i + 1], (
                f"Variance ordering violated: "
                f"E[X^2|H={H_seq[i]}] = {variances[i]:.6f} "
                f"<= E[X^2|H={H_seq[i+1]}] = {variances[i+1]:.6f}."
            )


# 3. Autocorrelation structure

class TestFGnAutocorrelation:
    """
    The theoretical lag-1 autocorrelation of fGn is:
        rho(1) = 2^(2H-1) - 1

    For H < 0.5: rho(1) < 0  (anti-persistent / rough)
    For H = 0.5: rho(1) = 0  (white noise)
    For H > 0.5: rho(1) > 0  (persistent / long-memory)

    As H decreases from 0.5 toward 0, |rho(1)| increases monotonically.
    """

    @staticmethod
    def _lag1_autocorr(path: np.ndarray) -> float:
        return float(np.corrcoef(path[:-1], path[1:])[0, 1])

    @staticmethod
    def _theoretical_lag1(H: float) -> float:
        return 2.0 ** (2 * H - 1) - 1.0

    @pytest.mark.parametrize("H", [0.10, 0.20, 0.35])
    def test_rough_fgn_has_negative_lag1_autocorr(self, H: float) -> None:
        """H < 0.5 must give a negative mean lag-1 autocorrelation."""
        paths = _many_fgn(H)
        mean_rho = float(np.mean([self._lag1_autocorr(p) for p in paths]))
        assert mean_rho < 0, (
            f"H={H} < 0.5 should give negative lag-1 autocorr; got {mean_rho:.4f}."
        )

    def test_autocorr_magnitude_monotone_as_H_decreases(self) -> None:
        """
        |rho(1)| increases monotonically as H decreases from 0.5 toward 0.
        Sequence H = 0.40, 0.30, 0.20, 0.10 should give strictly increasing |rho|.
        """
        H_seq = [0.40, 0.30, 0.20, 0.10]
        magnitudes = []
        for H in H_seq:
            rhos = [self._lag1_autocorr(p) for p in _many_fgn(H, n_paths=30)]
            magnitudes.append(abs(float(np.mean(rhos))))

        for i in range(len(magnitudes) - 1):
            assert magnitudes[i] < magnitudes[i + 1], (
                f"|rho(1)| not monotone: "
                f"H={H_seq[i]} -> {magnitudes[i]:.4f} "
                f">= H={H_seq[i+1]} -> {magnitudes[i+1]:.4f}."
            )

    @pytest.mark.parametrize("H", [0.10, 0.25, 0.40])
    def test_lag1_autocorr_close_to_theoretical_value(self, H: float) -> None:
        """
        Mean empirical rho(1) should match 2^(2H-1) - 1 within +/- 0.08.
        With 50 paths, the SE of the mean rho is ~0.01-0.02, so +/-0.08 ~= 4-8 SE.
        """
        paths = _many_fgn(H)
        empirical_rho = float(np.mean([self._lag1_autocorr(p) for p in paths]))
        theoretical_rho = self._theoretical_lag1(H)
        err = abs(empirical_rho - theoretical_rho)
        assert err < 0.08, (
            f"H={H}: empirical rho(1)={empirical_rho:.4f}, "
            f"theoretical={theoretical_rho:.4f}, error={err:.4f} > 0.08."
        )


# 4. Spectral power law

class TestSpectralPowerLaw:
    """
    fGn has spectral density S(f) ~ f^(1 - 2H) for small f (Robinson 1995).

    Consequently the log-periodogram slope (log I vs log f) is approximately:
        slope ~= 1 - 2H

    For H < 0.5: slope > 0  (more power at high frequencies, anti-persistent)
    For H = 0.5: slope ~= 0  (flat spectrum, white noise)
    For H > 0.5: slope < 0  (more power at low frequencies, long-memory)

    We estimate the slope via OLS on the lowest-quarter of Fourier frequencies,
    then average across 50 independent paths.
    """

    @staticmethod
    def _log_periodogram_slope(path: np.ndarray, low_frac: float = 0.25) -> float:
        """OLS slope of log I(f) vs log f, restricted to lowest-frequency ordinates."""
        n = len(path)
        fft = np.fft.rfft(path - path.mean())
        I = (np.abs(fft[1:]) ** 2) / n
        freqs = np.fft.rfftfreq(n)[1:]
        cutoff = max(4, int(len(freqs) * low_frac))
        log_f = np.log(freqs[:cutoff])
        log_I = np.log(I[:cutoff] + 1e-30)
        slope, _ = np.polyfit(log_f, log_I, 1)
        return float(slope)

    @pytest.mark.parametrize("H", [0.10, 0.20, 0.35])
    def test_log_periodogram_slope_positive_for_rough_fgn(self, H: float) -> None:
        """
        For H < 0.5 (rough), the theoretical slope 1-2H is positive.
        The mean slope over 30 paths should also be positive.
        """
        slopes = [self._log_periodogram_slope(p) for p in _many_fgn(H, n_paths=30)]
        mean_slope = float(np.mean(slopes))
        assert mean_slope > 0, (
            f"H={H}: mean log-periodogram slope {mean_slope:.4f} <= 0; "
            f"expected > 0 for rough (H < 0.5) fGn."
        )

    def test_slope_magnitude_increases_as_H_moves_away_from_half(self) -> None:
        """
        |slope| = |1-2H| increases as H moves away from 0.5.
        For H in {0.35, 0.20, 0.10}, |slope| should be strictly increasing.
        """
        H_seq = [0.35, 0.20, 0.10]
        magnitudes = []
        for H in H_seq:
            slopes = [self._log_periodogram_slope(p) for p in _many_fgn(H, n_paths=30)]
            magnitudes.append(abs(float(np.mean(slopes))))

        for i in range(len(magnitudes) - 1):
            assert magnitudes[i] < magnitudes[i + 1], (
                f"Slope magnitude not monotone: "
                f"|slope(H={H_seq[i]})| = {magnitudes[i]:.4f} "
                f">= |slope(H={H_seq[i+1]})| = {magnitudes[i+1]:.4f}."
            )

    @pytest.mark.parametrize("H", [0.10, 0.25, 0.40])
    def test_slope_value_within_tolerance_of_theory(self, H: float) -> None:
        """
        Mean slope across 50 paths should be within +/- 0.30 of (1-2H).
        The within-path SD of slope estimates is typically ~0.4-0.6, so
        the SE of the mean across 50 paths is ~0.06-0.09; +/-0.30 ~= 3-5 SE.
        """
        slopes = [self._log_periodogram_slope(p) for p in _many_fgn(H)]
        mean_slope = float(np.mean(slopes))
        theoretical = 1.0 - 2.0 * H
        err = abs(mean_slope - theoretical)
        assert err < 0.30, (
            f"H={H}: mean slope {mean_slope:.4f} vs theoretical {theoretical:.4f}; "
            f"error {err:.4f} > 0.30."
        )


# 5. Whittle H recovery

class TestWhittleHRecovery:
    """
    Gold-standard Phase 1 gate test.

    Whittle MLE (bandwidth_exp=0.75) applied to pure fGn paths generated by
    the simulator should recover the true H with high accuracy.

    With T=252, m ~= 252^0.75 ~= 76 frequencies, asymptotic SE ~= 0.057.
    We require:
        - Median |error| < 0.10  (~1.8 SE)  across 50 paths per H value.
        - Mean Whittle estimate < 0.5 for all tested H (all are rough).
        - Monotone ordering: H_hat(0.10) < H_hat(0.40).
        - 95% CIs cover the true H in >= 80% of paths.
    """

    @pytest.mark.parametrize("H", [0.10, 0.20, 0.30, 0.40])
    def test_whittle_median_absolute_error_below_010(self, H: float) -> None:
        """Median |H_hat - H_true| < 0.10 over 50 independent fGn paths."""
        paths = _many_fgn(H)
        estimates = np.array([_WHITTLE.fit(p).H for p in paths])
        mae = float(np.median(np.abs(estimates - H)))
        assert mae < 0.10, (
            f"H={H}: median |error| = {mae:.4f} > 0.10. "
            f"Mean estimate = {estimates.mean():.4f}."
        )

    @pytest.mark.parametrize("H", [0.10, 0.20, 0.30, 0.40])
    def test_whittle_classifies_paths_as_rough(self, H: float) -> None:
        """Mean Whittle estimate should be < 0.5 for all H < 0.5 test values."""
        paths = _many_fgn(H)
        mean_est = float(np.mean([_WHITTLE.fit(p).H for p in paths]))
        assert mean_est < 0.5, (
            f"H={H}: mean Whittle estimate {mean_est:.4f} >= 0.5; "
            "should be classified as rough (anti-persistent)."
        )

    def test_whittle_ordering_preserved_across_hurst_values(self) -> None:
        """
        Mean Whittle estimate for H=0.10 paths < mean for H=0.40 paths.
        This confirms the estimator has discriminatory power across the rough regime.
        """
        est_low = float(np.mean([_WHITTLE.fit(p).H for p in _many_fgn(0.10, n_paths=30)]))
        est_high = float(np.mean([_WHITTLE.fit(p).H for p in _many_fgn(0.40, n_paths=30)]))
        assert est_low < est_high, (
            f"Whittle should give H_hat(H=0.10) < H_hat(H=0.40); "
            f"got {est_low:.4f} vs {est_high:.4f}."
        )

    def test_whittle_ordering_across_four_h_values(self) -> None:
        """
        Mean Whittle estimates should be strictly increasing across
        H in {0.10, 0.20, 0.30, 0.40} -- a 4-point monotonicity check.
        """
        H_seq = [0.10, 0.20, 0.30, 0.40]
        means = [
            float(np.mean([_WHITTLE.fit(p).H for p in _many_fgn(H, n_paths=30)]))
            for H in H_seq
        ]
        for i in range(len(means) - 1):
            assert means[i] < means[i + 1], (
                f"Whittle ordering violated: "
                f"H_hat(H={H_seq[i]}) = {means[i]:.4f} "
                f">= H_hat(H={H_seq[i+1]}) = {means[i+1]:.4f}."
            )

    @pytest.mark.parametrize("H", [0.15, 0.30])
    def test_whittle_95_ci_coverage_at_least_80_percent(self, H: float) -> None:
        """
        The nominal 95% CI should contain H_true in >= 80% of paths.
        The lower threshold (80% vs 95%) reflects the small-T approximation.
        """
        paths = _many_fgn(H)
        results = [_WHITTLE.fit(p) for p in paths]
        covered = sum(1 for r in results if r.ci_lower <= H <= r.ci_upper)
        coverage = covered / len(paths)
        assert coverage >= 0.80, (
            f"H={H}: 95% CI coverage = {coverage:.2%} < 80%. "
            "Too many paths have CIs that miss the true H."
        )


# 6. End-to-end SPDE regime ordering

class TestSPDERegimeOrdering:
    """
    End-to-end test on the full rough Heston simulator (public API).

    The roughness of the Heston model is encoded in the *variance process*
    V(t), not the log-returns.  The drift term kappa*(theta - V)*dt adds
    positive serial correlation to the variance *level*, so Whittle applied
    directly to V(t) will typically give H_hat > 0.5 regardless of the input H.

    Instead, we use the variance *increments* dV = np.diff(V(t)), which strip
    out the first-order mean-reversion and expose the fGn noise driver.
    Empirically:
        H_input=0.10 -> mean Whittle H_hat(dV) ~= 0.21
        H_input=0.40 -> mean Whittle H_hat(dV) ~= 0.38

    We verify:
        (a) Ordering: H_hat(dV | H=H_low) < H_hat(dV | H=H_high).
        (b) Classification: H_hat(dV | H=0.10) < 0.5 (genuinely rough).
    """

    @pytest.fixture()
    def sim(self) -> SPDESimulator:
        """Minimal simulator configured for fast test execution."""
        return SPDESimulator(T=_T, dt=_DT, n_paths_per_hurst=1, num_workers=1)

    @pytest.mark.parametrize("H_low,H_high", [(0.10, 0.35), (0.15, 0.40)])
    def test_lower_hurst_yields_lower_whittle_on_variance_increments(
        self, sim: SPDESimulator, H_low: float, H_high: float
    ) -> None:
        """
        Mean Whittle H_hat on variance increments (dV) for H_low should be
        strictly less than for H_high.
        """
        n_compare = 20
        ests_low = [
            _WHITTLE.fit(np.diff(sim.simulate_path(H=H_low, seed=i)["variance"])).H
            for i in range(n_compare)
        ]
        ests_high = [
            _WHITTLE.fit(np.diff(sim.simulate_path(H=H_high, seed=i)["variance"])).H
            for i in range(n_compare)
        ]
        mean_low = float(np.mean(ests_low))
        mean_high = float(np.mean(ests_high))
        assert mean_low < mean_high, (
            f"SPDE dV Whittle ordering violated: "
            f"H_input={H_low} -> H_hat={mean_low:.4f}, "
            f"H_input={H_high} -> H_hat={mean_high:.4f}."
        )

    def test_variance_increments_classified_as_rough(
        self, sim: SPDESimulator
    ) -> None:
        """
        Whittle MLE on variance increments (dV = np.diff(V)) should classify
        paths simulated with H=0.10 as rough (H_hat < 0.5).

        Why increments, not levels?  The Heston drift adds positive serial
        correlation to V(t), making Whittle on V appear persistent even for
        very rough inputs.  Differencing strips this out and exposes the fGn
        noise driver, giving H_hat ~= 0.21 for H_input=0.10.
        """
        n_paths = 20
        ests = [
            _WHITTLE.fit(np.diff(sim.simulate_path(H=0.10, seed=i)["variance"])).H
            for i in range(n_paths)
        ]
        mean_est = float(np.mean(ests))
        assert mean_est < 0.5, (
            f"H_input=0.10 (deeply rough): mean Whittle H_hat on dV = {mean_est:.4f} >= 0.5; "
            "variance increments should be classified as rough (anti-persistent)."
        )

