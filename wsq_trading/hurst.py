"""Classical Hurst-exponent estimators: rescaled range, DFA and Whittle MLE."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar

from wsq_trading import config
from wsq_trading.constants import HURST_MAX, HURST_MIN
from wsq_trading.utils import get_logger

log = get_logger(__name__)


@dataclass
class HurstResult:
    """Container for a Hurst exponent estimate with diagnostics.

    Attributes
    ----------
    H : float
        Point estimate of the Hurst exponent.
    ci_lower : float
        Lower bound of the confidence interval.
    ci_upper : float
        Upper bound of the confidence interval.
    r_squared : float
        R² of the log-log linear fit (goodness of fit).
    n_scales : int
        Number of scale levels used in the regression.
    method : str
        Name of the estimator.
    """

    H: float
    ci_lower: float
    ci_upper: float
    r_squared: float
    n_scales: int
    method: str

    def __repr__(self) -> str:
        return (
            f"HurstResult(H={self.H:.4f}, "
            f"CI=[{self.ci_lower:.4f}, {self.ci_upper:.4f}], "
            f"R²={self.r_squared:.4f}, method='{self.method}')"
        )


class RSAnalysis:
    """Rescaled Range (R/S) Hurst exponent estimator.

    Parameters
    ----------
    min_window : int, optional
        Smallest block size to use.  Defaults to 10.
    max_window_frac : float, optional
        Largest block size as a fraction of series length.
        Defaults to 0.5 (half the series).
    n_scales : int, optional
        Number of logarithmically-spaced scales between ``min_window``
        and ``max_window``.  Defaults to 20.
    min_blocks : int, optional
        Discard any scale where fewer than ``min_blocks`` complete blocks
        are available.  Defaults to 4.
    apply_bias_correction : bool, optional
        Apply the Anis–Lloyd–Peters small-sample correction.  Defaults to
        ``True``.
    """

    def __init__(
        self,
        min_window: int = 10,
        max_window_frac: float = 0.5,
        n_scales: int = 20,
        min_blocks: int = 4,
        apply_bias_correction: bool = False,
    ) -> None:
        self._min_window = min_window
        self._max_window_frac = max_window_frac
        self._n_scales = n_scales
        self._min_blocks = min_blocks
        self._bias_correction = apply_bias_correction


    def fit(self, series: np.ndarray | pd.Series) -> float:
        """Estimate H from a 1-D time series of stationary increments.

        Parameters
        ----------
        series : array-like
            Log-returns or other stationary increments.  NaNs are dropped.

        Returns
        -------
        float
            Hurst exponent estimate, clipped to ``[HURST_MIN, HURST_MAX]``.
        """
        x = _prepare(series)
        scales, rs_vals = self._compute_rs_curve(x)
        if len(scales) < 3:
            log.warning("RSAnalysis.fit: too few scales (%d); returning 0.5.", len(scales))
            return 0.5
        H, _, _, _, _ = _log_log_fit(scales, rs_vals)
        return float(np.clip(H, HURST_MIN, HURST_MAX))

    def fit_with_ci(
        self,
        series: np.ndarray | pd.Series,
        confidence: float = 0.95,
        n_jackknife: int | None = None,
    ) -> HurstResult:
        """Estimate H with a jackknife confidence interval.

        The jackknife sequentially drops each observation and re-estimates H,
        giving a distribution of estimates from which standard error and CI
        are derived.  This is fast and requires no distributional assumptions.

        Parameters
        ----------
        series : array-like
            Stationary increments.
        confidence : float
            Confidence level for the interval (default 0.95).
        n_jackknife : int, optional
            Number of jackknife replications.  Defaults to min(len(series), 100).
        """
        x = _prepare(series)
        scales, rs_vals = self._compute_rs_curve(x)
        H, _, r_sq, _, _ = _log_log_fit(scales, rs_vals)

        # Jackknife by dropping one scale at a time (stable & fast)
        n_drop = min(len(scales), n_jackknife or min(len(x), 100))
        jack_H = np.zeros(n_drop)
        idx = np.round(np.linspace(0, len(scales) - 1, n_drop)).astype(int)
        for i, drop in enumerate(idx):
            mask = np.ones(len(scales), dtype=bool)
            mask[drop] = False
            if mask.sum() >= 2:
                h_j, *_ = _log_log_fit(scales[mask], rs_vals[mask])
                jack_H[i] = h_j
            else:
                jack_H[i] = H

        se = np.std(jack_H, ddof=1) * np.sqrt(n_drop)
        z = stats.norm.ppf((1 + confidence) / 2)
        H_clipped = float(np.clip(H, HURST_MIN, HURST_MAX))
        return HurstResult(
            H=H_clipped,
            ci_lower=float(np.clip(H - z * se, HURST_MIN, HURST_MAX)),
            ci_upper=float(np.clip(H + z * se, HURST_MIN, HURST_MAX)),
            r_squared=float(r_sq),
            n_scales=len(scales),
            method="R/S",
        )

    def fit_rolling(
        self,
        series: np.ndarray | pd.Series,
        window: int | None = None,
        step: int = 1,
    ) -> pd.Series:
        """Compute a rolling H estimate using a sliding window.

        Parameters
        ----------
        series : array-like
            Full log-return series.
        window : int, optional
            Number of observations per window.  Defaults to
            ``config.ROLLING_STATS_WINDOW``.
        step : int
            Stride between windows (default 1 = fully rolling).

        Returns
        -------
        pd.Series
            H estimates indexed like ``series``.  Leading values are NaN
            until the first full window is available.
        """
        s = pd.Series(_prepare(series)) if not isinstance(series, pd.Series) else series.dropna()
        window = window or config.ROLLING_STATS_WINDOW
        result = pd.Series(index=s.index, dtype=float)

        positions = range(window - 1, len(s), step)
        for end in positions:
            start = end - window + 1
            h = self.fit(s.iloc[start : end + 1].to_numpy())
            result.iloc[end] = h

        return result

    def rs_curve(
        self,
        series: np.ndarray | pd.Series,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the raw (scales, R/S values) for plotting or diagnostics.

        Parameters
        ----------
        series : array-like
            Stationary increments.

        Returns
        -------
        scales : np.ndarray
            Block sizes used.
        rs_vals : np.ndarray
            Mean R/S at each scale.
        """
        x = _prepare(series)
        return self._compute_rs_curve(x)

    # Internal helpers

    def _compute_rs_curve(
        self, x: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the R/S statistic at each scale."""
        n = len(x)
        max_window = max(self._min_window + 1, int(n * self._max_window_frac))
        raw_scales = np.unique(
            np.round(
                np.logspace(
                    np.log10(self._min_window),
                    np.log10(max_window),
                    self._n_scales,
                )
            ).astype(int)
        )

        scales_out, rs_out = [], []
        for m in raw_scales:
            n_blocks = n // m
            if n_blocks < self._min_blocks:
                continue
            rs_blocks = []
            for k in range(n_blocks):
                block = x[k * m : (k + 1) * m]
                rs = _rs_single_block(block)
                if rs is not None:
                    rs_blocks.append(rs)
            if rs_blocks:
                mean_rs = np.mean(rs_blocks)
                if self._bias_correction:
                    mean_rs -= _anis_lloyd_correction(m)
                scales_out.append(m)
                rs_out.append(mean_rs)

        return np.array(scales_out, dtype=float), np.array(rs_out, dtype=float)


# Module-level helpers (no class state needed)

def _prepare(series: np.ndarray | pd.Series) -> np.ndarray:
    """Convert to a clean 1-D float64 array, dropping NaNs."""
    if isinstance(series, pd.Series):
        arr = series.dropna().to_numpy(dtype=float)
    else:
        arr = np.asarray(series, dtype=float)
        arr = arr[np.isfinite(arr)]
    if len(arr) < 8:
        raise ValueError(f"Series too short for Hurst estimation: {len(arr)} observations.")
    return arr


def _rs_single_block(block: np.ndarray) -> float | None:
    """Compute R/S for a single block.  Returns None if std is zero."""
    mean = block.mean()
    dev = block - mean
    cumdev = np.cumsum(dev)
    R = cumdev.max() - cumdev.min()
    S = block.std(ddof=1)
    if S < 1e-12:
        return None
    return R / S


def _anis_lloyd_correction(m: int) -> float:
    """Anis–Lloyd (1976) expected R/S for an i.i.d. process of length m.

    This is subtracted from the empirical R/S before log-log regression
    to remove the upward bias present in short samples.
    """
    if m <= 340:
        # Exact formula using Gamma function
        from scipy.special import gamma as g_fn
        k = np.arange(1, m)
        expected_range = np.sum(np.sqrt((m - k) / k))
        return (g_fn((m - 1) / 2) / (np.sqrt(np.pi) * g_fn(m / 2))) * expected_range
    else:
        # Asymptotic approximation for large m
        return np.sqrt(np.pi * m / 2)


def _log_log_fit(
    scales: np.ndarray, rs_vals: np.ndarray
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """OLS regression of log(R/S) on log(scale)."""
    valid = (scales > 0) & (rs_vals > 0)
    ls = np.log(scales[valid])
    lr = np.log(rs_vals[valid])
    if len(ls) < 2:
        return 0.5, 0.0, 0.0, ls, lr
    slope, intercept, r, _, _ = stats.linregress(ls, lr)
    return float(slope), float(intercept), float(r**2), ls, lr


class DFA:
    """Detrended Fluctuation Analysis Hurst estimator.

    Parameters
    ----------
    order : int, optional
        Polynomial order for local detrending.  1 = linear (DFA-1, default),
        2 = quadratic (DFA-2).
    min_window : int, optional
        Smallest segment size.  Must be > ``order``.  Defaults to 10.
    max_window_frac : float, optional
        Largest segment size as a fraction of series length.  Defaults to 0.5.
    n_scales : int, optional
        Number of logarithmically-spaced scales.  Defaults to 25.
    min_blocks : int, optional
        Discard scales with fewer complete segments than this.  Defaults to 4.
    overlap : bool, optional
        If ``True``, use overlapping windows (slower but uses more data per
        scale).  Defaults to ``False`` (non-overlapping, faster).
    """

    def __init__(
        self,
        order: int = 1,
        min_window: int = 10,
        max_window_frac: float = 0.5,
        n_scales: int = 25,
        min_blocks: int = 4,
        overlap: bool = False,
    ) -> None:
        if order < 1:
            raise ValueError("DFA order must be >= 1.")
        self._order = order
        self._min_window = max(min_window, order + 2)
        self._max_window_frac = max_window_frac
        self._n_scales = n_scales
        self._min_blocks = min_blocks
        self._overlap = overlap


    def fit(self, series: np.ndarray | pd.Series) -> float:
        """Estimate H from a 1-D time series of stationary increments.

        Parameters
        ----------
        series : array-like
            Log-returns or other stationary increments.  NaNs are dropped.

        Returns
        -------
        float
            Hurst exponent clipped to ``[HURST_MIN, HURST_MAX]``.
        """
        x = _prepare(series)
        scales, F = self._compute_fluctuation_curve(x)
        if len(scales) < 3:
            log.warning("DFA.fit: too few valid scales (%d); returning 0.5.", len(scales))
            return 0.5
        H, _, _, _, _ = _log_log_fit(scales, F)
        return float(np.clip(H, HURST_MIN, HURST_MAX))

    def fit_with_ci(
        self,
        series: np.ndarray | pd.Series,
        confidence: float = 0.95,
    ) -> HurstResult:
        """Estimate H with a jackknife confidence interval.

        Parameters
        ----------
        series : array-like
            Stationary increments.
        confidence : float
            Confidence level (default 0.95).
        """
        x = _prepare(series)
        scales, F = self._compute_fluctuation_curve(x)
        H, _, r_sq, _, _ = _log_log_fit(scales, F)

        # Jackknife over scales
        jack_H = []
        for drop_idx in range(len(scales)):
            mask = np.ones(len(scales), dtype=bool)
            mask[drop_idx] = False
            if mask.sum() >= 2:
                h_j, *_ = _log_log_fit(scales[mask], F[mask])
                jack_H.append(h_j)

        se = np.std(jack_H, ddof=1) * np.sqrt(len(jack_H)) if jack_H else 0.0
        z = stats.norm.ppf((1 + confidence) / 2)
        H_clip = float(np.clip(H, HURST_MIN, HURST_MAX))
        return HurstResult(
            H=H_clip,
            ci_lower=float(np.clip(H - z * se, HURST_MIN, HURST_MAX)),
            ci_upper=float(np.clip(H + z * se, HURST_MIN, HURST_MAX)),
            r_squared=float(r_sq),
            n_scales=int(len(scales)),
            method=f"DFA-{self._order}",
        )

    def fit_rolling(
        self,
        series: np.ndarray | pd.Series,
        window: int | None = None,
        step: int = 1,
    ) -> pd.Series:
        """Compute a rolling H estimate.

        Parameters
        ----------
        series : array-like
            Full log-return series.
        window : int, optional
            Window length.  Defaults to ``config.ROLLING_STATS_WINDOW``.
        step : int
            Stride between windows (default 1).

        Returns
        -------
        pd.Series
            H estimates indexed like ``series``.
        """
        s = (
            pd.Series(_prepare(series))
            if not isinstance(series, pd.Series)
            else series.dropna()
        )
        window = window or config.ROLLING_STATS_WINDOW
        result = pd.Series(index=s.index, dtype=float)

        for end in range(window - 1, len(s), step):
            start = end - window + 1
            h = self.fit(s.iloc[start : end + 1].to_numpy())
            result.iloc[end] = h

        return result

    def fluctuation_curve(
        self,
        series: np.ndarray | pd.Series,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the raw (scales, F values) for diagnostics or plotting.

        Parameters
        ----------
        series : array-like
            Stationary increments.

        Returns
        -------
        scales : np.ndarray
        F : np.ndarray
            RMS fluctuation at each scale.
        """
        x = _prepare(series)
        return self._compute_fluctuation_curve(x)

    # Internal helpers

    def _compute_fluctuation_curve(
        self, x: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Core DFA computation."""
        n = len(x)

        # 1. Profile (cumulative sum of mean-adjusted series)
        profile = np.cumsum(x - x.mean())

        max_window = max(self._min_window + 1, int(n * self._max_window_frac))
        raw_scales = np.unique(
            np.round(
                np.logspace(
                    np.log10(self._min_window),
                    np.log10(max_window),
                    self._n_scales,
                )
            ).astype(int)
        )

        scales_out, F_out = [], []
        for m in raw_scales:
            n_complete = n // m
            if n_complete < self._min_blocks:
                continue

            # 2-3. Segment and detrend
            if self._overlap:
                sq_residuals = _dfa_overlapping(profile, m, self._order)
            else:
                sq_residuals = _dfa_non_overlapping(profile, n_complete, m, self._order)

            if len(sq_residuals) == 0:
                continue

            # 4. Fluctuation = RMS of all residuals
            F_m = np.sqrt(np.mean(sq_residuals))
            if np.isfinite(F_m) and F_m > 0:
                scales_out.append(float(m))
                F_out.append(F_m)

        return np.array(scales_out), np.array(F_out)


# Module-level computation helpers

def _dfa_non_overlapping(
    profile: np.ndarray, n_blocks: int, m: int, order: int
) -> np.ndarray:
    """Non-overlapping DFA: detrend each segment and collect squared residuals."""
    t = np.arange(m)
    vandermonde = np.vander(t, N=order + 1, increasing=True)   # shape (m, order+1)
    all_sq: list[np.ndarray] = []

    for k in range(n_blocks):
        seg = profile[k * m : (k + 1) * m]
        # Least-squares polynomial fit
        try:
            coeffs, _, _, _ = np.linalg.lstsq(vandermonde, seg, rcond=None)
            trend = vandermonde @ coeffs
            all_sq.append((seg - trend) ** 2)
        except np.linalg.LinAlgError:
            continue

    return np.concatenate(all_sq) if all_sq else np.array([])


def _dfa_overlapping(
    profile: np.ndarray, m: int, order: int
) -> np.ndarray:
    """Overlapping DFA (stride = 1): slower but uses every possible window."""
    n = len(profile)
    t = np.arange(m)
    vandermonde = np.vander(t, N=order + 1, increasing=True)
    all_sq: list[float] = []

    for start in range(0, n - m + 1):
        seg = profile[start : start + m]
        try:
            coeffs, _, _, _ = np.linalg.lstsq(vandermonde, seg, rcond=None)
            trend = vandermonde @ coeffs
            all_sq.append(np.mean((seg - trend) ** 2))
        except np.linalg.LinAlgError:
            continue

    return np.array(all_sq)


@dataclass
class WhittleResult:
    """Whittle MLE result with asymptotic standard error.

    Attributes
    ----------
    H : float
        Point estimate of the Hurst exponent.
    se : float
        Asymptotic standard error  (= 1 / (2 * sqrt(m))).
    ci_lower : float
        Lower 95% confidence interval bound.
    ci_upper : float
        Upper 95% confidence interval bound.
    objective : float
        Value of the minimised Whittle objective Q(H).
    n_freqs : int
        Number of Fourier frequencies used (bandwidth m).
    n_obs : int
        Length of the series.
    """

    H: float
    se: float
    ci_lower: float
    ci_upper: float
    objective: float
    n_freqs: int
    n_obs: int

    @property
    def t_statistic(self) -> float:
        """t-statistic for H != 0.5 (test for long-range dependence)."""
        return (self.H - 0.5) / self.se

    def __repr__(self) -> str:
        return (
            f"WhittleResult(H={self.H:.4f}, SE={self.se:.4f}, "
            f"CI=[{self.ci_lower:.4f}, {self.ci_upper:.4f}], "
            f"t={self.t_statistic:.2f}, m={self.n_freqs})"
        )


class WhittleMLE:
    """Semiparametric Whittle MLE for the Hurst exponent of fGn.

    Parameters
    ----------
    bandwidth_exp : float, optional
        Exponent controlling bandwidth ``m = floor(n ** bandwidth_exp)``.
        Must be in (0.5, 1.0).  Default 0.65 follows Robinson (1995) and
        the rough vol literature.
    demean : bool, optional
        Subtract the sample mean before computing the periodogram.
        Almost always correct for financial returns.  Default ``True``.
    n_grid : int, optional
        Number of points in the initial grid search before scalar
        optimisation.  Default 50.
    """

    def __init__(
        self,
        bandwidth_exp: float = 0.65,
        demean: bool = True,
        n_grid: int = 50,
    ) -> None:
        if not (0.5 < bandwidth_exp < 1.0):
            raise ValueError("bandwidth_exp must be in (0.5, 1.0).")
        self._bw_exp = bandwidth_exp
        self._demean = demean
        self._n_grid = n_grid


    def fit(self, series: np.ndarray | pd.Series) -> WhittleResult:
        """Estimate H and its standard error for a stationary return series.

        Parameters
        ----------
        series : array-like
            Log-returns or other stationary increments.  NaNs are dropped.

        Returns
        -------
        WhittleResult
            Contains H, SE, 95% CI, and diagnostic information.
        """
        x = _prepare(series)
        n = len(x)

        freqs, I = _periodogram(x, demean=self._demean)
        m = max(4, int(n ** self._bw_exp))   # bandwidth
        m = min(m, len(freqs))

        freqs_m = freqs[:m]
        I_m = I[:m]

        # Guard: ensure no zero or negative frequencies
        valid = freqs_m > 0
        freqs_m = freqs_m[valid]
        I_m = I_m[valid]
        m = len(freqs_m)

        if m < 4:
            log.warning("WhittleMLE.fit: too few valid frequencies (%d).", m)
            return WhittleResult(
                H=0.5, se=0.5, ci_lower=0.05, ci_upper=0.95,
                objective=np.nan, n_freqs=m, n_obs=n,
            )

        def objective(H: float) -> float:
            return _whittle_objective(H, freqs_m, I_m)

        # Grid search to find a good starting point, then refine
        h_grid = np.linspace(HURST_MIN + 0.01, HURST_MAX - 0.01, self._n_grid)
        q_grid = np.array([objective(h) for h in h_grid])
        h_init = h_grid[np.argmin(q_grid)]

        result = minimize_scalar(
            objective,
            bounds=(HURST_MIN, HURST_MAX),
            method="bounded",
            options={"xatol": 1e-5, "maxiter": 500},
        )

        H_hat = float(np.clip(result.x, HURST_MIN, HURST_MAX))
        Q_hat = float(result.fun)

        # Asymptotic SE from Robinson (1995) Theorem 1
        se = 1.0 / (2.0 * np.sqrt(m))
        z = 1.96   # 95% CI
        return WhittleResult(
            H=H_hat,
            se=se,
            ci_lower=float(np.clip(H_hat - z * se, HURST_MIN, HURST_MAX)),
            ci_upper=float(np.clip(H_hat + z * se, HURST_MIN, HURST_MAX)),
            objective=Q_hat,
            n_freqs=m,
            n_obs=n,
        )

    def fit_rolling(
        self,
        series: np.ndarray | pd.Series,
        window: int | None = None,
        step: int = 1,
    ) -> pd.DataFrame:
        """Compute rolling Whittle H estimates with standard errors.

        Parameters
        ----------
        series : array-like
            Full log-return series.
        window : int, optional
            Window length.  Defaults to ``config.ROLLING_STATS_WINDOW``.
        step : int
            Stride between windows (default 1).

        Returns
        -------
        pd.DataFrame
            Columns: ``H``, ``se``, ``ci_lower``, ``ci_upper``.
            Indexed like ``series``; leading rows are NaN.
        """
        s = (
            pd.Series(_prepare(series))
            if not isinstance(series, pd.Series)
            else series.dropna()
        )
        window = window or config.ROLLING_STATS_WINDOW
        records: dict[int, dict] = {}

        for end in range(window - 1, len(s), step):
            start = end - window + 1
            try:
                res = self.fit(s.iloc[start : end + 1].to_numpy())
                records[s.index[end]] = {
                    "H": res.H,
                    "se": res.se,
                    "ci_lower": res.ci_lower,
                    "ci_upper": res.ci_upper,
                }
            except Exception as exc:
                log.debug("WhittleMLE rolling failed at index %d: %s", end, exc)

        out = pd.DataFrame(records).T
        out.index = pd.Index(out.index)   # ensure original index type
        return out.reindex(s.index)       # re-align with full series index

    def periodogram(
        self,
        series: np.ndarray | pd.Series,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (frequencies, periodogram ordinates) for diagnostics.

        Parameters
        ----------
        series : array-like
            Stationary increments.

        Returns
        -------
        freqs : np.ndarray
            Normalised frequencies in (0, pi].
        I : np.ndarray
            Periodogram values.
        """
        x = _prepare(series)
        return _periodogram(x, demean=self._demean)


# Module-level helpers

def _periodogram(x: np.ndarray, demean: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Compute the one-sided periodogram of x via FFT."""
    n = len(x)
    if demean:
        x = x - x.mean()
    # FFT of the full series
    fft_vals = np.fft.rfft(x)
    # Periodogram ordinates (one-sided, j = 1, …, floor(n/2))
    m = n // 2
    I = (np.abs(fft_vals[1 : m + 1]) ** 2) / (2.0 * np.pi * n)
    freqs = 2.0 * np.pi * np.arange(1, m + 1) / n
    return freqs, I


def _whittle_objective(
    H: float,
    freqs: np.ndarray,
    I: np.ndarray,
) -> float:
    """Semiparametric Whittle objective Q(H) to minimise.

    Q(H) = log[ mean(I_j * lambda_j^{2H-1}) ]
           - (2H - 1) * mean(log(lambda_j))

    The spectral density shape is g(lambda; H) = lambda^{1-2H},
    so I_j / g(lambda_j) = I_j * lambda_j^{2H-1}.
    """
    alpha = 2.0 * H - 1.0
    # Weighted periodogram (I_j / g_j = I_j * lambda_j^{2H-1} = I_j * lambda_j^alpha)
    weighted = I * freqs ** alpha
    sigma_sq = np.mean(weighted)
    if sigma_sq <= 0:
        return 1e10   # infeasible
    return float(np.log(sigma_sq) - alpha * np.mean(np.log(freqs)))

