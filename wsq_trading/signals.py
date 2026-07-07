"""Regime-switching signal generation from Hurst estimates.

Strategy logic
--------------
H(t) is estimated from log-realized-volatility (not raw returns) to match the
rough Heston model where H characterises the *variance* process roughness.
A short rolling-std window proxies sqrt(V(t)); Whittle MLE is then applied to
windows of log(sqrt(V)) to recover H.

    H < HURST_ROUGH_THRESHOLD  (default 0.43)  ->  MEAN_REVERT
        Signal: -sign(price_z_score)  when z_threshold < |z| < mr_z_upper_cap,
        else 0.  The upper cap guards against treating structural breaks as
        reversion opportunities.

    H > HURST_TREND_THRESHOLD  (default 0.57)  ->  MOMENTUM
        Signal: sign(fast_MA - slow_MA) when that direction is confirmed by
        the N-day rolling return momentum, else 0.  Requiring two independent
        direction signals materially reduces false entries.

    Otherwise  (0.43 <= H <= 0.57)            ->  NEUTRAL
        Signal: -sign(daily_return_z) * NEUTRAL_MR_POSITION_SIZE
        when |daily_return_z| > NEUTRAL_MR_ENTRY_ZSCORE, else 0.
        Turns dead neutral time into a short-term contrarian alpha source.

Regime persistence filter
-------------------------
Raw H(t) values are classified bar by bar, but the *confirmed* regime only
switches after the new regime persists for ``regime_persistence`` consecutive
bars.  This single-parameter change is the most effective way to reduce
whipsaw trading from noisy Hurst estimates.

All signals are computed with a one-bar lag so that position t+1 is taken
using only information available up to and including bar t (no look-ahead).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from wsq_trading import config
from wsq_trading.constants import Regime
from wsq_trading.hurst import DFA, WhittleMLE
from wsq_trading.utils import get_logger

log = get_logger(__name__)


class HurstRegimeSignalGenerator:
    """Convert a return series into directional signals via rolling Hurst estimation.

    Parameters
    ----------
    hurst_window : int
        Rolling window length (bars) used for the Hurst estimator.
        Default: ``config.WINDOW_SIZE`` (63 bars, ~3 months).
    vol_window : int
        Short rolling-std window used to compute the realized-vol proxy
        before applying the Whittle estimator.  Default: 5 bars.
    z_window : int
        Look-back for the price z-score used in the mean-reversion signal.
        Default: 20 bars.
    z_threshold : float
        Lower entry threshold for the z-score signal.  A position is entered
        only when |z| > z_threshold.  Default: 1.5.
    mr_z_upper_cap : float
        Upper cap for the z-score signal.  |z| values above this level are
        treated as potential structural breaks and produce no signal.
        Default: ``config.MR_Z_UPPER_CAP`` (4.0).
    fast_ma : int
        Fast moving-average length for the momentum signal.  Default: 10.
    slow_ma : int
        Slow moving-average length for the momentum signal.  Default: 30.
    momentum_confirm_window : int
        Rolling return sum window used to confirm the MA-crossover direction.
        Default: ``config.MOMENTUM_CONFIRM_WINDOW`` (21 bars).
    regime_persistence : int
        Number of consecutive bars H(t) must remain in a new regime before
        the confirmed regime switches.  Set to 1 to disable.
        Default: ``config.REGIME_PERSISTENCE`` (5 bars).
    rough_threshold : float
        H below this value triggers MEAN_REVERT.
        Default: ``config.HURST_ROUGH_THRESHOLD``.
    trend_threshold : float
        H above this value triggers MOMENTUM.
        Default: ``config.HURST_TREND_THRESHOLD``.
    estimator : str
        Which Hurst estimator to use: ``'whittle'`` (default) or ``'dfa'``.
    neutral_mr_enabled : bool
        Whether to emit a contrarian signal in the neutral regime.
        Default: ``config.NEUTRAL_MR_ENABLED``.
    neutral_mr_return_window : int
        Rolling window (bars) for the return z-score in neutral MR.
        Default: ``config.NEUTRAL_MR_RETURN_WINDOW``.
    neutral_mr_entry_zscore : float
        Minimum |return z-score| to enter a neutral MR trade.
        Default: ``config.NEUTRAL_MR_ENTRY_ZSCORE``.
    neutral_mr_position_size : float
        Position size fraction for neutral MR (0.5 = half the nominal size).
        Default: ``config.NEUTRAL_MR_POSITION_SIZE``.
    """

    def __init__(
        self,
        hurst_window: int | None = None,
        vol_window: int = 5,
        z_window: int = 20,
        z_threshold: float = 1.5,
        mr_z_upper_cap: float | None = None,
        fast_ma: int = 10,
        slow_ma: int = 30,
        momentum_confirm_window: int | None = None,
        regime_persistence: int | None = None,
        rough_threshold: float | None = None,
        trend_threshold: float | None = None,
        estimator: str = "whittle",
        neutral_mr_enabled: bool | None = None,
        neutral_mr_return_window: int | None = None,
        neutral_mr_entry_zscore: float | None = None,
        neutral_mr_position_size: float | None = None,
    ) -> None:
        self._hurst_window = hurst_window or config.WINDOW_SIZE
        self._vol_window = vol_window
        self._z_window = z_window
        self._z_threshold = z_threshold
        self._mr_z_upper_cap = (
            mr_z_upper_cap if mr_z_upper_cap is not None else config.MR_Z_UPPER_CAP
        )
        self._fast_ma = fast_ma
        self._slow_ma = slow_ma
        self._momentum_confirm_window = (
            momentum_confirm_window
            if momentum_confirm_window is not None
            else config.MOMENTUM_CONFIRM_WINDOW
        )
        self._regime_persistence = (
            regime_persistence
            if regime_persistence is not None
            else config.REGIME_PERSISTENCE
        )
        self._rough_threshold = rough_threshold or config.HURST_ROUGH_THRESHOLD
        self._trend_threshold = trend_threshold or config.HURST_TREND_THRESHOLD
        self._neutral_mr_enabled = (
            neutral_mr_enabled if neutral_mr_enabled is not None
            else config.NEUTRAL_MR_ENABLED
        )
        self._neutral_mr_return_window = (
            neutral_mr_return_window if neutral_mr_return_window is not None
            else config.NEUTRAL_MR_RETURN_WINDOW
        )
        self._neutral_mr_entry_zscore = (
            neutral_mr_entry_zscore if neutral_mr_entry_zscore is not None
            else config.NEUTRAL_MR_ENTRY_ZSCORE
        )
        self._neutral_mr_position_size = (
            neutral_mr_position_size if neutral_mr_position_size is not None
            else config.NEUTRAL_MR_POSITION_SIZE
        )

        if estimator == "whittle":
            self._est = WhittleMLE()
        elif estimator == "dfa":
            self._est = DFA()
        else:
            raise ValueError(f"estimator must be 'whittle' or 'dfa'; got {estimator!r}")

        self._estimator_name = estimator


    def classify_regime(self, H: float) -> Regime:
        """Map a Hurst estimate to a Regime enum value.

        Parameters
        ----------
        H : float
            Hurst exponent estimate in (0, 1).

        Returns
        -------
        Regime
            MEAN_REVERT, NEUTRAL, or MOMENTUM.
        """
        if H < self._rough_threshold:
            return Regime.MEAN_REVERT
        if H > self._trend_threshold:
            return Regime.MOMENTUM
        return Regime.NEUTRAL

    def compute_rolling_hurst(self, returns: pd.Series) -> pd.Series:
        """Estimate H(t) from raw log-returns with a rolling window.

        The first ``hurst_window - 1`` values are NaN (warm-up period).

        Parameters
        ----------
        returns : pd.Series
            Log-return series with a DatetimeIndex (or integer index).

        Returns
        -------
        pd.Series
            Rolling H estimates, same index as *returns*.
        """
        n = len(returns)
        w = self._hurst_window
        h_values = np.full(n, np.nan)

        for i in range(w, n + 1):
            window = returns.iloc[i - w: i].to_numpy()
            try:
                h_values[i - 1] = self._est.fit(window).H
            except Exception as exc:
                log.debug("Hurst estimate failed at bar %d: %s", i, exc)
                h_values[i - 1] = np.nan

        return pd.Series(h_values, index=returns.index, name="H_estimate")

    def compute_signals(self, returns: pd.Series) -> pd.DataFrame:
        """Generate regime-based directional signals from a return series.

        The pipeline:
            1. Compute rolling H estimates.
            2. Classify each bar's raw regime from H.
            3. Apply the persistence filter to get the confirmed regime.
            4. Compute the raw signal bar by bar:
               - MEAN_REVERT: -sign(z_score) when z_threshold < |z| < mr_z_upper_cap
               - MOMENTUM:    sign(MA crossover) when confirmed by N-day momentum
               - NEUTRAL:     -sign(daily_return_z) * position_size when |z| > threshold
            5. Apply a one-bar lag so there is no look-ahead.

        Parameters
        ----------
        returns : pd.Series
            Daily log-returns.

        Returns
        -------
        pd.DataFrame
            Columns:
                ``H``          -- rolling Hurst estimate (NaN during warm-up)
                ``regime``     -- confirmed (persistence-filtered) Regime enum
                ``signal``     -- lagged directional signal in [-1, 1]
                ``raw_signal`` -- unlagged signal (for diagnostics)
        """
        if len(returns) < self._hurst_window:
            raise ValueError(
                f"Need at least hurst_window={self._hurst_window} bars; "
                f"got {len(returns)}."
            )

        log.info(
            "SignalGenerator (%s): computing signals for %d bars "
            "(persistence=%d, mom_confirm=%d, mr_z_cap=%.1f, neutral_mr=%s).",
            self._estimator_name,
            len(returns),
            self._regime_persistence,
            self._momentum_confirm_window,
            self._mr_z_upper_cap,
            self._neutral_mr_enabled,
        )

        # Step 1: rolling H
        H_series = self.compute_rolling_hurst(returns)

        # Cumulative price - used for both z-score and MA crossover.
        price = (1 + returns).cumprod()

        # Z-score on PRICE LEVEL (not individual returns) for mean reversion.
        rolling_mean = price.rolling(self._z_window, min_periods=self._z_window).mean()
        rolling_std  = price.rolling(self._z_window, min_periods=self._z_window).std(ddof=1)
        z_score = (price - rolling_mean) / rolling_std.replace(0, np.nan)

        # MA crossover on price level for momentum signal.
        fast_ma = price.rolling(self._fast_ma, min_periods=self._fast_ma).mean()
        slow_ma = price.rolling(self._slow_ma, min_periods=self._slow_ma).mean()
        ma_diff = fast_ma - slow_ma

        # Rolling cumulative return for momentum confirmation.
        mom_confirm = returns.rolling(
            self._momentum_confirm_window,
            min_periods=self._momentum_confirm_window,
        ).sum()

        # Daily return z-score for neutral-regime mean reversion.
        # Fade extreme daily returns: large up-day -> short tomorrow, large down-day -> long.
        _nmr_w = self._neutral_mr_return_window
        _nmr_min = max(5, _nmr_w // 4)
        ret_roll_std  = returns.rolling(_nmr_w, min_periods=_nmr_min).std(ddof=1).replace(0, np.nan)
        ret_roll_mean = returns.rolling(_nmr_w, min_periods=_nmr_min).mean()
        returns_z = (returns - ret_roll_mean) / ret_roll_std

        # Step 2: raw regime classification
        raw_regimes: list[Regime | None] = [
            None if np.isnan(h) else self.classify_regime(h)
            for h in H_series
        ]

        # Step 3: persistence filter -> confirmed regimes
        effective_regimes = self._apply_persistence_filter(raw_regimes)

        # Step 4: signal computation
        raw = np.zeros(len(returns))

        for i, (regime, z_val, ma_val, mom_val, ret_z) in enumerate(
            zip(effective_regimes, z_score, ma_diff, mom_confirm, returns_z)
        ):
            if regime is None:
                raw[i] = 0.0
                continue

            if regime == Regime.MEAN_REVERT:
                # Enter only in the valid z-score window [threshold, upper_cap).
                if (
                    not np.isnan(z_val)
                    and self._z_threshold < abs(z_val) < self._mr_z_upper_cap
                ):
                    raw[i] = -np.sign(z_val)
                else:
                    raw[i] = 0.0

            elif regime == Regime.MOMENTUM:
                if not np.isnan(ma_val):
                    ma_signal = np.sign(ma_val)
                    if np.isnan(mom_val):
                        # Confirm window not yet warm - accept MA direction alone
                        raw[i] = ma_signal
                    elif np.sign(mom_val) == ma_signal:
                        raw[i] = ma_signal
                    else:
                        raw[i] = 0.0
                else:
                    raw[i] = 0.0

            else:  # NEUTRAL - short-term mean reversion on extreme daily returns
                if (
                    self._neutral_mr_enabled
                    and not np.isnan(ret_z)
                    and abs(ret_z) > self._neutral_mr_entry_zscore
                ):
                    raw[i] = -np.sign(ret_z) * self._neutral_mr_position_size
                else:
                    raw[i] = 0.0

        raw_series = pd.Series(raw, index=returns.index, name="raw_signal")

        # Step 5: one-bar lag
        lagged = raw_series.shift(1).fillna(0.0).rename("signal")

        regime_series = pd.Series(effective_regimes, index=returns.index, name="regime")

        out = pd.DataFrame(
            {
                "H": H_series,
                "regime": regime_series,
                "raw_signal": raw_series,
                "signal": lagged,
            }
        )

        active_frac = (lagged.abs() > 0).mean()
        log.info(
            "SignalGenerator: %.1f%% of bars have a non-zero position.",
            active_frac * 100,
        )
        return out

    # Internal helpers

    def _apply_persistence_filter(
        self, raw_regimes: list[Regime | None]
    ) -> list[Regime | None]:
        """Apply hysteresis: only switch confirmed regime after N consecutive bars.

        Uses a state machine with two states: *confirmed* (the current live
        regime) and *candidate* (a new regime that is accumulating evidence).
        The confirmed regime changes only when the candidate has been observed
        for ``regime_persistence`` consecutive bars.

        ``None`` values (warm-up period) propagate unchanged and reset the
        state machine.
        """
        if self._regime_persistence <= 1:
            return list(raw_regimes)

        n = len(raw_regimes)
        filtered: list[Regime | None] = [None] * n

        confirmed: Regime | None = None
        candidate: Regime | None = None
        candidate_count: int = 0

        for i, r in enumerate(raw_regimes):
            if r is None:
                filtered[i] = None
                confirmed = None
                candidate = None
                candidate_count = 0
                continue

            if r == confirmed:
                filtered[i] = confirmed
                candidate = None
                candidate_count = 0

            elif r == candidate:
                candidate_count += 1
                if candidate_count >= self._regime_persistence:
                    confirmed = candidate
                    candidate = None
                    candidate_count = 0
                filtered[i] = confirmed

            else:
                candidate = r
                candidate_count = 1
                filtered[i] = confirmed

        return filtered

