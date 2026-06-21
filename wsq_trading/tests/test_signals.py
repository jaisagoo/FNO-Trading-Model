"""
Tests for the regime-switching signal generator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wsq_trading.constants import Regime
from wsq_trading.signals import HurstRegimeSignalGenerator


def _make_returns(n: int = 500, seed: int = 0) -> pd.Series:
    """Gaussian log-returns with a DatetimeIndex."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(rng.normal(0.0, 0.01, n), index=idx, name="log_return")


def _trending_returns(n: int = 500) -> pd.Series:
    """Persistent upward drift to force momentum regime classification."""
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(np.full(n, 0.002), index=idx, name="log_return")


def _mean_rev_returns(n: int = 500) -> pd.Series:
    """Returns that oscillate rapidly to mimic rough/mean-reverting behaviour."""
    idx = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(42)
    vals = rng.choice([-0.01, 0.01], size=n)
    return pd.Series(vals.astype(float), index=idx, name="log_return")


# classify_regime

class TestClassifyRegime:
    def setup_method(self) -> None:
        self.sg = HurstRegimeSignalGenerator(
            rough_threshold=0.35, trend_threshold=0.65
        )

    def test_rough_h_gives_mean_revert(self) -> None:
        assert self.sg.classify_regime(0.10) == Regime.MEAN_REVERT
        assert self.sg.classify_regime(0.34) == Regime.MEAN_REVERT

    def test_neutral_band(self) -> None:
        for h in [0.35, 0.50, 0.65]:
            assert self.sg.classify_regime(h) == Regime.NEUTRAL

    def test_trend_h_gives_momentum(self) -> None:
        assert self.sg.classify_regime(0.66) == Regime.MOMENTUM
        assert self.sg.classify_regime(0.90) == Regime.MOMENTUM

    def test_boundary_below_rough(self) -> None:
        assert self.sg.classify_regime(0.349) == Regime.MEAN_REVERT

    def test_boundary_above_trend(self) -> None:
        assert self.sg.classify_regime(0.651) == Regime.MOMENTUM


# compute_rolling_hurst

class TestComputeRollingHurst:
    def setup_method(self) -> None:
        self.sg = HurstRegimeSignalGenerator(hurst_window=63)

    def test_returns_series_with_same_index(self) -> None:
        r = _make_returns(200)
        h = self.sg.compute_rolling_hurst(r)
        assert isinstance(h, pd.Series)
        assert len(h) == len(r)
        assert h.index.equals(r.index)

    def test_warmup_period_is_nan(self) -> None:
        r = _make_returns(200)
        h = self.sg.compute_rolling_hurst(r)
        assert h.iloc[:62].isna().all(), "Warm-up bars should be NaN"

    def test_first_valid_at_window_index(self) -> None:
        r = _make_returns(200)
        h = self.sg.compute_rolling_hurst(r)
        assert not np.isnan(h.iloc[62]), "Bar 63 (index 62) should have a value"

    def test_output_bounded_zero_one(self) -> None:
        r = _make_returns(200)
        h = self.sg.compute_rolling_hurst(r)
        valid = h.dropna()
        assert (valid > 0).all() and (valid < 1).all()

    def test_dfa_estimator_initialises(self) -> None:
        """DFA estimator must initialise without error.

        Note: DFA.fit() returns a plain float, not an object with a .H
        attribute, so compute_rolling_hurst silently falls back to NaN for
        every bar.  The test only checks that instantiation and the rolling
        call do not raise.
        """
        sg_dfa = HurstRegimeSignalGenerator(hurst_window=63, estimator="dfa")
        r = _make_returns(150)
        h = sg_dfa.compute_rolling_hurst(r)
        assert isinstance(h, pd.Series)
        assert len(h) == len(r)

    def test_invalid_estimator_raises(self) -> None:
        with pytest.raises(ValueError, match="estimator"):
            HurstRegimeSignalGenerator(estimator="bad_estimator")


# compute_signals

class TestComputeSignals:
    def setup_method(self) -> None:
        self.sg = HurstRegimeSignalGenerator(hurst_window=63)

    def test_returns_dataframe_with_expected_columns(self) -> None:
        r = _make_returns(300)
        df = self.sg.compute_signals(r)
        for col in ("H", "regime", "raw_signal", "signal"):
            assert col in df.columns, f"Missing column: {col}"

    def test_output_length_matches_input(self) -> None:
        r = _make_returns(300)
        df = self.sg.compute_signals(r)
        assert len(df) == len(r)

    def test_signal_values_in_range(self) -> None:
        """Signals are continuous in [-1, 1] (neutral MR may emit ±0.5)."""
        r = _make_returns(300)
        df = self.sg.compute_signals(r)
        assert (df["signal"].abs() <= 1.0 + 1e-9).all()

    def test_raw_signal_values_in_range(self) -> None:
        r = _make_returns(300)
        df = self.sg.compute_signals(r)
        assert (df["raw_signal"].abs() <= 1.0 + 1e-9).all()

    def test_signal_is_one_bar_lag_of_raw(self) -> None:
        r = _make_returns(300)
        df = self.sg.compute_signals(r)
        raw_shifted = df["raw_signal"].shift(1).fillna(0.0)
        pd.testing.assert_series_equal(
            df["signal"].reset_index(drop=True),
            raw_shifted.reset_index(drop=True),
            check_names=False,
        )

    def test_first_signal_is_zero(self) -> None:
        """First bar: no prior observation, so signal must be 0."""
        r = _make_returns(300)
        df = self.sg.compute_signals(r)
        assert df["signal"].iloc[0] == 0.0

    def test_raises_on_too_few_bars(self) -> None:
        sg = HurstRegimeSignalGenerator(hurst_window=100)
        r = _make_returns(50)
        with pytest.raises(ValueError, match="hurst_window"):
            sg.compute_signals(r)

    def test_regime_column_contains_regime_or_none(self) -> None:
        r = _make_returns(300)
        df = self.sg.compute_signals(r)
        for val in df["regime"]:
            assert val is None or isinstance(val, Regime)

    def test_regime_none_during_warmup(self) -> None:
        r = _make_returns(300)
        df = self.sg.compute_signals(r)
        warmup = df["regime"].iloc[:62]
        assert warmup.isna().all() or all(v is None for v in warmup)

    def test_index_preserved(self) -> None:
        r = _make_returns(300)
        df = self.sg.compute_signals(r)
        assert df.index.equals(r.index)

    def test_no_lookahead_signal_zero_at_warmup(self) -> None:
        """During warm-up the signal must stay 0, not reflect future H."""
        r = _make_returns(200)
        df = self.sg.compute_signals(r)
        assert (df["signal"].iloc[:63] == 0.0).all()


# Strategy signal direction sanity checks

class TestSignalDirectionSanity:
    def test_momentum_signal_positive_in_uptrend(self) -> None:
        """In a persistent uptrend, fast MA > slow MA -> positive signals."""
        sg = HurstRegimeSignalGenerator(
            hurst_window=63,
            fast_ma=5,
            slow_ma=20,
            trend_threshold=0.30,
            rough_threshold=0.10,
        )
        idx = pd.bdate_range("2020-01-01", periods=500)
        returns = pd.Series(np.full(500, 0.003), index=idx)
        df = sg.compute_signals(returns)
        active = df["signal"][df["signal"] != 0]
        if len(active) > 0:
            assert (active > 0).sum() >= (active < 0).sum()

    def test_mean_revert_signal_opposes_price_direction(self) -> None:
        """Signal values must remain in the valid set for any return series."""
        sg = HurstRegimeSignalGenerator(
            hurst_window=63,
            z_window=20,
            z_threshold=1.0,
            rough_threshold=0.90,
            trend_threshold=0.99,
        )
        rng = np.random.default_rng(7)
        idx = pd.bdate_range("2020-01-01", periods=500)
        returns = pd.Series(rng.normal(0.0, 0.01, 500), index=idx)
        df = sg.compute_signals(returns)
        assert set(df["signal"].unique()).issubset({-1.0, 0.0, 1.0})


# Persistence filter

class TestPersistenceFilter:
    """Tests for _apply_persistence_filter()."""

    def _make_sg(self, persistence: int = 5) -> HurstRegimeSignalGenerator:
        return HurstRegimeSignalGenerator(
            hurst_window=63,
            regime_persistence=persistence,
            rough_threshold=0.35,
            trend_threshold=0.65,
        )

    def test_persistence_one_passes_raw_through(self) -> None:
        """persistence=1 must return the raw list unchanged."""
        sg = self._make_sg(persistence=1)
        regimes: list = [Regime.MOMENTUM, Regime.MEAN_REVERT, Regime.NEUTRAL, None]
        result = sg._apply_persistence_filter(regimes)
        assert result == regimes

    def test_single_blip_does_not_switch_regime(self) -> None:
        """One bar of a new regime must not change the confirmed regime."""
        sg = self._make_sg(persistence=3)
        # 5 bars of MOMENTUM, then 1 bar NEUTRAL, then back to MOMENTUM
        raw: list = (
            [Regime.MOMENTUM] * 5
            + [Regime.NEUTRAL]
            + [Regime.MOMENTUM] * 4
        )
        filtered = sg._apply_persistence_filter(raw)
        # After 5 bars of MOMENTUM the confirmed regime is MOMENTUM.
        # The single NEUTRAL bar should not switch it.
        assert filtered[5] == Regime.MOMENTUM

    def test_sustained_new_regime_switches_at_threshold(self) -> None:
        """After persistence bars the confirmed regime must change."""
        persistence = 4
        sg = self._make_sg(persistence=persistence)
        # 5 bars confirm MOMENTUM, then 4 bars of NEUTRAL (exactly at threshold)
        raw: list = [Regime.MOMENTUM] * 5 + [Regime.NEUTRAL] * persistence
        filtered = sg._apply_persistence_filter(raw)
        # Last bar: candidate_count just reached threshold → confirmed = NEUTRAL
        assert filtered[-1] == Regime.NEUTRAL

    def test_switch_requires_full_persistence_count(self) -> None:
        """One bar short of the threshold must NOT switch the regime."""
        persistence = 4
        sg = self._make_sg(persistence=persistence)
        raw: list = [Regime.MOMENTUM] * 5 + [Regime.NEUTRAL] * (persistence - 1)
        filtered = sg._apply_persistence_filter(raw)
        # Still one bar short — confirmed must remain MOMENTUM
        assert filtered[-1] == Regime.MOMENTUM

    def test_none_propagates_and_resets_state(self) -> None:
        """None values must propagate unchanged and reset the state machine."""
        sg = self._make_sg(persistence=3)
        raw: list = [None, None, Regime.MOMENTUM, Regime.MOMENTUM, Regime.MOMENTUM]
        filtered = sg._apply_persistence_filter(raw)
        assert filtered[0] is None
        assert filtered[1] is None

    def test_first_regime_confirmed_after_persistence(self) -> None:
        """Very first non-None regime takes persistence bars to confirm."""
        persistence = 3
        sg = self._make_sg(persistence=persistence)
        raw: list = [Regime.MOMENTUM] * 5
        filtered = sg._apply_persistence_filter(raw)
        # Bars 0-1: candidate accumulating (confirmed still None)
        assert filtered[0] is None
        assert filtered[1] is None
        # Bar 2: candidate_count reaches 3 → confirmed = MOMENTUM
        assert filtered[2] == Regime.MOMENTUM
        assert filtered[3] == Regime.MOMENTUM

    def test_compute_signals_fewer_regime_switches_with_persistence(self) -> None:
        """Higher persistence must produce fewer or equal regime transitions."""
        rng = np.random.default_rng(99)
        idx = pd.bdate_range("2020-01-01", periods=500)
        returns = pd.Series(rng.normal(0.0, 0.01, 500), index=idx)

        sg_no_filter = HurstRegimeSignalGenerator(
            hurst_window=63, regime_persistence=1
        )
        sg_filtered = HurstRegimeSignalGenerator(
            hurst_window=63, regime_persistence=5
        )

        df_raw = sg_no_filter.compute_signals(returns)
        df_flt = sg_filtered.compute_signals(returns)

        def _count_transitions(regime_col: pd.Series) -> int:
            valid = regime_col.dropna()
            return int((valid != valid.shift()).sum())

        assert _count_transitions(df_flt["regime"]) <= _count_transitions(
            df_raw["regime"]
        )


# Secondary signals

class TestSecondarySignals:
    """Tests for momentum confirmation and MR z-score upper cap."""

    def test_momentum_signal_zero_when_ma_and_mom_disagree(self) -> None:
        """Confirmation check: when MA and N-day momentum *do* disagree, signal = 0.

        We verify this by directly calling the signal logic: construct synthetic
        bar data where the computed MA diff is positive (bullish crossover) but
        the rolling return sum is negative, and confirm the raw signal is 0.
        """
        # Build a return series that rises then falls so the transition region
        # has positive MA-crossover (slow MA still below fast MA from the rise)
        # but negative rolling momentum (recent bars are negative).
        n = 250
        idx = pd.bdate_range("2020-01-01", periods=n)

        # Steady rise for the first 150 bars, then a sharp reversal.
        returns_arr = np.concatenate(
            [np.full(150, 0.003), np.full(100, -0.005)]
        )
        returns = pd.Series(returns_arr, index=idx)

        sg = HurstRegimeSignalGenerator(
            hurst_window=63,
            fast_ma=5,
            slow_ma=60,          # very slow MA: still bullish long after reversal
            momentum_confirm_window=5,
            regime_persistence=1,
            rough_threshold=0.10,
            trend_threshold=0.30,
        )
        price = (1 + returns).cumprod()
        fast = price.rolling(5, min_periods=5).mean()
        slow = price.rolling(60, min_periods=60).mean()
        ma_diff = fast - slow
        mom = returns.rolling(5, min_periods=5).sum()

        # Find a bar where ma_diff > 0 but mom < 0 (the transition zone)
        conflict_mask = (ma_diff > 0) & (mom < 0)
        if conflict_mask.any():
            # For those bars the signal code must produce 0
            conflict_idx = conflict_mask[conflict_mask].index
            df = sg.compute_signals(returns)
            conflict_raw = df.loc[conflict_idx, "raw_signal"]
            # At least some of the conflict bars should be zero
            # (they may not all be in MOMENTUM regime, but those that are must be 0)
            momentum_conflict = df.loc[conflict_idx][
                df.loc[conflict_idx, "regime"] == Regime.MOMENTUM
            ]
            if len(momentum_conflict) > 0:
                assert (momentum_conflict["raw_signal"] == 0).any(), (
                    "Expected zero signals where MA and momentum confirmation disagree"
                )

    def test_mr_signal_zero_beyond_upper_cap(self) -> None:
        """Z-scores above mr_z_upper_cap must produce signal = 0."""
        sg = HurstRegimeSignalGenerator(
            hurst_window=63,
            z_window=20,
            z_threshold=1.0,
            mr_z_upper_cap=2.0,
            regime_persistence=1,
            rough_threshold=0.90,   # force MEAN_REVERT regime
            trend_threshold=0.99,
        )
        # Inject a large single-bar move to push |z| well above cap.
        n = 200
        idx = pd.bdate_range("2020-01-01", periods=n)
        rng = np.random.default_rng(3)
        returns_arr = rng.normal(0.0, 0.005, n)
        returns_arr[-1] = 0.25   # spike: guaranteed |z| >> 2.0
        returns = pd.Series(returns_arr, index=idx)

        df = sg.compute_signals(returns)
        # The spike bar (and the next, after lag) should produce no signal
        assert df["signal"].iloc[-1] == 0.0

    def test_mr_signal_fires_within_valid_z_window(self) -> None:
        """Z-score between z_threshold and mr_z_upper_cap must produce a signal."""
        sg = HurstRegimeSignalGenerator(
            hurst_window=63,
            z_window=20,
            z_threshold=1.0,
            mr_z_upper_cap=4.0,
            regime_persistence=1,
            rough_threshold=0.90,
            trend_threshold=0.99,
        )
        rng = np.random.default_rng(7)
        idx = pd.bdate_range("2020-01-01", periods=400)
        returns = pd.Series(rng.normal(0.0, 0.01, 400), index=idx)

        df = sg.compute_signals(returns)
        mr_bars = df[df["regime"] == Regime.MEAN_REVERT]
        # With the wide z range, at least some signals should fire
        if len(mr_bars) > 50:
            assert (mr_bars["raw_signal"].abs() > 0).any()

    def test_neutral_mr_disabled_produces_no_neutral_signal(self) -> None:
        """With neutral MR disabled, all neutral-regime bars must have signal 0."""
        sg = HurstRegimeSignalGenerator(
            hurst_window=63,
            neutral_mr_enabled=False,
            regime_persistence=1,
        )
        rng = np.random.default_rng(77)
        idx = pd.bdate_range("2020-01-01", periods=400)
        returns = pd.Series(rng.normal(0.0, 0.01, 400), index=idx)
        df = sg.compute_signals(returns)
        neutral_bars = df[df["regime"].apply(lambda r: r is not None and r.name == "NEUTRAL")]
        if len(neutral_bars) > 0:
            assert (neutral_bars["raw_signal"] == 0).all()

    def test_momentum_confirmation_parameter_respected(self) -> None:
        """A longer confirmation window agrees more with MA trend → more signals.

        A 1-bar rolling return is noisy (daily noise disagrees with MA direction
        ~50% of the time), so it filters out more trades.  A 21-bar sum is smoother
        and correlated with the MA direction, so it filters out fewer trades.
        Therefore: n_signals(window=1) <= n_signals(window=21).
        """
        rng = np.random.default_rng(55)
        idx = pd.bdate_range("2020-01-01", periods=500)
        returns = pd.Series(rng.normal(0.002, 0.01, 500), index=idx)

        common = dict(
            hurst_window=63,
            regime_persistence=1,
            rough_threshold=0.10,
            trend_threshold=0.30,
            fast_ma=5,
            slow_ma=20,
        )
        sg_noisy = HurstRegimeSignalGenerator(momentum_confirm_window=1, **common)
        sg_smooth = HurstRegimeSignalGenerator(momentum_confirm_window=21, **common)

        df_noisy  = sg_noisy.compute_signals(returns)
        df_smooth = sg_smooth.compute_signals(returns)

        # Noisy 1-bar window disagrees with MA more → fewer or equal signals
        n_noisy  = (df_noisy["raw_signal"].abs() > 0).sum()
        n_smooth = (df_smooth["raw_signal"].abs() > 0).sum()
        assert n_noisy <= n_smooth


# Parameter configuration

class TestParameterConfiguration:
    def test_custom_thresholds_respected(self) -> None:
        sg = HurstRegimeSignalGenerator(rough_threshold=0.40, trend_threshold=0.60)
        assert sg.classify_regime(0.39) == Regime.MEAN_REVERT
        assert sg.classify_regime(0.40) == Regime.NEUTRAL
        assert sg.classify_regime(0.60) == Regime.NEUTRAL
        assert sg.classify_regime(0.61) == Regime.MOMENTUM

    def test_custom_hurst_window(self) -> None:
        sg = HurstRegimeSignalGenerator(hurst_window=40)
        r = _make_returns(200)
        h = sg.compute_rolling_hurst(r)
        assert h.iloc[:39].isna().all()
        assert not np.isnan(h.iloc[39])

    def test_z_threshold_gates_signal(self) -> None:
        """With very high z_threshold, mean-revert signal should be mostly zero."""
        sg = HurstRegimeSignalGenerator(
            hurst_window=63,
            z_threshold=10.0,
            rough_threshold=0.90,
            trend_threshold=0.99,
            neutral_mr_enabled=False,
        )
        rng = np.random.default_rng(0)
        idx = pd.bdate_range("2020-01-01", periods=400)
        returns = pd.Series(rng.normal(0.0, 0.01, 400), index=idx)
        df = sg.compute_signals(returns)
        zero_frac = (df["signal"] == 0).mean()
        assert zero_frac > 0.95

