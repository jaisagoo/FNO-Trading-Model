"""
Tests for the feature engineer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wsq_trading.features import FeatureEngineer


def _make_df(n: int = 300) -> pd.DataFrame:
    idx = pd.bdate_range("2021-01-04", periods=n)
    rng = np.random.default_rng(42)
    close = 100.0 * np.cumprod(1 + rng.normal(0, 0.012, n))
    return pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.012,
            "Low": close * 0.988,
            "Close": close,
            "Volume": rng.integers(1_000, 20_000, n).astype(float),
            "log_return": np.concatenate([[np.nan], np.log(close[1:] / close[:-1])]),
        },
        index=idx,
    )


@pytest.fixture()
def fe() -> FeatureEngineer:
    return FeatureEngineer(vol_window=21, stats_window=63, long_window=252, lags=[1, 5, 21])


class TestLogReturns:
    def test_column_added(self, fe: FeatureEngineer) -> None:
        df = _make_df().drop(columns=["log_return"])
        out = fe.log_returns(df)
        assert "log_return" in out.columns

    def test_no_duplicate_if_already_present(self, fe: FeatureEngineer) -> None:
        df = _make_df()
        original_lr = df["log_return"].copy()
        out = fe.log_returns(df)
        pd.testing.assert_series_equal(out["log_return"], original_lr)

    def test_first_value_nan(self, fe: FeatureEngineer) -> None:
        df = _make_df().drop(columns=["log_return"])
        out = fe.log_returns(df)
        assert pd.isna(out["log_return"].iloc[0])

    def test_correctness(self, fe: FeatureEngineer) -> None:
        df = _make_df().drop(columns=["log_return"])
        out = fe.log_returns(df)
        expected = np.log(df["Close"].iloc[2] / df["Close"].iloc[1])
        assert abs(out["log_return"].iloc[2] - expected) < 1e-10


class TestRealizedVol:
    def test_columns_added(self, fe: FeatureEngineer) -> None:
        out = fe.realized_vol(_make_df())
        assert f"rv_{fe._vol_window}d" in out.columns
        assert f"rv_{fe._long_window}d" in out.columns
        assert "vol_ratio" in out.columns

    def test_vol_nonneg(self, fe: FeatureEngineer) -> None:
        out = fe.realized_vol(_make_df())
        col = f"rv_{fe._vol_window}d"
        assert (out[col].dropna() >= 0).all()

    def test_annualised_magnitude(self, fe: FeatureEngineer) -> None:
        """Annualised vol of a daily 1% process should be ~15–20%."""
        out = fe.realized_vol(_make_df())
        col = f"rv_{fe._vol_window}d"
        median_vol = out[col].median()
        assert 0.05 < median_vol < 0.40, f"Unexpected annualised vol: {median_vol:.3f}"


class TestRollingStats:
    def test_all_columns_added(self, fe: FeatureEngineer) -> None:
        out = fe.rolling_stats(_make_df())
        for col in ("ret_mean", "ret_std", "ret_skew", "ret_kurt"):
            assert col in out.columns

    def test_no_nan_after_warmup(self, fe: FeatureEngineer) -> None:
        out = fe.rolling_stats(_make_df(300))
        warmup = fe._stats_window
        subset = out.iloc[warmup:][["ret_mean", "ret_std"]]
        assert subset.isna().sum().sum() == 0


class TestParkinsonVol:
    def test_column_added(self, fe: FeatureEngineer) -> None:
        out = fe.parkinson_vol(_make_df())
        assert "parkinson_vol" in out.columns

    def test_nonneg(self, fe: FeatureEngineer) -> None:
        out = fe.parkinson_vol(_make_df())
        assert (out["parkinson_vol"].dropna() >= 0).all()

    def test_skip_if_no_high_low(self, fe: FeatureEngineer) -> None:
        df = _make_df().drop(columns=["High", "Low"])
        out = fe.parkinson_vol(df)
        assert "parkinson_vol" not in out.columns


class TestVolumeFeatures:
    def test_columns_added(self, fe: FeatureEngineer) -> None:
        out = fe.volume_features(_make_df())
        for col in ("volume_ma", "volume_ratio", "amihud"):
            assert col in out.columns

    def test_volume_ratio_around_one(self, fe: FeatureEngineer) -> None:
        out = fe.volume_features(_make_df(300))
        median_ratio = out["volume_ratio"].dropna().median()
        assert 0.5 < median_ratio < 2.0

    def test_skip_if_no_volume(self, fe: FeatureEngineer) -> None:
        df = _make_df().drop(columns=["Volume"])
        out = fe.volume_features(df)
        assert "volume_ratio" not in out.columns


class TestLaggedReturns:
    def test_lag_columns_added(self, fe: FeatureEngineer) -> None:
        out = fe.lagged_returns(_make_df(), lags=[1, 5])
        assert "ret_lag_1" in out.columns
        assert "ret_lag_5" in out.columns

    def test_lag_1_equals_shift(self, fe: FeatureEngineer) -> None:
        df = _make_df()
        out = fe.lagged_returns(df, lags=[1])
        expected = df["log_return"].shift(1)
        pd.testing.assert_series_equal(out["ret_lag_1"], expected, check_names=False)


class TestRunPipeline:
    def test_returns_dataframe(self, fe: FeatureEngineer) -> None:
        out = fe.run_pipeline(_make_df())
        assert isinstance(out, pd.DataFrame)

    def test_no_column_loss(self, fe: FeatureEngineer) -> None:
        df = _make_df()
        out = fe.run_pipeline(df)
        for col in df.columns:
            assert col in out.columns

    def test_more_columns_than_input(self, fe: FeatureEngineer) -> None:
        df = _make_df()
        out = fe.run_pipeline(df)
        assert len(out.columns) > len(df.columns)

    def test_does_not_mutate_input(self, fe: FeatureEngineer) -> None:
        df = _make_df()
        original_cols = list(df.columns)
        fe.run_pipeline(df)
        assert list(df.columns) == original_cols

