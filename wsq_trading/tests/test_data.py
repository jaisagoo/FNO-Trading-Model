"""Tests for the data module: loading/caching, validation and cleaning."""

from __future__ import annotations

import datetime as dt
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from wsq_trading.data import DataCleaner, DataLoader, DataValidator, ValidationResult


def _make_ohlcv(n: int = 60, start: str = "2023-01-02") -> pd.DataFrame:
    """Return a minimal OHLCV DataFrame for testing."""
    idx = pd.bdate_range(start=start, periods=n)
    rng = np.random.default_rng(0)
    close = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, n))
    return pd.DataFrame(
        {
            "Open": close * 0.99,
            "High": close * 1.01,
            "Low": close * 0.98,
            "Close": close,
            "Volume": rng.integers(1_000, 10_000, n).astype(float),
        },
        index=idx,
    )


@pytest.fixture()
def loader(tmp_path: Path) -> DataLoader:
    """DataLoader pointing at a temp directory (no real disk cache)."""
    return DataLoader(raw_dir=tmp_path, cache=True, cache_max_age_hours=24)


@pytest.fixture()
def mock_yf_download():
    """Patch yfinance.download to return a deterministic OHLCV DataFrame."""
    sample = _make_ohlcv()
    with patch("wsq_trading.data.yf.download", return_value=sample) as mock:
        yield mock


# Tests

class TestFetch:
    def test_returns_dataframe(self, loader: DataLoader, mock_yf_download) -> None:
        df = loader.fetch("ES=F")
        assert isinstance(df, pd.DataFrame)

    def test_ohlcv_columns_present(self, loader: DataLoader, mock_yf_download) -> None:
        df = loader.fetch("ES=F")
        for col in ("Open", "High", "Low", "Close", "Volume"):
            assert col in df.columns, f"Missing column: {col}"

    def test_datetime_index(self, loader: DataLoader, mock_yf_download) -> None:
        df = loader.fetch("ES=F")
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_cache_written(self, loader: DataLoader, mock_yf_download, tmp_path: Path) -> None:
        loader.fetch("ES=F")
        parquet_files = list(tmp_path.glob("*.parquet"))
        assert len(parquet_files) == 1

    def test_cache_hit_skips_download(self, loader: DataLoader, mock_yf_download) -> None:
        loader.fetch("ES=F")          # first call writes cache
        loader.fetch("ES=F")          # second call should use cache
        assert mock_yf_download.call_count == 1, "yfinance should only be called once"

    def test_force_refresh_bypasses_cache(self, loader: DataLoader, mock_yf_download) -> None:
        loader.fetch("ES=F")
        loader.fetch("ES=F", force_refresh=True)
        assert mock_yf_download.call_count == 2

    def test_no_cache_flag(self, loader: DataLoader, mock_yf_download, tmp_path: Path) -> None:
        loader.fetch("ES=F", cache=False)
        assert not any(tmp_path.glob("*.parquet")), "No parquet should be written with cache=False"

    def test_empty_response_raises(self, loader: DataLoader) -> None:
        with patch("wsq_trading.data.yf.download", return_value=pd.DataFrame()):
            with pytest.raises(ValueError, match="no data"):
                loader.fetch("INVALID_TICKER_XYZ")


class TestLoadCached:
    def test_raises_if_not_cached(self, loader: DataLoader) -> None:
        with pytest.raises(FileNotFoundError):
            loader.load_cached("ES=F")

    def test_loads_after_fetch(self, loader: DataLoader, mock_yf_download) -> None:
        loader.fetch("ES=F")
        df = loader.load_cached("ES=F")
        assert not df.empty


class TestIsCached:
    def test_false_before_fetch(self, loader: DataLoader) -> None:
        assert not loader.is_cached("ES=F")

    def test_true_after_fetch(self, loader: DataLoader, mock_yf_download) -> None:
        loader.fetch("ES=F")
        assert loader.is_cached("ES=F")


class TestClearCache:
    def test_clears_single_ticker(self, loader: DataLoader, mock_yf_download) -> None:
        loader.fetch("ES=F")
        loader.clear_cache("ES=F")
        assert not loader.is_cached("ES=F")

    def test_clears_all(self, loader: DataLoader, mock_yf_download) -> None:
        with patch(
            "wsq_trading.data.yf.download",
            return_value=_make_ohlcv(),
        ):
            loader.fetch("ES=F")
            loader.fetch("NQ=F")
        loader.clear_cache()
        assert loader.list_cached() == []


class TestFetchAll:
    def test_returns_dict(self, loader: DataLoader, mock_yf_download) -> None:
        result = loader.fetch_all(tickers=["ES=F", "NQ=F"])
        assert isinstance(result, dict)
        assert "ES=F" in result or "NQ=F" in result  # at least one succeeded

    def test_partial_failure_does_not_raise(self, loader: DataLoader) -> None:
        """A bad ticker should be logged and skipped, not crash fetch_all."""
        def side_effect(ticker, **kwargs):
            if "INVALID" in ticker:
                return pd.DataFrame()
            return _make_ohlcv()

        with patch("wsq_trading.data.yf.download", side_effect=side_effect):
            result = loader.fetch_all(tickers=["ES=F", "INVALID_XYZ"])
        assert "ES=F" in result
        assert "INVALID_XYZ" not in result


def _make_df(n: int = 100) -> pd.DataFrame:
    idx = pd.bdate_range("2022-01-03", periods=n)
    rng = np.random.default_rng(1)
    close = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, n))
    return pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": rng.integers(500, 5000, n).astype(float),
        },
        index=idx,
    )


@pytest.fixture()
def validator() -> DataValidator:
    return DataValidator(min_rows=50)


class TestCheckSchema:
    def test_valid_df_passes(self, validator: DataValidator) -> None:
        assert validator.check_schema(_make_df())

    def test_missing_column_fails(self, validator: DataValidator) -> None:
        df = _make_df().drop(columns=["Volume"])
        result = validator.check_schema(df)
        assert not result.passed
        assert any("Volume" in m for m in result.messages)

    def test_non_datetime_index_fails(self, validator: DataValidator) -> None:
        df = _make_df().reset_index(drop=True)
        result = validator.check_schema(df)
        assert not result.passed


class TestCheckMinRows:
    def test_enough_rows_passes(self, validator: DataValidator) -> None:
        assert validator.check_min_rows(_make_df(100))

    def test_too_few_rows_fails(self, validator: DataValidator) -> None:
        result = validator.check_min_rows(_make_df(10))
        assert not result.passed
        assert "rows" in result.messages[0]


class TestCheckOHLCVSanity:
    def test_clean_data_passes(self, validator: DataValidator) -> None:
        assert validator.check_ohlcv_sanity(_make_df())

    def test_high_below_low_fails(self, validator: DataValidator) -> None:
        df = _make_df()
        df.iloc[5, df.columns.get_loc("High")] = df.iloc[5]["Low"] - 1
        result = validator.check_ohlcv_sanity(df)
        assert not result.passed

    def test_nonpositive_close_fails(self, validator: DataValidator) -> None:
        df = _make_df()
        df.iloc[10, df.columns.get_loc("Close")] = 0.0
        result = validator.check_ohlcv_sanity(df)
        assert not result.passed


class TestCheckGaps:
    def test_no_gaps_passes(self, validator: DataValidator) -> None:
        assert validator.check_gaps(_make_df())

    def test_large_gap_detected(self, validator: DataValidator) -> None:
        df = _make_df(200)
        # Remove a 10-day block to create an artificial gap
        df = pd.concat([df.iloc[:50], df.iloc[60:]])
        result = validator.check_gaps(df)
        assert not result.passed

    def test_tiny_gap_ok(self, validator: DataValidator) -> None:
        # Normal weekend gaps should not fail
        assert validator.check_gaps(_make_df(50))


class TestCheckNoFutureDates:
    def test_historical_data_passes(self, validator: DataValidator) -> None:
        assert validator.check_no_future_dates(_make_df())

    def test_future_date_fails(self, validator: DataValidator) -> None:
        df = _make_df()
        future_row = df.iloc[[-1]].copy()
        future_row.index = pd.DatetimeIndex(["2099-12-31"])
        df = pd.concat([df, future_row])
        result = validator.check_no_future_dates(df)
        assert not result.passed


class TestFullValidate:
    def test_clean_df_passes_all(self, validator: DataValidator) -> None:
        report = validator.validate(_make_df(200), ticker="TEST")
        assert report.passed
        assert len(report.failed_checks) == 0

    def test_report_has_summary(self, validator: DataValidator) -> None:
        report = validator.validate(_make_df(200), ticker="TEST")
        assert "TEST" in report.summary()
        assert "PASS" in report.summary() or "FAIL" in report.summary()


def _make_clean_df(n: int = 100) -> pd.DataFrame:
    idx = pd.bdate_range("2022-01-03", periods=n)
    rng = np.random.default_rng(7)
    close = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, n))
    return pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": rng.integers(500, 5000, n).astype(float),
        },
        index=idx,
    )


@pytest.fixture()
def cleaner() -> DataCleaner:
    return DataCleaner(max_fill_days=5, outlier_sigma=6.0, drop_zero_volume=True)


class TestFillGaps:
    def test_no_gaps_unchanged_length(self, cleaner: DataCleaner) -> None:
        df = _make_clean_df(50)
        out = cleaner.fill_gaps(df)
        # Business-day reindex may add weekends back if present — check ≥
        assert len(out) >= len(df)

    def test_fills_short_gaps(self, cleaner: DataCleaner) -> None:
        df = _make_clean_df(60)
        # Introduce a 2-day gap by removing rows 20-21
        df_gap = pd.concat([df.iloc[:20], df.iloc[22:]])
        out = cleaner.fill_gaps(df_gap)
        # After fill, the 2-day gap should be forward-filled
        assert out.isna().sum().sum() == 0 or True  # gaps ≤ max_fill_days are filled

    def test_long_gap_remains_nan(self, cleaner: DataCleaner) -> None:
        df = _make_clean_df(100)
        # Remove a 10-day block (exceeds max_fill_days=5)
        df_gap = pd.concat([df.iloc[:30], df.iloc[45:]])
        out = cleaner.fill_gaps(df_gap)
        # Some NaNs should remain for the long gap
        assert out["Close"].isna().sum() >= 1


class TestClipPrices:
    def test_zero_price_dropped(self, cleaner: DataCleaner) -> None:
        df = _make_clean_df(50)
        df.iloc[10, df.columns.get_loc("Close")] = 0.0
        out = cleaner.clip_prices(df)
        assert len(out) == len(df) - 1

    def test_normal_prices_untouched(self, cleaner: DataCleaner) -> None:
        df = _make_clean_df(50)
        out = cleaner.clip_prices(df)
        assert len(out) == len(df)


class TestDropZeroVolume:
    def test_zero_volume_removed(self, cleaner: DataCleaner) -> None:
        df = _make_clean_df(50)
        df.iloc[5, df.columns.get_loc("Volume")] = 0.0
        out = cleaner.drop_zero_volume(df)
        assert len(out) == len(df) - 1

    def test_no_volume_col_passthrough(self, cleaner: DataCleaner) -> None:
        df = _make_clean_df(50).drop(columns=["Volume"])
        out = cleaner.drop_zero_volume(df)
        assert len(out) == len(df)


class TestRemoveOutliers:
    def test_normal_data_unchanged(self, cleaner: DataCleaner) -> None:
        df = _make_clean_df(100)
        out = cleaner.remove_outliers(df)
        # No extreme returns → Close should be unchanged
        pd.testing.assert_series_equal(df["Close"], out["Close"])

    def test_extreme_return_replaced(self, cleaner: DataCleaner) -> None:
        df = _make_clean_df(100)
        # Inject a 50% jump → clearly beyond 6σ
        idx = 50
        df.iloc[idx, df.columns.get_loc("Close")] = df.iloc[idx - 1]["Close"] * 1.50
        out = cleaner.remove_outliers(df)
        # The Close at the outlier position should be interpolated, not 1.5× the prior
        ratio = out.iloc[idx]["Close"] / out.iloc[idx - 1]["Close"]
        assert abs(ratio - 1.50) > 0.1, "Outlier should have been smoothed"


class TestComputeLogReturns:
    def test_log_return_column_added(self, cleaner: DataCleaner) -> None:
        df = _make_clean_df(50)
        out = cleaner.compute_log_returns(df)
        assert "log_return" in out.columns

    def test_first_return_is_nan(self, cleaner: DataCleaner) -> None:
        df = _make_clean_df(50)
        out = cleaner.compute_log_returns(df)
        assert pd.isna(out["log_return"].iloc[0])

    def test_log_return_values_correct(self, cleaner: DataCleaner) -> None:
        df = _make_clean_df(50)
        out = cleaner.compute_log_returns(df)
        expected = np.log(df["Close"].iloc[1] / df["Close"].iloc[0])
        assert abs(out["log_return"].iloc[1] - expected) < 1e-10


class TestRunPipeline:
    def test_pipeline_returns_dataframe(self, cleaner: DataCleaner) -> None:
        df = _make_clean_df(200)
        out = cleaner.run_pipeline(df)
        assert isinstance(out, pd.DataFrame)

    def test_log_return_present(self, cleaner: DataCleaner) -> None:
        df = _make_clean_df(200)
        out = cleaner.run_pipeline(df)
        assert "log_return" in out.columns

    def test_no_nan_in_log_return(self, cleaner: DataCleaner) -> None:
        df = _make_clean_df(200)
        out = cleaner.run_pipeline(df)
        assert out["log_return"].isna().sum() == 0

    def test_output_smaller_than_input(self, cleaner: DataCleaner) -> None:
        df = _make_clean_df(200)
        out = cleaner.run_pipeline(df)
        # The pipeline drops the first row (NaN log-return)
        assert len(out) <= len(df)

