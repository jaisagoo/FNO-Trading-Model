"""Market data: loading and caching, schema/sanity validation, and cleaning."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from wsq_trading import config
from wsq_trading.constants import OHLCV_COLUMNS
from wsq_trading.utils import ensure_dir, get_logger, parquet_path, timer

log = get_logger(__name__)


class DataLoader:
    """Fetches OHLCV market data from Yahoo Finance with optional parquet caching.

    Parameters
    ----------
    raw_dir : Path | str, optional
        Directory for parquet cache files.  Defaults to ``config.RAW_DIR``.
    cache : bool, optional
        Master switch: if ``False``, never read or write cache files.
        Defaults to ``config.CACHE_PARQUET``.
    cache_max_age_hours : float, optional
        Re-download if cached file is older than this.
        Defaults to ``config.CACHE_MAX_AGE_HOURS``.
    """

    def __init__(
        self,
        raw_dir: Path | str | None = None,
        cache: bool | None = None,
        cache_max_age_hours: float | None = None,
    ) -> None:
        self._raw_dir = Path(raw_dir) if raw_dir else config.RAW_DIR
        self._cache = config.CACHE_PARQUET if cache is None else cache
        self._max_age = (
            config.CACHE_MAX_AGE_HOURS if cache_max_age_hours is None else cache_max_age_hours
        )
        if self._cache:
            ensure_dir(self._raw_dir)


    def fetch(
        self,
        ticker: str,
        start: str | None = None,
        end: str | None = None,
        force_refresh: bool = False,
        cache: bool | None = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV data for *ticker*, using the parquet cache when valid.

        Parameters
        ----------
        ticker : str
            Yahoo Finance symbol, e.g. ``'ES=F'``, ``'SPY'``, ``'AAPL'``.
        start : str, optional
            Start date ``'YYYY-MM-DD'``.  Defaults to ``config.DEFAULT_START``.
        end : str, optional
            End date ``'YYYY-MM-DD'``.  Defaults to today.
        force_refresh : bool
            Bypass the cache and re-download even if a fresh file exists.
        cache : bool, optional
            Override the instance-level cache setting for this call only.

        Returns
        -------
        pd.DataFrame
            OHLCV DataFrame with a ``pd.DatetimeIndex`` (UTC-naive daily bars).
            Columns: ``Open``, ``High``, ``Low``, ``Close``, ``Volume``.

        Raises
        ------
        ValueError
            If yfinance returns an empty DataFrame for the requested ticker/range.
        """
        use_cache = self._cache if cache is None else cache
        start = start or config.DEFAULT_START
        end = end or dt.date.today().isoformat()

        pq_file = parquet_path(self._raw_dir, ticker)

        # Serve from cache if valid
        if use_cache and not force_refresh and self._cache_is_valid(pq_file):
            log.info("Loading '%s' from cache: %s", ticker, pq_file)
            return self._read_parquet(pq_file)

        # Download from yfinance
        log.info("Downloading '%s' from Yahoo Finance (%s -> %s)…", ticker, start, end)
        with timer(f"yfinance download [{ticker}]"):
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,   # adjust OHLC for splits/dividends
                progress=False,
                threads=False,
            )

        if raw.empty:
            raise ValueError(
                f"yfinance returned no data for ticker '{ticker}' "
                f"between {start} and {end}. "
                "Check the symbol or try a different date range."
            )

        df = self._normalise(raw, ticker)

        # Persist to parquet
        if use_cache:
            ensure_dir(pq_file.parent)
            df.to_parquet(pq_file, index=True)
            log.info("Cached '%s' -> %s  (%d rows)", ticker, pq_file, len(df))

        return df

    def fetch_all(
        self,
        tickers: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        force_refresh: bool = False,
        cache: bool | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV data for multiple tickers.

        Parameters
        ----------
        tickers : list[str], optional
            Symbols to fetch.  Defaults to ``config.TICKERS``.
        start, end, force_refresh, cache
            Forwarded to :meth:`fetch`.

        Returns
        -------
        dict[str, pd.DataFrame]
            Mapping ``ticker -> DataFrame``.  Failed tickers are logged and
            omitted (not raised) so a partial failure doesn't abort the run.
        """
        tickers = tickers or config.TICKERS
        results: dict[str, pd.DataFrame] = {}

        for ticker in tickers:
            try:
                results[ticker] = self.fetch(
                    ticker,
                    start=start,
                    end=end,
                    force_refresh=force_refresh,
                    cache=cache,
                )
            except Exception as exc:
                log.warning("Failed to fetch '%s': %s", ticker, exc)

        log.info(
            "fetch_all complete: %d/%d tickers loaded.", len(results), len(tickers)
        )
        return results

    def load_cached(self, ticker: str) -> pd.DataFrame:
        """Load a ticker from parquet cache without any network call.

        Parameters
        ----------
        ticker : str
            Yahoo Finance symbol.

        Returns
        -------
        pd.DataFrame
            Cached OHLCV DataFrame.

        Raises
        ------
        FileNotFoundError
            If no cached file exists for *ticker*.
        """
        pq_file = parquet_path(self._raw_dir, ticker)
        if not pq_file.exists():
            raise FileNotFoundError(
                f"No cached file for '{ticker}' at {pq_file}. "
                "Call fetch() first to download and cache data."
            )
        log.info("Loading '%s' from cache: %s", ticker, pq_file)
        return self._read_parquet(pq_file)

    def is_cached(self, ticker: str) -> bool:
        """Return ``True`` if a parquet cache file exists for *ticker*.

        Parameters
        ----------
        ticker : str
            Yahoo Finance symbol.
        """
        return parquet_path(self._raw_dir, ticker).exists()

    def cache_age_hours(self, ticker: str) -> float:
        """Return the age of the parquet cache file in hours.

        Parameters
        ----------
        ticker : str
            Yahoo Finance symbol.

        Returns
        -------
        float
            Age in hours.  Returns ``float('inf')`` if no cache exists.
        """
        pq_file = parquet_path(self._raw_dir, ticker)
        if not pq_file.exists():
            return float("inf")
        mtime = dt.datetime.fromtimestamp(pq_file.stat().st_mtime)
        return (dt.datetime.now() - mtime).total_seconds() / 3600.0

    def clear_cache(self, ticker: str | None = None) -> None:
        """Delete parquet cache file(s).

        Parameters
        ----------
        ticker : str, optional
            If given, delete only this ticker's cache file.
            If ``None``, delete all ``.parquet`` files in ``raw_dir``.
        """
        if ticker:
            pq_file = parquet_path(self._raw_dir, ticker)
            if pq_file.exists():
                pq_file.unlink()
                log.info("Cleared cache for '%s'.", ticker)
        else:
            removed = 0
            for pq_file in self._raw_dir.glob("*.parquet"):
                pq_file.unlink()
                removed += 1
            log.info("Cleared %d cached parquet file(s) from %s.", removed, self._raw_dir)

    def list_cached(self) -> list[str]:
        """Return tickers that currently have a valid parquet cache file.

        Returns
        -------
        list[str]
            Ticker symbols inferred from the filenames in ``raw_dir``.
        """
        if not self._raw_dir.exists():
            return []
        return [
            p.stem.replace("_F", "=F")   # reverse safe_name transformation
            for p in sorted(self._raw_dir.glob("*.parquet"))
        ]

    # Internal helpers

    def _cache_is_valid(self, pq_file: Path) -> bool:
        """Return True if pq_file exists and is younger than max cache age."""
        if not pq_file.exists():
            return False
        age_hours = (
            dt.datetime.now()
            - dt.datetime.fromtimestamp(pq_file.stat().st_mtime)
        ).total_seconds() / 3600.0
        if age_hours > self._max_age:
            log.debug("Cache stale (%.1f h > %.1f h): %s", age_hours, self._max_age, pq_file)
            return False
        return True

    @staticmethod
    def _read_parquet(path: Path) -> pd.DataFrame:
        """Read a parquet file and ensure a clean DatetimeIndex."""
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "Date"
        return df

    @staticmethod
    def _normalise(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Standardise the yfinance output to a clean OHLCV DataFrame.

        yfinance ≥ 0.2.x returns a MultiIndex column when ``auto_adjust=True``
        with a 'Price' level and a 'Ticker' level.  This method flattens that
        and selects the canonical five columns.
        """
        df = raw.copy()

        # Flatten MultiIndex columns produced by newer yfinance versions
        if isinstance(df.columns, pd.MultiIndex):
            # (Price, Ticker) -> Price
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

        # Ensure canonical column names
        df.columns = [str(c).strip() for c in df.columns]

        # Select OHLCV subset; Volume may be absent for some instruments
        keep = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
        df = df[keep].copy()

        # Drop rows where all price columns are NaN
        price_cols = [c for c in ("Open", "High", "Low", "Close") if c in df.columns]
        df.dropna(subset=price_cols, how="all", inplace=True)

        # Ensure a clean DatetimeIndex
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "Date"
        df.sort_index(inplace=True)

        log.debug("'%s': %d rows, %s -> %s", ticker, len(df), df.index[0].date(), df.index[-1].date())
        return df


@dataclass
class ValidationResult:
    """Outcome of a single validation check.

    Attributes
    ----------
    passed : bool
        ``True`` if the check passed.
    messages : list[str]
        Human-readable diagnostics (warnings or errors).
    check_name : str
        Name of the check for logging.
    """

    passed: bool
    messages: list[str] = field(default_factory=list)
    check_name: str = ""

    def __bool__(self) -> bool:
        return self.passed

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        msgs = "; ".join(self.messages) if self.messages else "-"
        return f"ValidationResult({status} | {self.check_name}: {msgs})"


@dataclass
class ValidationReport:
    """Aggregated results across all checks for one DataFrame.

    Attributes
    ----------
    ticker : str
        Ticker symbol this report refers to.
    results : list[ValidationResult]
        Individual check outcomes.
    """

    ticker: str
    results: list[ValidationResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """``True`` only if every check passed."""
        return all(r.passed for r in self.results)

    @property
    def failed_checks(self) -> list[ValidationResult]:
        return [r for r in self.results if not r.passed]

    def summary(self) -> str:
        n_pass = sum(1 for r in self.results if r.passed)
        n_fail = len(self.results) - n_pass
        status = "✓ PASS" if self.passed else "✗ FAIL"
        return (
            f"[{status}] {self.ticker}: {n_pass} passed, {n_fail} failed "
            f"({len(self.results)} checks total)"
        )


# Validator

class DataValidator:
    """Validates OHLCV DataFrames produced by DataLoader.

    Parameters
    ----------
    min_rows : int, optional
        Minimum acceptable row count.  Defaults to ``config.MIN_ROWS``.
    max_gap_days : int, optional
        Maximum tolerated consecutive-day gap (weekends excluded).
        Defaults to 5 trading days.
    outlier_sigma : float, optional
        Flag daily log-returns beyond this many standard deviations.
        Defaults to 8.
    """

    def __init__(
        self,
        min_rows: int | None = None,
        max_gap_days: int = 5,
        outlier_sigma: float = 8.0,
    ) -> None:
        self._min_rows = min_rows or config.MIN_ROWS
        self._max_gap_days = max_gap_days
        self._outlier_sigma = outlier_sigma

    # Full pipeline

    def validate(self, df: pd.DataFrame, ticker: str = "") -> ValidationReport:
        """Run all checks and return an aggregated report.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame to validate.
        ticker : str, optional
            Symbol name used for logging only.

        Returns
        -------
        ValidationReport
            Contains individual ``ValidationResult`` objects.
        """
        report = ValidationReport(ticker=ticker)
        checks = [
            self.check_schema,
            self.check_min_rows,
            self.check_ohlcv_sanity,
            self.check_gaps,
            self.check_no_future_dates,
            self.check_outliers,
        ]
        for check in checks:
            result = check(df)
            report.results.append(result)
            if result.passed:
                log.debug("[%s] %s", ticker, result)
            else:
                log.warning("[%s] %s", ticker, result)

        log.info(report.summary())
        return report

    # Individual checks

    def check_schema(self, df: pd.DataFrame) -> ValidationResult:
        """Assert that required OHLCV columns are present and numeric.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to inspect.
        """
        msgs: list[str] = []
        required = list(OHLCV_COLUMNS)

        missing = [c for c in required if c not in df.columns]
        if missing:
            msgs.append(f"Missing columns: {missing}")

        non_numeric = [
            c for c in required if c in df.columns and not pd.api.types.is_numeric_dtype(df[c])
        ]
        if non_numeric:
            msgs.append(f"Non-numeric columns: {non_numeric}")

        if not isinstance(df.index, pd.DatetimeIndex):
            msgs.append("Index is not a DatetimeIndex.")

        return ValidationResult(
            passed=len(msgs) == 0,
            messages=msgs,
            check_name="schema",
        )

    def check_min_rows(self, df: pd.DataFrame) -> ValidationResult:
        """Verify the DataFrame has at least ``min_rows`` rows.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to inspect.
        """
        n = len(df)
        passed = n >= self._min_rows
        msgs = [] if passed else [f"Only {n} rows (need ≥ {self._min_rows})."]
        return ValidationResult(passed=passed, messages=msgs, check_name="min_rows")

    def check_ohlcv_sanity(self, df: pd.DataFrame) -> ValidationResult:
        """Check fundamental OHLCV relationships.

        Rules
        -----
        - High ≥ Open, High ≥ Close, High ≥ Low
        - Low ≤ Open, Low ≤ Close
        - Close > 0 everywhere
        - Volume ≥ 0 everywhere (if present)

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to inspect.
        """
        msgs: list[str] = []

        if not all(c in df.columns for c in ("Open", "High", "Low", "Close")):
            return ValidationResult(
                passed=False,
                messages=["Required OHLC columns missing; cannot run sanity check."],
                check_name="ohlcv_sanity",
            )

        high_low = (df["High"] < df["Low"]).sum()
        if high_low:
            msgs.append(f"High < Low on {high_low} rows.")

        high_open = (df["High"] < df["Open"]).sum()
        if high_open:
            msgs.append(f"High < Open on {high_open} rows.")

        high_close = (df["High"] < df["Close"]).sum()
        if high_close:
            msgs.append(f"High < Close on {high_close} rows.")

        non_pos = (df["Close"] <= 0).sum()
        if non_pos:
            msgs.append(f"Close ≤ 0 on {non_pos} rows.")

        if "Volume" in df.columns:
            neg_vol = (df["Volume"] < 0).sum()
            if neg_vol:
                msgs.append(f"Volume < 0 on {neg_vol} rows.")

        return ValidationResult(
            passed=len(msgs) == 0,
            messages=msgs,
            check_name="ohlcv_sanity",
        )

    def check_gaps(self, df: pd.DataFrame) -> ValidationResult:
        """Detect large gaps in the time series (calendar days, not business days).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with DatetimeIndex.
        """
        if len(df) < 2:
            return ValidationResult(
                passed=True,
                messages=["Too few rows to check gaps."],
                check_name="gaps",
            )

        # Use calendar-day gaps; multiply max_gap_days by ~1.5 for weekends
        cal_threshold = self._max_gap_days * 2   # generous calendar buffer
        diffs = df.index.to_series().diff().dt.days.dropna()
        large_gaps = diffs[diffs > cal_threshold]

        msgs: list[str] = []
        if len(large_gaps):
            details = ", ".join(
                f"{idx.date()} (gap={int(v)} days)"
                for idx, v in large_gaps.items()
            )
            msgs.append(f"{len(large_gaps)} large gap(s): {details}")

        return ValidationResult(
            passed=len(msgs) == 0,
            messages=msgs,
            check_name="gaps",
        )

    def check_no_future_dates(self, df: pd.DataFrame) -> ValidationResult:
        """Ensure the DataFrame contains no future-dated rows.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with DatetimeIndex.
        """
        today = pd.Timestamp.today().normalize()
        future = df.index[df.index > today]
        passed = len(future) == 0
        msgs = [] if passed else [f"{len(future)} future date(s) found: {list(future[:3])}."]
        return ValidationResult(passed=passed, messages=msgs, check_name="no_future_dates")

    def check_outliers(self, df: pd.DataFrame) -> ValidationResult:
        """Flag extreme log-return outliers.

        Uses a rolling z-score to catch both global and local extremes.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing a ``Close`` column.
        """
        if "Close" not in df.columns or len(df) < 5:
            return ValidationResult(
                passed=True,
                messages=["Skipped outlier check (insufficient Close data)."],
                check_name="outliers",
            )

        log_ret = np.log(df["Close"] / df["Close"].shift(1)).dropna()
        z = (log_ret - log_ret.mean()) / (log_ret.std() + 1e-10)
        outliers = z.abs()[z.abs() > self._outlier_sigma]

        msgs: list[str] = []
        if len(outliers):
            details = ", ".join(
                f"{idx.date()} (z={v:.1f})" for idx, v in outliers.items()
            )
            msgs.append(
                f"{len(outliers)} outlier(s) beyond {self._outlier_sigma}σ: {details}"
            )

        # Outliers are a warning, not a hard failure - return passed=True
        # so the pipeline continues; callers can inspect messages themselves.
        return ValidationResult(
            passed=True,
            messages=msgs,
            check_name="outliers",
        )


class DataCleaner:
    """Clean and normalise raw OHLCV DataFrames.

    Parameters
    ----------
    max_fill_days : int, optional
        Maximum consecutive NaN days to forward-fill.  Gaps longer than this
        are left as NaN and flagged.  Defaults to ``5``.
    outlier_sigma : float, optional
        Log-return z-score threshold beyond which the Close price is replaced
        by linear interpolation.  Defaults to ``6.0``.
    min_price : float, optional
        Rows with ``Close < min_price`` are dropped (handles zero/negative
        artefacts in raw data).  Defaults to ``1e-6``.
    drop_zero_volume : bool, optional
        Drop rows where ``Volume == 0`` (if Volume column is present).
        Defaults to ``True``.
    """

    def __init__(
        self,
        max_fill_days: int = 5,
        outlier_sigma: float = 6.0,
        min_price: float = 1e-6,
        drop_zero_volume: bool = True,
    ) -> None:
        self._max_fill = max_fill_days
        self._sigma = outlier_sigma
        self._min_price = min_price
        self._drop_zero_vol = drop_zero_volume


    def run_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the full cleaning pipeline in order.

        Steps
        -----
        1. Reindex to a business-day frequency to surface implicit gaps.
        2. Forward-fill gaps up to ``max_fill_days``.
        3. Drop rows with ``Close < min_price``.
        4. Optionally drop zero-volume rows.
        5. Winsorise extreme log-return outliers.
        6. Append a ``log_return`` column.
        7. Drop leading NaN rows (artefact of log-return shift).

        Parameters
        ----------
        df : pd.DataFrame
            Raw OHLCV DataFrame from ``DataLoader``.

        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame with ``log_return`` column appended.
        """
        n_raw = len(df)
        log.info("Cleaning pipeline started (%d rows in).", n_raw)

        df = self.fill_gaps(df)
        df = self.clip_prices(df)
        if self._drop_zero_vol and "Volume" in df.columns:
            df = self.drop_zero_volume(df)
        df = self.remove_outliers(df)
        df = self.compute_log_returns(df)
        df = df.dropna(subset=["log_return"])

        log.info(
            "Cleaning pipeline complete: %d -> %d rows (dropped %d).",
            n_raw,
            len(df),
            n_raw - len(df),
        )
        return df

    def fill_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reindex to business-day frequency and forward-fill short gaps.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame with DatetimeIndex.

        Returns
        -------
        pd.DataFrame
            DataFrame on a complete business-day grid; gaps longer than
            ``max_fill_days`` remain as NaN.
        """
        if df.empty:
            return df

        full_idx = pd.bdate_range(start=df.index.min(), end=df.index.max())
        n_implicit_gaps = len(full_idx) - len(df)
        if n_implicit_gaps:
            log.debug("fill_gaps: %d implicit business-day gap(s) found.", n_implicit_gaps)

        df = df.reindex(full_idx)
        df.index.name = "Date"

        # Forward-fill up to max_fill_days consecutive NaNs
        df = df.ffill(limit=self._max_fill)
        return df

    def clip_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows where ``Close`` is below ``min_price``.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame.
        """
        if "Close" not in df.columns:
            return df

        mask = df["Close"] >= self._min_price
        n_dropped = (~mask).sum()
        if n_dropped:
            log.warning("clip_prices: dropping %d rows with Close < %g.", n_dropped, self._min_price)
        return df.loc[mask].copy()

    def drop_zero_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows where ``Volume`` is zero or NaN.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame.
        """
        if "Volume" not in df.columns:
            return df

        mask = df["Volume"].fillna(0) > 0
        n_dropped = (~mask).sum()
        if n_dropped:
            log.debug("drop_zero_volume: dropping %d zero-volume rows.", n_dropped)
        return df.loc[mask].copy()

    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace extreme log-return outliers with linear interpolation.

        Extreme returns are winsorised rather than dropped, preserving the
        business-day grid required by the backtest engine.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame.
        """
        if "Close" not in df.columns or len(df) < 5:
            return df

        log_ret = np.log(df["Close"] / df["Close"].shift(1))
        mu = log_ret.mean()
        sigma = log_ret.std()
        z = (log_ret - mu) / (sigma + 1e-12)
        outlier_mask = z.abs() > self._sigma

        n_outliers = outlier_mask.sum()
        if n_outliers:
            log.warning(
                "remove_outliers: replacing %d outlier row(s) (|z| > %.1f) "
                "via linear interpolation.",
                n_outliers,
                self._sigma,
            )
            # Replace Close with NaN at outlier positions then interpolate
            df = df.copy()
            df.loc[outlier_mask, "Close"] = np.nan
            df["Close"] = df["Close"].interpolate(method="time")

        return df

    def compute_log_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Append a ``log_return`` column (natural log of daily price ratio).

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame with a ``Close`` column.

        Returns
        -------
        pd.DataFrame
            Same DataFrame with ``log_return`` appended.
        """
        if "Close" not in df.columns:
            log.warning("compute_log_returns: no Close column found; skipping.")
            return df

        df = df.copy()
        df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
        return df

