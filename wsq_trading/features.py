"""Feature engineering for OHLCV market data.

Operates entirely in memory.  Input: a cleaned OHLCV DataFrame (from
DataCleaner).  Output: the same DataFrame with additional columns appended.

All methods return a new DataFrame; the original is never mutated.
Call ``run_pipeline`` to apply the full feature set in one shot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from wsq_trading import config
from wsq_trading.constants import SQRT_252
from wsq_trading.utils import get_logger

log = get_logger(__name__)


class FeatureEngineer:
    """Compute and append features to cleaned OHLCV DataFrames.

    Parameters
    ----------
    vol_window : int, optional
        Rolling window for realized volatility.  Defaults to ``config.VOL_WINDOW``.
    stats_window : int, optional
        Window for rolling statistics.  Defaults to ``config.ROLLING_STATS_WINDOW``.
    long_window : int, optional
        Window for long-horizon statistics.  Defaults to ``config.LONG_ROLLING_WINDOW``.
    lags : list[int], optional
        Return lags to compute.  Defaults to ``[1, 2, 3, 5, 10, 21]``.
    """

    def __init__(
        self,
        vol_window: int | None = None,
        stats_window: int | None = None,
        long_window: int | None = None,
        lags: list[int] | None = None,
    ) -> None:
        self._vol_window = vol_window or config.VOL_WINDOW
        self._stats_window = stats_window or config.ROLLING_STATS_WINDOW
        self._long_window = long_window or config.LONG_ROLLING_WINDOW
        self._lags = lags or [1, 2, 3, 5, 10, 21]


    def run_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the full feature engineering pipeline.

        Steps (in order)
        ----------------
        1. Log returns (if not already present)
        2. Realized volatility (short and long window)
        3. Rolling statistics (mean, std, skewness, kurtosis)
        4. Parkinson high-low volatility estimator
        5. Volume features (if Volume is present)
        6. Lagged returns

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned OHLCV DataFrame (output of ``DataCleaner``).

        Returns
        -------
        pd.DataFrame
            Original columns plus all engineered features.
        """
        log.info("FeatureEngineer.run_pipeline: %d rows in.", len(df))
        df = df.copy()
        df = self.log_returns(df)
        df = self.realized_vol(df)
        df = self.rolling_stats(df)
        df = self.parkinson_vol(df)
        if "Volume" in df.columns:
            df = self.volume_features(df)
        df = self.lagged_returns(df)
        log.info(
            "FeatureEngineer.run_pipeline: done. %d columns added.",
            len(df.columns),
        )
        return df

    def log_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Append ``log_return`` column if not already present.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame with a ``Close`` column.
        """
        df = df.copy()
        if "log_return" not in df.columns:
            if "Close" not in df.columns:
                log.warning("log_returns: no Close column; skipping.")
                return df
            df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
        return df

    def realized_vol(self, df: pd.DataFrame) -> pd.DataFrame:
        """Append short- and long-window annualised realized volatility.

        New columns
        -----------
        ``rv_{vol_window}d``  : Annualised realized vol over ``vol_window`` days.
        ``rv_{long_window}d`` : Annualised realized vol over ``long_window`` days.
        ``vol_ratio``         : Short-window vol / long-window vol (vol-of-vol proxy).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing ``log_return``.
        """
        df = df.copy()
        if "log_return" not in df.columns:
            df = self.log_returns(df)

        short_col = f"rv_{self._vol_window}d"
        long_col = f"rv_{self._long_window}d"

        df[short_col] = (
            df["log_return"]
            .rolling(self._vol_window, min_periods=max(5, self._vol_window // 4))
            .std()
            * SQRT_252
        )
        df[long_col] = (
            df["log_return"]
            .rolling(self._long_window, min_periods=max(21, self._long_window // 4))
            .std()
            * SQRT_252
        )
        df["vol_ratio"] = df[short_col] / (df[long_col] + 1e-10)

        return df

    def rolling_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Append rolling mean, std, skewness, and excess kurtosis of log-returns.

        New columns (all over ``stats_window`` days)
        --------------------------------------------
        ``ret_mean``  : Rolling mean log-return (drift proxy).
        ``ret_std``   : Rolling std (already in rv, kept for convenience).
        ``ret_skew``  : Rolling skewness (tail-direction indicator).
        ``ret_kurt``  : Rolling excess kurtosis (fat-tail indicator).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing ``log_return``.
        """
        df = df.copy()
        if "log_return" not in df.columns:
            df = self.log_returns(df)

        w = self._stats_window
        min_p = max(10, w // 4)
        r = df["log_return"]

        df["ret_mean"] = r.rolling(w, min_periods=min_p).mean()
        df["ret_std"] = r.rolling(w, min_periods=min_p).std()
        df["ret_skew"] = r.rolling(w, min_periods=min_p).skew()
        df["ret_kurt"] = r.rolling(w, min_periods=min_p).kurt()

        return df

    def parkinson_vol(self, df: pd.DataFrame) -> pd.DataFrame:
        """Append Parkinson (1980) high-low range volatility estimator.

        The Parkinson estimator uses intraday high-low range and is more
        efficient than close-to-close volatility:

            σ²_P = 1 / (4 ln 2) × E[(ln(H/L))²]

        New column: ``parkinson_vol``

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame with ``High`` and ``Low`` columns.
        """
        df = df.copy()
        if "High" not in df.columns or "Low" not in df.columns:
            log.debug("parkinson_vol: High or Low column missing; skipping.")
            return df

        hl_sq = (np.log(df["High"] / df["Low"])) ** 2
        factor = 1.0 / (4.0 * np.log(2.0))
        w = self._vol_window
        df["parkinson_vol"] = (
            np.sqrt(factor * hl_sq.rolling(w, min_periods=max(3, w // 4)).mean()) * SQRT_252
        )
        return df

    def volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Append volume-based features.

        New columns
        -----------
        ``volume_ma``    : Rolling mean volume over ``stats_window`` days.
        ``volume_ratio`` : Volume / volume_ma (relative activity indicator).
        ``amihud``       : Amihud (2002) illiquidity ratio = |ret| / Volume.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame with ``Volume`` and ``log_return`` columns.
        """
        df = df.copy()
        if "Volume" not in df.columns:
            return df
        if "log_return" not in df.columns:
            df = self.log_returns(df)

        w = self._stats_window
        min_p = max(5, w // 4)

        df["volume_ma"] = df["Volume"].rolling(w, min_periods=min_p).mean()
        df["volume_ratio"] = df["Volume"] / (df["volume_ma"] + 1e-10)
        df["amihud"] = (
            df["log_return"].abs() / (df["Volume"].replace(0, np.nan) + 1e-10)
        ).rolling(w, min_periods=min_p).mean()

        return df

    def lagged_returns(
        self,
        df: pd.DataFrame,
        lags: list[int] | None = None,
    ) -> pd.DataFrame:
        """Append lagged log-return columns.

        New columns: ``ret_lag_1``, ``ret_lag_2``, … (one per lag).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing ``log_return``.
        lags : list[int], optional
            Lag periods.  Defaults to ``self._lags``.
        """
        df = df.copy()
        if "log_return" not in df.columns:
            df = self.log_returns(df)
        lags = lags or self._lags
        for lag in lags:
            df[f"ret_lag_{lag}"] = df["log_return"].shift(lag)
        return df

