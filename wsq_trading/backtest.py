"""Walk-forward backtest engine and performance metrics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from wsq_trading import config
from wsq_trading.constants import TRADING_DAYS_PER_YEAR, Regime
from wsq_trading.signals import HurstRegimeSignalGenerator
from wsq_trading.utils import get_logger, timer

log = get_logger(__name__)


_NAN = float("nan")


# Internal helpers

def _to_array(returns: pd.Series | np.ndarray) -> np.ndarray:
    """Coerce input to a clean float64 array with NaNs dropped."""
    if isinstance(returns, pd.Series):
        arr = returns.dropna().to_numpy(dtype=float)
    else:
        arr = np.asarray(returns, dtype=float)
        arr = arr[~np.isnan(arr)]
    return arr


def _cum_returns(arr: np.ndarray) -> np.ndarray:
    """Cumulative growth factor from a series of log-like arithmetic returns."""
    return np.cumprod(1.0 + arr)


# Core metrics

def sharpe_ratio(
    returns: pd.Series | np.ndarray,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    risk_free_rate: float = 0.0,
) -> float:
    """Annualised Sharpe ratio.

    Parameters
    ----------
    returns : array-like
        Daily arithmetic returns (e.g. 0.01 for +1%).
    periods_per_year : int
        Number of return observations per year.  Default 252 (daily).
    risk_free_rate : float
        Annual risk-free rate.  Default 0.

    Returns
    -------
    float
        Annualised Sharpe, or NaN if fewer than 2 observations.
    """
    arr = _to_array(returns)
    if len(arr) < 2:
        return _NAN
    daily_rf = risk_free_rate / periods_per_year
    excess = arr - daily_rf
    std = excess.std(ddof=1)
    if std == 0:
        return _NAN
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series | np.ndarray,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    target_return: float = 0.0,
) -> float:
    """Annualised Sortino ratio (penalises only downside deviation).

    Parameters
    ----------
    returns : array-like
        Daily arithmetic returns.
    periods_per_year : int
        Annualisation factor.
    target_return : float
        Minimum acceptable daily return.  Default 0.

    Returns
    -------
    float
        Annualised Sortino ratio, or NaN if downside deviation is zero.
    """
    arr = _to_array(returns)
    if len(arr) < 2:
        return _NAN
    daily_target = target_return / periods_per_year
    downside = arr[arr < daily_target] - daily_target
    downside_std = np.sqrt(np.mean(downside ** 2))
    if downside_std == 0:
        return _NAN
    return float((arr.mean() - daily_target) / downside_std * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series | np.ndarray) -> float:
    """Maximum peak-to-trough drawdown (expressed as a positive fraction).

    Parameters
    ----------
    returns : array-like
        Daily arithmetic returns.

    Returns
    -------
    float
        Maximum drawdown in [0, 1], e.g. 0.15 means 15% peak-to-trough loss.
        Returns NaN for empty series.
    """
    arr = _to_array(returns)
    if len(arr) == 0:
        return _NAN
    cum = _cum_returns(arr)
    running_max = np.maximum.accumulate(cum)
    drawdowns = (running_max - cum) / running_max
    return float(drawdowns.max())


def calmar_ratio(
    returns: pd.Series | np.ndarray,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Calmar ratio = annualised return / max drawdown.

    Parameters
    ----------
    returns : array-like
        Daily arithmetic returns.
    periods_per_year : int
        Annualisation factor.

    Returns
    -------
    float
        Calmar ratio, or NaN if max drawdown is zero or undefined.
    """
    mdd = max_drawdown(returns)
    if np.isnan(mdd) or mdd == 0:
        return _NAN
    ann_ret = annualized_return(returns, periods_per_year)
    return float(ann_ret / mdd)


def win_rate(returns: pd.Series | np.ndarray) -> float:
    """Fraction of return observations that are strictly positive.

    Parameters
    ----------
    returns : array-like
        Daily arithmetic returns.

    Returns
    -------
    float
        Win rate in [0, 1], or NaN for empty series.
    """
    arr = _to_array(returns)
    if len(arr) == 0:
        return _NAN
    return float((arr > 0).sum() / len(arr))


def annualized_return(
    returns: pd.Series | np.ndarray,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Compound annualised return (geometric mean).

    Parameters
    ----------
    returns : array-like
        Daily arithmetic returns.
    periods_per_year : int
        Annualisation factor.

    Returns
    -------
    float
        Annualised compound return, or NaN for empty series.
    """
    arr = _to_array(returns)
    if len(arr) == 0:
        return _NAN
    total = float(_cum_returns(arr)[-1])
    if total <= 0:
        # Cumulative wealth went to zero or negative; return NaN to avoid complex numbers.
        return _NAN
    n_years = len(arr) / periods_per_year
    return float(total ** (1.0 / n_years) - 1.0)


def annualized_volatility(
    returns: pd.Series | np.ndarray,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualised standard deviation of returns.

    Parameters
    ----------
    returns : array-like
        Daily arithmetic returns.
    periods_per_year : int
        Annualisation factor.

    Returns
    -------
    float
        Annualised volatility, or NaN for fewer than 2 observations.
    """
    arr = _to_array(returns)
    if len(arr) < 2:
        return _NAN
    return float(arr.std(ddof=1) * np.sqrt(periods_per_year))


def profit_factor(returns: pd.Series | np.ndarray) -> float:
    """Gross profit / gross loss (always positive).

    Parameters
    ----------
    returns : array-like
        Daily arithmetic returns.

    Returns
    -------
    float
        Profit factor >= 0, or NaN if there are no losing days.
    """
    arr = _to_array(returns)
    gross_profit = arr[arr > 0].sum()
    gross_loss = abs(arr[arr < 0].sum())
    if gross_loss == 0:
        return _NAN
    return float(gross_profit / gross_loss)


def turnover(
    positions: pd.Series | np.ndarray,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualised turnover as a fraction of NAV.

    Turnover = sum of absolute daily position changes / number of observations,
    then scaled to an annual figure.

    Parameters
    ----------
    positions : array-like
        Daily position sizes (e.g. {-1, 0, 1}).
    periods_per_year : int
        Annualisation factor.

    Returns
    -------
    float
        Annualised turnover (0 = no trading, 2 = fully long/short every day).
    """
    if isinstance(positions, pd.Series):
        pos = positions.dropna().to_numpy(dtype=float)
    else:
        pos = np.asarray(positions, dtype=float)
    if len(pos) < 2:
        return _NAN
    daily_turnover = np.mean(np.abs(np.diff(pos)))
    return float(daily_turnover * periods_per_year)


# Convenience wrapper

def compute_all(
    returns: pd.Series | np.ndarray,
    positions: pd.Series | np.ndarray | None = None,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict[str, float]:
    """Compute all standard backtest metrics and return them as a dict.

    Parameters
    ----------
    returns : array-like
        Daily strategy returns (net of costs).
    positions : array-like, optional
        Daily position sizes; used for turnover calculation.
    periods_per_year : int
        Annualisation factor.  Default 252.

    Returns
    -------
    dict[str, float]
        Keys: sharpe, sortino, max_drawdown, calmar, win_rate,
              annualized_return, annualized_vol, profit_factor,
              turnover (if positions provided), n_obs.
    """
    result: dict[str, float] = {
        "sharpe": sharpe_ratio(returns, periods_per_year),
        "sortino": sortino_ratio(returns, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "calmar": calmar_ratio(returns, periods_per_year),
        "win_rate": win_rate(returns),
        "annualized_return": annualized_return(returns, periods_per_year),
        "annualized_vol": annualized_volatility(returns, periods_per_year),
        "profit_factor": profit_factor(returns),
        "n_obs": float(len(_to_array(returns))),
    }
    if positions is not None:
        result["turnover"] = turnover(positions, periods_per_year)

    log.debug(
        "metrics: sharpe=%.3f  ann_ret=%.1f%%  mdd=%.1f%%  win=%.1f%%",
        result["sharpe"],
        result["annualized_return"] * 100,
        result["max_drawdown"] * 100,
        result["win_rate"] * 100,
    )
    return result


@dataclass
class BacktestResult:
    """All outputs from a single-instrument backtest run.

    Attributes
    ----------
    ticker : str
        Instrument identifier.
    strategy_returns : pd.Series
        Daily net-of-cost strategy returns.
    positions : pd.Series
        Daily position sizes in [-1, 1].  Values are continuous when
        vol-targeting or H-conviction sizing is enabled.
    H_estimates : pd.Series
        Rolling Hurst exponent estimates.
    regimes : pd.Series
        Regime labels (Regime enum) aligned with position dates.
    gross_returns : pd.Series
        Raw instrument returns (before signal application).
    metrics : dict[str, float]
        Summary statistics (Sharpe, Sortino, drawdown, ...).
    """

    ticker: str
    strategy_returns: pd.Series
    positions: pd.Series
    H_estimates: pd.Series
    regimes: pd.Series
    gross_returns: pd.Series
    metrics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.metrics:
            self.metrics = compute_all(
                self.strategy_returns,
                positions=self.positions,
            )

    @property
    def sharpe(self) -> float:
        """Shortcut to the annualised Sharpe ratio."""
        return self.metrics.get("sharpe", float("nan"))

    @property
    def max_drawdown(self) -> float:
        """Shortcut to the maximum drawdown (positive fraction)."""
        return self.metrics.get("max_drawdown", float("nan"))


# Engine

class WalkForwardEngine:
    """Apply a signal generator to OHLCV data with strict temporal ordering.

    Parameters
    ----------
    signal_generator : HurstRegimeSignalGenerator, optional
        Pre-configured signal generator.  A default one is created if omitted.
    cost_bps : float, optional
        Round-trip transaction cost in basis points applied to each position
        change.  Default: ``config.TRANSACTION_COST_BPS``.
    return_col : str
        Column name for the daily log-return series inside each DataFrame.
        Default ``'log_return'``.
    long_bias_tickers : list[str], optional
        Tickers for which the long-bias trend filter is applied.
        Default: ``config.LONG_BIAS_TICKERS``.
    vol_target_enabled : bool, optional
        Whether to apply volatility targeting.
        Default: ``config.VOL_TARGET_ENABLED``.
    vol_target_window : int, optional
        Bars used for the realized-vol estimate in vol targeting.
        Default: ``config.VOL_TARGET_WINDOW``.
    vol_floor_scalar : float, optional
        Minimum position scalar from vol targeting.
        Default: ``config.VOL_FLOOR_SCALAR``.
    vol_cap_scalar : float, optional
        Maximum position scalar from vol targeting (cap at 1 = no leverage).
        Default: ``config.VOL_CAP_SCALAR``.
    h_conviction_sizing : bool, optional
        Whether to scale position magnitude by H-conviction.
        Default: ``config.H_CONVICTION_SIZING``.
    """

    def __init__(
        self,
        signal_generator: HurstRegimeSignalGenerator | None = None,
        cost_bps: float | None = None,
        return_col: str = "log_return",
        long_bias_tickers: list[str] | None = None,
        vol_target_enabled: bool | None = None,
        vol_target_window: int | None = None,
        vol_floor_scalar: float | None = None,
        vol_cap_scalar: float | None = None,
        h_conviction_sizing: bool | None = None,
    ) -> None:
        self._sig_gen = signal_generator or HurstRegimeSignalGenerator()
        self._cost = (cost_bps if cost_bps is not None else config.TRANSACTION_COST_BPS) / 10_000
        self._return_col = return_col

        self._long_bias_tickers = (
            long_bias_tickers if long_bias_tickers is not None
            else list(config.LONG_BIAS_TICKERS)
        )
        self._vol_target_enabled = (
            vol_target_enabled if vol_target_enabled is not None
            else config.VOL_TARGET_ENABLED
        )
        self._vol_target_window = (
            vol_target_window if vol_target_window is not None
            else config.VOL_TARGET_WINDOW
        )
        self._vol_floor_scalar = (
            vol_floor_scalar if vol_floor_scalar is not None
            else config.VOL_FLOOR_SCALAR
        )
        self._vol_cap_scalar = (
            vol_cap_scalar if vol_cap_scalar is not None
            else config.VOL_CAP_SCALAR
        )
        self._h_conviction_sizing = (
            h_conviction_sizing if h_conviction_sizing is not None
            else config.H_CONVICTION_SIZING
        )

        log.info(
            "WalkForwardEngine: long_bias=%s  vol_target=%s  h_conviction=%s",
            self._long_bias_tickers,
            self._vol_target_enabled,
            self._h_conviction_sizing,
        )


    def run_single(
        self,
        returns: pd.Series,
        ticker: str = "instrument",
    ) -> BacktestResult:
        """Backtest the strategy on a single return series.

        Parameters
        ----------
        returns : pd.Series
            Daily arithmetic (or log-approximate) returns with a DatetimeIndex.
            Must contain at least ``signal_generator.hurst_window`` observations.
        ticker : str
            Label for the instrument (used in logging and result).

        Returns
        -------
        BacktestResult
            Full backtest output including strategy returns and metrics.
        """
        log.info("WalkForwardEngine: running '%s' (%d bars).", ticker, len(returns))

        with timer(f"backtest [{ticker}]"):
            signal_df = self._sig_gen.compute_signals(returns)

        positions = signal_df["signal"].copy().astype(float)
        H_estimates = signal_df["H"]
        regimes = signal_df["regime"]

        # 1. Long-bias trend filter
        if ticker in self._long_bias_tickers:
            positions = self._apply_long_bias_filter(positions, returns)

        # 2. H-conviction position scaling
        if self._h_conviction_sizing:
            positions = self._apply_h_conviction(positions, H_estimates)

        # 3. Volatility targeting
        if self._vol_target_enabled:
            positions = self._apply_vol_target(positions, returns)

        # Strategy returns
        gross_strat = positions * returns
        position_changes = positions.diff().abs().fillna(0.0)
        costs = position_changes * self._cost
        net_strat = gross_strat - costs

        result = BacktestResult(
            ticker=ticker,
            strategy_returns=net_strat,
            positions=positions,
            H_estimates=H_estimates,
            regimes=regimes,
            gross_returns=returns,
        )

        log.info(
            "WalkForwardEngine '%s': Sharpe=%.3f  MaxDD=%.1f%%  WinRate=%.1f%%",
            ticker,
            result.sharpe,
            result.max_drawdown * 100,
            result.metrics.get("win_rate", float("nan")) * 100,
        )
        return result

    def run(self, data: dict[str, pd.DataFrame]) -> dict[str, BacktestResult]:
        """Backtest across multiple instruments."""
        results: dict[str, BacktestResult] = {}
        for ticker, df in data.items():
            if self._return_col not in df.columns:
                log.warning("Ticker '%s' missing '%s' column; skipping.", ticker, self._return_col)
                continue
            returns = df[self._return_col].dropna()
            if len(returns) < self._sig_gen._hurst_window:
                log.warning("Ticker '%s': only %d rows (need >= %d); skipping.",
                            ticker, len(returns), self._sig_gen._hurst_window)
                continue
            try:
                results[ticker] = self.run_single(returns, ticker=ticker)
            except Exception as exc:
                log.warning("Backtest failed for '%s': %s", ticker, exc)
        log.info("WalkForwardEngine.run: completed %d/%d tickers.", len(results), len(data))
        return results

    def summary(self, results: dict[str, BacktestResult]) -> pd.DataFrame:
        """Return a DataFrame of metrics for every instrument in *results*."""
        rows = {ticker: res.metrics for ticker, res in results.items()}
        return pd.DataFrame(rows).T

    # Position transformation helpers

    def _apply_long_bias_filter(self, positions: pd.Series, returns: pd.Series) -> pd.Series:
        """Suppress counter-trend positions and add baseline long in uptrend.

        For instruments with a structural upward bias (equity indices):
          1. Zero any short positions while price is above the trend MA.
          2. Zero any long positions while price is below the trend MA.
          3. Set a baseline long (``config.LONG_BIAS_BASE_POSITION``) whenever
             the net position is still zero in an uptrend.  This prevents the
             strategy from leaving systematic alpha on the table during periods
             where the momentum signal conflicts (MA vs. momentum confirm
             disagree) but the broad trend is intact.
        """
        price = (1 + returns).cumprod()
        trend_ma = price.rolling(
            config.TREND_FILTER_WINDOW,
            min_periods=config.TREND_FILTER_WINDOW // 2,
        ).mean()
        in_uptrend = (price > trend_ma).reindex(positions.index).fillna(False)
        in_downtrend = ~in_uptrend
        pos = positions.copy()

        # Step 1 & 2: kill counter-trend positions
        pos.loc[in_uptrend   & (pos < 0)] = 0.0
        pos.loc[in_downtrend & (pos > 0)] = 0.0

        # Step 3: baseline long during confirmed uptrend when no signal fires
        base = config.LONG_BIAS_BASE_POSITION
        if base > 0.0:
            pos.loc[in_uptrend & (pos == 0.0)] = base

        return pos

    def _apply_h_conviction(self, positions: pd.Series, H_estimates: pd.Series) -> pd.Series:
        """Scale position magnitude by H-conviction."""
        trend_thresh = self._sig_gen._trend_threshold
        rough_thresh = self._sig_gen._rough_threshold
        floor = config.H_CONVICTION_FLOOR
        h = H_estimates.reindex(positions.index).fillna((trend_thresh + rough_thresh) / 2)
        pos_mask = positions > 0
        neg_mask = positions < 0
        scalar = pd.Series(1.0, index=positions.index)
        if pos_mask.any():
            band = max(1e-6, 1.0 - trend_thresh)
            conviction = ((h - trend_thresh) / band).clip(floor, 1.0)
            scalar.loc[pos_mask] = conviction.loc[pos_mask]
        if neg_mask.any():
            band = max(1e-6, rough_thresh)
            conviction = ((rough_thresh - h) / band).clip(floor, 1.0)
            scalar.loc[neg_mask] = conviction.loc[neg_mask]
        return (positions * scalar).clip(-1.0, 1.0)

    def _apply_vol_target(self, positions: pd.Series, returns: pd.Series) -> pd.Series:
        """Scale positions to target a constant annualised portfolio volatility."""
        realized_vol = (
            returns
            .rolling(self._vol_target_window,
                     min_periods=max(5, self._vol_target_window // 4))
            .std()
            .mul(np.sqrt(TRADING_DAYS_PER_YEAR))
        )
        full_sample_vol = float(returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
        realized_vol = realized_vol.fillna(full_sample_vol).clip(lower=0.01)
        scalar = (config.TARGET_VOL / realized_vol).clip(self._vol_floor_scalar, self._vol_cap_scalar)
        return (positions * scalar).clip(-1.0, 1.0)

