"""End-to-end orchestration of the trading strategy.

Loads and cleans OHLCV data, engineers features, estimates H(t) (classically
via Whittle MLE, or with the trained FNO), converts that into regime-switched
signals, sizes positions with HRP, and runs the walk-forward backtest.

``Pipeline(mode="classical" | "fno")`` exposes ``run()``, ``summary()`` and
``run_and_summarise()``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from wsq_trading import config
from wsq_trading.backtest import (
    BacktestResult,
    WalkForwardEngine,
    compute_all,
    compute_portfolio_metrics,
    cost_sensitivity,
    regime_attribution,
)
from wsq_trading.data import DataCleaner, DataLoader, DataValidator
from wsq_trading.features import FeatureEngineer
from wsq_trading.portfolio import HRPAllocator
from wsq_trading.signals import HurstRegimeSignalGenerator
from wsq_trading.utils import get_logger, timer

log = get_logger(__name__)

# Optional FNO imports (only needed in 'fno' mode)
try:
    from wsq_trading.fno import FNO1d, FNOTrainer
    _FNO_AVAILABLE = True
except ImportError:
    _FNO_AVAILABLE = False


@dataclass
class OOSReport:
    """True out-of-sample evaluation report.

    Attributes
    ----------
    train_end : pd.Timestamp
        Last date of the train+val window.
    test_start : pd.Timestamp
        First date of the pure test window.
    test_end : pd.Timestamp
        Last date of the test window.
    per_instrument : pd.DataFrame
        Index = ticker, columns = OOS (test-period) metrics.
    portfolio : dict[str, float]
        Portfolio-level OOS metrics (from :class:`PortfolioMetrics`).
    hrp_weights : pd.Series
        HRP weights computed from the train window only.
    regime_attribution : pd.DataFrame
        Test-period P&L decomposed by Hurst regime.
    cost_sensitivity : pd.DataFrame
        Test-period Sharpe / return / drawdown at a grid of cost levels.
    """

    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    per_instrument: pd.DataFrame
    portfolio: dict[str, float]
    hrp_weights: pd.Series
    regime_attribution: pd.DataFrame
    cost_sensitivity: pd.DataFrame


class Pipeline:
    """End-to-end pipeline from raw data to backtest results and HRP weights.

    Parameters
    ----------
    mode : 'classical' | 'fno'
        'classical'  -- Use rolling Whittle MLE to estimate H(t).
        'fno'        -- Use a trained FNO to estimate H(t).
                        Requires PyTorch and a trained checkpoint.
    cost_bps : float
        Round-trip transaction cost in basis points.
        Default: ``config.TRANSACTION_COST_BPS``.
    signal_kwargs : dict
        Extra keyword arguments forwarded to ``HurstRegimeSignalGenerator``.
    fno_checkpoint : Path, optional
        Path to the FNO checkpoint file (``fno_best.pt``).
        Required only when ``mode='fno'``.
    allocator : HRPAllocator, optional
        Pre-configured allocator.  A default one is created if omitted.
    """

    def __init__(
        self,
        mode: Literal["classical", "fno"] = "classical",
        cost_bps: float | None = None,
        signal_kwargs: dict | None = None,
        fno_checkpoint: Path | None = None,
        allocator: HRPAllocator | None = None,
        long_bias_tickers: list[str] | None = None,
        vol_target_enabled: bool | None = None,
        h_conviction_sizing: bool | None = None,
    ) -> None:
        if mode not in ("classical", "fno"):
            raise ValueError(f"mode must be 'classical' or 'fno'; got {mode!r}")
        if mode == "fno" and not _FNO_AVAILABLE:
            raise ImportError(
                "PyTorch is required for mode='fno'. "
                "Install it with: pip install torch"
            )

        self.mode        = mode
        self.cost_bps    = cost_bps if cost_bps is not None else config.TRANSACTION_COST_BPS
        self.signal_kwargs = signal_kwargs or {}
        self.fno_checkpoint = fno_checkpoint
        self.allocator   = allocator or HRPAllocator()

        # Build the engine with classical signal generator by default;
        # replaced in _make_signal_generator for FNO mode.
        self._sig_gen = self._make_signal_generator()
        self._engine  = WalkForwardEngine(
            signal_generator=self._sig_gen,
            cost_bps=self.cost_bps,
            long_bias_tickers=long_bias_tickers,
            vol_target_enabled=vol_target_enabled,
            h_conviction_sizing=h_conviction_sizing,
        )

        log.info("Pipeline: mode=%s  cost_bps=%.1f", mode, self.cost_bps)


    def run(
        self,
        tickers: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> dict[str, BacktestResult]:
        """Download, clean, and backtest all tickers.

        Parameters
        ----------
        tickers : list[str], optional
            Instruments to trade.  Default: ``config.TICKERS``.
        start, end : str, optional
            ISO date strings for the historical range.
            Defaults: ``config.DEFAULT_START`` / today.
        """
        tickers = tickers or config.TICKERS
        start   = start   or config.DEFAULT_START
        end     = end     or config.DEFAULT_END

        log.info("Pipeline.run: %d tickers  %s -> %s", len(tickers), start, end)

        with timer("Pipeline: data loading + cleaning"):
            data = self._load_and_clean(tickers, start, end)

        if not data:
            log.warning("No usable data after cleaning; returning empty results.")
            return {}

        with timer("Pipeline: backtest"):
            results = self._engine.run(data)

        log.info(
            "Pipeline.run: %d/%d tickers backtested successfully.",
            len(results), len(tickers),
        )
        return results

    def allocate(
        self,
        results: dict[str, BacktestResult],
        train_end_date: pd.Timestamp | str | None = None,
    ) -> pd.Series:
        """Compute HRP weights from backtest strategy returns.

        Parameters
        ----------
        results : dict[str, BacktestResult]
            Per-instrument backtest results.
        train_end_date : pd.Timestamp or str, optional
            Forwarded to :meth:`HRPAllocator.allocate_backtest` so the quality
            gate only sees returns up to this date (avoids look-ahead).

        Returns
        -------
        pd.Series
            Ticker -> HRP weight (sum to 1).
        """
        return self.allocator.allocate_backtest(results, train_end_date=train_end_date)

    def summary(
        self,
        results: dict[str, BacktestResult],
        weights: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Produce a metrics summary DataFrame.

        Parameters
        ----------
        results : dict[str, BacktestResult]
        weights : pd.Series, optional
            HRP weights to append as an extra column.

        Returns
        -------
        pd.DataFrame
            Index = ticker, columns = metrics (+ 'hrp_weight' if weights given).
        """
        df = self._engine.summary(results)
        if weights is not None:
            df["hrp_weight"] = weights.reindex(df.index)
        return df

    def run_and_summarise(
        self,
        tickers: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        oos: bool = False,
    ) -> tuple[dict[str, BacktestResult], pd.Series, pd.DataFrame] | OOSReport:
        """One-shot convenience: run -> allocate -> summarise.

        Parameters
        ----------
        tickers : list[str], optional
            Instruments to trade.  Default: ``config.TICKERS``.
        start, end : str, optional
            ISO date strings for the historical range.
        oos : bool
            If ``True``, route to :meth:`run_oos` and return an
            :class:`OOSReport` instead of the ``(results, weights, summary)``
            tuple.  Default ``False`` (backward compatible).

        Returns
        -------
        tuple[dict[str, BacktestResult], pd.Series, pd.DataFrame] or OOSReport
            The three-tuple when ``oos=False``; an :class:`OOSReport` otherwise.
        """
        if oos:
            return self.run_oos(tickers=tickers, start=start, end=end)

        results = self.run(tickers=tickers, start=start, end=end)
        if not results:
            empty = pd.Series(dtype=float)
            return results, empty, pd.DataFrame()

        # Compute the train cutoff from config.TRAIN_SPLIT and the data date
        # range so the quality gate does not use the (held-out) tail of history.
        train_end = self._date_at_fraction(results, config.TRAIN_SPLIT)
        weights = self.allocate(results, train_end_date=train_end)
        summary = self.summary(results, weights)
        return results, weights, summary

    def run_oos(
        self,
        tickers: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> OOSReport:
        """Run a clean out-of-sample evaluation.

        Signal generation uses the full return series (Hurst estimation needs
        warm-up history), but every reported metric is computed on the pure test
        window only.  The HRP weights and quality gate are derived from the
        train+val window so the test period never influences allocation.

        Parameters
        ----------
        tickers : list[str], optional
            Instruments to trade.  Default: ``config.TICKERS``.
        start, end : str, optional
            ISO date strings for the historical range.

        Returns
        -------
        OOSReport
            Out-of-sample per-instrument, portfolio, regime, and cost results.

        Raises
        ------
        ValueError
            If no usable data is available or no ticker backtests successfully.
        """
        tickers = tickers or config.TICKERS
        start = start or config.DEFAULT_START
        end = end or config.DEFAULT_END

        log.info("Pipeline.run_oos: %d tickers  %s -> %s", len(tickers), start, end)

        with timer("Pipeline.run_oos: data loading + cleaning"):
            data = self._load_and_clean(tickers, start, end)
        if not data:
            raise ValueError("No usable data after cleaning; cannot run OOS evaluation.")

        # Signals run on FULL history; metrics are sliced to the test window below.
        with timer("Pipeline.run_oos: backtest"):
            results = self._engine.run(data)
        if not results:
            raise ValueError("No tickers backtested successfully; cannot run OOS.")

        # Date-based split using the union of all instrument dates.
        all_dates = sorted(set().union(*[r.strategy_returns.index for r in results.values()]))
        n = len(all_dates)
        train_val_n = int(n * (config.TRAIN_SPLIT + config.VAL_SPLIT))
        train_val_n = min(max(train_val_n, 1), n - 1)
        train_end = pd.Timestamp(all_dates[train_val_n - 1])
        test_start = pd.Timestamp(all_dates[train_val_n])
        test_end = pd.Timestamp(all_dates[-1])
        log.info(
            "OOS split: train_end=%s  test=[%s .. %s]",
            train_end.date(), test_start.date(), test_end.date(),
        )

        # Weights from train window only: gate AND weights see no test data.
        train_results = {
            t: self._slice_result(r, None, train_end) for t, r in results.items()
        }
        hrp_weights = self.allocate(train_results, train_end_date=train_end)

        # OOS metrics: slice every series to the pure test window.
        oos_results = {
            t: self._slice_result(r, test_start, test_end) for t, r in results.items()
        }

        per_rows = {
            t: compute_all(r.strategy_returns, positions=r.positions)
            for t, r in oos_results.items()
        }
        per_instrument = pd.DataFrame(per_rows).T
        per_instrument["hrp_weight"] = hrp_weights.reindex(per_instrument.index)

        # Equity benchmark for alpha/beta/IR: ES=F gross returns where available.
        benchmark = (
            oos_results["ES=F"].gross_returns if "ES=F" in oos_results else None
        )

        port = compute_portfolio_metrics(
            oos_results, hrp_weights, benchmark_returns=benchmark, cost_bps=self.cost_bps
        )
        attribution = regime_attribution(oos_results, hrp_weights)
        costs = cost_sensitivity(oos_results, hrp_weights)

        return OOSReport(
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            per_instrument=per_instrument,
            portfolio=asdict(port),
            hrp_weights=hrp_weights,
            regime_attribution=attribution,
            cost_sensitivity=costs,
        )

    def print_oos_report(self, report: OOSReport) -> None:
        """Log a formatted summary of an :class:`OOSReport`.

        Parameters
        ----------
        report : OOSReport
            The report produced by :meth:`run_oos`.
        """
        p = report.portfolio
        lines = [
            "",
            "=== OUT-OF-SAMPLE REPORT ===",
            f"Train+Val window: ... -> {report.train_end.date()}",
            f"Test window:      {report.test_start.date()} -> {report.test_end.date()}",
            "",
            "--- Portfolio (OOS) ---",
            f"Sharpe:           {p['sharpe']:.3f}",
            f"Annual Return:    {p['annualized_return']:.1%}",
            f"Annual Vol:       {p['annualized_vol']:.1%}",
            f"Max Drawdown:     {p['max_drawdown']:.1%}",
            f"Calmar:           {p['calmar']:.3f}",
            f"Turnover (ann):   {p['turnover']:.2f}x",
            f"Alpha vs ES=F:    {p['alpha']:.1%}",
            f"Beta vs ES=F:     {p['beta']:.3f}",
            f"Info Ratio:       {p['information_ratio']:.3f}",
            "",
            "--- Per-Instrument (OOS) ---",
            report.per_instrument.to_string(),
            "",
            "--- HRP Weights (from train window) ---",
            report.hrp_weights.to_string(),
            "",
            "--- Regime Attribution ---",
            report.regime_attribution.to_string(),
            "",
            "--- Cost Sensitivity ---",
            report.cost_sensitivity.to_string(),
        ]
        log.info("\n".join(lines))

    # Internal helpers

    @staticmethod
    def _date_at_fraction(
        results: dict[str, BacktestResult],
        fraction: float,
    ) -> pd.Timestamp:
        """Return the date at ``fraction`` of the union of all result dates.

        Parameters
        ----------
        results : dict[str, BacktestResult]
            Per-instrument backtest results.
        fraction : float
            Fraction in (0, 1] of the history to include up to the cutoff.

        Returns
        -------
        pd.Timestamp
            The cutoff date (inclusive).
        """
        all_dates = sorted(set().union(*[r.strategy_returns.index for r in results.values()]))
        n = len(all_dates)
        idx = min(max(int(n * fraction) - 1, 0), n - 1)
        return pd.Timestamp(all_dates[idx])

    @staticmethod
    def _slice_result(
        res: BacktestResult,
        start: pd.Timestamp | None,
        end: pd.Timestamp | None,
    ) -> BacktestResult:
        """Return a copy of ``res`` with all series sliced to ``[start, end]``.

        Metrics are recomputed from the sliced series by
        ``BacktestResult.__post_init__``, giving window-local statistics.

        Parameters
        ----------
        res : BacktestResult
            Source result spanning the full history.
        start, end : pd.Timestamp or None
            Inclusive date bounds; ``None`` means open-ended on that side.

        Returns
        -------
        BacktestResult
            Result restricted to the requested window.
        """
        sl = slice(start, end)
        return BacktestResult(
            ticker=res.ticker,
            strategy_returns=res.strategy_returns.loc[sl],
            positions=res.positions.loc[sl],
            H_estimates=res.H_estimates.loc[sl],
            regimes=res.regimes.loc[sl],
            gross_returns=res.gross_returns.loc[sl],
        )

    def _load_and_clean(
        self,
        tickers: list[str],
        start: str,
        end: str | None,
    ) -> dict[str, pd.DataFrame]:
        """Load, validate, clean, and engineer features for each ticker."""
        loader    = DataLoader()
        validator = DataValidator()
        cleaner   = DataCleaner()
        engineer  = FeatureEngineer()

        data: dict[str, pd.DataFrame] = {}

        for ticker in tickers:
            try:
                raw = loader.fetch(ticker, start=start, end=end)
                issues = validator.validate(raw, ticker=ticker)
                if issues:
                    log.warning("Validation issues for %s: %s", ticker, issues)

                cleaned  = cleaner.run_pipeline(raw)
                featured = engineer.run_pipeline(cleaned)
                data[ticker] = featured

            except Exception as exc:
                log.warning("Skipping %s: %s", ticker, exc)

        log.info("Loaded %d/%d tickers.", len(data), len(tickers))
        return data

    def _make_signal_generator(self) -> HurstRegimeSignalGenerator:
        """Build the appropriate signal generator for the chosen mode."""
        if self.mode == "classical":
            return HurstRegimeSignalGenerator(**self.signal_kwargs)

        # FNO mode: wrap FNO predictions inside a custom signal generator
        if not _FNO_AVAILABLE:
            raise ImportError("PyTorch required for mode='fno'.")

        # Build the FNO model from config, then wrap in a trainer
        model = FNO1d(
            in_channels=2,            # log_return + variance proxy
            width=config.FNO_WIDTH,
            n_modes=config.FNO_MODES,
            n_layers=config.FNO_LAYERS,
        )
        trainer = FNOTrainer(
            model=model,
            window_size=config.WINDOW_SIZE,
            checkpoint_dir=config.CHECKPOINTS_DIR,
        )
        if self.fno_checkpoint:
            trainer.load_checkpoint(self.fno_checkpoint)
        else:
            log.warning(
                "No FNO checkpoint provided; using randomly initialised weights. "
                "Train the model first with FNOTrainer.train()."
            )

        return _FNOSignalGenerator(trainer, **self.signal_kwargs)


# FNO-backed signal generator (wraps FNOTrainer.predict_rolling)

class _FNOSignalGenerator(HurstRegimeSignalGenerator):
    """HurstRegimeSignalGenerator subclass that replaces the Whittle MLE
    rolling Hurst estimator with FNO predictions.

    The rest of the regime classification and signal logic is inherited
    unchanged from the parent class.
    """

    def __init__(self, trainer: "FNOTrainer", **kwargs) -> None:  # type: ignore[name-defined]
        # Initialise parent (Whittle estimator created but unused - overridden below)
        super().__init__(**{k: v for k, v in kwargs.items()
                            if k in (
                                "hurst_window", "z_window", "z_threshold",
                                "mr_z_upper_cap", "fast_ma", "slow_ma",
                                "momentum_confirm_window", "regime_persistence",
                                "rough_threshold", "trend_threshold",
                                "neutral_mr_enabled", "neutral_mr_return_window",
                                "neutral_mr_entry_zscore", "neutral_mr_position_size",
                            )})
        self._trainer = trainer
        # Override: use the FNO window size and update the log label
        self._hurst_window = trainer.window_size
        self._estimator_name = "fno"

    def compute_rolling_hurst(self, returns: pd.Series) -> pd.Series:
        """Replace Whittle MLE with FNO rolling predictions."""
        return self._trainer.predict_rolling(returns)
