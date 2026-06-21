"""End-to-end orchestration of the trading strategy.

Loads and cleans OHLCV data, engineers features, estimates H(t) (classically
via Whittle MLE, or with the trained FNO), converts that into regime-switched
signals, sizes positions with HRP, and runs the walk-forward backtest.

``Pipeline(mode="classical" | "fno")`` exposes ``run()``, ``summary()`` and
``run_and_summarise()``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from wsq_trading import config
from wsq_trading.backtest import BacktestResult, WalkForwardEngine
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
    ) -> pd.Series:
        """Compute HRP weights from backtest strategy returns.

        Parameters
        ----------
        results : dict[str, BacktestResult]

        Returns
        -------
        pd.Series
            Ticker -> HRP weight (sum to 1).
        """
        return self.allocator.allocate_backtest(results)

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
    ) -> tuple[dict[str, BacktestResult], pd.Series, pd.DataFrame]:
        """One-shot convenience: run -> allocate -> summarise."""
        results = self.run(tickers=tickers, start=start, end=end)
        if not results:
            empty = pd.Series(dtype=float)
            return results, empty, pd.DataFrame()

        weights = self.allocate(results)
        summary = self.summary(results, weights)
        return results, weights, summary

    # Internal helpers

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
