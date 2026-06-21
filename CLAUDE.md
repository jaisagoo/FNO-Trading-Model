# CLAUDE.md — context for AI assistants

## What this is

A quantitative trading system that:

1. Simulates rough-Heston SPDE paths to generate labelled training data.
2. Trains a Fourier Neural Operator (FNO) to predict a time-varying Hurst exponent `H(t)`.
3. Switches between mean-reversion, trend-following and neutral strategies based on `H(t)`.
4. Allocates across instruments with Hierarchical Risk Parity (HRP).
5. Validates everything with a walk-forward backtest.

The importable package is `wsq_trading`; all configuration lives in `wsq_trading/config.py`.

## Layout

```
wsq_trading/
    config.py       single source of truth for settings
    constants.py    enums, math constants
    utils.py        logging, seeding, timing
    data.py         OHLCV load / validate / clean
    spde.py         rough-Heston SPDE simulator
    features.py     feature engineering
    hurst.py        classical Hurst estimators (R/S, DFA, Whittle MLE)
    fno.py          FNO architecture + trainer
    signals.py      regime-switching signal generation
    portfolio.py    HRP allocator
    backtest.py     walk-forward engine + metrics
    pipeline.py     orchestration
    tests/          test_<module>.py
```

## Conventions

- Python >= 3.10, type hints throughout.
- Line length 100 (Black + Ruff, configured in `pyproject.toml`).
- Public functions get a NumPy-style docstring.
- Import directly from the relevant module, e.g. `from wsq_trading.hurst import DFA`.
- Use `wsq_trading.utils.get_logger(__name__)` for logging, never `print()`.
- `fno.py` is the only module that imports torch; keep it lazy elsewhere.

## Design notes

- Config-driven: change `config.py` to swap data sources, hyperparameters or thresholds.
- Synthetic data first: real OHLCV is supplemented with rough-Heston SPDE paths so the
  FNO has enough labelled signal.
- Regime thresholds: `H < 0.35` mean-reversion, `0.35–0.65` neutral, `H > 0.65` momentum.
- HRP over mean-variance optimisation — more robust to estimation error out of sample.
- The backtest engine enforces strict temporal splits (no look-ahead).

## Don't

- Hardcode paths — use the constants in `config.py`.
- Use `print()` — use the logger.
- Commit large data files or `.pt` checkpoints (gitignored).
