# Prompt for Opus

---

You are implementing a set of specific, scoped code changes to the `wsq_trading` Python package — a quantitative trading system built around Fourier Neural Operator Hurst estimation and regime-switching signals.

**Your reference document is `OPUS_EXECUTION_PROMPT.md` in the repo root. Read it in full before writing a single line of code.**

---

## Before you start

Read these files in this order. Do not edit anything until you have read all of them:

1. `OPUS_EXECUTION_PROMPT.md` — the full spec (read this first and completely)
2. `wsq_trading/config.py`
3. `wsq_trading/constants.py`
4. `wsq_trading/backtest.py`
5. `wsq_trading/portfolio.py`
6. `wsq_trading/pipeline.py`
7. `wsq_trading/tests/` — list all test files, then read each one

---

## Implementation order

Work strictly in this sequence. Complete and verify each step before moving to the next:

**Step 1 — `wsq_trading/backtest.py`**
Add at the bottom of the file (after the existing `WalkForwardEngine` class):
- `PortfolioMetrics` dataclass
- `compute_portfolio_metrics()` function
- `regime_attribution()` function
- `cost_sensitivity()` function

**Step 2 — `wsq_trading/portfolio.py`**
Modify `HRPAllocator.allocate_backtest()`:
- Add `train_end_date: pd.Timestamp | str | None = None` parameter
- When set, restrict the quality-gate Sharpe calculation to returns up to that date only
- HRP weight computation still uses all returns — only the *filter decision* is time-limited

**Step 3 — `wsq_trading/pipeline.py`**
- Add `OOSReport` dataclass (at module level, before the `Pipeline` class)
- Add `run_oos()` method to `Pipeline`
- Add `print_oos_report()` method to `Pipeline`
- Update `Pipeline.allocate()` to accept and forward `train_end_date`
- Update `Pipeline.run_and_summarise()` to accept `oos: bool = False`

**Step 4 — Tests**
Add tests for the new functionality per the spec. Extend existing test files rather than creating new ones where a suitable file already exists.

**Step 5 — Verify**
Run `python -m pytest wsq_trading/tests/ -q` and confirm all tests pass, including the original ones.

---

## Hard rules — do not violate these

- **Read every file before editing it.** Never edit a file you haven't read in this session.
- **No look-ahead in OOS metrics.** Signal generation may use full history (Hurst needs warm-up), but `BacktestResult` metrics and all `OOSReport` fields must be computed on test-period rows only.
- **No `print()`.** All output goes through `wsq_trading.utils.get_logger(__name__)`.
- **No new dependencies.** Use `np.linalg.lstsq` for alpha/beta OLS — do not import `statsmodels` or `sklearn`.
- **`fno.py` is the only module that imports torch.** Do not add torch imports elsewhere.
- **Backward compatibility.** `Pipeline.run()`, `Pipeline.allocate()`, and `Pipeline.run_and_summarise()` must still work with their existing call signatures. All new parameters are optional with sensible defaults.
- **Line length 100.** Black-formatted.
- **Full type hints and NumPy-style docstrings on all public functions.**

---

## Definition of done

You are finished when:
1. All five changes from the spec are implemented.
2. `python -m pytest wsq_trading/tests/ -q` exits with zero failures.
3. The following smoke test runs without raising an exception (network access required):

```python
from wsq_trading.pipeline import Pipeline
p = Pipeline(mode='classical')
report = p.run_oos()
p.print_oos_report(report)
```

The log output must include an OOS test window, portfolio Sharpe, regime attribution table, and cost sensitivity table.
