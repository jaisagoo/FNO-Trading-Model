# Execution Prompt for Claude Opus — Rough-Heston Trading Model

You are working inside the `wsq_trading` Python package, a quantitative trading system that:
1. Simulates rough-Heston SPDE paths to generate labelled training data.
2. Trains a Fourier Neural Operator (FNO) to predict a time-varying Hurst exponent `H(t)`.
3. Switches between mean-reversion, trend-following, and neutral strategies based on `H(t)`.
4. Allocates across instruments with Hierarchical Risk Parity (HRP).
5. Validates everything with a walk-forward backtest.

**Importable package:** `wsq_trading`  
**Single config source:** `wsq_trading/config.py`  
**Python >= 3.10, type hints throughout, line length 100 (Black + Ruff)**  
**Never use `print()` — use `wsq_trading.utils.get_logger(__name__)`**  

---

## Current Layout

```
wsq_trading/
    config.py       — all settings (thresholds, paths, costs, tickers, splits)
    constants.py    — Regime enum, TRADING_DAYS_PER_YEAR
    utils.py        — get_logger, timer, seeding
    data.py         — OHLCV load / validate / clean
    spde.py         — rough-Heston SPDE simulator
    features.py     — feature engineering
    hurst.py        — classical Hurst estimators (R/S, DFA, Whittle MLE)
    fno.py          — FNO architecture + trainer (imports torch; keep lazy elsewhere)
    signals.py      — HurstRegimeSignalGenerator (regime-switching signal)
    portfolio.py    — HRPAllocator (Hierarchical Risk Parity)
    backtest.py     — WalkForwardEngine, BacktestResult, metric functions
    pipeline.py     — Pipeline orchestration class
    tests/          — test_<module>.py
```

---

## Key Types You Will Touch

### `BacktestResult` (backtest.py)
```python
@dataclass
class BacktestResult:
    ticker: str
    strategy_returns: pd.Series   # daily net-of-cost returns
    positions: pd.Series          # daily position sizes in [-1, 1]
    H_estimates: pd.Series        # rolling Hurst estimates
    regimes: pd.Series            # Regime enum per bar
    gross_returns: pd.Series      # raw instrument returns (before signal)
    metrics: dict[str, float]     # sharpe, sortino, max_drawdown, ...
```

### `HRPAllocator.allocate_backtest` (portfolio.py, line 122)
Currently receives the **full** `dict[str, BacktestResult]` and computes Sharpe over the
**entire** `strategy_returns` series to decide which instruments pass the quality gate.
This is the look-ahead bias bug.

### `WalkForwardEngine` (backtest.py, line 376)
`run_single(returns, ticker)` runs the full return series through signal generation and
returns a `BacktestResult`. There is no internal train/test split — the split is the
caller's responsibility.

### `Pipeline` (pipeline.py)
`run()` → `allocate()` → `summary()` — no OOS boundary is enforced anywhere in the call
chain. `config.TRAIN_SPLIT=0.70`, `config.VAL_SPLIT=0.15` exist but are unused.

### `config.py` — relevant constants
```python
TRANSACTION_COST_BPS: float = 2.0
QUALITY_SHARPE_THRESHOLD: float = 0.30
TRAIN_SPLIT: float = 0.70
VAL_SPLIT: float = 0.15
TICKERS: list[str] = ["ES=F","NQ=F","CL=F","6E=F","NG=F","HG=F","ZN=F"]
```

---

## Five Changes Required

---

### Change 1 — Fix HRP quality gate look-ahead bias

**File:** `wsq_trading/portfolio.py`  
**Function:** `HRPAllocator.allocate_backtest` (line 122)

**Problem:** The quality gate currently computes Sharpe over the full `strategy_returns`
series for every instrument. When `allocate_backtest` is called after a backtest whose
`strategy_returns` span the entire history (including the test period), the gate uses
future information to decide which instruments receive allocation. This means test-set
returns influence the instrument universe, which is look-ahead bias.

**Fix:** Accept an optional `train_end_date: pd.Timestamp | str | None` parameter.
When provided, only compute the quality-gate Sharpe from returns **up to and including**
`train_end_date`. The HRP allocation itself (after filtering) still uses all available
returns for weight computation — only the *filter decision* must be time-limited.

**Exact signature change:**
```python
def allocate_backtest(
    self,
    results: "dict[str, BacktestResult]",
    train_end_date: pd.Timestamp | str | None = None,  # NEW
) -> pd.Series:
```

**Logic change inside the quality gate block (currently around line 154):**
```python
# If train_end_date provided, restrict Sharpe calculation to training window
if train_end_date is not None:
    cutoff = pd.Timestamp(train_end_date)
    sharpe_returns = returns_df.loc[:cutoff]
else:
    sharpe_returns = returns_df

# Use sharpe_returns (not returns_df) for the Sharpe calculation
for col in sharpe_returns.columns:
    r = sharpe_returns[col].dropna()
    ...
```

After this change, `Pipeline.allocate()` must pass `train_end_date` through. Update
`Pipeline.allocate()` to accept and forward it. Also update `Pipeline.run_and_summarise()`
to compute `train_end_date` from `config.TRAIN_SPLIT` and the data date range.

---

### Change 2 — True out-of-sample final report

**File:** `wsq_trading/pipeline.py`

**Problem:** `Pipeline.run()` and `Pipeline.run_and_summarise()` have no OOS boundary.
All thresholds, ticker choices, and filters were calibrated/observed on the same data
that is used to report performance. There is no clean final report showing true OOS results.

**Fix:** Add a `run_oos()` method to `Pipeline` that:
1. Splits each ticker's return series into train+val (`TRAIN_SPLIT + VAL_SPLIT`) and test
   (`1 - TRAIN_SPLIT - VAL_SPLIT`) using the date-based split from `config`.
2. The `WalkForwardEngine` runs signals on the **full** series (Hurst needs warm-up history),
   but metrics are computed **only on the test period** rows of `strategy_returns`.
3. The quality gate Sharpe uses only train+val rows (via `train_end_date`).
4. Returns a `OOSReport` dataclass (defined below).

**Add to `pipeline.py`:**
```python
@dataclass
class OOSReport:
    """True out-of-sample evaluation report."""
    train_end: pd.Timestamp       # last date of train+val window
    test_start: pd.Timestamp      # first date of pure test window
    test_end: pd.Timestamp
    per_instrument: pd.DataFrame  # index=ticker, cols=OOS metrics
    portfolio: dict[str, float]   # portfolio-level OOS metrics
    hrp_weights: pd.Series        # weights computed from train window only
    regime_attribution: pd.DataFrame  # P&L by regime (see Change 4)
    cost_sensitivity: pd.DataFrame    # Sharpe at various cost_bps (see Change 5)
```

```python
def run_oos(
    self,
    tickers: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> OOSReport:
    """
    Run a clean OOS evaluation.

    Steps:
      1. Load full history for each ticker.
      2. Compute train_end = start + TRAIN_SPLIT fraction of total days.
      3. Run WalkForwardEngine on FULL history (signal warm-up needs history).
      4. Slice strategy_returns to test period only for metric computation.
      5. Quality gate uses train_end to avoid look-ahead.
      6. Compute portfolio-level OOS metrics from HRP-weighted returns.
      7. Attach regime attribution and cost sensitivity.
    """
```

The date split logic (step 2):
```python
# Get date range from any loaded ticker
all_dates = sorted(data[next(iter(data))].index)
n = len(all_dates)
train_val_n = int(n * (config.TRAIN_SPLIT + config.VAL_SPLIT))
train_end = all_dates[train_val_n - 1]
test_start = all_dates[train_val_n]
```

---

### Change 3 — Portfolio-level performance report

**File:** `wsq_trading/backtest.py` (add a new section at the bottom)

**Problem:** The existing `BacktestResult.metrics` dict has per-instrument stats.
There is no portfolio-level view: no combined Sharpe, no aggregate alpha/beta vs
a benchmark, no transaction cost total, no portfolio drawdown.

**Add a `PortfolioMetrics` dataclass and `compute_portfolio_metrics()` function:**

```python
@dataclass
class PortfolioMetrics:
    """Aggregate statistics for the HRP-weighted portfolio."""
    sharpe: float
    sortino: float
    annualized_return: float
    annualized_vol: float
    max_drawdown: float
    calmar: float
    win_rate: float
    turnover: float          # weighted average instrument turnover
    total_cost_bps: float    # realised round-trip cost over the period
    alpha: float             # CAPM alpha vs benchmark (annualised)
    beta: float              # CAPM beta vs benchmark
    information_ratio: float # (port_return - bench_return) / tracking_error
    n_obs: int


def compute_portfolio_metrics(
    results: dict[str, BacktestResult],
    weights: pd.Series,
    benchmark_returns: pd.Series | None = None,
    cost_bps: float | None = None,
) -> PortfolioMetrics:
    """
    Combine per-instrument results into a portfolio-level performance summary.

    Parameters
    ----------
    results : dict[str, BacktestResult]
        Per-instrument backtest results.
    weights : pd.Series
        HRP weights (index = ticker).
    benchmark_returns : pd.Series, optional
        Daily benchmark returns for alpha/beta/IR calculation.
        If None, alpha=NaN, beta=NaN, IR=NaN.
    cost_bps : float, optional
        Transaction cost assumption for the total cost calculation.

    Returns
    -------
    PortfolioMetrics
    """
    # Align strategy returns on a common date index
    returns_df = pd.DataFrame(
        {t: r.strategy_returns for t, r in results.items()}
    ).dropna(how="all")

    # Weighted portfolio return
    w = weights.reindex(returns_df.columns).fillna(0.0)
    port_returns = returns_df.mul(w, axis=1).sum(axis=1)

    # Weighted turnover
    turnover_total = sum(
        weights.get(t, 0.0) * r.metrics.get("turnover", 0.0)
        for t, r in results.items()
    )

    # Alpha / Beta via OLS against benchmark
    alpha, beta, ir = float("nan"), float("nan"), float("nan")
    if benchmark_returns is not None:
        bench = benchmark_returns.reindex(port_returns.index).dropna()
        port_aligned = port_returns.reindex(bench.index).dropna()
        if len(port_aligned) > 30:
            import numpy as np
            X = np.column_stack([np.ones(len(bench)), bench.to_numpy()])
            y = port_aligned.to_numpy()
            coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
            alpha_daily, beta = float(coeffs[0]), float(coeffs[1])
            alpha = alpha_daily * TRADING_DAYS_PER_YEAR
            active = port_aligned - bench
            te = float(active.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
            ir = float(active.mean() * TRADING_DAYS_PER_YEAR / te) if te > 0 else float("nan")

    metrics = compute_all(port_returns)
    return PortfolioMetrics(
        sharpe=metrics["sharpe"],
        sortino=metrics["sortino"],
        annualized_return=metrics["annualized_return"],
        annualized_vol=metrics["annualized_vol"],
        max_drawdown=metrics["max_drawdown"],
        calmar=metrics["calmar"],
        win_rate=metrics["win_rate"],
        turnover=turnover_total,
        total_cost_bps=cost_bps if cost_bps is not None else float("nan"),
        alpha=alpha,
        beta=beta,
        information_ratio=ir,
        n_obs=len(port_returns),
    )
```

Then integrate this into `Pipeline.run_oos()` so the `OOSReport.portfolio` field is populated
by calling `compute_portfolio_metrics()` with OOS-period returns and benchmark
(use `ES=F` gross returns as the equity benchmark where available).

---

### Change 4 — Regime attribution

**File:** `wsq_trading/backtest.py` (new function) and consumed in `pipeline.py`

**Problem:** There is no breakdown of how much P&L comes from each regime (MEAN_REVERT,
NEUTRAL, MOMENTUM). Without this the Hurst-regime thesis is unverifiable.

**Add a `regime_attribution()` function:**

```python
def regime_attribution(
    results: dict[str, BacktestResult],
    weights: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Decompose strategy returns by Hurst regime.

    For each instrument (and optionally the HRP-weighted portfolio), compute:
      - Fraction of bars spent in each regime
      - Annualised return while in each regime
      - Annualised Sharpe while in each regime
      - Number of observations in each regime

    Parameters
    ----------
    results : dict[str, BacktestResult]
    weights : pd.Series, optional
        If provided, also compute portfolio-level attribution.

    Returns
    -------
    pd.DataFrame
        MultiIndex rows: (ticker, regime).
        Columns: frac_bars, ann_return, sharpe, n_obs.
    """
    from wsq_trading.constants import Regime
    rows = []
    for ticker, res in results.items():
        for regime in Regime:
            mask = res.regimes == regime
            r = res.strategy_returns[mask].dropna()
            n = len(r)
            rows.append({
                "ticker": ticker,
                "regime": regime.name,
                "frac_bars": mask.sum() / max(len(res.regimes), 1),
                "ann_return": annualized_return(r) if n > 5 else float("nan"),
                "sharpe": sharpe_ratio(r) if n > 5 else float("nan"),
                "n_obs": n,
            })

    # Portfolio-level attribution
    if weights is not None:
        returns_df = pd.DataFrame({t: r.strategy_returns for t, r in results.items()})
        regimes_df = pd.DataFrame({t: r.regimes for t, r in results.items()})
        w = weights.reindex(returns_df.columns).fillna(0.0)
        port_returns = returns_df.mul(w, axis=1).sum(axis=1)

        # Regime of portfolio = majority vote across instruments on each bar
        # (weighted by position magnitude)
        for regime in Regime:
            # Days where majority of weighted exposure is in this regime
            regime_mask = regimes_df.apply(lambda col: col == regime)
            weighted_mask = regime_mask.mul(w, axis=1).sum(axis=1)
            total_weight = regime_mask.notna().mul(w, axis=1).sum(axis=1).replace(0, float("nan"))
            dominant = (weighted_mask / total_weight) > 0.5
            r = port_returns[dominant].dropna()
            n = len(r)
            rows.append({
                "ticker": "PORTFOLIO",
                "regime": regime.name,
                "frac_bars": dominant.sum() / max(len(port_returns), 1),
                "ann_return": annualized_return(r) if n > 5 else float("nan"),
                "sharpe": sharpe_ratio(r) if n > 5 else float("nan"),
                "n_obs": n,
            })

    df = pd.DataFrame(rows).set_index(["ticker", "regime"])
    return df
```

---

### Change 5 — Transaction cost sensitivity

**File:** `wsq_trading/backtest.py` (new function) and consumed in `pipeline.py`

**Problem:** The default of 2 bps may be optimistic for some futures. There is no
sensitivity analysis showing how Sharpe and annual return degrade at higher costs.
Reviewers cannot assess robustness.

**Add a `cost_sensitivity()` function:**

```python
def cost_sensitivity(
    results: dict[str, BacktestResult],
    weights: pd.Series,
    cost_bps_grid: list[float] | None = None,
) -> pd.DataFrame:
    """
    Re-price strategy returns at a range of round-trip transaction costs.

    The function does NOT re-run the signal generator — it takes the existing
    gross strategy returns and position series and subtracts cost at each
    cost level.

    Parameters
    ----------
    results : dict[str, BacktestResult]
    weights : pd.Series
        HRP weights.
    cost_bps_grid : list[float]
        Costs in bps to evaluate.  Default: [0, 2, 5, 10, 20, 30].

    Returns
    -------
    pd.DataFrame
        Index = cost_bps, columns = [portfolio_sharpe, portfolio_ann_return,
        portfolio_max_drawdown] + per-ticker sharpes.
    """
    if cost_bps_grid is None:
        cost_bps_grid = [0, 2, 5, 10, 20, 30]

    w = weights.reindex([t for t in results]).fillna(0.0)
    rows = []
    for cost_bps in cost_bps_grid:
        cost_rate = cost_bps / 10_000
        repriced: dict[str, pd.Series] = {}
        ticker_sharpes: dict[str, float] = {}
        for ticker, res in results.items():
            # Gross strategy return = positions * gross_returns (no costs)
            gross_strat = res.positions * res.gross_returns
            position_changes = res.positions.diff().abs().fillna(0.0)
            net = gross_strat - position_changes * cost_rate
            repriced[ticker] = net
            ticker_sharpes[ticker] = sharpe_ratio(net)

        returns_df = pd.DataFrame(repriced).dropna(how="all")
        port = returns_df.mul(w.reindex(returns_df.columns).fillna(0.0), axis=1).sum(axis=1)

        row = {
            "cost_bps": cost_bps,
            "portfolio_sharpe": sharpe_ratio(port),
            "portfolio_ann_return": annualized_return(port),
            "portfolio_max_drawdown": max_drawdown(port),
        }
        row.update({f"sharpe_{t}": s for t, s in ticker_sharpes.items()})
        rows.append(row)

    return pd.DataFrame(rows).set_index("cost_bps")
```

---

### Change 6 — Wire everything into `Pipeline` and update `OOSReport`

**File:** `wsq_trading/pipeline.py`

After implementing Changes 1–5:

1. `Pipeline.run_oos()` must call all new functions and populate `OOSReport`.
2. `Pipeline.allocate()` must accept and forward `train_end_date`.
3. `Pipeline.run_and_summarise()` optionally gets an `oos: bool = False` parameter
   that routes to `run_oos()` when `True`.
4. Add a `Pipeline.print_oos_report(report: OOSReport)` method that logs a formatted
   summary table to the logger (not `print()`), covering all fields in `OOSReport`.

**`print_oos_report` structure:**
```
=== OUT-OF-SAMPLE REPORT ===
Train+Val window: {start} → {train_end}
Test window:      {test_start} → {test_end}

--- Portfolio (OOS) ---
Sharpe:           {sharpe:.3f}
Annual Return:    {ann_return:.1%}
Annual Vol:       {ann_vol:.1%}
Max Drawdown:     {max_dd:.1%}
Calmar:           {calmar:.3f}
Turnover (ann):   {turnover:.2f}x
Alpha vs ES=F:    {alpha:.1%}
Beta vs ES=F:     {beta:.3f}
Info Ratio:       {ir:.3f}

--- Per-Instrument (OOS) ---
{per_instrument dataframe as tabular log}

--- HRP Weights (from train window) ---
{weights as tabular log}

--- Regime Attribution ---
{regime_attribution dataframe as tabular log}

--- Cost Sensitivity ---
{cost_sensitivity dataframe as tabular log}
```

---

## Constraints and Anti-Patterns to Avoid

1. **No look-ahead in the OOS test** — signal generation may use the full return series
   (Hurst estimation needs history), but ALL metric computations must slice to the test
   window AFTER signal generation.
2. **No `print()`** — every output goes through `get_logger(__name__)`.
3. **No new dependencies** — `numpy`, `pandas`, `scipy` are already present. Do not add
   `statsmodels` or `sklearn`; implement OLS for alpha/beta with `np.linalg.lstsq`.
4. **No hardcoded paths** — use `config.*_DIR` constants.
5. **`fno.py` is the only file that imports torch** — do not add torch imports elsewhere.
6. **Line length 100** — Black-formatted; keep all code under 100 chars per line.
7. **Type hints everywhere** — all new public functions must have full type annotations
   and NumPy-style docstrings.
8. **Backward compatibility** — `Pipeline.run()`, `Pipeline.allocate()`, and
   `Pipeline.run_and_summarise()` must still work with their existing signatures.
   New parameters must be optional with sensible defaults.

---

## Tests to Add

Add tests to `wsq_trading/tests/` following the existing `test_<module>.py` convention.
Tests must not import `torch` or require network access.

### `test_portfolio.py` — add to existing

```python
def test_quality_gate_no_lookahead():
    """Quality gate Sharpe must only use pre-cutoff returns."""
    # Create returns with clearly different train vs test Sharpe
    np.random.seed(42)
    idx = pd.date_range("2015-01-01", periods=500)
    r_good = pd.Series(np.random.normal(0.001, 0.01, 500), index=idx)  # high Sharpe train
    r_bad  = pd.Series(np.random.normal(-0.001, 0.01, 500), index=idx)  # low Sharpe overall

    # Make r_bad have high Sharpe in the LAST 100 days only (test period)
    r_bad.iloc[-100:] += 0.005

    from wsq_trading.backtest import BacktestResult, compute_all
    # Build minimal BacktestResult stubs
    def _stub(ticker, ret):
        return BacktestResult(
            ticker=ticker, strategy_returns=ret,
            positions=pd.Series(np.sign(ret), index=ret.index),
            H_estimates=pd.Series(0.5, index=ret.index),
            regimes=pd.Series([None]*len(ret), index=ret.index),
            gross_returns=ret,
        )

    results = {"good": _stub("good", r_good), "bad": _stub("bad", r_bad)}
    from wsq_trading.portfolio import HRPAllocator
    alloc = HRPAllocator(quality_sharpe_threshold=0.0)

    # With train_end_date set to first 400 days: bad should fail gate (low train Sharpe)
    train_end = idx[399]
    w = alloc.allocate_backtest(results, train_end_date=train_end)
    assert w["bad"] == 0.0, "bad instrument should be gated out using train-only Sharpe"
```

### `test_backtest.py` — add to existing

```python
def test_regime_attribution_sums_to_one():
    """Fraction of bars across all regimes should sum to ~1 per ticker."""
    ...  # create a minimal BacktestResult with mixed regimes, verify frac_bars sum ≈ 1


def test_cost_sensitivity_monotone():
    """Higher costs must produce lower or equal Sharpe."""
    ...  # verify portfolio_sharpe is non-increasing as cost_bps increases


def test_portfolio_metrics_alpha_beta():
    """Alpha and beta computed via OLS should be numerically correct."""
    ...  # feed known port and benchmark returns, verify against manual calculation
```

---

## Order of Implementation

Work in this order to avoid forward references:

1. `backtest.py` — add `PortfolioMetrics`, `compute_portfolio_metrics()`,
   `regime_attribution()`, `cost_sensitivity()`.
2. `portfolio.py` — add `train_end_date` parameter to `allocate_backtest()`.
3. `pipeline.py` — add `OOSReport` dataclass, `run_oos()`, `print_oos_report()`,
   update `allocate()` and `run_and_summarise()`.
4. Tests — add/extend test files for new functionality.
5. Verify the existing test suite still passes (`python -m pytest wsq_trading/tests/ -q`).

---

## Verification Checklist

After implementation, run the following and confirm no failures:

```bash
# Full test suite (no torch required)
python -m pytest wsq_trading/tests/ -q

# Quick OOS smoke test (downloads data — needs network)
python -c "
from wsq_trading.pipeline import Pipeline
p = Pipeline(mode='classical')
report = p.run_oos()
p.print_oos_report(report)
"
```

Confirm the smoke test log output includes:
- `=== OUT-OF-SAMPLE REPORT ===`
- A test window that starts after the train+val cutoff
- Portfolio Sharpe, alpha, beta fields (may be NaN if data insufficient — that is fine)
- Regime attribution table with rows for MEAN_REVERT, NEUTRAL, MOMENTUM
- Cost sensitivity table with rows for [0, 2, 5, 10, 20, 30] bps
