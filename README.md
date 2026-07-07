# Rough-Heston Trading Model

A quantitative trading system that predicts a time-varying Hurst exponent with a
Fourier Neural Operator (FNO) and switches between mean-reversion, trend-following
and neutral strategies depending on the estimated roughness regime.

## Idea

Volatility is rough: realised variance behaves like a fractional process with a
small Hurst exponent. If the local roughness `H(t)` can be estimated in real time,
it tells you which regime the price is in:

- `H < 0.43` — rough / anti-persistent → mean-reversion
- `0.43 ≤ H ≤ 0.57` — close to Brownian → neutral (short-horizon contrarian overlay)
- `H > 0.57` — persistent / trending → momentum

Classical Whittle-MLE estimates on daily futures data cluster in roughly
[0.40, 0.56], so the band is asymmetric around 0.5; the wide neutral band cuts
false momentum entries from noisy `H` estimates. The calibration notes live next
to the thresholds in `wsq_trading/config.py`, which is the single source of truth.

The FNO is trained on rough-Heston SPDE paths (where the true `H` is known), then
used to label live return windows. Signals are combined across instruments with
Hierarchical Risk Parity and evaluated with a walk-forward backtest.

## Layout

```
wsq_trading/
    config.py       all settings live here
    constants.py    enums and math constants
    utils.py        logging, seeding, timing helpers
    data.py         load / validate / clean OHLCV
    spde.py         rough-Heston SPDE path simulator
    features.py     log-returns, realised vol, rolling stats
    hurst.py        classical estimators (R/S, DFA, Whittle MLE)
    fno.py          Fourier Neural Operator + trainer
    signals.py      H(t) -> regime -> long/short/flat
    portfolio.py    Hierarchical Risk Parity allocator
    backtest.py     walk-forward engine + performance metrics
    pipeline.py     end-to-end orchestration
    reporting.py    persist OOS results to CSV + markdown
    tests/          pytest suite
scripts/
    run_backtest.py       CLI entry point for the pipeline
    run_oos_report.py     out-of-sample report + overlay ablation
    tune_validation.py    parameter search scored on the validation window only
    plot_oos_results.py   OOS figure suite (11 figures)
    plot_results.py       data-layer / SPDE diagnostics
```

The FNO code in `fno.py` is the only part that needs PyTorch, so it is imported
lazily and torch ships as an optional extra (`pip install -e ".[fno]"`).

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
make test          # full CPU-only suite; FNO tests skip unless torch is installed
```

Market data downloads automatically from Yahoo Finance on first run and is cached
as parquet under `data/raw/futures/`, so a fresh clone has an empty `data/` until
the first backtest.

```bash
# Walk-forward backtest over the configured universe (classical Hurst estimator)
python scripts/run_backtest.py

# True out-of-sample evaluation: fixed train/test split, metrics saved to
# results/metrics/*.csv and a dated report to results/backtest_reports/*.md
python scripts/run_oos_report.py

# Same, plus an ablation of the sizing/bias overlays (long-bias, vol-target,
# H-conviction on/off) to separate regime-switching alpha from the overlays
python scripts/run_oos_report.py --ablation

# Regenerate the OOS figure suite into results/plots/
python scripts/plot_oos_results.py
```

FNO mode needs torch and a trained checkpoint — none ships with the repo:

```bash
pip install -e ".[fno]"
python scripts/run_backtest.py --mode fno --train-fno   # trains from synthetic SPDE data
```

## Results

True out-of-sample test window **2024-01-03 → 2026-07-02** (645 trading days).
Thresholds, universe, filters and overlay configuration were all fixed before this
window was evaluated: the overlay configuration was selected on the validation
window (H-conviction sizing off — it reduced validation-window Sharpe), and the
test window was evaluated once after freezing. The full 54-candidate validation
search is committed at `results/metrics/2026-07-07_validation_tuning.csv`.

| Metric | OOS value |
|---|---|
| Sharpe | 1.60 |
| Sortino | 1.50 |
| Annual return | 11.9% |
| Annual vol | 7.2% |
| Max drawdown | 4.5% |
| Calmar | 2.64 |
| Win rate | 50.1% |
| Turnover (ann.) | 16.6x |
| Cost assumption | 2.0 bps |
| Alpha vs ES=F | 5.5% |
| Beta vs ES=F | 0.34 |
| Information ratio | -0.57 |

What the book actually is: the HRP quality gate (fitted on the train window)
admits ES and NQ at 50/50 over this period, so the traded portfolio is a
concentrated equity-futures book with beta ≈ 0.34 — an equity-tilted
absolute-return profile, not a market-neutral one. The negative information
ratio against a long-ES benchmark is the expected consequence of running ~0.3
beta through a strong bull test window; alpha is +5.5%/yr against that same
benchmark.

Overlay ablation on the test window (each overlay toggled from the frozen
config) separates the return sources:

| variant | Sharpe | ann. return | ann. vol |
|---|---|---|---|
| baseline (frozen) | 1.60 | 11.9% | 7.2% |
| no_long_bias | 0.17 | 0.5% | 3.3% |
| no_vol_target | 1.44 | 11.7% | 7.9% |
| h_conviction_on | 0.98 | 1.5% | 1.6% |
| signal_only | -0.20 | -1.1% | 4.9% |

The long-bias trend filter is the largest single contributor and the raw regime
signal alone is unprofitable after costs at ~48x turnover. The H(t) regime layer
earns its place in *timing*: while confirmed regimes are active the portfolio
runs at Sharpe 3.39 (momentum bars) and 3.21 (neutral bars) versus 0.37 in
mean-revert bars (regime attribution tables in `results/metrics/`).

Cost sensitivity (round-trip): Sharpe 1.64 / 1.60 / 1.53 / 1.42 / 1.19 / 0.96 at
0 / 2 / 5 / 10 / 20 / 30 bps. The decay is material at 16.6x annual turnover and
worth weighing against realistic execution assumptions, though the strategy
remains positive at 30 bps.

One known artefact, documented rather than patched after the fact: realised vol
(7.2%) still runs below the 15% `TARGET_VOL` intent, because per-instrument vol
scaling is capped at 1.5x and HRP splits weight across only the two gated
instruments — deployment, not signal quality, caps the headline return.

## Make targets

`make test`, `make test-cov`, `make lint`, `make format`, `make typecheck`,
`make data-dirs`, `make clean`.

## References

- Gatheral, Jaisson & Rosenbaum (2018), *Volatility is rough*
- El Euch & Rosenbaum (2019), *The characteristic function of rough Heston models*
- Li, Kovachki et al. (2021), *Fourier Neural Operator for Parametric PDEs*
- López de Prado (2016), *Building diversified portfolios that outperform out of sample* (HRP)

## License

MIT — see [LICENSE](LICENSE).
