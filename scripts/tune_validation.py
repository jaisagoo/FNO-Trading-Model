"""Search signal parameters on the VALIDATION window only.

Motivation: the regime thresholds and filters are flexible, so any tuning done
while looking at test-period results silently overfits the final report.  This
script keeps the protocol honest:

- the walk-forward engine runs on full history (Hurst estimation needs warm-up),
- HRP weights and the quality gate are fitted on the TRAIN window only,
- every candidate is scored on the VALIDATION window only
  (``config.TRAIN_SPLIT`` .. ``TRAIN_SPLIT + VAL_SPLIT``),
- the TEST window is never touched, printed, or saved here.

Workflow: run this, pick a candidate, set the winning values in
``wsq_trading/config.py``, then run ``scripts/run_oos_report.py`` exactly once
for the final out-of-sample numbers.  The full candidate table is saved to
``results/metrics/`` so the number of configurations tried is on the record.

Hurst estimates are cached across candidates (all searched knobs act on the
regime classification, signal, and position-sizing stages, downstream of H
estimation), which makes the grid roughly free compared with re-estimating H
per candidate.  The grid may include the engine-level overlay flags
(``h_conviction_sizing``, ``vol_target_enabled``, ``long_bias_tickers``), so
overlay on/off decisions can be made on validation evidence rather than on
the test-window ablation.

Usage
-----
    python scripts/tune_validation.py                    # default grid
    python scripts/tune_validation.py --offline --tickers ES=F NQ=F
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wsq_trading import config
from wsq_trading.backtest import WalkForwardEngine, compute_portfolio_metrics
from wsq_trading.pipeline import Pipeline
from wsq_trading.signals import HurstRegimeSignalGenerator
from wsq_trading.utils import get_logger

log = get_logger(__name__)

# Search space: post-H signal knobs and engine-level overlay flags.  Edit here.
# Keys in ENGINE_KEYS go to WalkForwardEngine; everything else goes to the
# signal generator.  All act downstream of Hurst estimation, so the H cache
# stays valid across candidates.
GRID: dict[str, list] = {
    "rough_threshold": [0.41, 0.43, 0.45],
    "trend_threshold": [0.55, 0.57, 0.59],
    "regime_persistence": [3, 5, 7],
    "h_conviction_sizing": [True, False],
}

ENGINE_KEYS = {"long_bias_tickers", "vol_target_enabled", "h_conviction_sizing"}


class _CachedHurstSignalGenerator(HurstRegimeSignalGenerator):
    """Signal generator that memoises rolling-H series across candidates.

    Valid only while the searched parameters all act downstream of Hurst
    estimation; the cache key includes the estimator inputs
    (``hurst_window``, ``vol_window``) as a guard.
    """

    _cache: dict[tuple, pd.Series] = {}

    def compute_rolling_hurst(self, returns: pd.Series) -> pd.Series:
        key = (
            int(pd.util.hash_pandas_object(returns).sum()),
            self._hurst_window,
            self._vol_window,
        )
        if key not in self._cache:
            self._cache[key] = super().compute_rolling_hurst(returns)
        return self._cache[key]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tune_validation",
        description="Grid-search signal parameters on the validation window only.",
    )
    p.add_argument("--tickers", nargs="+", default=None, metavar="TICKER",
                   help=f"Default: {config.TICKERS}")
    p.add_argument("--start", default=None, metavar="YYYY-MM-DD")
    p.add_argument("--end", default=None, metavar="YYYY-MM-DD")
    p.add_argument("--cost-bps", type=float, default=None, metavar="BPS")
    p.add_argument("--offline", action="store_true",
                   help="Use the parquet cache only (no network).")
    p.add_argument("--top", type=int, default=10, help="Rows to log (default 10).")
    p.add_argument("--metrics-dir", type=Path, default=None,
                   help=f"Default: {config.METRICS_DIR}")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.offline:
        config.CACHE_MAX_AGE_HOURS = 1e12
        log.info("offline mode: using parquet cache only")

    tickers = args.tickers or config.TICKERS
    cost_bps = args.cost_bps if args.cost_bps is not None else config.TRANSACTION_COST_BPS

    # Load and feature-engineer once; candidates only re-run signals + backtest.
    base = Pipeline(mode="classical", cost_bps=cost_bps)
    data = base._load_and_clean(tickers, args.start or config.DEFAULT_START,
                                args.end or config.DEFAULT_END)
    if not data:
        raise SystemExit("No usable data - check tickers / cache / network.")

    names = list(GRID)
    combos = list(itertools.product(*GRID.values()))
    log.info("Searching %d candidates over %s", len(combos), names)

    rows = []
    for combo in combos:
        params = dict(zip(names, combo))
        sig_params = {k: v for k, v in params.items() if k not in ENGINE_KEYS}
        eng_params = {k: v for k, v in params.items() if k in ENGINE_KEYS}
        sig_gen = _CachedHurstSignalGenerator(**sig_params)
        engine = WalkForwardEngine(signal_generator=sig_gen, cost_bps=cost_bps, **eng_params)
        results = engine.run(data)
        if not results:
            log.warning("candidate %s produced no results; skipping", params)
            continue

        # Date splits mirror Pipeline.run_oos, but stop at the validation edge.
        all_dates = sorted(set().union(*[r.strategy_returns.index for r in results.values()]))
        n = len(all_dates)
        train_n = min(max(int(n * config.TRAIN_SPLIT), 1), n - 2)
        val_n = min(max(int(n * (config.TRAIN_SPLIT + config.VAL_SPLIT)), train_n + 1), n - 1)
        train_end = pd.Timestamp(all_dates[train_n - 1])
        val_start, val_end = pd.Timestamp(all_dates[train_n]), pd.Timestamp(all_dates[val_n - 1])

        # Weights and gate from the train window only (validation stays unseen).
        train_results = {
            t: Pipeline._slice_result(r, None, train_end) for t, r in results.items()
        }
        weights = base.allocator.allocate_backtest(train_results, train_end_date=train_end)

        val_results = {
            t: Pipeline._slice_result(r, val_start, val_end) for t, r in results.items()
        }
        benchmark = val_results["ES=F"].gross_returns if "ES=F" in val_results else None
        port = compute_portfolio_metrics(
            val_results, weights, benchmark_returns=benchmark, cost_bps=cost_bps
        )
        rows.append({
            **params,
            "val_sharpe": port.sharpe,
            "val_ann_return": port.annualized_return,
            "val_ann_vol": port.annualized_vol,
            "val_max_drawdown": port.max_drawdown,
            "val_turnover": port.turnover,
            "n_gated_in": int((weights > 0).sum()),
        })
        log.info("%s -> val Sharpe %.3f", params, port.sharpe)

    table = pd.DataFrame(rows).sort_values("val_sharpe", ascending=False).reset_index(drop=True)

    metrics_dir = Path(args.metrics_dir if args.metrics_dir is not None else config.METRICS_DIR)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out = metrics_dir / f"{pd.Timestamp.today().date().isoformat()}_validation_tuning.csv"
    table.to_csv(out, index=False)

    log.info("\nValidation window: %s -> %s  (test window untouched)\n%s",
             val_start.date(), val_end.date(), table.head(args.top).to_string())
    log.info("Saved all %d candidates to %s", len(table), out)
    log.info("Next: set the chosen values in config.py, then run "
             "scripts/run_oos_report.py ONCE for the final test-window numbers.")


if __name__ == "__main__":
    main()
