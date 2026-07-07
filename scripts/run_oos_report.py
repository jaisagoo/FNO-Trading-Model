"""Run a true out-of-sample evaluation and save the results to disk.

Runs ``Pipeline.run_oos()`` on the configured universe, prints the report, and
persists it via :mod:`wsq_trading.reporting`:

- ``results/metrics/<date>_oos_*.csv``           machine-readable tables
- ``results/backtest_reports/<date>_oos_report.md``  dated written report,
  including a ready-to-paste README results section

With ``--ablation`` it re-runs the evaluation with the sizing/bias overlays
(long-bias filter, vol targeting, H-conviction sizing) toggled off one at a
time, so the H(t) regime-switching contribution can be judged separately from
the overlays.

Usage
-----
    python scripts/run_oos_report.py
    python scripts/run_oos_report.py --ablation
    python scripts/run_oos_report.py --offline --tickers ES=F NQ=F CL=F
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wsq_trading import config
from wsq_trading.reporting import (
    append_ablation_section,
    portfolio_summary_frame,
    save_oos_report,
)
from wsq_trading.utils import get_logger

log = get_logger(__name__)

# Overlay ablation variants: name -> Pipeline keyword overrides.
# The conviction variant toggles AWAY from the config default, so the table
# stays meaningful whichever way H_CONVICTION_SIZING is frozen.
_CONV_NAME = "no_h_conviction" if config.H_CONVICTION_SIZING else "h_conviction_on"
ABLATION_VARIANTS: list[tuple[str, dict]] = [
    ("baseline", {}),
    ("no_long_bias", {"long_bias_tickers": []}),
    ("no_vol_target", {"vol_target_enabled": False}),
    (_CONV_NAME, {"h_conviction_sizing": not config.H_CONVICTION_SIZING}),
    (
        "signal_only",
        {
            "long_bias_tickers": [],
            "vol_target_enabled": False,
            "h_conviction_sizing": False,
        },
    ),
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_oos_report",
        description="Out-of-sample evaluation with saved CSV + markdown report.",
    )
    p.add_argument("--mode", choices=["classical", "fno"], default="classical")
    p.add_argument("--tickers", nargs="+", default=None, metavar="TICKER",
                   help=f"Default: {config.TICKERS}")
    p.add_argument("--start", default=None, metavar="YYYY-MM-DD")
    p.add_argument("--end", default=None, metavar="YYYY-MM-DD")
    p.add_argument("--cost-bps", type=float, default=None, metavar="BPS",
                   help=f"Round-trip cost in bps (default {config.TRANSACTION_COST_BPS}).")
    p.add_argument("--checkpoint", type=Path, default=None, metavar="PATH",
                   help="FNO checkpoint (.pt); only used with --mode fno.")
    p.add_argument("--offline", action="store_true",
                   help="Use the parquet cache only (no network).")
    p.add_argument("--ablation", action="store_true",
                   help="Also run the overlay on/off ablation (slower).")
    p.add_argument("--label", default="oos",
                   help="Filename slug for the saved artefacts (default 'oos').")
    p.add_argument("--metrics-dir", type=Path, default=None,
                   help=f"Default: {config.METRICS_DIR}")
    p.add_argument("--reports-dir", type=Path, default=None,
                   help=f"Default: {config.REPORTS_DIR}")
    return p


def _make_pipeline(args: argparse.Namespace, overrides: dict | None = None):
    from wsq_trading.pipeline import Pipeline

    overrides = overrides or {}
    return Pipeline(
        mode=args.mode,
        cost_bps=args.cost_bps,
        fno_checkpoint=args.checkpoint,
        **overrides,
    )


def main() -> None:
    args = build_parser().parse_args()

    if args.offline:
        config.CACHE_MAX_AGE_HOURS = 1e12  # treat any cached parquet as fresh
        log.info("offline mode: using parquet cache only")

    tickers = args.tickers or config.TICKERS
    run_config = {
        "mode": args.mode,
        "tickers": tickers,
        "start": args.start or config.DEFAULT_START,
        "end": args.end or "today",
        "cost_bps": args.cost_bps or config.TRANSACTION_COST_BPS,
        "hurst_rough_threshold": config.HURST_ROUGH_THRESHOLD,
        "hurst_trend_threshold": config.HURST_TREND_THRESHOLD,
        "train_split": config.TRAIN_SPLIT,
        "val_split": config.VAL_SPLIT,
    }

    pipe = _make_pipeline(args)
    report = pipe.run_oos(tickers=tickers, start=args.start, end=args.end)
    pipe.print_oos_report(report)

    paths = save_oos_report(
        report,
        metrics_dir=args.metrics_dir,
        reports_dir=args.reports_dir,
        label=args.label,
        run_config=run_config,
    )

    if args.ablation:
        rows = {}
        for name, overrides in ABLATION_VARIANTS:
            if name == "baseline":
                rows[name] = portfolio_summary_frame(report).iloc[0]
                continue
            log.info("ablation variant: %s  (%s)", name, overrides)
            variant_pipe = _make_pipeline(args, overrides)
            variant_report = variant_pipe.run_oos(
                tickers=tickers, start=args.start, end=args.end
            )
            rows[name] = portfolio_summary_frame(variant_report).iloc[0]

        import pandas as pd

        ablation = pd.DataFrame(rows).T
        ablation.index.name = "variant"
        append_ablation_section(
            ablation,
            report_md_path=paths["report_md"],
            metrics_dir=args.metrics_dir,
            label=args.label,
        )
        log.info("\nOverlay ablation (portfolio OOS):\n%s", ablation.to_string())

    log.info("Report saved: %s", paths["report_md"])


if __name__ == "__main__":
    main()
