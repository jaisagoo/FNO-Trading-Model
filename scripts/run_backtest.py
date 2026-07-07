"""run_backtest.py - Entry point for the Rough-Heston trading pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make sure the package is importable when run from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wsq_trading import config
from wsq_trading.utils import get_logger

log = get_logger(__name__)


#  CLI

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_backtest",
        description="Rough-Heston walk-forward backtest with H(t) regime switching.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        "--mode",
        choices=["classical", "fno"],
        default="classical",
        help=(
            "Hurst estimator to use.\n"
            "  classical  - rolling Whittle MLE (fast, no GPU required)\n"
            "  fno        - Fourier Neural Operator (requires a trained checkpoint)"
        ),
    )
    p.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        metavar="TICKER",
        help=f"Space-separated list of tickers.  Default: {config.TICKERS}",
    )
    p.add_argument(
        "--start",
        default=None,
        metavar="YYYY-MM-DD",
        help=f"Start date.  Default: {config.DEFAULT_START}",
    )
    p.add_argument(
        "--end",
        default=None,
        metavar="YYYY-MM-DD",
        help="End date.  Default: today",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to FNO checkpoint (.pt file).  Required for --mode fno.",
    )
    p.add_argument(
        "--train-fno",
        action="store_true",
        help=(
            "Generate synthetic SPDE data and train the FNO before backtesting. "
            "Ignored in classical mode."
        ),
    )
    p.add_argument(
        "--cost-bps",
        type=float,
        default=None,
        metavar="BPS",
        help=f"Round-trip transaction cost in basis points.  Default: {config.TRANSACTION_COST_BPS}",
    )
    return p


#  FNO training helper

def train_fno() -> Path:
    """Generate synthetic paths and train the FNO.  Returns checkpoint path."""
    log.info("=== FNO Training Mode ===")

    from wsq_trading.spde import SPDESimulator
    from wsq_trading.fno import FNO1d
    from wsq_trading.fno import FNOTrainer

    log.info("Generating %d synthetic SPDE paths …", config.N_SYNTHETIC_PATHS)
    sim = SPDESimulator()
    dataset = sim.generate_dataset()
    log.info("Dataset ready: %d samples.", len(dataset))

    model = FNO1d(
        in_channels=2,
        width=config.FNO_WIDTH,
        n_modes=config.FNO_MODES,
        n_layers=config.FNO_LAYERS,
    )

    config.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    trainer = FNOTrainer(
        model=model,
        window_size=config.WINDOW_SIZE,
        checkpoint_dir=config.CHECKPOINTS_DIR,
    )

    log.info(
        "Training FNO: epochs=%d  lr=%.4f  batch=%d",
        config.NUM_EPOCHS, config.LEARNING_RATE, config.BATCH_SIZE,
    )
    trainer.train(dataset)

    ckpt = config.CHECKPOINTS_DIR / "fno_best.pt"
    log.info("Training complete.  Checkpoint saved to %s", ckpt)
    return ckpt


#  Main

def main() -> None:
    args = build_parser().parse_args()

    # Resolve FNO checkpoint
    checkpoint: Path | None = args.checkpoint

    if args.mode == "fno":
        if args.train_fno:
            checkpoint = train_fno()
        elif checkpoint is None:
            # Try the default location before erroring
            default_ckpt = config.CHECKPOINTS_DIR / "fno_best.pt"
            if default_ckpt.exists():
                log.info("Using existing checkpoint: %s", default_ckpt)
                checkpoint = default_ckpt
            else:
                log.error(
                    "No FNO checkpoint found.  Either:\n"
                    "  1. Run with --train-fno to train from scratch, or\n"
                    "  2. Provide --checkpoint <path> to an existing .pt file."
                )
                sys.exit(1)

    # Run pipeline
    from wsq_trading.pipeline import Pipeline

    log.info("=== Backtest: mode=%s ===", args.mode)

    pipe = Pipeline(
        mode=args.mode,
        cost_bps=args.cost_bps,
        fno_checkpoint=checkpoint,
    )

    results, weights, summary = pipe.run_and_summarise(
        tickers=args.tickers,
        start=args.start,
        end=args.end,
    )

    if summary.empty:
        log.warning("No results produced - check logs above for errors.")
        sys.exit(1)

    # Print results
    print("\n" + "=" * 70)
    print("  BACKTEST SUMMARY")
    print("=" * 70)
    print(summary.to_string())
    print("=" * 70)

    print("\nHRP Portfolio Weights:")
    for ticker, w in weights.sort_values(ascending=False).items():
        print(f"  {ticker:<10} {w:.2%}")

    log.info("Done.")


if __name__ == "__main__":
    main()
