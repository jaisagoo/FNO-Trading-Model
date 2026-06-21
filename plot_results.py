"""plot_results.py - Visualise backtest results for the Rough-Heston strategy.

Generates seven figures and saves them to results/plots/:
  1. equity_curves.png      - cumulative strategy vs buy-and-hold per ticker
  2. hurst_regimes.png      - H(t) estimates with regime shading per ticker
  3. drawdowns.png          - drawdown curves per ticker
  4. positions.png          - daily position (+1 / 0 / -1) per ticker
  5. regime_breakdown.png   - pie charts of time spent in each regime
  6. metrics_summary.png    - bar chart comparison of key metrics
  7. portfolio_summary.png  - combined portfolio equity curve + rolling Sharpe + metric dashboard
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # non-interactive backend - works without a display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from wsq_trading import config
from wsq_trading.backtest import BacktestResult
from wsq_trading.constants import Regime
from wsq_trading.utils import get_logger

log = get_logger(__name__)

# Colour palette
REGIME_COLORS = {
    Regime.MEAN_REVERT: "#4e9af1",   # blue
    Regime.NEUTRAL:     "#aaaaaa",   # grey
    Regime.MOMENTUM:    "#f4845f",   # orange
}
REGIME_LABELS = {
    Regime.MEAN_REVERT: f"Mean Revert  (H < {config.HURST_ROUGH_THRESHOLD})",
    Regime.NEUTRAL:     (
        f"Neutral  ({config.HURST_ROUGH_THRESHOLD} ≤ H ≤ {config.HURST_TREND_THRESHOLD})"
    ),
    Regime.MOMENTUM:    f"Momentum  (H > {config.HURST_TREND_THRESHOLD})",
}
STRATEGY_COLOR  = "#2ecc71"
BH_COLOR        = "#95a5a6"
DRAWDOWN_COLOR  = "#e74c3c"


#  Helpers

def _cum_return(returns: pd.Series) -> pd.Series:
    return (1 + returns.fillna(0)).cumprod()


def _drawdown_series(returns: pd.Series) -> pd.Series:
    cum = _cum_return(returns)
    running_max = cum.cummax()
    return (cum - running_max) / running_max


def _shade_regimes(ax: plt.Axes, regimes: pd.Series) -> None:
    """Shade background by regime using fast span logic."""
    if regimes.empty:
        return
    prev_regime = regimes.iloc[0]
    start_date  = regimes.index[0]
    for date, regime in regimes.items():
        if regime != prev_regime:
            ax.axvspan(start_date, date,
                       alpha=0.12, color=REGIME_COLORS.get(prev_regime, "#cccccc"),
                       linewidth=0)
            prev_regime = regime
            start_date  = date
    ax.axvspan(start_date, regimes.index[-1],
               alpha=0.12, color=REGIME_COLORS.get(prev_regime, "#cccccc"),
               linewidth=0)


def _apply_style(fig: plt.Figure, title: str) -> None:
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()


#  Figure 1 - Equity curves

def plot_equity_curves(
    results: dict[str, BacktestResult],
    weights: pd.Series,
    save_dir: Path | None,
) -> plt.Figure:
    tickers = list(results)
    n = len(tickers)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows),
                              squeeze=False)

    for idx, ticker in enumerate(tickers):
        res = results[ticker]
        ax  = axes[idx // ncols][idx % ncols]

        strat_cum = _cum_return(res.strategy_returns)
        bh_cum    = _cum_return(res.gross_returns)

        _shade_regimes(ax, res.regimes)

        ax.plot(bh_cum.index, bh_cum.values,
                color=BH_COLOR, linewidth=1.2, label="Buy & Hold", alpha=0.8)
        ax.plot(strat_cum.index, strat_cum.values,
                color=STRATEGY_COLOR, linewidth=1.6, label="Strategy")

        w = weights.get(ticker, float("nan"))
        ax.set_title(
            f"{ticker}   Sharpe={res.sharpe:.2f}   MaxDD={res.max_drawdown:.1%}"
            f"   HRP wt={w:.1%}",
            fontsize=9,
        )
        ax.set_ylabel("Growth of $1")
        ax.axhline(1.0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    # Shared regime legend
    patches = [mpatches.Patch(color=c, alpha=0.4, label=REGIME_LABELS[r])
               for r, c in REGIME_COLORS.items()]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               fontsize=8, bbox_to_anchor=(0.5, -0.02))

    _apply_style(fig, "Equity Curves - Strategy vs Buy & Hold")
    if save_dir:
        out = save_dir / "equity_curves.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        log.info("Saved %s", out)
    return fig


#  Figure 2 - H(t) with regime shading

def plot_hurst_regimes(
    results: dict[str, BacktestResult],
    save_dir: Path | None,
) -> plt.Figure:
    tickers = list(results)
    n = len(tickers)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), squeeze=False)

    for idx, ticker in enumerate(tickers):
        res = results[ticker]
        ax  = axes[idx][0]
        H   = res.H_estimates.dropna()

        _shade_regimes(ax, res.regimes)

        ax.plot(H.index, H.values, color="#333333", linewidth=1.0, alpha=0.85)
        ax.axhline(config.HURST_ROUGH_THRESHOLD, color=REGIME_COLORS[Regime.MEAN_REVERT],
                   linewidth=1.2, linestyle="--", alpha=0.8,
                   label=f"Rough threshold ({config.HURST_ROUGH_THRESHOLD})")
        ax.axhline(config.HURST_TREND_THRESHOLD, color=REGIME_COLORS[Regime.MOMENTUM],
                   linewidth=1.2, linestyle="--", alpha=0.8,
                   label=f"Trend threshold ({config.HURST_TREND_THRESHOLD})")
        ax.axhline(0.5, color="black", linewidth=0.7, linestyle=":", alpha=0.4,
                   label="H = 0.5 (Brownian)")

        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("H(t)")
        ax.set_title(ticker, fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.3)

    _apply_style(fig, "Rolling Hurst Exponent H(t) with Regime Shading")
    if save_dir:
        out = save_dir / "hurst_regimes.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        log.info("Saved %s", out)
    return fig


#  Figure 3 - Drawdowns

def plot_drawdowns(
    results: dict[str, BacktestResult],
    save_dir: Path | None,
) -> plt.Figure:
    tickers = list(results)
    n = len(tickers)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.5 * n), squeeze=False)

    for idx, ticker in enumerate(tickers):
        res = results[ticker]
        ax  = axes[idx][0]
        dd  = _drawdown_series(res.strategy_returns)

        ax.fill_between(dd.index, dd.values, 0,
                        color=DRAWDOWN_COLOR, alpha=0.5, linewidth=0)
        ax.plot(dd.index, dd.values, color=DRAWDOWN_COLOR, linewidth=0.8)
        ax.set_ylabel("Drawdown")
        ax.set_title(f"{ticker}   Max DD = {res.max_drawdown:.1%}", fontsize=9)
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:.0%}")
        )
        ax.grid(alpha=0.3)

    _apply_style(fig, "Strategy Drawdowns")
    if save_dir:
        out = save_dir / "drawdowns.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        log.info("Saved %s", out)
    return fig


#  Figure 4 - Positions

def plot_positions(
    results: dict[str, BacktestResult],
    save_dir: Path | None,
) -> plt.Figure:
    tickers = list(results)
    n = len(tickers)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2 * n), squeeze=False)

    for idx, ticker in enumerate(tickers):
        res = results[ticker]
        ax  = axes[idx][0]
        pos = res.positions

        ax.fill_between(pos.index, pos.values, 0,
                        where=pos.values > 0,
                        color=STRATEGY_COLOR, alpha=0.6, label="Long (+1)")
        ax.fill_between(pos.index, pos.values, 0,
                        where=pos.values < 0,
                        color=DRAWDOWN_COLOR, alpha=0.6, label="Short (−1)")

        ax.set_ylim(-1.5, 1.5)
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels(["Short", "Flat", "Long"])
        ax.set_title(ticker, fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.2)

    _apply_style(fig, "Daily Positions")
    if save_dir:
        out = save_dir / "positions.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        log.info("Saved %s", out)
    return fig


#  Figure 5 - Regime breakdown (pie charts)

def plot_regime_breakdown(
    results: dict[str, BacktestResult],
    save_dir: Path | None,
) -> plt.Figure:
    tickers = list(results)
    n = len(tickers)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                              squeeze=False)

    for idx, ticker in enumerate(tickers):
        res     = results[ticker]
        ax      = axes[idx // ncols][idx % ncols]
        regimes = res.regimes.dropna()

        counts = {r: (regimes == r).sum() for r in Regime}
        labels = [REGIME_LABELS[r] for r in Regime]
        sizes  = [counts[r] for r in Regime]
        colors = [REGIME_COLORS[r] for r in Regime]

        wedges, _, autotexts = ax.pie(
            sizes, labels=None, colors=colors,
            autopct=lambda p: f"{p:.1f}%" if p > 1 else "",
            startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 1},
        )
        for at in autotexts:
            at.set_fontsize(8)

        ax.legend(wedges, labels, fontsize=7, loc="lower center",
                  bbox_to_anchor=(0.5, -0.15))
        ax.set_title(ticker, fontsize=9)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    _apply_style(fig, "Time Spent in Each Regime")
    if save_dir:
        out = save_dir / "regime_breakdown.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        log.info("Saved %s", out)
    return fig


#  Figure 6 - Metrics summary bar chart

def plot_metrics_summary(
    results: dict[str, BacktestResult],
    weights: pd.Series,
    save_dir: Path | None,
) -> plt.Figure:
    METRICS = ["sharpe", "sortino", "calmar", "annualized_return",
               "annualized_vol", "max_drawdown", "win_rate"]
    LABELS  = ["Sharpe", "Sortino", "Calmar", "Ann. Return",
               "Ann. Vol", "Max DD", "Win Rate"]

    tickers = list(results)
    df = pd.DataFrame(
        {t: results[t].metrics for t in tickers}
    ).T[METRICS]

    n_metrics = len(METRICS)
    fig, axes = plt.subplots(1, n_metrics, figsize=(3.2 * n_metrics, 5),
                              squeeze=False)

    for col_idx, (metric, label) in enumerate(zip(METRICS, LABELS)):
        ax  = axes[0][col_idx]
        vals = df[metric]
        colors = [STRATEGY_COLOR if v >= 0 else DRAWDOWN_COLOR for v in vals]

        bars = ax.barh(tickers, vals, color=colors, edgecolor="white",
                       linewidth=0.5)
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.axvline(0, color="black", linewidth=0.7, alpha=0.5)
        ax.grid(axis="x", alpha=0.3)
        ax.tick_params(axis="y", labelsize=8)

        # Format x-axis as % for return/vol/dd metrics
        if metric in {"annualized_return", "annualized_volatility",
                      "max_drawdown", "win_rate"}:
            ax.xaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:.0%}")
            )

        # Annotate bars
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                fmt = f"{val:.2f}" if metric not in {
                    "annualized_return", "annualized_volatility",
                    "max_drawdown", "win_rate"} else f"{val:.1%}"
                ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                        fmt, va="center", fontsize=7)

    _apply_style(fig, "Per-Ticker Metrics Summary")
    if save_dir:
        out = save_dir / "metrics_summary.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        log.info("Saved %s", out)
    return fig


#  Figure 7 - Combined portfolio metrics

def _compute_portfolio_returns(
    results: dict[str, BacktestResult],
    weights: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Compute HRP-weighted portfolio strategy returns and equal-weight B&H returns."""
    # Only include instruments with a non-zero HRP weight
    active = {t: r for t, r in results.items() if weights.get(t, 0.0) > 0}
    if not active:
        # Fall back to all instruments equally weighted
        active = results

    all_dates = pd.DatetimeIndex(
        sorted(set().union(*[r.strategy_returns.index for r in active.values()]))
    )

    strat_df = pd.DataFrame(
        {t: r.strategy_returns.reindex(all_dates, fill_value=0.0)
         for t, r in active.items()}
    )
    bh_df = pd.DataFrame(
        {t: r.gross_returns.reindex(all_dates, fill_value=0.0)
         for t, r in active.items()}
    )

    # Normalise weights to sum to 1 over active instruments
    w = weights.reindex(list(active.keys()), fill_value=0.0)
    if w.sum() > 0:
        w = w / w.sum()
    else:
        w = pd.Series(1.0 / len(active), index=list(active.keys()))

    port_strat = strat_df.mul(w, axis=1).sum(axis=1)
    port_bh    = bh_df.mul(1.0 / len(active), axis=1).sum(axis=1)

    return port_strat, port_bh


def _rolling_sharpe(returns: pd.Series, window: int = 252) -> pd.Series:
    """Rolling annualised Sharpe ratio."""
    roll_mean = returns.rolling(window, min_periods=window // 2).mean()
    roll_std  = returns.rolling(window, min_periods=window // 2).std(ddof=1)
    return (roll_mean / roll_std.replace(0, np.nan)) * np.sqrt(252)


def _portfolio_metrics(returns: pd.Series) -> dict[str, float]:
    """Compute annualised portfolio-level metrics."""
    r = returns.dropna()
    n = len(r)
    if n < 30 or r.std() == 0:
        return {}

    ann_ret = r.mean() * 252
    ann_vol = r.std(ddof=1) * np.sqrt(252)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0.0

    downside = r[r < 0]
    ann_down = downside.std(ddof=1) * np.sqrt(252) if len(downside) > 1 else ann_vol
    sortino  = ann_ret / ann_down if ann_down > 0 else 0.0

    cum = (1 + r.fillna(0)).cumprod()
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max
    max_dd = float(-dd.min())
    calmar = ann_ret / max_dd if max_dd > 0 else 0.0

    win_rate = float((r > 0).mean())

    return {
        "ann_return": ann_ret,
        "ann_vol":    ann_vol,
        "sharpe":     sharpe,
        "sortino":    sortino,
        "calmar":     calmar,
        "max_dd":     max_dd,
        "win_rate":   win_rate,
    }


def plot_portfolio_summary(
    results: dict[str, BacktestResult],
    weights: pd.Series,
    save_dir: Path | None,
) -> plt.Figure:
    """Figure 7: combined portfolio equity curve + metrics dashboard.

    Layout
    ------
    Row 0 (wide): Portfolio equity curve vs equal-weight B&H
    Row 0 (wide): Rolling 1-year Sharpe
    Row 1: Bar charts - Portfolio vs each instrument for 6 key metrics
    Row 2: Metric scorecard table
    """
    port_strat, port_bh = _compute_portfolio_returns(results, weights)
    port_metrics = _portfolio_metrics(port_strat)

    # Layout: 3 rows
    fig = plt.figure(figsize=(18, 16))
    gs  = fig.add_gridspec(
        3, 6,
        height_ratios=[2.5, 2.2, 1.2],
        hspace=0.45, wspace=0.35,
    )

    # Row 0 left: portfolio equity curve
    ax_eq = fig.add_subplot(gs[0, :3])
    cum_strat = _cum_return(port_strat)
    cum_bh    = _cum_return(port_bh)
    ax_eq.plot(cum_bh.index,    cum_bh.values,    color=BH_COLOR,       linewidth=1.2,
               label="Equal-weight B&H", alpha=0.8)
    ax_eq.plot(cum_strat.index, cum_strat.values, color=STRATEGY_COLOR, linewidth=2.0,
               label="HRP Portfolio Strategy")
    ax_eq.axhline(1.0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
    ax_eq.set_title("HRP Portfolio - Equity Curve", fontsize=10, fontweight="bold")
    ax_eq.set_ylabel("Growth of $1")
    ax_eq.legend(fontsize=8)
    ax_eq.grid(alpha=0.3)

    # Row 0 right: rolling 1-year Sharpe
    ax_rs = fig.add_subplot(gs[0, 3:])
    roll_sh = _rolling_sharpe(port_strat)
    ax_rs.fill_between(roll_sh.index, roll_sh.values, 0,
                       where=roll_sh.values >= 0,
                       color=STRATEGY_COLOR, alpha=0.4, label="Positive")
    ax_rs.fill_between(roll_sh.index, roll_sh.values, 0,
                       where=roll_sh.values < 0,
                       color=DRAWDOWN_COLOR, alpha=0.4, label="Negative")
    ax_rs.plot(roll_sh.index, roll_sh.values, color="#333333", linewidth=0.8)
    ax_rs.axhline(0, color="black", linewidth=0.7, alpha=0.5)
    ax_rs.axhline(1, color=STRATEGY_COLOR, linewidth=0.8, linestyle="--",
                  alpha=0.6, label="Sharpe = 1")
    ax_rs.set_title("Rolling 12-Month Portfolio Sharpe", fontsize=10, fontweight="bold")
    ax_rs.set_ylabel("Sharpe")
    ax_rs.legend(fontsize=7)
    ax_rs.grid(alpha=0.3)

    # Row 1: per-metric comparison bars
    COMP_METRICS = [
        ("sharpe",     "Sharpe",         False),
        ("sortino",    "Sortino",        False),
        ("calmar",     "Calmar",         False),
        ("ann_return", "Ann. Return",    True),
        ("max_dd",     "Max DD",         True),
        ("win_rate",   "Win Rate",       True),
    ]

    # Build data table: portfolio + each instrument
    all_labels = ["Portfolio"] + list(results.keys())
    all_metrics: list[dict[str, float]] = [port_metrics] + [r.metrics for r in results.values()]

    for col_idx, (key, label, is_pct) in enumerate(COMP_METRICS):
        ax = fig.add_subplot(gs[1, col_idx])

        vals   = [m.get(key, np.nan) for m in all_metrics]
        colors = []
        for i, (lbl, val) in enumerate(zip(all_labels, vals)):
            if lbl == "Portfolio":
                colors.append("#2c3e50")          # dark navy for portfolio
            elif np.isnan(val) or val < 0:
                colors.append(DRAWDOWN_COLOR)
            else:
                colors.append(STRATEGY_COLOR)

        bars = ax.barh(all_labels, vals, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.axvline(0, color="black", linewidth=0.6, alpha=0.5)
        ax.grid(axis="x", alpha=0.3)
        ax.tick_params(axis="y", labelsize=7)

        # Format x-axis
        if is_pct:
            ax.xaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:.0%}")
            )

        # Annotate
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                fmt = f"{val:.1%}" if is_pct else f"{val:.2f}"
                ax.text(
                    bar.get_width() + abs(bar.get_width()) * 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    fmt, va="center", fontsize=6.5,
                )

        # Highlight portfolio bar with a border
        bars[0].set_edgecolor("#2c3e50")
        bars[0].set_linewidth(1.5)

    # Row 2: scorecard text table
    ax_tbl = fig.add_subplot(gs[2, :])
    ax_tbl.axis("off")

    pm = port_metrics
    col_names = ["Sharpe", "Sortino", "Calmar",
                 "Ann. Return", "Ann. Vol", "Max DD", "Win Rate"]
    col_vals  = [
        f"{pm.get('sharpe',    np.nan):.2f}",
        f"{pm.get('sortino',   np.nan):.2f}",
        f"{pm.get('calmar',    np.nan):.2f}",
        f"{pm.get('ann_return',np.nan):.2%}",
        f"{pm.get('ann_vol',   np.nan):.2%}",
        f"{pm.get('max_dd',    np.nan):.2%}",
        f"{pm.get('win_rate',  np.nan):.2%}",
    ]
    n_active = sum(1 for t in results if weights.get(t, 0.0) > 0)
    active_str = ", ".join(
        f"{t} ({weights.get(t,0):.1%})" for t in results if weights.get(t, 0.0) > 0
    )

    tbl = ax_tbl.table(
        cellText=[col_vals],
        colLabels=col_names,
        cellLoc="center",
        loc="center",
        bbox=[0, 0.3, 1, 0.55],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#ecf0f1")

    ax_tbl.text(
        0.5, 0.05,
        f"Portfolio composition ({n_active} active instruments):  {active_str}",
        ha="center", va="center", fontsize=8, style="italic",
        transform=ax_tbl.transAxes,
    )

    fig.suptitle("Combined Portfolio - Performance Dashboard",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()

    if save_dir:
        out = save_dir / "portfolio_summary.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        log.info("Saved %s", out)
    return fig


#  CLI

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="plot_results",
        description="Run backtest and plot results.",
    )
    p.add_argument("--mode", choices=["classical", "fno"], default="fno")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--tickers", nargs="+", default=None, metavar="TICKER")
    p.add_argument("--start", default=None, metavar="YYYY-MM-DD")
    p.add_argument("--end", default=None, metavar="YYYY-MM-DD")
    p.add_argument("--cost-bps", type=float, default=None)
    p.add_argument("--no-save", action="store_true",
                   help="Show plots interactively instead of saving to disk.")
    return p


def main() -> None:
    args = build_parser().parse_args()

    # Resolve checkpoint for FNO mode
    checkpoint = args.checkpoint
    if args.mode == "fno" and checkpoint is None:
        default = config.CHECKPOINTS_DIR / "fno_best.pt"
        if default.exists():
            checkpoint = default
        else:
            log.error(
                "No FNO checkpoint found at %s.\n"
                "Pass --checkpoint <path> or use --mode classical.", default
            )
            sys.exit(1)

    # Run pipeline
    from wsq_trading.pipeline import Pipeline

    log.info("Running pipeline: mode=%s", args.mode)
    pipe = Pipeline(mode=args.mode, cost_bps=args.cost_bps,
                    fno_checkpoint=checkpoint)
    results, weights, summary = pipe.run_and_summarise(
        tickers=args.tickers, start=args.start, end=args.end
    )

    if not results:
        log.error("No backtest results produced.")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("  BACKTEST SUMMARY")
    print("=" * 70)
    print(summary.to_string())
    print("=" * 70 + "\n")

    # Output directory
    save_dir: Path | None = None
    if not args.no_save:
        save_dir = config.PLOTS_DIR
        save_dir.mkdir(parents=True, exist_ok=True)
        log.info("Saving plots to %s", save_dir)

    # Generate figures
    plot_equity_curves(results, weights, save_dir)
    plot_hurst_regimes(results, save_dir)
    plot_drawdowns(results, save_dir)
    plot_positions(results, save_dir)
    plot_regime_breakdown(results, save_dir)
    plot_metrics_summary(results, weights, save_dir)
    plot_portfolio_summary(results, weights, save_dir)

    if args.no_save:
        matplotlib.use("TkAgg")
        plt.show()
    else:
        print(f"All plots saved to:  {save_dir}/")
        print("  equity_curves.png")
        print("  hurst_regimes.png")
        print("  drawdowns.png")
        print("  positions.png")
        print("  regime_breakdown.png")
        print("  metrics_summary.png")
        print("  portfolio_summary.png")


if __name__ == "__main__":
    main()
