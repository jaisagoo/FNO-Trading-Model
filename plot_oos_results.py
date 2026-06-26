"""Out-of-sample result plots for the rough-Heston regime-switching strategy.

This script renders the figures an experienced quant would expect to see when
judging a strategy on a *true* out-of-sample (OOS) window, addressing the five
points raised in ``OPUS_EXECUTION_PROMPT.md``:

  1. Look-ahead control  -> equity curve with the train/test boundary marked.
  2. True OOS report      -> OOS-only equity, drawdown, rolling Sharpe.
  3. Portfolio metrics    -> per-instrument metric bars, alpha/beta scatter.
  4. Regime attribution   -> P&L, Sharpe and time-in-regime by Hurst regime.
  5. Cost sensitivity     -> Sharpe / return decay versus transaction cost.

Plus standard diagnostics: HRP weights, Hurst path with regime shading, OOS
strategy-return correlation, and a monthly-return heatmap.

Everything is CPU-only and needs no torch.  By default it pulls fresh data via
``Pipeline``; pass ``--offline`` to reuse the parquet cache without any network
access (handy on a locked-down box or for reproducible figures).

Usage
-----
    python plot_oos_results.py                      # default tickers, live data
    python plot_oos_results.py --offline            # use cached parquet only
    python plot_oos_results.py --tickers ES=F NQ=F CL=F --offline
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / server-safe
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from wsq_trading import config  # noqa: E402
from wsq_trading.backtest import (  # noqa: E402
    BacktestResult,
    annualized_return,
    compute_all,
    compute_portfolio_metrics,
    cost_sensitivity,
    max_drawdown,
    regime_attribution,
    sharpe_ratio,
)
from wsq_trading.constants import TRADING_DAYS_PER_YEAR, Regime  # noqa: E402
from wsq_trading.pipeline import OOSReport, Pipeline  # noqa: E402
from wsq_trading.utils import get_logger  # noqa: E402

log = get_logger(__name__)

# Consistent regime colours throughout
REGIME_COLOURS = {
    "MEAN_REVERT": "#2c7fb8",  # blue
    "NEUTRAL": "#969696",      # grey
    "MOMENTUM": "#d95f0e",     # orange
}
PORT_COLOUR = "#1a1a1a"
BENCH_COLOUR = "#7a7a7a"


# Data assembly

def _portfolio_returns(
    results: dict[str, BacktestResult], weights: pd.Series
) -> pd.Series:
    """Weighted daily portfolio return over the index spanned by *results*."""
    df = pd.DataFrame({t: r.strategy_returns for t, r in results.items()})
    w = weights.reindex(df.columns).fillna(0.0)
    return df.mul(w, axis=1).sum(axis=1)


def _equity(returns: pd.Series) -> pd.Series:
    """Cumulative growth curve (start = 1.0) from daily arithmetic returns."""
    return (1.0 + returns.fillna(0.0)).cumprod()


def _drawdown(returns: pd.Series) -> pd.Series:
    """Drawdown path (<= 0) from daily arithmetic returns."""
    eq = _equity(returns)
    return eq / eq.cummax() - 1.0


def build_context(
    tickers: list[str], start: str | None, end: str | None, mode: str
) -> dict:
    """Run the pipeline once and assemble everything the figures need.

    Returns a dict with full-history results, the OOS report, the train/test
    split dates, HRP weights, the benchmark series, and the OOS-sliced results.
    """
    p = Pipeline(mode=mode)
    full = p.run(tickers=tickers, start=start, end=end)
    if not full:
        raise SystemExit("No data returned - try --offline or check tickers.")

    train_end = p._date_at_fraction(full, config.TRAIN_SPLIT + config.VAL_SPLIT)
    all_dates = sorted(set().union(*[r.strategy_returns.index for r in full.values()]))
    test_start = pd.Timestamp(all_dates[all_dates.index(pd.Timestamp(train_end)) + 1])
    test_end = pd.Timestamp(all_dates[-1])

    train_results = {t: p._slice_result(r, None, train_end) for t, r in full.items()}
    weights = p.allocate(train_results, train_end_date=train_end)
    oos_results = {t: p._slice_result(r, test_start, test_end) for t, r in full.items()}

    bench = full["ES=F"].gross_returns if "ES=F" in full else None
    bench_oos = bench.loc[test_start:test_end] if bench is not None else None

    # Build the OOS report from the single engine pass above (no re-run).
    per_rows = {t: compute_all(r.strategy_returns, positions=r.positions)
                for t, r in oos_results.items()}
    per_instrument = pd.DataFrame(per_rows).T
    per_instrument["hrp_weight"] = weights.reindex(per_instrument.index)
    port_metrics = compute_portfolio_metrics(
        oos_results, weights, benchmark_returns=bench_oos, cost_bps=p.cost_bps)
    report = OOSReport(
        train_end=train_end, test_start=test_start, test_end=test_end,
        per_instrument=per_instrument, portfolio=asdict(port_metrics),
        hrp_weights=weights,
        regime_attribution=regime_attribution(oos_results, weights),
        cost_sensitivity=cost_sensitivity(oos_results, weights),
    )
    return {
        "pipeline": p,
        "full": full,
        "oos": oos_results,
        "report": report,
        "weights": weights,
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
        "benchmark": bench,
    }


# Figures

def fig_equity_curve(ctx: dict, outdir: Path) -> Path:
    """Full-history portfolio equity with the OOS test window shaded."""
    full, w = ctx["full"], ctx["weights"]
    port = _portfolio_returns(full, w)
    eq = _equity(port)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(eq.index, eq.values, color=PORT_COLOUR, lw=1.6, label="HRP portfolio")

    if ctx["benchmark"] is not None:
        beq = _equity(ctx["benchmark"].reindex(eq.index).fillna(0.0))
        ax.plot(beq.index, beq.values, color=BENCH_COLOUR, lw=1.2, ls="--",
                label="ES=F (benchmark)")

    ax.axvspan(ctx["test_start"], ctx["test_end"], color="#fde0dd", alpha=0.6,
               label="OOS test window")
    ax.axvline(ctx["train_end"], color="#c51b8a", lw=1.2, ls=":")
    ax.text(ctx["train_end"], ax.get_ylim()[1], "  train/val | test",
            color="#c51b8a", va="top", fontsize=8)

    ax.set_title("Equity curve - full history with out-of-sample window shaded",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Growth of $1")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    return _save(fig, outdir, "oos_01_equity_curve")


def fig_oos_equity_drawdown(ctx: dict, outdir: Path) -> Path:
    """OOS-only equity curve and the underwater (drawdown) plot."""
    oos, w = ctx["oos"], ctx["weights"]
    port = _portfolio_returns(oos, w).loc[ctx["test_start"]:ctx["test_end"]]
    eq = _equity(port)
    dd = _drawdown(port)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(eq.index, eq.values, color=PORT_COLOUR, lw=1.6, label="HRP portfolio")
    if ctx["benchmark"] is not None:
        b = ctx["benchmark"].reindex(port.index).fillna(0.0)
        ax1.plot(_equity(b).index, _equity(b).values, color=BENCH_COLOUR, lw=1.2,
                 ls="--", label="ES=F")
    ax1.set_title("Out-of-sample portfolio - equity & drawdown",
                  fontsize=12, fontweight="bold")
    ax1.set_ylabel("Growth of $1")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.25)

    ax2.fill_between(dd.index, dd.values * 100, 0, color="#cb181d", alpha=0.5)
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(alpha=0.25)
    ax2.text(0.01, 0.08, f"Max DD: {max_drawdown(port) * 100:.1f}%",
             transform=ax2.transAxes, fontsize=9, color="#cb181d")
    return _save(fig, outdir, "oos_02_equity_drawdown")


def fig_rolling_sharpe(ctx: dict, outdir: Path) -> Path:
    """Rolling annualised Sharpe across the OOS window."""
    oos, w = ctx["oos"], ctx["weights"]
    port = _portfolio_returns(oos, w).loc[ctx["test_start"]:ctx["test_end"]].dropna()
    win = min(63, max(21, len(port) // 4))
    roll = (port.rolling(win).mean() / port.rolling(win).std()) * np.sqrt(
        TRADING_DAYS_PER_YEAR)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(roll.index, roll.values, color="#238b45", lw=1.4)
    ax.axhline(0, color="k", lw=0.8)
    full_sharpe = sharpe_ratio(port)
    ax.axhline(full_sharpe, color="#c51b8a", lw=1.0, ls="--",
               label=f"OOS Sharpe = {full_sharpe:.2f}")
    ax.set_title(f"Rolling {win}-day annualised Sharpe (OOS)",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Sharpe")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)
    return _save(fig, outdir, "oos_03_rolling_sharpe")


def fig_per_instrument_metrics(ctx: dict, outdir: Path) -> Path:
    """Per-instrument OOS Sharpe, annual return, max drawdown + HRP weight."""
    per = ctx["report"].per_instrument.copy()
    tickers = list(per.index)
    x = np.arange(len(tickers))

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.5))
    specs = [
        ("sharpe", "OOS Sharpe", "#2c7fb8"),
        ("annualized_return", "OOS Annual Return", "#238b45"),
        ("max_drawdown", "OOS Max Drawdown", "#cb181d"),
        ("hrp_weight", "HRP Weight (train)", "#756bb1"),
    ]
    for ax, (col, label, colour) in zip(axes, specs):
        vals = per[col].reindex(tickers).astype(float)
        disp = vals * 100 if col in ("annualized_return", "max_drawdown",
                                      "hrp_weight") else vals
        ax.bar(x, disp.values, color=colour, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(tickers, rotation=45, ha="right", fontsize=8)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.axhline(0, color="k", lw=0.6)
        ax.grid(alpha=0.2, axis="y")
        unit = "%" if col != "sharpe" else ""
        for xi, v in zip(x, disp.values):
            if np.isfinite(v):
                ax.text(xi, v, f"{v:.1f}{unit}", ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=7)
    fig.suptitle("Per-instrument out-of-sample performance",
                 fontsize=12, fontweight="bold", y=1.02)
    return _save(fig, outdir, "oos_04_per_instrument_metrics")


def fig_hrp_weights(ctx: dict, outdir: Path) -> Path:
    """HRP weights computed from the train window (gated names show as 0)."""
    w = ctx["weights"].sort_values(ascending=False)
    colours = ["#756bb1" if v > 0 else "#d9d9d9" for v in w.values]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(np.arange(len(w)), w.values * 100, color=colours)
    ax.set_xticks(np.arange(len(w)))
    ax.set_xticklabels(w.index, rotation=45, ha="right")
    ax.set_ylabel("Weight (%)")
    ax.set_title("HRP weights (train+val window only - no look-ahead)",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.2, axis="y")
    for xi, v in enumerate(w.values):
        ax.text(xi, v * 100, f"{v * 100:.0f}%", ha="center", va="bottom",
                fontsize=8)
    gated = [t for t, v in w.items() if v == 0]
    if gated:
        ax.text(0.99, 0.95, "Gated out: " + ", ".join(gated),
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                color="#636363")
    return _save(fig, outdir, "oos_05_hrp_weights")


def fig_regime_attribution(ctx: dict, outdir: Path) -> Path:
    """Annualised return, Sharpe and time-in-regime by Hurst regime."""
    ra = ctx["report"].regime_attribution.reset_index()
    order = ["MEAN_REVERT", "NEUTRAL", "MOMENTUM"]
    tickers = list(dict.fromkeys(ra["ticker"]))
    x = np.arange(len(tickers))
    width = 0.26

    def _pivot(col: str) -> pd.DataFrame:
        return (ra.pivot(index="ticker", columns="regime", values=col)
                .reindex(index=tickers, columns=order))

    ann = _pivot("ann_return")
    shp = _pivot("sharpe")
    frac = _pivot("frac_bars").fillna(0.0)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for i, reg in enumerate(order):
        axes[0].bar(x + (i - 1) * width, ann[reg].values * 100, width,
                    label=reg, color=REGIME_COLOURS[reg])
        axes[1].bar(x + (i - 1) * width, shp[reg].values, width,
                    label=reg, color=REGIME_COLOURS[reg])
    axes[0].set_title("Annualised return by regime (%)", fontsize=10, fontweight="bold")
    axes[1].set_title("Sharpe by regime", fontsize=10, fontweight="bold")
    for ax in axes[:2]:
        ax.set_xticks(x)
        ax.set_xticklabels(tickers, rotation=45, ha="right", fontsize=8)
        ax.axhline(0, color="k", lw=0.6)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2, axis="y")

    bottom = np.zeros(len(tickers))
    for reg in order:
        axes[2].bar(x, frac[reg].values, 0.6, bottom=bottom, label=reg,
                    color=REGIME_COLOURS[reg])
        bottom += frac[reg].values
    axes[2].set_title("Fraction of bars in each regime", fontsize=10, fontweight="bold")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(tickers, rotation=45, ha="right", fontsize=8)
    axes[2].set_ylim(0, 1)
    axes[2].legend(fontsize=7)
    fig.suptitle("Regime attribution - does the Hurst thesis pay off where it should?",
                 fontsize=12, fontweight="bold", y=1.03)
    return _save(fig, outdir, "oos_06_regime_attribution")


def fig_cost_sensitivity(ctx: dict, outdir: Path) -> Path:
    """Portfolio Sharpe & annual return versus round-trip cost, with per-ticker."""
    cs = ctx["report"].cost_sensitivity
    fig, ax1 = plt.subplots(figsize=(10, 5.5))

    for col in [c for c in cs.columns if c.startswith("sharpe_")]:
        ax1.plot(cs.index, cs[col], color="#bdbdbd", lw=1.0, alpha=0.8)
        ax1.text(cs.index[-1], cs[col].iloc[-1], " " + col.replace("sharpe_", ""),
                 fontsize=7, color="#969696", va="center")
    ax1.plot(cs.index, cs["portfolio_sharpe"], color="#2c7fb8", lw=2.4, marker="o",
             label="Portfolio Sharpe")
    ax1.set_xlabel("Round-trip transaction cost (bps)")
    ax1.set_ylabel("Sharpe", color="#2c7fb8")
    ax1.tick_params(axis="y", labelcolor="#2c7fb8")
    ax1.axhline(0, color="k", lw=0.6)
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(cs.index, cs["portfolio_ann_return"] * 100, color="#238b45", lw=2.0,
             marker="s", ls="--", label="Portfolio ann. return")
    ax2.set_ylabel("Annual return (%)", color="#238b45")
    ax2.tick_params(axis="y", labelcolor="#238b45")

    base = config.TRANSACTION_COST_BPS
    ax1.axvline(base, color="#c51b8a", lw=1.0, ls=":")
    ax1.text(base, ax1.get_ylim()[1], f" assumed {base:.0f} bps",
             color="#c51b8a", fontsize=8, va="top")
    ax1.set_title("Transaction-cost sensitivity (grey = individual instruments)",
                  fontsize=12, fontweight="bold")
    return _save(fig, outdir, "oos_07_cost_sensitivity")


def fig_alpha_beta(ctx: dict, outdir: Path) -> Path:
    """Scatter of OOS portfolio returns vs benchmark with the CAPM fit."""
    if ctx["benchmark"] is None:
        return Path()
    oos, w = ctx["oos"], ctx["weights"]
    port = _portfolio_returns(oos, w).loc[ctx["test_start"]:ctx["test_end"]]
    bench = ctx["benchmark"].reindex(port.index)
    d = pd.concat([port, bench], axis=1, keys=["port", "bench"]).dropna()
    if len(d) < 10:
        return Path()

    X = np.column_stack([np.ones(len(d)), d["bench"].to_numpy()])
    alpha_d, beta = np.linalg.lstsq(X, d["port"].to_numpy(), rcond=None)[0]
    xs = np.linspace(d["bench"].min(), d["bench"].max(), 50)

    pm = ctx["report"].portfolio
    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.scatter(d["bench"] * 100, d["port"] * 100, s=12, alpha=0.4, color="#2c7fb8")
    ax.plot(xs * 100, (alpha_d + beta * xs) * 100, color="#cb181d", lw=2,
            label=f"fit: beta={beta:.3f}")
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("ES=F daily return (%)")
    ax.set_ylabel("Portfolio daily return (%)")
    ax.set_title("CAPM fit - portfolio vs benchmark (OOS)",
                 fontsize=12, fontweight="bold")
    txt = (f"Annualised alpha: {pm['alpha'] * 100:.2f}%\n"
           f"Beta: {pm['beta']:.3f}\n"
           f"Information ratio: {pm['information_ratio']:.2f}")
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="#fff7bc", ec="#cccccc"))
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.25)
    return _save(fig, outdir, "oos_08_alpha_beta_scatter")


def fig_hurst_regimes(ctx: dict, outdir: Path) -> Path:
    """Hurst path with regime shading over the OOS window (weighted names)."""
    full, w = ctx["full"], ctx["weights"]
    names = [t for t in full if w.get(t, 0.0) > 0] or list(full)[:3]
    names = names[:4]
    fig, axes = plt.subplots(len(names), 1, figsize=(13, 2.6 * len(names)),
                             squeeze=False, sharex=True)
    ts, te = ctx["test_start"], ctx["test_end"]
    for ax, t in zip(axes[:, 0], names):
        res = full[t]
        h = res.H_estimates.loc[ts:te]
        reg = res.regimes.loc[ts:te]
        ax.plot(h.index, h.values, color="#252525", lw=1.0)
        ax.axhline(config.HURST_ROUGH_THRESHOLD, color=REGIME_COLOURS["MEAN_REVERT"],
                   lw=0.8, ls="--")
        ax.axhline(config.HURST_TREND_THRESHOLD, color=REGIME_COLOURS["MOMENTUM"],
                   lw=0.8, ls="--")
        for regime in Regime:
            mask = (reg == regime).reindex(h.index).fillna(False).to_numpy()
            ax.fill_between(h.index, 0, 1, where=mask, transform=ax.get_xaxis_transform(),
                            color=REGIME_COLOURS[regime.name], alpha=0.12)
        ax.set_ylim(0, 1)
        ax.set_ylabel("H(t)")
        ax.set_title(f"{t}  (weight {w.get(t, 0.0) * 100:.0f}%)", fontsize=9,
                     loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=REGIME_COLOURS[r.name], alpha=0.3)
               for r in Regime]
    axes[0, 0].legend(handles, [r.name for r in Regime], ncol=3, fontsize=7,
                      loc="upper right")
    fig.suptitle("Estimated Hurst exponent with regime shading (OOS)",
                 fontsize=12, fontweight="bold", y=1.0)
    return _save(fig, outdir, "oos_09_hurst_regimes")


def fig_return_correlation(ctx: dict, outdir: Path) -> Path:
    """Heatmap of OOS strategy-return correlations across instruments."""
    oos = ctx["oos"]
    df = pd.DataFrame({t: r.strategy_returns for t, r in oos.items()}).dropna(how="all")
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.index, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="k")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("OOS strategy-return correlation", fontsize=12, fontweight="bold")
    return _save(fig, outdir, "oos_10_return_correlation")


def fig_monthly_heatmap(ctx: dict, outdir: Path) -> Path:
    """Calendar heatmap of OOS monthly portfolio returns."""
    oos, w = ctx["oos"], ctx["weights"]
    port = _portfolio_returns(oos, w).loc[ctx["test_start"]:ctx["test_end"]].dropna()
    monthly = (1 + port).resample("ME").prod() - 1
    tbl = monthly.to_frame("r")
    tbl["year"] = tbl.index.year
    tbl["month"] = tbl.index.month
    grid = tbl.pivot_table(index="year", columns="month", values="r")
    grid = grid.reindex(columns=range(1, 13))

    fig, ax = plt.subplots(figsize=(11, 1.2 + 0.6 * len(grid)))
    vmax = np.nanmax(np.abs(grid.values)) or 0.01
    im = ax.imshow(grid.values * 100, cmap="RdYlGn", vmin=-vmax * 100, vmax=vmax * 100,
                   aspect="auto")
    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug",
                        "Sep", "Oct", "Nov", "Dec"], fontsize=8)
    ax.set_yticks(range(len(grid)))
    ax.set_yticklabels(grid.index, fontsize=8)
    for i in range(len(grid)):
        for j in range(12):
            v = grid.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v * 100:.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Monthly return (%)")
    ax.set_title("OOS monthly portfolio returns (%)", fontsize=12, fontweight="bold")
    return _save(fig, outdir, "oos_11_monthly_returns")


# IO helpers + entry point

def _save(fig, outdir: Path, name: str) -> Path:
    """Save *fig* as a 150-dpi PNG into *outdir* and close it."""
    fig.tight_layout()
    out = outdir / f"{name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("saved %s", out)
    return out


def main() -> None:
    """Parse arguments, run the pipeline, and render every OOS figure."""
    ap = argparse.ArgumentParser(description="Plot out-of-sample strategy results.")
    ap.add_argument("--tickers", nargs="+", default=None,
                    help="Tickers (default: config.TICKERS).")
    ap.add_argument("--mode", choices=["classical", "fno"], default="classical")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--offline", action="store_true",
                    help="Use the parquet cache only (no network).")
    ap.add_argument("--outdir", default=str(config.PLOTS_DIR))
    args = ap.parse_args()

    if args.offline:
        config.CACHE_MAX_AGE_HOURS = 1e12  # treat any cached parquet as fresh
        log.info("offline mode: using parquet cache only")

    tickers = args.tickers or config.TICKERS
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    mirror = config.BASE_DIR / "plotting" / "plot_images"
    mirror.mkdir(parents=True, exist_ok=True)

    ctx = build_context(tickers, args.start, args.end, args.mode)
    ctx["pipeline"].print_oos_report(ctx["report"])

    figs = [
        fig_equity_curve, fig_oos_equity_drawdown, fig_rolling_sharpe,
        fig_per_instrument_metrics, fig_hrp_weights, fig_regime_attribution,
        fig_cost_sensitivity, fig_alpha_beta, fig_hurst_regimes,
        fig_return_correlation, fig_monthly_heatmap,
    ]
    written = []
    for fn in figs:
        try:
            p = fn(ctx, outdir)
            if p and p.exists():
                written.append(p)
                (mirror / p.name).write_bytes(p.read_bytes())
        except Exception as exc:  # keep going; one bad figure shouldn't abort
            log.warning("figure %s failed: %s", fn.__name__, exc)

    log.info("wrote %d figures to %s (mirrored to %s)", len(written), outdir, mirror)
    for p in written:
        log.info("  %s", p.name)


if __name__ == "__main__":
    main()
