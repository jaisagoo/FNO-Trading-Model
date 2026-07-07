"""Persist out-of-sample evaluation results to disk.

Writes the :class:`~wsq_trading.pipeline.OOSReport` produced by
``Pipeline.run_oos()`` to two places:

- one CSV per table under ``results/metrics/`` (machine-readable), and
- a dated, human-readable markdown report under ``results/backtest_reports/``.

Every figure in the markdown file is rendered directly from the in-memory
report object, so the ``.md`` and ``.csv`` outputs always agree and any number
quoted elsewhere (e.g. the README results section) can be traced back to the
saved artefacts of a single run.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from wsq_trading import config
from wsq_trading.utils import get_logger

if TYPE_CHECKING:  # avoid a circular import at runtime
    from wsq_trading.pipeline import OOSReport

log = get_logger(__name__)

# Portfolio metrics rendered in the markdown summary, in display order:
# (key in OOSReport.portfolio, label, format spec)
_PORTFOLIO_ROWS: list[tuple[str, str, str]] = [
    ("sharpe", "Sharpe", "{:.2f}"),
    ("sortino", "Sortino", "{:.2f}"),
    ("annualized_return", "Annual return", "{:.1%}"),
    ("annualized_vol", "Annual vol", "{:.1%}"),
    ("max_drawdown", "Max drawdown", "{:.1%}"),
    ("calmar", "Calmar", "{:.2f}"),
    ("win_rate", "Win rate", "{:.1%}"),
    ("turnover", "Turnover (ann.)", "{:.2f}x"),
    ("total_cost_bps", "Cost assumption", "{:.1f} bps"),
    ("alpha", "Alpha vs benchmark", "{:.1%}"),
    ("beta", "Beta vs benchmark", "{:.2f}"),
    ("information_ratio", "Information ratio", "{:.2f}"),
    ("n_obs", "Test observations", "{:.0f}"),
]


def portfolio_summary_frame(report: "OOSReport") -> pd.DataFrame:
    """Return the portfolio-level metrics of *report* as a one-row DataFrame.

    Parameters
    ----------
    report : OOSReport
        Report produced by ``Pipeline.run_oos()``.

    Returns
    -------
    pd.DataFrame
        Single row indexed ``'PORTFOLIO'`` with one column per metric.
    """
    return pd.DataFrame([report.portfolio], index=["PORTFOLIO"])


def dataframe_to_markdown(df: pd.DataFrame, float_format: str = "{:.4f}") -> str:
    """Render *df* as a GitHub-flavoured markdown pipe table.

    Implemented locally so the package does not need ``tabulate``.

    Parameters
    ----------
    df : pd.DataFrame
        Table to render.  MultiIndex rows are flattened with ``' / '``.
    float_format : str
        ``str.format`` spec applied to float cells.

    Returns
    -------
    str
        Markdown table, no trailing newline.
    """
    flat = df.reset_index()

    def _cell(v: Any) -> str:
        if isinstance(v, float):
            return "" if pd.isna(v) else float_format.format(v)
        return str(v)

    header = [str(c) if not isinstance(c, tuple) else " / ".join(map(str, c)) for c in flat.columns]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    for _, row in flat.iterrows():
        lines.append("| " + " | ".join(_cell(v) for v in row) + " |")
    return "\n".join(lines)


def readme_results_section(report: "OOSReport", run_config: dict[str, Any] | None = None) -> str:
    """Build a ready-to-paste ``## Results`` section for the README.

    Parameters
    ----------
    report : OOSReport
        Report produced by ``Pipeline.run_oos()``.
    run_config : dict, optional
        Run settings (tickers, mode, dates, ...) to cite alongside the numbers.

    Returns
    -------
    str
        Markdown text for the README, including the OOS window and headline
        portfolio metrics.
    """
    p = report.portfolio
    rc = run_config or {}
    lines = [
        "## Results",
        "",
        f"True out-of-sample test window **{report.test_start.date()} to "
        f"{report.test_end.date()}** (trained/calibrated on data up to "
        f"{report.train_end.date()} only; thresholds, universe and filters fixed "
        "before the test window was evaluated).",
        "",
        "| Metric | OOS value |",
        "|---|---|",
    ]
    for key, label, fmt in _PORTFOLIO_ROWS:
        val = p.get(key)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        lines.append(f"| {label} | {fmt.format(val)} |")
    if rc:
        cited = ", ".join(f"{k}={v}" for k, v in rc.items())
        lines += ["", f"Run settings: {cited}."]
    lines += [
        "",
        "Full tables (per-instrument metrics, regime attribution, cost sensitivity,",
        "HRP weights) are saved per run under `results/metrics/` and",
        "`results/backtest_reports/`.",
    ]
    return "\n".join(lines)


def save_oos_report(
    report: "OOSReport",
    metrics_dir: Path | str | None = None,
    reports_dir: Path | str | None = None,
    label: str = "oos",
    run_config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Save *report* as CSV tables plus a dated markdown report.

    Parameters
    ----------
    report : OOSReport
        Report produced by ``Pipeline.run_oos()``.
    metrics_dir : Path or str, optional
        Directory for the CSV tables.  Default: ``config.METRICS_DIR``.
    reports_dir : Path or str, optional
        Directory for the markdown report.  Default: ``config.REPORTS_DIR``.
    label : str
        Slug included in every filename, e.g. ``oos`` ->
        ``2026-07-02_oos_portfolio.csv``.
    run_config : dict, optional
        Run settings recorded in the markdown header so the report is
        self-describing (tickers, mode, cost assumption, date range, ...).

    Returns
    -------
    dict[str, Path]
        Mapping of table name -> written file path (plus ``'report_md'``).
    """
    metrics_dir = Path(metrics_dir if metrics_dir is not None else config.METRICS_DIR)
    reports_dir = Path(reports_dir if reports_dir is not None else config.REPORTS_DIR)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    stamp = dt.date.today().isoformat()
    tables: dict[str, pd.DataFrame] = {
        "portfolio": portfolio_summary_frame(report),
        "per_instrument": report.per_instrument,
        "hrp_weights": report.hrp_weights.rename("weight").to_frame(),
        "regime_attribution": report.regime_attribution,
        "cost_sensitivity": report.cost_sensitivity,
    }

    paths: dict[str, Path] = {}
    for name, df in tables.items():
        path = metrics_dir / f"{stamp}_{label}_{name}.csv"
        df.to_csv(path)
        paths[name] = path
        log.info("wrote %s", path)

    md_path = reports_dir / f"{stamp}_{label}_report.md"
    md_path.write_text(
        _render_markdown(report, tables, stamp, run_config), encoding="utf-8"
    )
    paths["report_md"] = md_path
    log.info("wrote %s", md_path)
    return paths


def append_ablation_section(
    ablation: pd.DataFrame,
    report_md_path: Path | str,
    metrics_dir: Path | str | None = None,
    label: str = "oos",
) -> Path:
    """Save an ablation table as CSV and append it to an existing report.

    Parameters
    ----------
    ablation : pd.DataFrame
        One row per variant (e.g. baseline / no_long_bias / ...), columns are
        portfolio metrics.
    report_md_path : Path or str
        Markdown report (from :func:`save_oos_report`) to append to.
    metrics_dir : Path or str, optional
        Directory for the CSV.  Default: ``config.METRICS_DIR``.
    label : str
        Filename slug, matching the parent report.

    Returns
    -------
    Path
        Path of the written CSV.
    """
    metrics_dir = Path(metrics_dir if metrics_dir is not None else config.METRICS_DIR)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.date.today().isoformat()

    csv_path = metrics_dir / f"{stamp}_{label}_ablation.csv"
    ablation.to_csv(csv_path)
    log.info("wrote %s", csv_path)

    section = (
        "\n## Overlay ablation\n\n"
        "Portfolio OOS metrics with the sizing/bias overlays toggled off one at a\n"
        "time.  This separates how much performance comes from H(t) regime\n"
        "switching itself versus the long-bias, vol-target and H-conviction\n"
        "overlays.\n\n"
        + dataframe_to_markdown(ablation)
        + "\n"
    )
    report_md_path = Path(report_md_path)
    report_md_path.write_text(
        report_md_path.read_text(encoding="utf-8") + section, encoding="utf-8"
    )
    log.info("appended ablation section to %s", report_md_path)
    return csv_path


def _render_markdown(
    report: "OOSReport",
    tables: dict[str, pd.DataFrame],
    stamp: str,
    run_config: dict[str, Any] | None,
) -> str:
    """Render the full markdown report body."""
    p = report.portfolio
    parts: list[str] = [
        f"# Out-of-sample report — {stamp}",
        "",
        f"- Train+val window: … → **{report.train_end.date()}**",
        f"- Test window: **{report.test_start.date()} → {report.test_end.date()}**",
    ]
    if run_config:
        parts.append("")
        parts.append("Run settings:")
        parts.append("")
        for k, v in run_config.items():
            parts.append(f"- {k}: `{v}`")

    parts += ["", "## Portfolio (OOS)", ""]
    for key, portfolio_label, fmt in _PORTFOLIO_ROWS:
        val = p.get(key)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        parts.append(f"- {portfolio_label}: **{fmt.format(val)}**")

    section_titles = {
        "per_instrument": "Per-instrument metrics (OOS)",
        "hrp_weights": "HRP weights (train window only)",
        "regime_attribution": "Regime attribution",
        "cost_sensitivity": "Cost sensitivity",
    }
    for name, title in section_titles.items():
        parts += ["", f"## {title}", "", dataframe_to_markdown(tables[name])]

    parts += [
        "",
        "## README results section",
        "",
        "Paste the block below into the README verbatim:",
        "",
        "```markdown",
        readme_results_section(report, run_config),
        "```",
        "",
    ]
    return "\n".join(parts)
