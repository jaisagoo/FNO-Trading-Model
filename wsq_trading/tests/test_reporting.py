"""Tests for wsq_trading.reporting (OOS report persistence)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wsq_trading.pipeline import OOSReport
from wsq_trading.reporting import (
    append_ablation_section,
    dataframe_to_markdown,
    portfolio_summary_frame,
    readme_results_section,
    save_oos_report,
)

# Fixture: a small, fully populated OOSReport


@pytest.fixture()
def report() -> OOSReport:
    per_instrument = pd.DataFrame(
        {
            "sharpe": [1.10, 0.85],
            "annualized_return": [0.12, 0.09],
            "max_drawdown": [0.08, 0.11],
            "hrp_weight": [0.6, 0.4],
        },
        index=["ES=F", "NQ=F"],
    )
    portfolio = {
        "sharpe": 1.23,
        "sortino": 1.80,
        "annualized_return": 0.115,
        "annualized_vol": 0.093,
        "max_drawdown": 0.076,
        "calmar": 1.51,
        "win_rate": 0.54,
        "turnover": 3.2,
        "total_cost_bps": 2.0,
        "alpha": 0.041,
        "beta": 0.35,
        "information_ratio": 0.62,
        "n_obs": 380,
    }
    hrp_weights = pd.Series({"ES=F": 0.6, "NQ=F": 0.4})
    regime_attribution = pd.DataFrame(
        {
            "frac_bars": [0.5, 0.3, 0.2],
            "ann_return": [0.10, 0.02, np.nan],
            "sharpe": [1.4, 0.3, np.nan],
            "n_obs": [190, 114, 4],
        },
        index=pd.MultiIndex.from_tuples(
            [("ES=F", "MOMENTUM"), ("ES=F", "NEUTRAL"), ("ES=F", "MEAN_REVERT")],
            names=["ticker", "regime"],
        ),
    )
    cost_sensitivity = pd.DataFrame(
        {
            "portfolio_sharpe": [1.30, 1.23, 1.10],
            "portfolio_ann_return": [0.12, 0.115, 0.10],
            "portfolio_max_drawdown": [0.075, 0.076, 0.08],
        },
        index=pd.Index([0, 2, 10], name="cost_bps"),
    )
    return OOSReport(
        train_end=pd.Timestamp("2023-06-30"),
        test_start=pd.Timestamp("2023-07-03"),
        test_end=pd.Timestamp("2024-12-31"),
        per_instrument=per_instrument,
        portfolio=portfolio,
        hrp_weights=hrp_weights,
        regime_attribution=regime_attribution,
        cost_sensitivity=cost_sensitivity,
    )


# portfolio_summary_frame


def test_portfolio_summary_frame_single_row(report):
    df = portfolio_summary_frame(report)
    assert list(df.index) == ["PORTFOLIO"]
    assert df.loc["PORTFOLIO", "sharpe"] == pytest.approx(1.23)
    assert "information_ratio" in df.columns


# dataframe_to_markdown


def test_dataframe_to_markdown_pipe_table():
    df = pd.DataFrame({"a": [1.23456, np.nan], "b": ["x", "y"]}, index=["r1", "r2"])
    md = dataframe_to_markdown(df, float_format="{:.2f}")
    lines = md.splitlines()
    assert lines[0].startswith("| ") and lines[0].endswith(" |")
    assert set(lines[1].replace("|", "")) <= {"-"}
    assert "1.23" in md
    # NaN renders as an empty cell, not the string 'nan'
    assert "nan" not in md


def test_dataframe_to_markdown_multiindex(report):
    md = dataframe_to_markdown(report.regime_attribution)
    assert "MOMENTUM" in md and "ticker" in md


# save_oos_report


def test_save_oos_report_writes_all_files(report, tmp_path):
    metrics = tmp_path / "metrics"
    reports = tmp_path / "reports"
    paths = save_oos_report(report, metrics_dir=metrics, reports_dir=reports, label="test")

    expected = {"portfolio", "per_instrument", "hrp_weights",
                "regime_attribution", "cost_sensitivity", "report_md"}
    assert set(paths) == expected
    for p in paths.values():
        assert p.exists() and p.stat().st_size > 0

    # CSV round-trip: the saved portfolio Sharpe is the report's Sharpe
    saved = pd.read_csv(paths["portfolio"], index_col=0)
    assert saved.loc["PORTFOLIO", "sharpe"] == pytest.approx(report.portfolio["sharpe"])


def test_save_oos_report_markdown_content(report, tmp_path):
    paths = save_oos_report(
        report,
        metrics_dir=tmp_path / "m",
        reports_dir=tmp_path / "r",
        label="test",
        run_config={"tickers": ["ES=F", "NQ=F"], "cost_bps": 2.0},
    )
    md = paths["report_md"].read_text(encoding="utf-8")
    assert "2023-07-03" in md and "2024-12-31" in md           # test window
    assert "1.23" in md                                        # Sharpe value
    assert "## Cost sensitivity" in md
    assert "## README results section" in md
    assert "cost_bps: `2.0`" in md                             # run config recorded


# readme_results_section


def test_readme_results_section(report):
    text = readme_results_section(report, run_config={"mode": "classical"})
    assert text.startswith("## Results")
    assert "| Sharpe | 1.23 |" in text
    assert "2023-07-03" in text
    assert "mode=classical" in text


# append_ablation_section


def test_append_ablation_section(report, tmp_path):
    paths = save_oos_report(
        report, metrics_dir=tmp_path / "m", reports_dir=tmp_path / "r", label="test"
    )
    ablation = pd.DataFrame(
        {"sharpe": [1.23, 0.9], "annualized_return": [0.115, 0.07]},
        index=pd.Index(["baseline", "signal_only"], name="variant"),
    )
    csv_path = append_ablation_section(
        ablation, report_md_path=paths["report_md"], metrics_dir=tmp_path / "m", label="test"
    )
    assert csv_path.exists()
    md = paths["report_md"].read_text(encoding="utf-8")
    assert "## Overlay ablation" in md
    assert "signal_only" in md
