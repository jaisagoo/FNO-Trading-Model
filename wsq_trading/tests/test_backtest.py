"""Tests for the walk-forward backtest engine and performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wsq_trading.backtest import (
    BacktestResult,
    WalkForwardEngine,
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    compute_all,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    turnover,
    win_rate,
)
from wsq_trading.constants import Regime

_WINDOW = 63  # default hurst_window


def _make_returns(n: int = 500, mu: float = 0.0, sigma: float = 0.01,
                  seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(rng.normal(mu, sigma, n), index=idx, name="log_return")


def _make_data(tickers: list[str], n: int = 500) -> dict[str, pd.DataFrame]:
    result = {}
    for i, ticker in enumerate(tickers):
        rng = np.random.default_rng(i)
        idx = pd.bdate_range("2020-01-01", periods=n)
        df = pd.DataFrame(
            {
                "log_return": rng.normal(0.0, 0.01, n),
                "close": 100 * np.cumprod(1 + rng.normal(0.0, 0.01, n)),
            },
            index=idx,
        )
        result[ticker] = df
    return result


# BacktestResult

class TestBacktestResult:
    def _make_result(self, n: int = 400) -> BacktestResult:
        engine = WalkForwardEngine()
        returns = _make_returns(n)
        return engine.run_single(returns)

    def test_has_expected_attributes(self) -> None:
        res = self._make_result()
        for attr in ("ticker", "strategy_returns", "positions", "H_estimates",
                     "regimes", "gross_returns", "metrics"):
            assert hasattr(res, attr), f"Missing attribute: {attr}"

    def test_sharpe_property_matches_metrics(self) -> None:
        res = self._make_result()
        assert res.sharpe == res.metrics.get("sharpe", float("nan")) or (
            np.isnan(res.sharpe) and np.isnan(res.metrics.get("sharpe", float("nan")))
        )

    def test_max_drawdown_property_matches_metrics(self) -> None:
        res = self._make_result()
        assert res.max_drawdown == res.metrics.get("max_drawdown", float("nan")) or (
            np.isnan(res.max_drawdown)
            and np.isnan(res.metrics.get("max_drawdown", float("nan")))
        )

    def test_metrics_auto_computed(self) -> None:
        """__post_init__ should populate metrics automatically."""
        res = self._make_result()
        assert isinstance(res.metrics, dict)
        assert len(res.metrics) > 0

    def test_metrics_contains_core_keys(self) -> None:
        res = self._make_result()
        for key in ("sharpe", "max_drawdown", "win_rate", "annualized_return"):
            assert key in res.metrics, f"Missing metric: {key}"

    def test_metrics_contains_turnover(self) -> None:
        res = self._make_result()
        assert "turnover" in res.metrics


# WalkForwardEngine.run_single

class TestRunSingle:
    def setup_method(self) -> None:
        self.engine = WalkForwardEngine()

    def test_returns_backtest_result(self) -> None:
        r = _make_returns(400)
        res = self.engine.run_single(r)
        assert isinstance(res, BacktestResult)

    def test_ticker_label_preserved(self) -> None:
        r = _make_returns(400)
        res = self.engine.run_single(r, ticker="AAPL")
        assert res.ticker == "AAPL"

    def test_strategy_returns_length_matches_input(self) -> None:
        r = _make_returns(400)
        res = self.engine.run_single(r)
        assert len(res.strategy_returns) == len(r)

    def test_positions_length_matches_input(self) -> None:
        r = _make_returns(400)
        res = self.engine.run_single(r)
        assert len(res.positions) == len(r)

    def test_positions_values_in_range_minus1_to_1(self) -> None:
        """Positions may be continuous (vol-targeting / H-conviction) but bounded."""
        r = _make_returns(400)
        res = self.engine.run_single(r)
        assert (res.positions >= -1.0 - 1e-9).all() and (res.positions <= 1.0 + 1e-9).all()

    def test_h_estimates_length_matches_input(self) -> None:
        r = _make_returns(400)
        res = self.engine.run_single(r)
        assert len(res.H_estimates) == len(r)

    def test_strategy_returns_index_matches_input(self) -> None:
        r = _make_returns(400)
        res = self.engine.run_single(r)
        assert res.strategy_returns.index.equals(r.index)

    def test_gross_returns_equals_input(self) -> None:
        r = _make_returns(400)
        res = self.engine.run_single(r)
        pd.testing.assert_series_equal(res.gross_returns, r)

    def test_strategy_returns_net_of_costs(self) -> None:
        """With non-zero costs, net returns <= gross strategy returns in magnitude."""
        engine_costly = WalkForwardEngine(cost_bps=100)  # 100 bps is large
        r = _make_returns(400)
        res = engine_costly.run_single(r)
        gross_strat = res.positions * r
        # costs can only decrease net returns vs. gross strategy
        diff = gross_strat - res.strategy_returns
        # diff should be >= 0 everywhere (costs subtracted)
        assert (diff >= -1e-12).all()

    def test_zero_cost_equals_gross(self) -> None:
        """With zero transaction costs, net = position * return."""
        engine_free = WalkForwardEngine(cost_bps=0)
        r = _make_returns(400)
        res = engine_free.run_single(r)
        expected = res.positions * r
        pd.testing.assert_series_equal(
            res.strategy_returns, expected, check_names=False, atol=1e-12
        )

    def test_no_lookahead_first_position_zero(self) -> None:
        """First position must be 0 (no prior H estimate available)."""
        r = _make_returns(400)
        res = self.engine.run_single(r)
        assert res.positions.iloc[0] == 0.0

    def test_regimes_are_regime_or_none(self) -> None:
        r = _make_returns(400)
        res = self.engine.run_single(r)
        for val in res.regimes:
            assert val is None or isinstance(val, Regime)


# WalkForwardEngine.run (multi-instrument)

class TestRunMulti:
    def setup_method(self) -> None:
        self.engine = WalkForwardEngine()

    def test_returns_dict_of_results(self) -> None:
        data = _make_data(["AAPL", "MSFT"])
        results = self.engine.run(data)
        assert isinstance(results, dict)

    def test_all_tickers_processed(self) -> None:
        tickers = ["AAPL", "MSFT", "GOOG"]
        data = _make_data(tickers)
        results = self.engine.run(data)
        assert set(results.keys()) == set(tickers)

    def test_each_result_is_backtest_result(self) -> None:
        data = _make_data(["AAPL", "MSFT"])
        results = self.engine.run(data)
        for ticker, res in results.items():
            assert isinstance(res, BacktestResult), f"{ticker} is not a BacktestResult"

    def test_missing_return_col_skipped(self) -> None:
        data = _make_data(["AAPL"])
        # Replace log_return col with something else
        data["BAD"] = data["AAPL"].rename(columns={"log_return": "price"})
        results = self.engine.run(data)
        assert "BAD" not in results
        assert "AAPL" in results

    def test_too_few_rows_skipped(self) -> None:
        data = _make_data(["AAPL"])
        short_idx = pd.bdate_range("2020-01-01", periods=10)
        rng = np.random.default_rng(99)
        data["SHORT"] = pd.DataFrame(
            {"log_return": rng.normal(0.0, 0.01, 10)}, index=short_idx
        )
        results = self.engine.run(data)
        assert "SHORT" not in results
        assert "AAPL" in results

    def test_empty_data_returns_empty_dict(self) -> None:
        results = self.engine.run({})
        assert results == {}


# WalkForwardEngine.summary

class TestSummary:
    def setup_method(self) -> None:
        self.engine = WalkForwardEngine()

    def test_summary_returns_dataframe(self) -> None:
        data = _make_data(["AAPL", "MSFT"])
        results = self.engine.run(data)
        df = self.engine.summary(results)
        assert isinstance(df, pd.DataFrame)

    def test_summary_index_contains_tickers(self) -> None:
        tickers = ["AAPL", "MSFT"]
        data = _make_data(tickers)
        results = self.engine.run(data)
        df = self.engine.summary(results)
        for ticker in tickers:
            assert ticker in df.index

    def test_summary_columns_contain_sharpe(self) -> None:
        data = _make_data(["AAPL"])
        results = self.engine.run(data)
        df = self.engine.summary(results)
        assert "sharpe" in df.columns

    def test_summary_empty_for_empty_results(self) -> None:
        df = self.engine.summary({})
        assert df.empty


# Transaction cost sensitivity

class TestTransactionCosts:
    def test_higher_costs_lower_or_equal_sharpe(self) -> None:
        """More expensive trading should not improve Sharpe in expectation."""
        r = _make_returns(600, mu=0.0002, seed=5)

        engine_low = WalkForwardEngine(cost_bps=0)
        engine_high = WalkForwardEngine(cost_bps=50)

        res_low = engine_low.run_single(r)
        res_high = engine_high.run_single(r)

        # Net returns under high costs should be <= net returns under low costs
        # at least in cumulative sum
        assert res_low.strategy_returns.sum() >= res_high.strategy_returns.sum() - 1e-10

    def test_cost_bps_uses_config_default_when_none(self) -> None:
        from wsq_trading import config
        engine = WalkForwardEngine()
        # The internal cost multiplier is cost_bps / 10000
        from wsq_trading import config as cfg
        expected = cfg.TRANSACTION_COST_BPS / 10_000
        assert abs(engine._cost - expected) < 1e-12


# Long-bias trend filter

class TestLongBiasFilter:
    def test_no_shorts_above_trend_ma(self) -> None:
        n = 500
        idx = pd.bdate_range("2020-01-01", periods=n)
        returns = pd.Series(np.full(n, 0.003), index=idx, name="log_return")
        engine = WalkForwardEngine(long_bias_tickers=["TEST"], vol_target_enabled=False, h_conviction_sizing=False)
        res = engine.run_single(returns, ticker="TEST")
        warmup = 200 // 2
        assert (res.positions.iloc[warmup:] >= -1e-9).all()

    def test_no_longs_below_trend_ma(self) -> None:
        n = 500
        idx = pd.bdate_range("2020-01-01", periods=n)
        returns = pd.Series(np.full(n, -0.003), index=idx, name="log_return")
        engine = WalkForwardEngine(long_bias_tickers=["TEST"], vol_target_enabled=False, h_conviction_sizing=False)
        res = engine.run_single(returns, ticker="TEST")
        warmup = 200 // 2
        assert (res.positions.iloc[warmup:] <= 1e-9).all()

    def test_non_bias_ticker_unaffected(self) -> None:
        n = 400
        idx = pd.bdate_range("2020-01-01", periods=n)
        rng = np.random.default_rng(7)
        returns = pd.Series(rng.normal(0.001, 0.01, n), index=idx)
        eng_bias    = WalkForwardEngine(long_bias_tickers=["ES=F"], vol_target_enabled=False, h_conviction_sizing=False)
        eng_no_bias = WalkForwardEngine(long_bias_tickers=[],       vol_target_enabled=False, h_conviction_sizing=False)
        res_bias    = eng_bias.run_single(returns, ticker="TEST")
        res_no_bias = eng_no_bias.run_single(returns, ticker="TEST")
        pd.testing.assert_series_equal(res_bias.positions, res_no_bias.positions)


# Volatility targeting

class TestVolTargeting:
    def test_positions_bounded_after_vol_target(self) -> None:
        rng = np.random.default_rng(42)
        idx = pd.bdate_range("2020-01-01", periods=500)
        returns = pd.Series(rng.normal(0.0, 0.05, 500), index=idx)
        engine = WalkForwardEngine(vol_target_enabled=True, long_bias_tickers=[], h_conviction_sizing=False)
        res = engine.run_single(returns)
        assert (res.positions.abs() <= 1.0 + 1e-9).all()

    def test_high_vol_reduces_position_size(self) -> None:
        n = 400
        idx = pd.bdate_range("2020-01-01", periods=n)
        rng = np.random.default_rng(9)
        low_vol  = pd.Series(rng.normal(0.0, 0.005, n), index=idx)
        high_vol = pd.Series(rng.normal(0.0, 0.05,  n), index=idx)
        engine = WalkForwardEngine(vol_target_enabled=True, long_bias_tickers=[], h_conviction_sizing=False)
        res_low  = engine.run_single(low_vol)
        res_high = engine.run_single(high_vol)
        assert res_high.positions.abs().mean() < res_low.positions.abs().mean()


# H-conviction sizing

class TestHConvictionSizing:
    def test_positions_bounded(self) -> None:
        rng = np.random.default_rng(1)
        idx = pd.bdate_range("2020-01-01", periods=400)
        returns = pd.Series(rng.normal(0.0, 0.01, 400), index=idx)
        engine = WalkForwardEngine(h_conviction_sizing=True, vol_target_enabled=False, long_bias_tickers=[])
        res = engine.run_single(returns)
        assert (res.positions.abs() <= 1.0 + 1e-9).all()

    def test_some_fractional_positions(self) -> None:
        rng = np.random.default_rng(22)
        idx = pd.bdate_range("2020-01-01", periods=500)
        returns = pd.Series(rng.normal(0.001, 0.01, 500), index=idx)
        engine = WalkForwardEngine(h_conviction_sizing=True, vol_target_enabled=False, long_bias_tickers=[])
        res = engine.run_single(returns)
        nonzero = res.positions[res.positions.abs() > 1e-9]
        if len(nonzero) > 0:
            assert (nonzero.abs() < 1.0 - 1e-9).any()


@pytest.fixture()
def flat_returns() -> pd.Series:
    """Constant positive returns: Sharpe is well-defined."""
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, 500))  # ~12% annual return
    return r


@pytest.fixture()
def zero_returns() -> pd.Series:
    return pd.Series(np.zeros(100))


@pytest.fixture()
def declining_returns() -> pd.Series:
    """Monotonically declining: max drawdown should be large."""
    return pd.Series(np.full(100, -0.005))


# Sharpe ratio

class TestSharpeRatio:
    def test_positive_for_positive_mean(self, flat_returns: pd.Series) -> None:
        assert sharpe_ratio(flat_returns) > 0

    def test_nan_for_empty(self) -> None:
        assert np.isnan(sharpe_ratio(pd.Series([], dtype=float)))

    def test_nan_for_zero_vol(self, zero_returns: pd.Series) -> None:
        assert np.isnan(sharpe_ratio(zero_returns))

    def test_numpy_array_input(self, flat_returns: pd.Series) -> None:
        s_series = sharpe_ratio(flat_returns)
        s_array = sharpe_ratio(flat_returns.to_numpy())
        assert abs(s_series - s_array) < 1e-10

    def test_negative_for_losing_strategy(self) -> None:
        assert sharpe_ratio(pd.Series(np.full(252, -0.001))) < 0


# Max drawdown

class TestMaxDrawdown:
    def test_zero_for_monotone_positive(self) -> None:
        returns = pd.Series(np.full(100, 0.001))
        assert max_drawdown(returns) == pytest.approx(0.0, abs=1e-9)

    def test_large_for_declining(self, declining_returns: pd.Series) -> None:
        mdd = max_drawdown(declining_returns)
        assert mdd > 0.3   # -0.5% / day for 100 days ~ 40% drawdown

    def test_positive_value(self, flat_returns: pd.Series) -> None:
        assert max_drawdown(flat_returns) >= 0

    def test_nan_for_empty(self) -> None:
        assert np.isnan(max_drawdown(pd.Series([], dtype=float)))


# Win rate

class TestWinRate:
    def test_one_for_all_positive(self) -> None:
        assert win_rate(pd.Series(np.full(50, 0.001))) == pytest.approx(1.0)

    def test_zero_for_all_negative(self) -> None:
        assert win_rate(pd.Series(np.full(50, -0.001))) == pytest.approx(0.0)

    def test_between_zero_and_one(self, flat_returns: pd.Series) -> None:
        wr = win_rate(flat_returns)
        assert 0 <= wr <= 1

    def test_nan_for_empty(self) -> None:
        assert np.isnan(win_rate(pd.Series([], dtype=float)))


# Annualized return and vol

class TestAnnualizedReturn:
    def test_positive_for_positive_returns(self) -> None:
        r = pd.Series(np.full(252, 0.001))
        assert annualized_return(r) > 0

    def test_rough_magnitude(self) -> None:
        # 0.1% per day -> ~28% annualised (compound)
        r = pd.Series(np.full(252, 0.001))
        ann = annualized_return(r)
        assert 0.20 < ann < 0.35

    def test_nan_for_empty(self) -> None:
        assert np.isnan(annualized_return(pd.Series([], dtype=float)))


class TestAnnualizedVolatility:
    def test_positive_for_nonzero(self, flat_returns: pd.Series) -> None:
        assert annualized_volatility(flat_returns) > 0

    def test_nan_for_single_obs(self) -> None:
        assert np.isnan(annualized_volatility(pd.Series([0.01])))


# Calmar ratio

class TestCalmarRatio:
    def test_positive_for_profitable_strategy(self, flat_returns: pd.Series) -> None:
        c = calmar_ratio(flat_returns)
        assert not np.isnan(c) and c > 0

    def test_nan_for_zero_drawdown(self) -> None:
        # monotone positive -> no drawdown
        assert np.isnan(calmar_ratio(pd.Series(np.full(252, 0.001))))


# Profit factor

class TestProfitFactor:
    def test_greater_than_one_for_profitable(self, flat_returns: pd.Series) -> None:
        pf = profit_factor(flat_returns)
        assert not np.isnan(pf) and pf > 1

    def test_nan_when_no_losses(self) -> None:
        assert np.isnan(profit_factor(pd.Series(np.full(50, 0.001))))


# Turnover

class TestTurnover:
    def test_zero_for_static_position(self) -> None:
        pos = pd.Series(np.ones(100))
        assert turnover(pos) == pytest.approx(0.0, abs=1e-9)

    def test_positive_when_positions_change(self) -> None:
        pos = pd.Series(np.tile([1, -1], 50))
        assert turnover(pos) > 0

    def test_nan_for_single_position(self) -> None:
        assert np.isnan(turnover(pd.Series([1.0])))


# compute_all

class TestComputeAll:
    def test_returns_dict(self, flat_returns: pd.Series) -> None:
        m = compute_all(flat_returns)
        assert isinstance(m, dict)

    def test_expected_keys_present(self, flat_returns: pd.Series) -> None:
        m = compute_all(flat_returns)
        for key in ("sharpe", "max_drawdown", "win_rate", "annualized_return"):
            assert key in m, f"Missing key: {key}"

    def test_turnover_included_when_positions_given(self, flat_returns: pd.Series) -> None:
        pos = pd.Series(np.ones(len(flat_returns)))
        m = compute_all(flat_returns, positions=pos)
        assert "turnover" in m

    def test_turnover_absent_when_positions_not_given(self, flat_returns: pd.Series) -> None:
        m = compute_all(flat_returns)
        assert "turnover" not in m



# Portfolio-level analytics: PortfolioMetrics / regime_attribution / cost_sensitivity

from wsq_trading.backtest import (  # noqa: E402
    PortfolioMetrics,
    compute_portfolio_metrics,
    cost_sensitivity,
    regime_attribution,
)
from wsq_trading.constants import TRADING_DAYS_PER_YEAR  # noqa: E402


def _stub_result(ticker: str, ret: pd.Series, regimes: pd.Series | None = None,
                 positions: pd.Series | None = None,
                 gross: pd.Series | None = None) -> BacktestResult:
    """Build a minimal BacktestResult for analytics tests."""
    if regimes is None:
        regimes = pd.Series([None] * len(ret), index=ret.index)
    if positions is None:
        positions = pd.Series(np.sign(ret), index=ret.index)
    if gross is None:
        gross = ret
    return BacktestResult(
        ticker=ticker,
        strategy_returns=ret,
        positions=positions,
        H_estimates=pd.Series(0.5, index=ret.index),
        regimes=regimes,
        gross_returns=gross,
    )


class TestRegimeAttribution:
    def test_frac_bars_sums_to_one_per_ticker(self) -> None:
        """Fraction of bars across all regimes should sum to ~1 per ticker."""
        idx = pd.bdate_range("2020-01-01", periods=300)
        rng = np.random.default_rng(0)
        ret = pd.Series(rng.normal(0.0, 0.01, 300), index=idx)
        regimes = pd.Series(
            rng.choice(list(Regime), size=300), index=idx
        )
        res = _stub_result("X", ret, regimes=regimes)
        df = regime_attribution({"X": res})
        assert abs(df.loc["X"]["frac_bars"].sum() - 1.0) < 1e-9

    def test_portfolio_rows_present_with_weights(self) -> None:
        idx = pd.bdate_range("2020-01-01", periods=300)
        rng = np.random.default_rng(1)
        regimes = pd.Series(rng.choice(list(Regime), size=300), index=idx)
        results = {
            "A": _stub_result("A", pd.Series(rng.normal(0, 0.01, 300), index=idx),
                              regimes=regimes),
            "B": _stub_result("B", pd.Series(rng.normal(0, 0.01, 300), index=idx),
                              regimes=regimes),
        }
        weights = pd.Series({"A": 0.5, "B": 0.5})
        df = regime_attribution(results, weights=weights)
        assert "PORTFOLIO" in df.index.get_level_values("ticker")

    def test_columns_present(self) -> None:
        idx = pd.bdate_range("2020-01-01", periods=100)
        rng = np.random.default_rng(2)
        regimes = pd.Series(rng.choice(list(Regime), size=100), index=idx)
        res = _stub_result("X", pd.Series(rng.normal(0, 0.01, 100), index=idx),
                           regimes=regimes)
        df = regime_attribution({"X": res})
        for col in ("frac_bars", "ann_return", "sharpe", "n_obs"):
            assert col in df.columns


class TestCostSensitivity:
    def _results(self):
        engine = WalkForwardEngine(cost_bps=0)
        data = _make_data(["AAPL", "MSFT"], n=500)
        return engine.run(data)

    def test_higher_cost_lower_or_equal_sharpe(self) -> None:
        """portfolio_sharpe must be non-increasing as cost_bps increases."""
        results = self._results()
        weights = pd.Series({t: 1.0 / len(results) for t in results})
        df = cost_sensitivity(results, weights)
        sharpes = df["portfolio_sharpe"].to_numpy()
        assert np.all(np.diff(sharpes) <= 1e-9)

    def test_default_grid(self) -> None:
        results = self._results()
        weights = pd.Series({t: 1.0 / len(results) for t in results})
        df = cost_sensitivity(results, weights)
        assert list(df.index) == [0, 2, 5, 10, 20, 30]

    def test_per_ticker_sharpe_columns(self) -> None:
        results = self._results()
        weights = pd.Series({t: 1.0 / len(results) for t in results})
        df = cost_sensitivity(results, weights)
        for t in results:
            assert f"sharpe_{t}" in df.columns


class TestPortfolioMetrics:
    def test_alpha_beta_via_ols(self) -> None:
        """Alpha and beta from OLS should match an exact linear relationship."""
        idx = pd.bdate_range("2020-01-01", periods=300)
        rng = np.random.default_rng(3)
        bench = pd.Series(rng.normal(0.0, 0.01, 300), index=idx)
        a, b = 0.0003, 1.5
        port = a + b * bench  # exact linear -> OLS recovers (a, b)
        res = _stub_result(
            "P", port,
            positions=pd.Series(np.zeros(300), index=idx),
            gross=bench,
        )
        pm = compute_portfolio_metrics(
            {"P": res}, pd.Series({"P": 1.0}),
            benchmark_returns=bench, cost_bps=2.0,
        )
        assert isinstance(pm, PortfolioMetrics)
        assert abs(pm.beta - b) < 1e-6
        assert abs(pm.alpha - a * TRADING_DAYS_PER_YEAR) < 1e-6
        assert pm.total_cost_bps == 2.0

    def test_alpha_beta_nan_without_benchmark(self) -> None:
        idx = pd.bdate_range("2020-01-01", periods=200)
        rng = np.random.default_rng(4)
        ret = pd.Series(rng.normal(0.0005, 0.01, 200), index=idx)
        res = _stub_result("P", ret)
        pm = compute_portfolio_metrics({"P": res}, pd.Series({"P": 1.0}))
        assert np.isnan(pm.alpha) and np.isnan(pm.beta)
        assert np.isnan(pm.information_ratio)

    def test_n_obs_matches_return_length(self) -> None:
        idx = pd.bdate_range("2020-01-01", periods=150)
        rng = np.random.default_rng(5)
        ret = pd.Series(rng.normal(0.0, 0.01, 150), index=idx)
        res = _stub_result("P", ret)
        pm = compute_portfolio_metrics({"P": res}, pd.Series({"P": 1.0}))
        assert pm.n_obs == 150
