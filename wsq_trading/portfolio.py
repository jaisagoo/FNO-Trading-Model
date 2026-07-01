"""Hierarchical Risk Parity portfolio allocation.

Algorithm (Lopez de Prado, 2016 -- "Building Diversified Portfolios that
Outperform Out-of-Sample"):

    1. Compute the correlation matrix from asset returns.
    2. Build a distance matrix: d(i,j) = sqrt(0.5 * (1 - rho(i,j))).
    3. Cluster assets using single-linkage hierarchical clustering on d.
    4. Quasi-diagonalise: reorder assets so correlated assets sit together.
    5. Recursive bisection: split the dendrogram into two halves; allocate
       capital inversely proportional to each half's cluster variance; recurse.

Why HRP over MVO
----------------
Mean-variance optimisation inverts the covariance matrix, amplifying
estimation noise.  HRP only uses the covariance matrix for inverse-variance
weighting within leaf clusters and is therefore more stable out-of-sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from wsq_trading import config
from wsq_trading.constants import TRADING_DAYS_PER_YEAR
from wsq_trading.utils import get_logger

log = get_logger(__name__)


class HRPAllocator:
    """Hierarchical Risk Parity portfolio weight calculator.

    Parameters
    ----------
    max_weight : float
        Maximum weight for any single asset.  Weights exceeding this cap
        are clipped and the excess redistributed proportionally.
        Default: ``config.MAX_WEIGHT``.
    linkage_method : str
        scipy linkage method (``'single'``, ``'complete'``, ``'ward'``, …).
        Default: ``'single'`` (original HRP paper recommendation).
    min_periods : int
        Minimum number of non-NaN observations required per pair of assets
        to include the pair in the covariance/correlation computation.
        Default: 30.
    """

    def __init__(
        self,
        max_weight: float | None = None,
        min_weight: float | None = None,
        quality_sharpe_threshold: float | None = None,
        linkage_method: str = "single",
        min_periods: int = 30,
    ) -> None:
        self.max_weight     = max_weight if max_weight is not None else config.MAX_WEIGHT
        self.min_weight     = min_weight if min_weight is not None else config.MIN_WEIGHT
        self.quality_sharpe_threshold = (
            quality_sharpe_threshold if quality_sharpe_threshold is not None
            else config.QUALITY_SHARPE_THRESHOLD
        )
        self.linkage_method = linkage_method
        self.min_periods    = min_periods


    def allocate(self, returns: pd.DataFrame) -> pd.Series:
        """Compute HRP weights from a DataFrame of asset returns.

        Parameters
        ----------
        returns : pd.DataFrame
            Rows = dates, columns = tickers.  May contain NaN (aligned on
            common dates or ffill-cleaned before passing in).

        Returns
        -------
        pd.Series
            Index = ticker, values = portfolio weight (sum to 1, all >= 0).
            Capped at ``max_weight`` with proportional redistribution.

        Raises
        ------
        ValueError
            If fewer than 2 valid assets remain after dropping all-NaN columns.
        """
        returns = returns.dropna(axis=1, how="all")
        if returns.shape[1] < 2:
            raise ValueError(
                f"Need at least 2 assets for HRP; got {returns.shape[1]}."
            )

        tickers = list(returns.columns)
        log.info("HRPAllocator.allocate: %d assets, %d bars", len(tickers), len(returns))

        cov, corr = self._cov_corr(returns)

        # Step 1: distance matrix
        dist = self._corr_to_distance(corr)

        # Step 2: hierarchical clustering
        link = self._cluster(dist, tickers)

        # Step 3: quasi-diagonalise
        sort_ix = self._quasi_diag(link, len(tickers))

        # Step 4: recursive bisection
        raw_weights = self._rec_bisect(cov, sort_ix, tickers)

        # Step 5: cap and normalise
        weights = self._apply_cap(raw_weights)

        log.info(
            "HRP weights: %s",
            "  ".join(f"{t}={w:.3f}" for t, w in weights.items()),
        )
        return weights

    def allocate_backtest(
        self,
        results: "dict[str, BacktestResult]",  # type: ignore[name-defined]
        train_end_date: "pd.Timestamp | str | None" = None,
    ) -> pd.Series:
        """Convenience wrapper: extract strategy returns, apply quality gate, then HRP.

        Instruments whose annualised strategy Sharpe is below
        ``quality_sharpe_threshold`` are excluded before running HRP.  The
        freed weight is redistributed proportionally among the survivors.
        Set ``quality_sharpe_threshold`` to ``-inf`` (or a very negative number)
        to disable the gate.

        The returned Series is always indexed by *all* tickers in ``results``,
        with 0.0 assigned to any instrument excluded by the quality gate.
        This avoids NaN weights appearing in downstream plots and summaries.

        Parameters
        ----------
        results : dict[str, BacktestResult]
            Output of :meth:`WalkForwardEngine.run`.
        train_end_date : pd.Timestamp or str, optional
            If provided, the quality-gate Sharpe is computed only from returns
            up to and including this date, avoiding look-ahead from the test
            period.  The HRP weight computation (after filtering) still uses all
            available returns -- only the *filter decision* is time-limited.

        Returns
        -------
        pd.Series
            HRP weights indexed by all tickers (sum ≤ 1; excluded tickers = 0).
        """
        all_tickers = list(results.keys())
        returns_df = pd.DataFrame(
            {ticker: res.strategy_returns for ticker, res in results.items()}
        )

        # Restrict the quality-gate Sharpe window if a cutoff was supplied.
        # HRP weights still use the full returns_df; only the gate decision
        # is limited to the training window.
        if train_end_date is not None:
            cutoff = pd.Timestamp(train_end_date)
            sharpe_returns = returns_df.loc[:cutoff]
        else:
            sharpe_returns = returns_df

        # Quality gate
        threshold = self.quality_sharpe_threshold
        if threshold > -1e9:  # effectively disabled if very negative
            sharpes: dict[str, float] = {}
            for col in sharpe_returns.columns:
                r = sharpe_returns[col].dropna()
                if len(r) > 30 and r.std(ddof=1) > 0:
                    sharpes[col] = (r.mean() / r.std(ddof=1)) * np.sqrt(TRADING_DAYS_PER_YEAR)
                else:
                    sharpes[col] = 0.0

            passing = [t for t, s in sharpes.items() if s >= threshold]
            failing = [t for t, s in sharpes.items() if s < threshold]

            if failing:
                log.info(
                    "Quality gate (threshold=%.2f): excluding %s  "
                    "[Sharpes: %s]",
                    threshold,
                    failing,
                    {t: f"{sharpes[t]:.3f}" for t in failing},
                )

            if len(passing) >= 2:
                returns_df = returns_df[passing]
            elif len(passing) == 1:
                # Only 1 passes - keep the 2 best by Sharpe to maintain HRP
                top2 = sorted(sharpes.items(), key=lambda kv: -kv[1])[:2]
                returns_df = returns_df[[t for t, _ in top2]]
                log.warning(
                    "Only 1 instrument passed quality gate; keeping top 2."
                )
            else:
                log.warning(
                    "No instruments passed quality gate; using all (gate disabled)."
                )

        weights = self.allocate(returns_df)

        # Re-index to ALL tickers so excluded instruments show 0.0 not NaN
        weights = weights.reindex(all_tickers, fill_value=0.0)
        return weights

    # Internal steps

    def _cov_corr(
        self, returns: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (covariance, correlation) matrices with small regularisation.

        Extra robustness: columns with near-zero variance (e.g. an instrument
        whose strategy returns are almost always zero after quality-gate
        filtering) receive a small variance floor before correlation is computed.
        This prevents division-by-zero NaN values from propagating into the
        distance matrix and producing NaN cluster weights.
        """
        # Variance floor: replace near-zero variances with a small epsilon so
        # that correlation computation never divides by zero.
        ret_arr = returns.to_numpy(dtype=float)
        col_stds = np.nanstd(ret_arr, axis=0, ddof=1)
        eps = 1e-8
        floor_needed = col_stds < eps
        if floor_needed.any():
            log.debug(
                "Covariance floor applied to near-zero-variance columns: %s",
                [returns.columns[i] for i, f in enumerate(floor_needed) if f],
            )
            ret_arr[:, floor_needed] += np.random.default_rng(0).normal(
                0, eps, size=(ret_arr.shape[0], floor_needed.sum())
            )
            returns = pd.DataFrame(ret_arr, index=returns.index, columns=returns.columns)

        cov  = returns.cov(min_periods=self.min_periods)
        corr = returns.corr(min_periods=self.min_periods)

        # Fill any remaining NaN (e.g. assets with < min_periods overlap) with 0
        corr = corr.fillna(0.0)
        cov  = cov.fillna(0.0)

        # Ensure diagonal of correlation is exactly 1
        # .values returns a read-only view in newer pandas/numpy - copy first
        corr_arr = corr.to_numpy().copy()
        np.fill_diagonal(corr_arr, 1.0)
        corr = pd.DataFrame(corr_arr, index=corr.index, columns=corr.columns)

        # Shrink correlations slightly to improve conditioning
        # (Ledoit-Wolf lite: blend with identity at epsilon)
        n = corr.shape[0]
        shrink = 0.05
        corr = (1 - shrink) * corr + shrink * pd.DataFrame(
            np.eye(n), index=corr.index, columns=corr.columns
        )
        return cov, corr

    @staticmethod
    def _corr_to_distance(corr: pd.DataFrame) -> np.ndarray:
        """d(i,j) = sqrt(0.5 * (1 - rho(i,j))), clipped to [0, 1]."""
        d = np.sqrt(np.clip(0.5 * (1.0 - corr.values), 0.0, 1.0))
        np.fill_diagonal(d, 0.0)
        return d

    def _cluster(self, dist: np.ndarray, tickers: list[str]) -> np.ndarray:
        """Run scipy hierarchical clustering on the condensed distance vector."""
        condensed = squareform(dist, checks=False)
        return linkage(condensed, method=self.linkage_method)

    @staticmethod
    def _quasi_diag(link: np.ndarray, n_assets: int) -> list[int]:
        """Return asset indices sorted by dendrogram leaf order.

        Performs an iterative in-order traversal of the linkage tree so that
        assets in the same cluster are adjacent in the reordered matrix.
        """
        # Map leaf node ids (0..n-1) to their positions
        # Use a stack-based traversal of the linkage matrix
        n = n_assets
        # Each internal node i (i >= n) merges link[i-n, 0] and link[i-n, 1]
        # Start from the root node
        root = 2 * n - 2
        stack = [int(root)]
        ordered: list[int] = []

        while stack:
            node = stack.pop()
            if node < n:
                ordered.append(node)
            else:
                row = link[node - n]
                # Push right then left so left is processed first
                stack.append(int(row[1]))
                stack.append(int(row[0]))

        return ordered

    def _rec_bisect(
        self,
        cov: pd.DataFrame,
        sort_ix: list[int],
        tickers: list[str],
    ) -> pd.Series:
        """Recursive bisection weight allocation.

        Splits the sorted asset list into two halves, weights each half
        proportionally to the inverse of its cluster variance, and recurses
        until each cluster is a single asset.
        """
        w = pd.Series(1.0, index=[tickers[i] for i in sort_ix])
        clusters = [sort_ix]   # list of asset index lists to process

        while clusters:
            clusters_new: list[list[int]] = []
            for cluster in clusters:
                if len(cluster) <= 1:
                    continue
                mid = len(cluster) // 2
                left  = cluster[:mid]
                right = cluster[mid:]

                left_names  = [tickers[i] for i in left]
                right_names = [tickers[i] for i in right]

                # Cluster variance: w^T * Sigma * w for equal-weight sub-portfolio
                var_left  = self._cluster_var(cov, left_names)
                var_right = self._cluster_var(cov, right_names)

                # Inverse-variance split
                total = var_left + var_right
                if total <= 0:
                    alpha = 0.5
                else:
                    alpha = 1.0 - var_left / total   # fraction going to left

                w[left_names]  *= alpha
                w[right_names] *= (1.0 - alpha)

                clusters_new.extend([left, right])
            clusters = clusters_new

        return w

    @staticmethod
    def _cluster_var(cov: pd.DataFrame, names: list[str]) -> float:
        """Variance of an equal-weight portfolio over ``names``."""
        sub = cov.loc[names, names].values
        n   = len(names)
        w   = np.full(n, 1.0 / n)
        return float(w @ sub @ w)

    def _apply_cap(self, weights: pd.Series) -> pd.Series:
        """Clip each weight to [min_weight, max_weight] and renormalise.

        First applies the upper cap (iterative redistribution), then applies
        the lower floor (raises small positive weights and takes proportionally
        from the largest weights).  A final renormalisation ensures sum = 1.
        """
        w = weights.copy()
        n = len(w)

        # Upper cap
        for _ in range(20):
            excess = (w - self.max_weight).clip(lower=0.0)
            if excess.sum() < 1e-10:
                break
            w = w.clip(upper=self.max_weight)
            uncapped_mask = w < self.max_weight
            if uncapped_mask.sum() == 0:
                break
            redistribute = excess.sum() / uncapped_mask.sum()
            w[uncapped_mask] += redistribute

        # Lower floor
        if self.min_weight > 0 and n > 0:
            effective_floor = min(self.min_weight, 1.0 / n * 0.9)
            for _ in range(20):
                below_floor = (w > 0) & (w < effective_floor)
                if not below_floor.any():
                    break
                deficit = (effective_floor - w[below_floor]).sum()
                w[below_floor] = effective_floor
                above_floor = w > effective_floor
                if above_floor.sum() == 0:
                    break
                total_above = w[above_floor].sum()
                if total_above <= deficit:
                    break
                w[above_floor] -= deficit * w[above_floor] / total_above

        # Final normalise
        total = w.sum()
        if total > 0:
            w = w / total
        return w
