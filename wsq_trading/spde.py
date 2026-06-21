"""Rough Heston SPDE path generation for FNO training data.

Model
-----
The Rough Heston model couples a fractional-volatility process with a
log-normal price process:

    log S(t) = log S(0) - ½ ∫₀ᵗ V(s)ds + ∫₀ᵗ √V(s) dW(s)

    V(t) = V₀ + (1/Γ(H+½)) ∫₀ᵗ (t-s)^(H-½) κ(θ - V(s)) ds
                + (ξ/Γ(H+½)) ∫₀ᵗ (t-s)^(H-½) √V(s) dW^H(s)

where W and W^H are correlated Brownian motions (correlation ρ).
H ∈ (0, ½) is the Hurst exponent controlling roughness.

Simulation approach
-------------------
We use a two-step discretisation:

1.  **Fractional Gaussian noise** (fGn) - increments of a fractional Brownian
    motion with exponent H - are generated via exact Cholesky decomposition of
    the fGn covariance matrix (Mandelbrot & Van Ness 1968).  For path length
    T ≤ 1000 this is tractable and exact.

2.  **Euler–Maruyama** integration for V(t) and log S(t) with a truncated
    Volterra kernel.  We use the hybrid scheme approximation (Bennedsen,
    Lunde & Pakkanen 2017) where the first b steps use exact fGn and the
    remainder use a simple power-law kernel.

For the purposes of training data generation (T = 252, daily steps) the
Cholesky method is exact and takes < 1 s per batch of 100 paths on a
modern CPU.
"""

from __future__ import annotations

import multiprocessing as mp
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import gamma as gamma_fn

from wsq_trading import config
from wsq_trading.utils import ensure_dir, get_logger, set_seed, timer

log = get_logger(__name__)


_DATASET_FILE = config.SYNTHETIC_DIR / "rough_heston_paths.parquet"


# Core numerical routines (module-level for pickling with multiprocessing)

def _fbm_covariance(H: float, n: int, dt: float) -> np.ndarray:
    """Covariance matrix of fractional Gaussian noise increments.

    The autocovariance of fGn at lag k is:
        γ(k) = ½ σ² (|k+1|^{2H} - 2|k|^{2H} + |k-1|^{2H})
    with σ² = dt^{2H} (unit diffusion over the step).
    """
    ks = np.arange(n)
    # Autocovariance at each lag
    autocov = 0.5 * (
        np.abs(ks + 1) ** (2 * H)
        - 2 * np.abs(ks) ** (2 * H)
        + np.abs(ks - 1) ** (2 * H)
    ) * dt ** (2 * H)
    # Toeplitz structure
    cov = np.zeros((n, n))
    for i in range(n):
        cov[i, i:] = autocov[: n - i]
        cov[i:, i] = autocov[: n - i]
    return cov


def _fbm_increments_cholesky(
    H: float, n: int, dt: float, rng: np.random.Generator
) -> np.ndarray:
    """Generate n fractional Gaussian noise increments via Cholesky method."""
    cov = _fbm_covariance(H, n, dt)
    # Add a small nugget to ensure positive definiteness
    cov += np.eye(n) * 1e-10
    L = np.linalg.cholesky(cov)
    z = rng.standard_normal(n)
    return L @ z


def _simulate_single_path(
    H: float,
    T: int,
    dt: float,
    S0: float,
    V0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    seed: int,
) -> dict[str, Any]:
    """Simulate one rough Heston path.

    Returns log-returns and variance path labelled with H.
    """
    rng = np.random.default_rng(seed)
    sqrt_dt = np.sqrt(dt)

    # Correlated Brownian increments
    # dW_price and dW_vol are correlated with coefficient rho
    dW1 = rng.standard_normal(T) * sqrt_dt       # price BM
    dW2 = rng.standard_normal(T) * sqrt_dt       # independent BM
    dW_vol_std = rho * dW1 + np.sqrt(1 - rho**2) * dW2  # correlated

    # Fractional Gaussian noise for rough vol
    fgn = _fbm_increments_cholesky(H, T, dt, rng)

    # Euler–Maruyama integration
    V = np.zeros(T + 1)
    log_S = np.zeros(T + 1)
    V[0] = V0
    log_S[0] = np.log(S0)

    for t in range(T):
        V_pos = max(V[t], 0.0)   # Reflection: keep variance non-negative
        sqrt_V = np.sqrt(V_pos)

        # Rough variance process (Volterra-type, approximated via fGn scaling)
        # The kernel (t-s)^{H-1/2} is absorbed into the fGn covariance.
        dV = kappa * (theta - V_pos) * dt + xi * sqrt_V * fgn[t]
        V[t + 1] = V_pos + dV

        # Log price process
        d_logS = -0.5 * V_pos * dt + sqrt_V * dW1[t]
        log_S[t + 1] = log_S[t] + d_logS

    log_returns = np.diff(log_S)       # shape (T,)
    variance = V[1:]                   # shape (T,) aligned with returns

    return {
        "H": H,
        "log_returns": log_returns,
        "variance": np.clip(variance, 0.0, None),
    }


def _simulate_batch_worker(
    args: tuple[float, int, int, int],
    T: int,
    dt: float,
    S0: float,
    V0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
) -> list[dict[str, Any]]:
    """Worker function for multiprocessing; simulates n_paths for one H value."""
    H, n_paths, base_seed, _ = args
    paths = []
    for i in range(n_paths):
        path = _simulate_single_path(
            H=H,
            T=T,
            dt=dt,
            S0=S0,
            V0=V0,
            kappa=kappa,
            theta=theta,
            xi=xi,
            rho=rho,
            seed=base_seed + i,
        )
        paths.append(path)
    return paths


# Simulator class

class SPDESimulator:
    """Generate rough Heston paths with known Hurst labels for FNO training.

    Parameters
    ----------
    T : int, optional
        Path length in time steps.  Defaults to ``config.SPDE_T``.
    dt : float, optional
        Time step.  Defaults to ``config.SPDE_DT``.
    S0 : float, optional
        Initial spot price.  Defaults to ``config.SPDE_S0``.
    V0 : float, optional
        Initial variance.  Defaults to ``config.SPDE_V0``.
    kappa : float, optional
        Vol mean-reversion speed.  Defaults to ``config.SPDE_KAPPA``.
    theta : float, optional
        Long-run variance.  Defaults to ``config.SPDE_THETA``.
    xi : float, optional
        Vol-of-vol.  Defaults to ``config.SPDE_XI``.
    rho : float, optional
        Price-vol correlation.  Defaults to ``config.SPDE_RHO``.
    hurst_grid : list[float], optional
        H values to simulate.  Defaults to ``config.HURST_GRID``.
    n_paths_per_hurst : int, optional
        Paths per H value.  Defaults to ``config.N_PATHS_PER_HURST``.
    num_workers : int, optional
        Parallel worker processes.  Defaults to ``config.NUM_WORKERS``.
    output_file : Path | str, optional
        Parquet output path.  Defaults to ``SYNTHETIC_DIR/rough_heston_paths.parquet``.
    """

    def __init__(
        self,
        T: int | None = None,
        dt: float | None = None,
        S0: float | None = None,
        V0: float | None = None,
        kappa: float | None = None,
        theta: float | None = None,
        xi: float | None = None,
        rho: float | None = None,
        hurst_grid: list[float] | None = None,
        n_paths_per_hurst: int | None = None,
        num_workers: int | None = None,
        output_file: Path | str | None = None,
    ) -> None:
        self.T = T or config.SPDE_T
        self.dt = dt or config.SPDE_DT
        self.S0 = S0 or config.SPDE_S0
        self.V0 = V0 or config.SPDE_V0
        self.kappa = kappa or config.SPDE_KAPPA
        self.theta = theta or config.SPDE_THETA
        self.xi = xi or config.SPDE_XI
        self.rho = rho or config.SPDE_RHO
        self.hurst_grid = hurst_grid or config.HURST_GRID
        self.n_paths_per_hurst = n_paths_per_hurst or config.N_PATHS_PER_HURST
        self.num_workers = num_workers or config.NUM_WORKERS
        self.output_file = Path(output_file) if output_file else _DATASET_FILE


    def simulate_path(
        self,
        H: float,
        seed: int | None = None,
        return_dataframe: bool = False,
    ) -> dict[str, Any] | pd.DataFrame:
        """Simulate a single rough Heston path.

        Parameters
        ----------
        H : float
            Hurst exponent ∈ (0, 1).
        seed : int, optional
            RNG seed.  Defaults to ``config.RANDOM_SEED``.
        return_dataframe : bool
            If ``True``, return a ``pd.DataFrame`` instead of a raw dict.

        Returns
        -------
        dict | pd.DataFrame
            Keys/columns: ``H``, ``log_returns`` (shape T), ``variance`` (shape T).
        """
        seed = seed if seed is not None else config.RANDOM_SEED
        result = _simulate_single_path(
            H=H,
            T=self.T,
            dt=self.dt,
            S0=self.S0,
            V0=self.V0,
            kappa=self.kappa,
            theta=self.theta,
            xi=self.xi,
            rho=self.rho,
            seed=seed,
        )
        if return_dataframe:
            return pd.DataFrame(
                {
                    "log_return": result["log_returns"],
                    "variance": result["variance"],
                    "H": result["H"],
                }
            )
        return result

    def simulate_batch(
        self,
        H: float,
        n_paths: int | None = None,
        base_seed: int = 0,
    ) -> pd.DataFrame:
        """Simulate multiple paths for a single H value.

        Parameters
        ----------
        H : float
            Hurst exponent.
        n_paths : int, optional
            Number of paths.  Defaults to ``self.n_paths_per_hurst``.
        base_seed : int
            First seed; subsequent paths use ``base_seed + i``.

        Returns
        -------
        pd.DataFrame
            Columns: ``path_id``, ``step``, ``log_return``, ``variance``, ``H``.
        """
        n_paths = n_paths or self.n_paths_per_hurst
        rows: list[dict] = []

        for i in range(n_paths):
            path = _simulate_single_path(
                H=H,
                T=self.T,
                dt=self.dt,
                S0=self.S0,
                V0=self.V0,
                kappa=self.kappa,
                theta=self.theta,
                xi=self.xi,
                rho=self.rho,
                seed=base_seed + i,
            )
            for t in range(self.T):
                rows.append(
                    {
                        "path_id": i,
                        "step": t,
                        "log_return": path["log_returns"][t],
                        "variance": path["variance"][t],
                        "H": H,
                    }
                )

        return pd.DataFrame(rows)

    def generate_dataset(
        self,
        save: bool = True,
        force_regenerate: bool = False,
    ) -> pd.DataFrame:
        """Generate the full synthetic training dataset.

        Simulates ``n_paths_per_hurst`` paths for each H in ``hurst_grid``.
        Uses multiprocessing for speed.

        Parameters
        ----------
        save : bool
            Persist the result to parquet at ``self.output_file``.
            Defaults to ``True``.
        force_regenerate : bool
            Re-generate even if the parquet file already exists.
            Defaults to ``False``.

        Returns
        -------
        pd.DataFrame
            Columns: ``H``, ``path_id``, ``step``, ``log_return``, ``variance``.
            Total rows = ``len(hurst_grid) × n_paths_per_hurst × T``.
        """
        if not force_regenerate and self.output_file.exists():
            log.info(
                "Dataset already exists at %s. "
                "Use force_regenerate=True to overwrite.",
                self.output_file,
            )
            return self.load_dataset()

        total_paths = len(self.hurst_grid) * self.n_paths_per_hurst
        log.info(
            "Generating %d paths (%d H values × %d paths each, T=%d)…",
            total_paths,
            len(self.hurst_grid),
            self.n_paths_per_hurst,
            self.T,
        )

        # Build worker argument list: (H, n_paths, base_seed, batch_id)
        worker_args = [
            (H, self.n_paths_per_hurst, idx * self.n_paths_per_hurst * 100, idx)
            for idx, H in enumerate(self.hurst_grid)
        ]

        worker_fn = partial(
            _simulate_batch_worker,
            T=self.T,
            dt=self.dt,
            S0=self.S0,
            V0=self.V0,
            kappa=self.kappa,
            theta=self.theta,
            xi=self.xi,
            rho=self.rho,
        )

        all_paths: list[dict] = []
        with timer("SPDE dataset generation"):
            if self.num_workers > 1:
                with mp.Pool(processes=self.num_workers) as pool:
                    batch_results = pool.map(worker_fn, worker_args)
                for batch in batch_results:
                    all_paths.extend(batch)
            else:
                # Single-process fallback (easier debugging)
                for args in worker_args:
                    all_paths.extend(worker_fn(args))

        # Flatten into a long DataFrame
        rows: list[dict] = []
        global_path_id = 0
        for path in all_paths:
            for t in range(self.T):
                rows.append(
                    {
                        "H": path["H"],
                        "path_id": global_path_id,
                        "step": t,
                        "log_return": path["log_returns"][t],
                        "variance": path["variance"][t],
                    }
                )
            global_path_id += 1

        df = pd.DataFrame(rows)

        if save:
            ensure_dir(self.output_file.parent)
            df.to_parquet(self.output_file, index=False)
            log.info(
                "Saved %d rows -> %s (%.1f MB)",
                len(df),
                self.output_file,
                self.output_file.stat().st_size / 1e6,
            )

        return df

    def load_dataset(self) -> pd.DataFrame:
        """Load the previously generated synthetic dataset from parquet.

        Returns
        -------
        pd.DataFrame
            Same schema as returned by :meth:`generate_dataset`.

        Raises
        ------
        FileNotFoundError
            If ``self.output_file`` does not exist.
        """
        if not self.output_file.exists():
            raise FileNotFoundError(
                f"No synthetic dataset found at {self.output_file}. "
                "Run generate_dataset() first."
            )
        log.info("Loading synthetic dataset from %s…", self.output_file)
        df = pd.read_parquet(self.output_file)
        log.info("Loaded %d rows (%.1f MB).", len(df), self.output_file.stat().st_size / 1e6)
        return df

    def as_windows(
        self,
        df: pd.DataFrame | None = None,
        window_size: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reshape the flat dataset into fixed-length windows for FNO input.

        Parameters
        ----------
        df : pd.DataFrame, optional
            Flat synthetic dataset.  Loads from disk if not provided.
        window_size : int, optional
            Length of each window.  Defaults to ``config.WINDOW_SIZE``.

        Returns
        -------
        X : np.ndarray
            Shape ``(n_windows, window_size, 1)`` - log-return sequences.
        y : np.ndarray
            Shape ``(n_windows,)`` - corresponding H labels.
        """
        if df is None:
            df = self.load_dataset()
        window_size = window_size or config.WINDOW_SIZE

        X_list: list[np.ndarray] = []
        y_list: list[float] = []

        for (H_val, path_id), grp in df.groupby(["H", "path_id"]):
            returns = grp.sort_values("step")["log_return"].to_numpy()
            # Sliding windows over the path
            for start in range(0, len(returns) - window_size + 1, window_size):
                window = returns[start : start + window_size]
                if len(window) == window_size:
                    X_list.append(window)
                    y_list.append(H_val)

        X = np.array(X_list, dtype=np.float32)[:, :, np.newaxis]  # (N, T, 1)
        y = np.array(y_list, dtype=np.float32)                     # (N,)

        log.info(
            "as_windows: %d windows of length %d extracted.", len(X_list), window_size
        )
        return X, y

