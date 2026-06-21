"""Shared utilities for wsq_trading.

Exports
-------
get_logger  : Module-level logger factory.
set_seed    : Reproducible RNG seeding (numpy + torch if available).
timer       : Context manager that logs elapsed wall-clock time.
ensure_dir  : Create a directory (and parents) if it does not exist.
"""

from __future__ import annotations

import logging
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

# Logging

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_root_configured = False


def _configure_root_logger() -> None:
    """Set up the root logger once (idempotent)."""
    global _root_configured
    if _root_configured:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root = logging.getLogger("wsq_trading")
    if not root.handlers:
        root.addHandler(handler)
    root.setLevel(logging.INFO)
    _root_configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``wsq_trading`` namespace.

    Parameters
    ----------
    name : str
        Typically ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
        Logger with the ``wsq_trading.<module>`` hierarchy.

    Examples
    --------
    >>> log = get_logger(__name__)
    >>> log.info("Fetching data…")
    """
    _configure_root_logger()
    return logging.getLogger(name)


# Reproducibility

def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and (optionally) PyTorch RNGs for reproducibility.

    Parameters
    ----------
    seed : int
        Integer seed value. Default ``42``.
    """
    random.seed(seed)

    import numpy as np
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass  # PyTorch is optional during data-phase work

    log = get_logger(__name__)
    log.debug("RNG seed set to %d.", seed)


# Timing

@contextmanager
def timer(label: str = "block") -> Generator[None, None, None]:
    """Context manager that logs elapsed time for any code block.

    Parameters
    ----------
    label : str
        Human-readable name shown in the log message.

    Examples
    --------
    >>> with timer("SPDE simulation"):
    ...     simulator.generate_dataset()
    2024-01-01 00:00:00 | INFO     | wsq_trading.utils | SPDE simulation finished in 12.34 s
    """
    log = get_logger(__name__)
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        log.info("%s finished in %.2f s.", label, elapsed)


# Path helpers

def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (and all parents) if it does not already exist.

    Parameters
    ----------
    path : str | Path
        Directory path to create.

    Returns
    -------
    Path
        The resolved ``Path`` object (always a directory).
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def parquet_path(directory: str | Path, ticker: str) -> Path:
    """Return the canonical parquet file path for a given ticker.

    Parameters
    ----------
    directory : str | Path
        Parent directory (e.g. ``config.RAW_DIR``).
    ticker : str
        Ticker symbol (e.g. ``'ES=F'``).

    Returns
    -------
    Path
        e.g. ``data/raw/futures/ES_F.parquet``
    """
    safe_name = ticker.replace("=", "_").replace("/", "_").replace("^", "_")
    return Path(directory) / f"{safe_name}.parquet"
