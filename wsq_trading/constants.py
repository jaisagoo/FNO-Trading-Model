"""Immutable enumerations and math constants for wsq_trading.

All values here are fixed by design or by convention.
Tunable parameters belong in config.py.
"""

from __future__ import annotations

import math
from enum import Enum, auto

# Regime labels

class Regime(Enum):
    """Trading regime inferred from the predicted Hurst exponent H(t).

    Boundaries are defined in config.HURST_ROUGH_THRESHOLD and
    config.HURST_TREND_THRESHOLD.
    """

    MEAN_REVERT = auto()   # H < HURST_ROUGH_THRESHOLD  (rough / anti-persistent)
    NEUTRAL = auto()       # HURST_ROUGH_THRESHOLD <= H <= HURST_TREND_THRESHOLD
    MOMENTUM = auto()      # H > HURST_TREND_THRESHOLD  (persistent / trending)


# Dataset splits

class DataSplit(Enum):
    """Temporal dataset partition labels used by the backtest engine."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


# Signal directions

class Signal(Enum):
    """Directional position signal emitted by signal_generation.py."""

    LONG = 1
    FLAT = 0
    SHORT = -1


# OHLCV column contract

OHLCV_COLUMNS: tuple[str, ...] = ("Open", "High", "Low", "Close", "Volume")
"""Required columns (exact capitalisation) for every raw market DataFrame."""

ADJUSTED_CLOSE_COL: str = "Adj Close"
"""Optional adjusted-close column that yfinance may return."""


# Math constants

SQRT_2PI: float = math.sqrt(2.0 * math.pi)
SQRT_252: float = math.sqrt(252.0)   # annualisation factor (daily -> annual)
TRADING_DAYS_PER_YEAR: int = 252
MINUTES_PER_TRADING_DAY: int = 390   # 9:30–16:00 ET


# Hurst exponent bounds

HURST_MIN: float = 0.05   # below this H -> numerical instability in fBm Cholesky
HURST_MAX: float = 0.95   # above this H -> near unit-root behaviour
