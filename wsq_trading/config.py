"""Single source of truth for all wsq_trading settings.

HOW TO USE
----------
Import this module wherever settings are needed:

    from wsq_trading import config
    print(config.TICKERS)
    print(config.HURST_ROUGH_THRESHOLD)

All paths are resolved relative to the repository root so the package works
regardless of the working directory from which it is run.

EDITING GUIDE
-------------
- Tickers          -> TICKERS
- Date range       -> DEFAULT_START, DEFAULT_END
- Hurst thresholds -> HURST_ROUGH_THRESHOLD, HURST_TREND_THRESHOLD
- FNO architecture -> FNO_MODES, FNO_WIDTH, FNO_LAYERS
- Training         -> LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS, …
- SPDE             -> N_SYNTHETIC_PATHS, HURST_GRID, SPDE_*
- Cache behaviour  -> CACHE_PARQUET, CACHE_MAX_AGE_HOURS
"""

from __future__ import annotations

from pathlib import Path

#  PATHS  (all derived from repo root - never hardcode elsewhere)

# Repository root = two levels above this file (wsq_trading/config.py)
BASE_DIR: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = BASE_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw" / "futures"
PROCESSED_DIR: Path = DATA_DIR / "processed" / "futures"
SYNTHETIC_DIR: Path = DATA_DIR / "synthetic"

MODELS_DIR: Path = BASE_DIR / "models"
CHECKPOINTS_DIR: Path = MODELS_DIR / "checkpoints"

RESULTS_DIR: Path = BASE_DIR / "results"
REPORTS_DIR: Path = RESULTS_DIR / "backtest_reports"
METRICS_DIR: Path = RESULTS_DIR / "metrics"
PLOTS_DIR: Path = RESULTS_DIR / "plots"


#  DATA  -  universe, dates, cache

# Free data source: Yahoo Finance via yfinance
# Continuous futures symbols recognised by yfinance
#
# Portfolio construction rationale
# ES=F / NQ=F  - equity indices with structural upward drift; baseline long filter
#                captures the secular bull market while the Hurst regime signal
#                adds timing edge.  Sharpe ~1.0-1.2 historically.
#
# CL=F         - WTI crude oil; strong bi-directional trends; Hurst momentum
#                signal outperforms buy-and-hold significantly.  Sharpe ~0.5.
#
# 6E=F         - Euro FX (EUR/USD).  The world's most liquid FX future.
#                EUR/USD has documented H ≈ 0.55-0.65 over medium timeframes
#                driven by ECB/Fed policy divergence cycles.  Good diversifier
#                against equity risk.
#
# NG=F         - Natural Gas.  Very high realised vol (50-80% annualised) but
#                the vol-targeting layer scales positions accordingly.  Exhibits
#                strong seasonal trends (winter demand) and supply-shock trends.
#                H signal is reliable because variance roughness is pronounced.
#
# HG=F         - Copper (High-Grade).  Industrial commodity; trends with the
#                global manufacturing / China-demand cycle.  Low correlation to
#                equities and gold.  Replaces GC=F which shows near-zero alpha.
#
# ZN=F         - 10-Year Treasury Note.  Shorter duration than ZB=F (30-year),
#                more liquid, cleaner Hurst signal during rate cycles.
#                Replaces ZB=F which showed negative Sharpe.
#
# Instruments removed: GC=F (Sharpe 0.08, HRP wt consistently 0%), ZB=F (Sharpe -0.19)
TICKERS: list[str] = [
    "ES=F",   # S&P 500 E-mini         - equity, long-bias
    "NQ=F",   # Nasdaq-100 E-mini      - equity, long-bias
    "CL=F",   # WTI Crude Oil          - energy, strong trends
    "6E=F",   # Euro FX                - currency, macro policy trends
    "NG=F",   # Natural Gas            - energy, seasonal + supply trends
    "HG=F",   # Copper (High-Grade)    - industrial metal, growth cycle trends
    "ZN=F",   # 10-Year Treasury Note  - rates, cleaner than ZB
]

# Legacy tickers kept here for reference / comparison runs
TICKERS_LEGACY: list[str] = [
    "ES=F", "NQ=F", "CL=F", "GC=F", "ZB=F",
]

# Historical range for backtesting and model training
DEFAULT_START: str = "2010-01-01"
DEFAULT_END: str | None = None   # None -> today

# Parquet cache: avoids re-downloading on every run.
# Set CACHE_PARQUET = False to always fetch live from yfinance.
CACHE_PARQUET: bool = True
CACHE_MAX_AGE_HOURS: float = 24.0   # Refresh cache if older than this

# Minimum required rows to consider a dataset usable
MIN_ROWS: int = 252   # ~1 trading year


#  FEATURE ENGINEERING

# Rolling window lengths (in trading days)
VOL_WINDOW: int = 21          # 1-month realized vol
ROLLING_STATS_WINDOW: int = 63  # 3-month rolling statistics
LONG_ROLLING_WINDOW: int = 252  # 1-year rolling statistics


#  HURST / REGIME  -  strategy switching thresholds

# H < HURST_ROUGH_THRESHOLD      -> Regime.MEAN_REVERT
# HURST_ROUGH_THRESHOLD ≤ H ≤ … -> Regime.NEUTRAL
# H > HURST_TREND_THRESHOLD      -> Regime.MOMENTUM
#
# Threshold calibration notes
# Classical mode (Whittle MLE on daily log-returns, bandwidth_exp=0.65):
#   Estimates cluster in [0.40, 0.56] across typical futures markets.
#   Thresholds are set asymmetrically around 0.5 to create a meaningful
#   neutral band.  Mean-reverting markets hit H < 0.43;
#   trending markets hit H > 0.57.
#
# FNO mode (neural operator predicts true rough-Heston H):
#   Predictions span [0.05, 0.95] matching the HURST_GRID training range.
#   The same thresholds apply - the wider neutral band (0.43–0.57) gives
#   a more balanced regime distribution (~15% MR / ~35% neutral / ~50% mom)
#   and avoids overcrowding into the momentum regime when H estimates are noisy.
#
# Calibration target: ~40% momentum, ~45% neutral, ~15% mean-revert.
# The 0.14-wide neutral band (vs the old 0.04-wide 0.47/0.51 band) is the
# single highest-leverage change for improving Sharpe: it reduces false
# momentum entries and lets vol-targeting do its job on confirmed signals.
HURST_ROUGH_THRESHOLD: float = 0.43   # widened from 0.47 - broader neutral band
HURST_TREND_THRESHOLD: float = 0.57   # widened from 0.51 - broader neutral band


#  SPDE SIMULATION  -  synthetic rough Heston training data

# Grid of Hurst values to simulate (covers rough to persistent regimes)
HURST_GRID: list[float] = [round(0.05 * k, 2) for k in range(1, 20)]
# -> [0.05, 0.10, 0.15, ..., 0.90, 0.95]

# Paths per H value; total paths = len(HURST_GRID) × N_PATHS_PER_HURST
N_PATHS_PER_HURST: int = 1_000
N_SYNTHETIC_PATHS: int = len(HURST_GRID) * N_PATHS_PER_HURST

# Path length and time step
SPDE_T: int = 252          # trading days per path
SPDE_DT: float = 1 / 252   # daily time step

# Initial conditions
SPDE_S0: float = 100.0     # initial spot price
SPDE_V0: float = 0.04      # initial variance (vol ≈ 20%)

# Rough Heston model parameters
SPDE_KAPPA: float = 1.0    # mean-reversion speed of variance
SPDE_THETA: float = 0.04   # long-run variance
SPDE_XI: float = 0.3       # vol-of-vol
SPDE_RHO: float = -0.7     # correlation between price and vol Brownian motions

NUM_WORKERS: int = 4       # parallel workers for batch simulation


#  NEURAL OPERATOR  -  FNO architecture & training

# Sliding window fed into the FNO (in trading days)
WINDOW_SIZE: int = 63   # ~3 months

# FNO spectral layer settings
FNO_MODES: int = 16      # number of Fourier modes kept per layer
FNO_WIDTH: int = 64      # channel width (lifting dimension)
FNO_LAYERS: int = 4      # number of spectral conv blocks

# Training
LEARNING_RATE: float = 1e-3
BATCH_SIZE: int = 64
NUM_EPOCHS: int = 200
EARLY_STOPPING_PATIENCE: int = 20
LR_SCHEDULER_FACTOR: float = 0.5
LR_SCHEDULER_PATIENCE: int = 10
WEIGHT_DECAY: float = 1e-4
GRADIENT_CLIP_NORM: float = 1.0


#  WALK-FORWARD BACKTEST  -  temporal splits

TRAIN_SPLIT: float = 0.70   # fraction of history used for training
VAL_SPLIT: float = 0.15     # fraction used for validation (test = remainder)
WALK_FORWARD_STEP: int = 63  # retrain every ~3 months


#  PORTFOLIO  -  signal & sizing

# Maximum absolute weight per instrument in HRP portfolio
MAX_WEIGHT: float = 0.30   # reduced from 0.40 to prevent low-vol instruments dominating

# Minimum absolute weight per instrument (floor applied after HRP bisection)
MIN_WEIGHT: float = 0.05

# Transaction cost model (round-trip in bps)
TRANSACTION_COST_BPS: float = 2.0

# Annualised target volatility for position sizing (not currently implemented)
TARGET_VOL: float = 0.15

# Regime persistence filter
REGIME_PERSISTENCE: int = 5

# Momentum confirmation window
MOMENTUM_CONFIRM_WINDOW: int = 21

# Mean-reversion z-score upper cap
MR_Z_UPPER_CAP: float = 4.0

# Long-bias trend filter
LONG_BIAS_TICKERS: list[str] = ["ES=F", "NQ=F"]
TREND_FILTER_WINDOW: int = 200
# Baseline long position applied to long-bias tickers when price > trend MA
# and the regime signal is zero (flat).  Prevents leaving alpha on the table
# during bull-market uptrend periods where ES/NQ drift up but no momentum
# signal fires (MA crossover conflicts with momentum confirm).
LONG_BIAS_BASE_POSITION: float = 0.30

# Volatility targeting
# Each position is scaled by TARGET_VOL / realized_21d_vol so ex-ante
# annualised vol contribution is constant across regimes and instruments.
# CAP_SCALAR raised from 1.0 to 1.5 so that in low-volatility trending
# periods (where conviction is high and risk is low) the position can scale
# above 1× nominal without going fully unconstrained.
VOL_TARGET_ENABLED: bool = True
VOL_TARGET_WINDOW: int = 21
VOL_FLOOR_SCALAR: float = 0.25   # never scale below 25% of nominal
VOL_CAP_SCALAR: float = 1.5      # allow up to 1.5× in low-vol trending regimes

# H-conviction position sizing
# Scales momentum positions by how far H is above the trend threshold,
# and MR positions by how far H is below the rough threshold.
# Produces continuous fractional positions rather than binary ±1.
# Works multiplicatively before vol-targeting: weak-signal bars get smaller
# positions, reducing transaction costs and drawdown from false entries.
H_CONVICTION_SIZING: bool = False   # off since 2026-07: hurt validation-window Sharpe
                                    # (see results/metrics/2026-07-07_validation_tuning.csv)
H_CONVICTION_FLOOR: float = 0.15   # minimum scalar (was 0.20)


#  MISCELLANEOUS

# Allocation quality gate
# Instruments whose full-sample annualised strategy Sharpe is below this
# threshold are excluded from HRP weight allocation (they still run in backtest
# but receive zero portfolio weight).  Raised from 0.15 to 0.30 to exclude
# poor-signal instruments (GC=F, ZB=F at current calibration) from dragging
# down the portfolio Sharpe via diversification into low-alpha strategies.
QUALITY_SHARPE_THRESHOLD: float = 0.30

# Neutral MR signal
NEUTRAL_MR_ENABLED: bool = True
NEUTRAL_MR_RETURN_WINDOW: int = 21
NEUTRAL_MR_ENTRY_ZSCORE: float = 1.0
NEUTRAL_MR_POSITION_SIZE: float = 0.5

# Miscellaneous
RANDOM_SEED: int = 42
LOG_LEVEL: str = "INFO"
