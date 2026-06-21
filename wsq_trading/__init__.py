"""Rough-Heston trading model with neural-operator Hurst estimation.

Modules are imported directly, e.g.::

    from wsq_trading.data import DataLoader, DataValidator, DataCleaner
    from wsq_trading.spde import SPDESimulator
    from wsq_trading.features import FeatureEngineer
    from wsq_trading.hurst import RSAnalysis, DFA, WhittleMLE
    from wsq_trading.fno import FNO1d, FNOTrainer
    from wsq_trading.signals import HurstRegimeSignalGenerator
    from wsq_trading.portfolio import HRPAllocator
    from wsq_trading.backtest import WalkForwardEngine

(The neural-operator code under ``fno`` pulls in torch, so it is only imported
when you actually need it rather than at package import time.)
"""

__version__ = "0.1.0"
