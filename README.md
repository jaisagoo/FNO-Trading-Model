# Rough-Heston Trading Model

A quantitative trading system that predicts a time-varying Hurst exponent with a
Fourier Neural Operator (FNO) and switches between mean-reversion, trend-following
and neutral strategies depending on the estimated roughness regime.

## Idea

Volatility is rough: realised variance behaves like a fractional process with a
small Hurst exponent. If the local roughness `H(t)` can be estimated in real time,
it tells you which regime the price is in:

- `H < 0.35` — rough / anti-persistent → mean-reversion
- `0.35 ≤ H ≤ 0.65` — close to Brownian → stay neutral
- `H > 0.65` — persistent / trending → momentum

The FNO is trained on rough-Heston SPDE paths (where the true `H` is known), then
used to label live return windows. Signals are combined across instruments with
Hierarchical Risk Parity and evaluated with a walk-forward backtest.

## Layout

```
wsq_trading/
    config.py       all settings live here
    constants.py    enums and math constants
    utils.py        logging, seeding, timing helpers
    data.py         load / validate / clean OHLCV
    spde.py         rough-Heston SPDE path simulator
    features.py     log-returns, realised vol, rolling stats
    hurst.py        classical estimators (R/S, DFA, Whittle MLE)
    fno.py          Fourier Neural Operator + trainer
    signals.py      H(t) -> regime -> long/short/flat
    portfolio.py    Hierarchical Risk Parity allocator
    backtest.py     walk-forward engine + performance metrics
    pipeline.py     end-to-end orchestration
    tests/          pytest suite
```

The FNO code in `fno.py` is the only part that needs PyTorch, so it is imported
lazily rather than at package import time.

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
make test
```

## Make targets

`make test`, `make test-cov`, `make lint`, `make format`, `make typecheck`,
`make data-dirs`, `make clean`.

## References

- Gatheral, Jaisson & Rosenbaum (2018), *Volatility is rough*
- El Euch & Rosenbaum (2019), *The characteristic function of rough Heston models*
- Li, Kovachki et al. (2021), *Fourier Neural Operator for Parametric PDEs*
- López de Prado (2016), *Building diversified portfolios that outperform out of sample* (HRP)
