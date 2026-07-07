# Out-of-sample report — 2026-07-07

- Train+val window: … → **2024-01-02**
- Test window: **2024-01-03 → 2026-07-02**

Run settings:

- mode: `classical`
- tickers: `['ES=F', 'NQ=F', 'CL=F', '6E=F', 'NG=F', 'HG=F', 'ZN=F']`
- start: `2010-01-01`
- end: `today`
- cost_bps: `2.0`
- hurst_rough_threshold: `0.43`
- hurst_trend_threshold: `0.57`
- train_split: `0.7`
- val_split: `0.15`
- h_conviction_sizing: `False`
- data: `parquet cache through 2026-07-02 (offline run)`

## Portfolio (OOS)

- Sharpe: **1.60**
- Sortino: **1.50**
- Annual return: **11.9%**
- Annual vol: **7.2%**
- Max drawdown: **4.5%**
- Calmar: **2.64**
- Win rate: **50.1%**
- Turnover (ann.): **16.57x**
- Cost assumption: **2.0 bps**
- Alpha vs benchmark: **5.5%**
- Beta vs benchmark: **0.34**
- Information ratio: **-0.57**
- Test observations: **645**

## Per-instrument metrics (OOS)

| index | sharpe | sortino | max_drawdown | calmar | win_rate | annualized_return | annualized_vol | profit_factor | n_obs | turnover | hrp_weight |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ES=F | 1.5513 | 1.4609 | 0.0556 | 1.9665 | 0.4922 | 0.1093 | 0.0684 | 1.3584 | 644.0000 | 16.2671 | 0.5000 |
| NQ=F | 1.5046 | 1.3882 | 0.0496 | 2.5815 | 0.5078 | 0.1280 | 0.0823 | 1.3642 | 644.0000 | 16.8759 | 0.5000 |
| CL=F | -0.6724 | -0.5181 | 0.2517 | -0.2885 | 0.1944 | -0.0726 | 0.1041 | 0.8382 | 643.0000 | 27.9921 | 0.0000 |
| 6E=F | 0.2443 | 0.2149 | 0.0545 | 0.1787 | 0.2422 | 0.0097 | 0.0436 | 1.0609 | 644.0000 | 57.8025 | 0.0000 |
| NG=F | -0.8754 | -0.6795 | 0.3588 | -0.3595 | 0.2224 | -0.1290 | 0.1455 | 0.7929 | 643.0000 | 17.5161 | 0.0000 |
| HG=F | 1.1252 | 1.0694 | 0.1203 | 1.1284 | 0.2955 | 0.1358 | 0.1195 | 1.2882 | 643.0000 | 27.8971 | 0.0000 |
| ZN=F | 0.2651 | 0.2269 | 0.0717 | 0.1198 | 0.2034 | 0.0086 | 0.0345 | 1.0694 | 644.0000 | 63.4899 | 0.0000 |

## HRP weights (train window only)

| index | weight |
|---|---|
| ES=F | 0.5000 |
| NQ=F | 0.5000 |
| CL=F | 0.0000 |
| 6E=F | 0.0000 |
| NG=F | 0.0000 |
| HG=F | 0.0000 |
| ZN=F | 0.0000 |

## Regime attribution

| ticker | regime | frac_bars | ann_return | sharpe | n_obs |
|---|---|---|---|---|---|
| ES=F | MEAN_REVERT | 0.4845 | 0.0242 | 0.4144 | 312 |
| ES=F | NEUTRAL | 0.2842 | 0.1011 | 2.0269 | 183 |
| ES=F | MOMENTUM | 0.2314 | 0.3228 | 2.9835 | 149 |
| NQ=F | MEAN_REVERT | 0.4457 | 0.0471 | 0.7021 | 287 |
| NQ=F | NEUTRAL | 0.2826 | 0.0927 | 1.7095 | 182 |
| NQ=F | MOMENTUM | 0.2717 | 0.3174 | 2.3722 | 175 |
| CL=F | MEAN_REVERT | 0.5397 | -0.0424 | -0.3955 | 347 |
| CL=F | NEUTRAL | 0.2317 | -0.0067 | -0.1757 | 149 |
| CL=F | MOMENTUM | 0.2286 | -0.1980 | -1.3560 | 147 |
| 6E=F | MEAN_REVERT | 0.2842 | -0.0009 | 0.0020 | 183 |
| 6E=F | NEUTRAL | 0.4006 | -0.0219 | -0.9571 | 258 |
| 6E=F | MOMENTUM | 0.3152 | 0.0615 | 1.0190 | 203 |
| NG=F | MEAN_REVERT | 0.2955 | 0.0923 | 1.0244 | 190 |
| NG=F | NEUTRAL | 0.3095 | -0.0035 | -0.0459 | 199 |
| NG=F | MOMENTUM | 0.3950 | -0.3383 | -1.8318 | 254 |
| HG=F | MEAN_REVERT | 0.2535 | -0.0217 | -0.1690 | 163 |
| HG=F | NEUTRAL | 0.2348 | -0.0105 | -0.2059 | 151 |
| HG=F | MOMENTUM | 0.5117 | 0.3029 | 1.8637 | 329 |
| ZN=F | MEAN_REVERT | 0.5311 | 0.0290 | 0.9029 | 342 |
| ZN=F | NEUTRAL | 0.2158 | -0.0016 | -0.0925 | 139 |
| ZN=F | MOMENTUM | 0.2531 | -0.0245 | -0.4924 | 163 |
| PORTFOLIO | MEAN_REVERT | 0.3814 | 0.0219 | 0.3696 | 246 |
| PORTFOLIO | NEUTRAL | 0.1690 | 0.1366 | 3.2054 | 109 |
| PORTFOLIO | MOMENTUM | 0.2016 | 0.4096 | 3.3871 | 130 |

## Cost sensitivity

| cost_bps | portfolio_sharpe | portfolio_ann_return | portfolio_max_drawdown | sharpe_ES=F | sharpe_NQ=F | sharpe_CL=F | sharpe_6E=F | sharpe_NG=F | sharpe_HG=F | sharpe_ZN=F |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.0000 | 1.6427 | 0.1224 | 0.0449 | 1.5977 | 1.5449 | -0.6188 | 0.5097 | -0.8512 | 1.1722 | 0.6308 |
| 2.0000 | 1.5976 | 0.1187 | 0.0450 | 1.5513 | 1.5047 | -0.6724 | 0.2443 | -0.8754 | 1.1252 | 0.2651 |
| 5.0000 | 1.5298 | 0.1132 | 0.0453 | 1.4815 | 1.4442 | -0.7527 | -0.1526 | -0.9117 | 1.0548 | -0.2868 |
| 10.0000 | 1.4163 | 0.1040 | 0.0457 | 1.3643 | 1.3429 | -0.8863 | -0.8056 | -0.9721 | 0.9373 | -1.2012 |
| 20.0000 | 1.1877 | 0.0860 | 0.0479 | 1.1279 | 1.1391 | -1.1520 | -2.0477 | -1.0930 | 0.7026 | -2.9237 |
| 30.0000 | 0.9575 | 0.0682 | 0.0525 | 0.8895 | 0.9339 | -1.4150 | -3.1604 | -1.2139 | 0.4687 | -4.3908 |

## README results section

Paste the block below into the README verbatim:

```markdown
## Results

True out-of-sample test window **2024-01-03 to 2026-07-02** (trained/calibrated on data up to 2024-01-02 only; thresholds, universe and filters fixed before the test window was evaluated).

| Metric | OOS value |
|---|---|
| Sharpe | 1.60 |
| Sortino | 1.50 |
| Annual return | 11.9% |
| Annual vol | 7.2% |
| Max drawdown | 4.5% |
| Calmar | 2.64 |
| Win rate | 50.1% |
| Turnover (ann.) | 16.57x |
| Cost assumption | 2.0 bps |
| Alpha vs benchmark | 5.5% |
| Beta vs benchmark | 0.34 |
| Information ratio | -0.57 |
| Test observations | 645 |

Run settings: mode=classical, tickers=['ES=F', 'NQ=F', 'CL=F', '6E=F', 'NG=F', 'HG=F', 'ZN=F'], start=2010-01-01, end=today, cost_bps=2.0, hurst_rough_threshold=0.43, hurst_trend_threshold=0.57, train_split=0.7, val_split=0.15, h_conviction_sizing=False, data=parquet cache through 2026-07-02 (offline run).

Full tables (per-instrument metrics, regime attribution, cost sensitivity,
HRP weights) are saved per run under `results/metrics/` and
`results/backtest_reports/`.
```

## Overlay ablation

Portfolio OOS metrics with the sizing/bias overlays toggled off one at a
time.  This separates how much performance comes from H(t) regime
switching itself versus the long-bias, vol-target and H-conviction
overlays.

| variant | sharpe | sortino | annualized_return | annualized_vol | max_drawdown | calmar | win_rate | turnover | total_cost_bps | alpha | beta | information_ratio | n_obs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 1.5976 | 1.5047 | 0.1187 | 0.0718 | 0.0450 | 2.6363 | 0.5008 | 16.5715 | 2.0000 | 0.0552 | 0.3374 | -0.5660 | 645.0000 |
| no_long_bias | 0.1703 | 0.1686 | 0.0051 | 0.0329 | 0.0549 | 0.0925 | 0.4868 | 45.7096 | 2.0000 | -0.0047 | 0.0580 | -1.2179 | 645.0000 |
| no_vol_target | 1.4413 | 1.3054 | 0.1165 | 0.0786 | 0.0453 | 2.5729 | 0.4977 | 16.6171 | 2.0000 | 0.0495 | 0.3616 | -0.5793 | 645.0000 |
| h_conviction_on | 0.9779 | 1.0074 | 0.0152 | 0.0156 | 0.0190 | 0.7992 | 0.4946 | 5.1236 | 2.0000 | 0.0073 | 0.0449 | -1.1579 | 645.0000 |
| signal_only | -0.1945 | -0.1794 | -0.0107 | 0.0491 | 0.1016 | -0.1052 | 0.4992 | 48.2639 | 2.0000 | -0.0308 | 0.1199 | -1.3712 | 645.0000 |
