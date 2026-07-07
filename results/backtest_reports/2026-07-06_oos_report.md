# Out-of-sample report — 2026-07-06

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

## Portfolio (OOS)

- Sharpe: **0.98**
- Sortino: **1.01**
- Annual return: **1.5%**
- Annual vol: **1.6%**
- Max drawdown: **1.9%**
- Calmar: **0.80**
- Win rate: **49.5%**
- Turnover (ann.): **5.12x**
- Cost assumption: **2.0 bps**
- Alpha vs benchmark: **0.7%**
- Beta vs benchmark: **0.04**
- Information ratio: **-1.16**
- Test observations: **645**

## Per-instrument metrics (OOS)

| index | sharpe | sortino | max_drawdown | calmar | win_rate | annualized_return | annualized_vol | profit_factor | n_obs | turnover | hrp_weight |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ES=F | 1.5475 | 1.5921 | 0.0156 | 1.7941 | 0.4922 | 0.0280 | 0.0179 | 1.4780 | 644.0000 | 6.0950 | 0.3333 |
| NQ=F | 1.6754 | 1.8532 | 0.0084 | 3.7040 | 0.5062 | 0.0311 | 0.0184 | 1.5141 | 644.0000 | 5.5676 | 0.3333 |
| CL=F | -0.4565 | -0.3635 | 0.0445 | -0.2600 | 0.1944 | -0.0116 | 0.0248 | 0.8624 | 643.0000 | 6.1296 | 0.0000 |
| 6E=F | 0.1146 | 0.0915 | 0.0163 | 0.0795 | 0.2407 | 0.0013 | 0.0119 | 1.0338 | 644.0000 | 12.2117 | 0.0000 |
| NG=F | -0.4172 | -0.3504 | 0.0613 | -0.2165 | 0.2224 | -0.0133 | 0.0309 | 0.8789 | 643.0000 | 3.7081 | 0.3333 |
| HG=F | 1.3275 | 1.4488 | 0.0400 | 1.3446 | 0.2955 | 0.0537 | 0.0400 | 1.4300 | 643.0000 | 9.1472 | 0.0000 |
| ZN=F | 0.0807 | 0.0682 | 0.0257 | 0.0371 | 0.2034 | 0.0010 | 0.0128 | 1.0259 | 644.0000 | 18.8615 | 0.0000 |

## HRP weights (train window only)

| index | weight |
|---|---|
| ES=F | 0.3333 |
| NQ=F | 0.3333 |
| CL=F | 0.0000 |
| 6E=F | 0.0000 |
| NG=F | 0.3333 |
| HG=F | 0.0000 |
| ZN=F | 0.0000 |

## Regime attribution

| ticker | regime | frac_bars | ann_return | sharpe | n_obs |
|---|---|---|---|---|---|
| ES=F | MEAN_REVERT | 0.4845 | 0.0029 | 0.2756 | 312 |
| ES=F | NEUTRAL | 0.2842 | 0.0148 | 2.0375 | 183 |
| ES=F | MOMENTUM | 0.2314 | 0.0999 | 2.9267 | 149 |
| NQ=F | MEAN_REVERT | 0.4457 | 0.0078 | 0.7488 | 287 |
| NQ=F | NEUTRAL | 0.2826 | 0.0102 | 0.9862 | 182 |
| NQ=F | MOMENTUM | 0.2717 | 0.0935 | 2.9340 | 175 |
| CL=F | MEAN_REVERT | 0.5397 | -0.0106 | -0.3771 | 347 |
| CL=F | NEUTRAL | 0.2317 | -0.0012 | -0.2344 | 149 |
| CL=F | MOMENTUM | 0.2286 | -0.0241 | -0.7966 | 147 |
| 6E=F | MEAN_REVERT | 0.2842 | -0.0093 | -0.5997 | 183 |
| 6E=F | NEUTRAL | 0.4006 | -0.0035 | -1.0052 | 258 |
| 6E=F | MOMENTUM | 0.3152 | 0.0171 | 1.1472 | 203 |
| NG=F | MEAN_REVERT | 0.2955 | 0.0318 | 1.5148 | 190 |
| NG=F | NEUTRAL | 0.3095 | -0.0026 | -0.3356 | 199 |
| NG=F | MOMENTUM | 0.3950 | -0.0537 | -1.2007 | 254 |
| HG=F | MEAN_REVERT | 0.2535 | 0.0146 | 0.5777 | 163 |
| HG=F | NEUTRAL | 0.2348 | 0.0028 | 0.3353 | 151 |
| HG=F | MOMENTUM | 0.5117 | 0.0983 | 1.8112 | 329 |
| ZN=F | MEAN_REVERT | 0.5311 | 0.0085 | 0.5777 | 342 |
| ZN=F | NEUTRAL | 0.2158 | -0.0011 | -0.4079 | 139 |
| ZN=F | MOMENTUM | 0.2531 | -0.0131 | -0.9761 | 163 |
| PORTFOLIO | MEAN_REVERT | 0.4372 | -0.0069 | -0.4478 | 282 |
| PORTFOLIO | NEUTRAL | 0.2140 | 0.0002 | 0.0245 | 138 |
| PORTFOLIO | MOMENTUM | 0.2248 | 0.0783 | 3.4443 | 145 |

## Cost sensitivity

| cost_bps | portfolio_sharpe | portfolio_ann_return | portfolio_max_drawdown | sharpe_ES=F | sharpe_NQ=F | sharpe_CL=F | sharpe_6E=F | sharpe_NG=F | sharpe_HG=F | sharpe_ZN=F |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.0000 | 1.0419 | 0.0162 | 0.0188 | 1.6115 | 1.7319 | -0.4073 | 0.3201 | -0.3932 | 1.3723 | 0.3740 |
| 2.0000 | 0.9780 | 0.0152 | 0.0190 | 1.5475 | 1.6755 | -0.4565 | 0.1146 | -0.4172 | 1.3275 | 0.0807 |
| 5.0000 | 0.8817 | 0.0137 | 0.0194 | 1.4508 | 1.5903 | -0.5301 | -0.1916 | -0.4532 | 1.2603 | -0.3591 |
| 10.0000 | 0.7200 | 0.0111 | 0.0200 | 1.2872 | 1.4463 | -0.6524 | -0.6936 | -0.5133 | 1.1477 | -1.0832 |
| 20.0000 | 0.3922 | 0.0059 | 0.0212 | 0.9529 | 1.1518 | -0.8954 | -1.6509 | -0.6333 | 0.9210 | -2.4432 |
| 30.0000 | 0.0599 | 0.0008 | 0.0224 | 0.6111 | 0.8498 | -1.1356 | -2.5245 | -0.7532 | 0.6928 | -3.6161 |

## README results section

Paste the block below into the README verbatim:

```markdown
## Results

True out-of-sample test window **2024-01-03 to 2026-07-02** (trained/calibrated on data up to 2024-01-02 only; thresholds, universe and filters fixed before the test window was evaluated).

| Metric | OOS value |
|---|---|
| Sharpe | 0.98 |
| Sortino | 1.01 |
| Annual return | 1.5% |
| Annual vol | 1.6% |
| Max drawdown | 1.9% |
| Calmar | 0.80 |
| Win rate | 49.5% |
| Turnover (ann.) | 5.12x |
| Cost assumption | 2.0 bps |
| Alpha vs benchmark | 0.7% |
| Beta vs benchmark | 0.04 |
| Information ratio | -1.16 |
| Test observations | 645 |

Run settings: mode=classical, tickers=['ES=F', 'NQ=F', 'CL=F', '6E=F', 'NG=F', 'HG=F', 'ZN=F'], start=2010-01-01, end=today, cost_bps=2.0, hurst_rough_threshold=0.43, hurst_trend_threshold=0.57, train_split=0.7, val_split=0.15.

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
| baseline | 0.9779 | 1.0074 | 0.0152 | 0.0156 | 0.0190 | 0.7992 | 0.4946 | 5.1236 | 2.0000 | 0.0073 | 0.0449 | -1.1579 | 645.0000 |
| no_long_bias | 0.8117 | 0.8729 | 0.0200 | 0.0247 | 0.0267 | 0.7465 | 0.3907 | 6.4277 | 2.0000 | 0.0177 | 0.0137 | -1.0775 | 645.0000 |
| no_vol_target | -0.0435 | -0.0398 | -0.0026 | 0.0412 | 0.0731 | -0.0361 | 0.4837 | 7.7074 | 2.0000 | -0.0103 | 0.0482 | -1.2387 | 645.0000 |
| no_h_conviction | 1.5976 | 1.5047 | 0.1187 | 0.0718 | 0.0450 | 2.6363 | 0.5008 | 16.5715 | 2.0000 | 0.0552 | 0.3374 | -0.5660 | 645.0000 |
| signal_only | -0.5908 | -0.5194 | -0.2029 | 0.3038 | 0.5389 | -0.3764 | 0.3767 | 51.4206 | 2.0000 | -0.2100 | 0.1707 | -1.0941 | 645.0000 |
