# Preferences Ablation: CJE vs Binary Calibration Results

## Summary

- Continuous experiments: 900
- Binary experiments: 450

## RMSE Comparison

| Method | Mean RMSE | Std RMSE | N |
|--------|-----------|----------|---|
| naive-direct | 0.0707 | 0.0021 | 450 |
| direct | 0.0043 | 0.0059 | 450 |
| binary-naive | 0.0569 | 0.0034 | 450 |
| binary-adjusted | 0.0894 | 0.0773 | 450 |

## Continuous Transport Audit

| Policy | Oracle | naive-direct Bias | direct Bias | direct Passes |
|--------|--------|-------------------|-------------|---------------|
| verbose | 0.505 | 0.064 | -0.000 | PASS |
| terse | 0.505 | 0.083 | -0.000 | PASS |
| contrarian | 0.505 | -0.064 | -0.000 | PASS |

## Bias-Corrected RMSE (Transport Residual Adjustment)

| Method | Raw RMSE | Bias-Corrected RMSE | Improvement |
|--------|----------|---------------------|-------------|
| CJE (direct) | 0.0043 | 0.0044 | -1.3% |

*Correction formula: `corrected = estimate - transport_bias`*

## Binary Confusion Matrix Audit

| Policy | Sensitivity | Specificity | Youden's J | Transport |
|--------|-------------|-------------|------------|-----------|
| base | 0.824 | 0.770 | 0.594 | PASS |
| verbose | 0.821 | 0.732 | 0.553 | PASS |
| terse | 0.704 | 0.568 | 0.272 | FAIL |
| contrarian | 0.627 | 0.788 | 0.415 | PASS |

### Youden's J Comparison

| Domain | Mean Youden's J | Binary Works? |
|--------|-----------------|---------------|
| MBPP (code) | 0.054 | No |
| Arena (prefs) | 0.458 | Yes |

## Ranking Accuracy (Kendall τ)

| Method | Mean τ | Std τ | N |
|--------|--------|-------|---|
| naive-direct | -0.333 | 0.000 | 450 |
| direct | -0.259 | 0.727 | 450 |
| binary-naive | -0.357 | 0.123 | 450 |
| binary-adjusted | 0.040 | 0.573 | 450 |

## Key Findings

1. **Best RMSE**: direct
2. **CJE beats binary** by 95.2%
3. **Youden's J = 0.458** (vs 0.054 on MBPP)
   - Binary calibration is valid on preferences (J > 0.3)
