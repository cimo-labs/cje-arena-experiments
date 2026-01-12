# Preferences Ablation: CJE vs Binary Calibration Results

## Summary

- Continuous experiments: 900
- Binary experiments: 450

## Experiment Setup

**Source Dataset:**
- lmsys/chatbot_arena_conversations (Hugging Face)
- Filtered to non-tie battles with responses > 10 chars
- Final: 9,999 preference pairs (50.55% A-wins, 49.45% B-wins)

**Judge Configuration:**
- Model: OpenAI gpt-4o-mini (temperature=0.0, deterministic)
- Continuous output: 0-100 score normalized to 0-1 scale
- Binary output: A/B categorical prediction
- 4 policy variants: base (neutral), verbose, terse, contrarian

**Oracle Protocol:**
- Oracle = human preference labels (metadata.oracle_label)
- Sampling: Random slice (k = n_samples × oracle_coverage)
- Same oracle labels used for both CJE and Rogan-Gladen (fair comparison)

**Experimental Grid:**
- Sample sizes: 2,000 / 5,000 / 10,000
- Oracle coverages: 1% / 2% / 5% / 10% / 25%
- Seeds: 30 independent random seeds (0-29)
- Total: 3 × 5 × 30 = 450 experiments per method

**CJE Inference Settings:**
- Bootstrap: 500 replicates per experiment
- Calibration: Monotone isotonic regression (fit on base policy)
- Transport correction: Residual augmentation on all target policies

**Binary CI Method:**
- Lee et al. (2026) Eq. 6 with Agresti-Coull adjustment
- Accounts for both test and calibration uncertainty
- Implementation: `binary_baseline.py:lee_et_al_confidence_interval()`

## RMSE Comparison

| Method | Mean RMSE | Std RMSE | N |
|--------|-----------|----------|---|
| naive-direct | 0.0707 | 0.0021 | 450 |
| direct | 0.0043 | 0.0059 | 450 |
| binary-naive | 0.0450 | 0.0039 | 450 |
| binary-adjusted | 0.0871 | 0.0802 | 450 |

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

### Youden's J Comparison

| Domain | Mean Youden's J | Binary Works? |
|--------|-----------------|---------------|
| MBPP (code) | 0.054 | No |
| Arena (prefs) | 0.473 | Yes |

## Ranking Accuracy (Kendall τ)

| Method | Mean τ | Std τ | N |
|--------|--------|-------|---|
| naive-direct | -0.333 | 0.000 | 450 |
| direct | -0.259 | 0.727 | 450 |
| binary-naive | 0.929 | 0.370 | 450 |
| binary-adjusted | -0.022 | 1.000 | 450 |

## 95% Confidence Interval Coverage

| Method | Coverage | N | Nominal |
|--------|----------|---|---------|
| CJE (direct) | 96.4% | 450 | 95% |
| Rogan-Gladen (Lee et al. CI) | 96.9% | 1340 | 95% |
| naive-direct | 0.0% | 450 | 95% |

*Both CJE and Rogan-Gladen achieve valid uncertainty quantification (~95% coverage).*

## CI Half-Widths (Precision)

| Method | Mean Half-Width | Median | Std | Valid Coverage? |
|--------|-----------------|--------|-----|-----------------|
| CJE (direct) | **0.015** | 0.014 | 0.005 | Yes (96.4%) |
| Rogan-Gladen | 0.172 | 0.130 | 0.125 | Yes (96.9%) |
| naive-direct | 0.009 | 0.008 | 0.003 | No (0.0%) |

*CJE achieves 11x narrower CIs than Rogan-Gladen while maintaining valid coverage.*

### Coverage by Policy (Rogan-Gladen)

| Policy | Coverage | N |
|--------|----------|---|
| base | 94.9% | 450 |
| verbose | 97.1% | 450 |
| terse | 98.9% | 440 |

## Key Findings

1. **Best RMSE**: CJE direct (0.0043) — 95% better than Rogan-Gladen
2. **Both methods have valid UQ**: CJE 96.4%, Rogan-Gladen 96.9% coverage
3. **CJE is 11x more precise**: Half-width 0.015 vs 0.172
4. **Youden's J = 0.473** (vs 0.054 on MBPP)
   - Binary calibration is valid on preferences (J > 0.3), but less efficient
