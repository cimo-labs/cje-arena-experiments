# Binary vs Continuous Calibration: MBPP Results

**To:** Analyst
**From:** Eddie Landesberg
**Date:** January 11, 2026
**Re:** Follow-up on Lee et al. binary baseline suggestion

---

## Executive Summary

We implemented your suggested binary baseline comparison (Rogan-Gladen misclassification adjustment). **Continuous calibration wins decisively** - but the reason is instructive.

| Method | RMSE (transport-valid) | Change |
|--------|------------------------|--------|
| naive-direct (continuous) | 0.048 | baseline |
| **direct (continuous)** | **0.029** | **-39%** |
| binary-naive | 0.16-0.23 | +235-373% |
| binary-adjusted | 0.20-0.51 | +323-958% |

Binary calibration *increases* error rather than reducing it.

---

## Why Binary Fails

### The Binary Judge Has Poor Discriminative Power

| Metric | Value | Problem |
|--------|-------|---------|
| Sensitivity | 70.3% | OK - says YES when code passes |
| **Specificity** | **35.2%** | **Bad - says YES even when code fails 65% of the time** |
| **Youden's J** | **0.054** | **At validity threshold** |

### The Math Breaks Down

Rogan-Gladen: θ̂ = (p̂ + Sp - 1) / J

With J = 0.054, a 1pp error in p̂ becomes an **18.5pp error** in θ̂.

| Youden's J | Error Amplification |
|------------|---------------------|
| 0.50 | 2x |
| 0.20 | 5x |
| 0.10 | 10x |
| **0.054** | **18.5x** |

---

## Transport Audit Comparison

Both continuous and binary methods need transport audits. Here's how they compare:

| Policy | Score Transport | Binary Transport | Agreement |
|--------|-----------------|------------------|-----------|
| enhanced | ✓ PASS | ✓ PASS | ✓ |
| terse | ✓ PASS | ✗ FAIL | ✗ |
| mini | ✗ FAIL | ✗ FAIL | ✓ |
| unhelpful | ✗ FAIL | ✗ FAIL | ✓ |

The binary audit is more conservative (fails terse) because the low base specificity makes the confusion matrix unstable.

---

## Key Takeaways

1. **Continuous beats binary when continuous is available**
   - Continuous scores contain information that binary thresholding loses
   - 39% RMSE reduction vs 235%+ RMSE increase

2. **Binary calibration has a validity precondition**
   - Youden's J must be >> 0 for Rogan-Gladen to work
   - Our binary judge has J = 0.054, making it borderline invalid
   - Recommendation: Check J > 0.3 before using binary adjustment

3. **Both methods need transport diagnostics**
   - Continuous: Mean residual E[Y - f(S)] = 0
   - Binary: Confusion matrix stability (Se, Sp constant across policies)
   - The diagnostic framework is method-agnostic

4. **The "continuous scores are sketchy" objection is backwards**
   - Continuous scores provide *more* information, not less
   - Binary thresholding discards the continuous signal
   - If anything, binary is sketchier (all-or-nothing bet)

---

## Paper Framing Suggestion

> We compared continuous calibration (isotonic regression) against binary misclassification adjustment (Rogan-Gladen). Continuous calibration reduced RMSE by 39% on transport-valid policies, while binary adjustment increased RMSE due to low judge specificity (Youden's J = 0.054). Both methods require transport diagnostics: mean-residual audits for continuous scores, confusion-matrix stability for binary. When continuous scores are available, they should be preferred.

---

## Ranking Accuracy

Level estimation aside, how well do the methods rank policies?

| Method | Kendall τ | Top-1* | Unhelpful Last | Mini Separated |
|--------|-----------|--------|----------------|----------------|
| Binary naive | 0.25 | 0% | 100% | 0% |
| Binary adjusted | 0.28 | 0% | 100% | 0% |
| Continuous naive | 0.33 | 98% | 100% | 0% |
| Continuous direct | 0.21 | 67% | 96% | 1% |

*Top-1 = correctly identifies terse OR enhanced as best (they're a statistical tie)

### Key Observations

1. **All methods rank unhelpful last** - the 31pp gap is easily detectable
2. **None reliably separate mini** from top tier (6.5pp gap ≈ MDE)
3. **Binary always ranks: enhanced > mini > terse > unhelpful**
4. **Calibrated continuous has transport failures** - 4% rank unhelpful NOT last

### Most Common Rankings

```
Binary (both):        enhanced > mini > terse > unhelpful (75-84%)
Continuous naive:     enhanced > mini > terse > unhelpful (98%)
Continuous direct:    enhanced > mini > terse > unhelpful (66%)
                      mini > enhanced > terse > unhelpful (28%)
```

### Why Calibrated Continuous Has Lower Ranking Accuracy

The calibrated method (direct) applies corrections to transport-invalid policies:
- mini: Calibration pushes estimate UP (wrong direction)
- unhelpful: Calibration pushes estimate UP significantly

This inflates estimates for policies that should rank lower, hurting ranking accuracy.

**Takeaway:** For ranking, use naive estimates OR stratify by transport validity.

---

## Files

- `binary_baseline.jsonl` - 1,250 experiment results
- `binary_baseline_summary.json` - Aggregated statistics
- `REPORT.md` Section 13 - Full technical details

---

## Cost

- Binary judge API calls: ~$12 (6,425 calls × gpt-4o-mini)
- Compute time: ~5 minutes for experiments

---

*This comparison directly addresses the "why not compare to Lee et al.?" reviewer question and strengthens the "diagnostics-first" narrative.*
