# Bootstrap OUA Evaluation

## Executive Summary

**Recommendation: Stick with jackknife, do NOT use bootstrap for OUA.**

Bootstrapping oracle samples (B=50) consistently **overestimates** oracle variance by 3.4x on average, compared to 1.5x for the current K-fold jackknife. While both methods have limitations, jackknife is substantially closer to the truth.

## Key Findings

### 1. Bootstrap Overestimates Variance

Across 8 scenarios (oracle 5-50%, n=250-500):

| Method | Average Ratio | Low Oracle (≤10%) |
|--------|--------------|-------------------|
| Jackknife (K=3-5) | 1.54x | 2.16x |
| Bootstrap (B=50) | **3.40x** | **3.26x** |

**Ratio interpretation**:
- < 1.0: Underestimates variance → under-coverage
- = 1.0: Accurate
- > 1.0: Overestimates variance → over-coverage (but overly wide CIs)

### 2. Why Bootstrap Doesn't Help

Both jackknife and bootstrap measure **resampling uncertainty** within a single oracle sample. They don't capture:

- **Extrapolation uncertainty**: Applying calibration trained on oracle slice to full evaluation set
- **Selection uncertainty**: Different random oracle selections give wildly different calibrations

Evidence from real data (n_oracle=12):
```
Oracle variance across 20 independent runs:
  Mean: 6.14e-04
  Std:  7.79e-04
  CV:   1.27 (127%!)
  Range: 8.4e-06 to 3.5e-03 (400x variation)
```

The oracle variance itself varies by **400x** across runs with the same setup. Neither jackknife nor bootstrap captures this cross-run variability—they only measure within-run resampling variance.

### 3. The Fundamental Problem

When oracle << sample, the calibration function f̂(S) varies wildly depending on which oracle samples were selected:

```
n_oracle=12 runs showing oracle contribution to total variance:
  Run 1: 10.6% (tiny oracle variance)
  Run 2: 91.3% (huge oracle variance)
  Run 3: 98.0% (almost all variance)
  Run 4:  8.4% (tiny oracle variance)
```

**This is why coverage is poor at low oracle percentage**: The variance of the calibration function (across different oracle selections) dominates, but K-fold jackknife only captures within-selection uncertainty.

## Detailed Results

### Simulation Comparison

| Oracle % | n_oracle | Jackknife Ratio | Bootstrap Ratio | Better |
|----------|----------|----------------|-----------------|---------|
| 5% | 12 | 4.80 | 3.43 | Bootstrap |
| 10% | 25 | 0.81 | 3.10 | Jackknife |
| 20% | 50 | 0.37 | 2.41 | Jackknife |
| 50% | 125 | 1.05 | 4.22 | Jackknife |
| 5% | 25 | 2.12 | 4.45 | Jackknife |
| 10% | 50 | 0.89 | 2.05 | Jackknife |
| 20% | 100 | 1.49 | 4.00 | Jackknife |
| 50% | 250 | 0.79 | 3.59 | Jackknife |

Bootstrap wins only 1/8 scenarios (n_oracle=12, 5%), and even there it still overestimates by 3.4x.

### Why Jackknife Is Still Better

While jackknife also has issues (sometimes under-estimates at 20-50%, sometimes over-estimates at 5%):

1. **Closer to truth on average**: 1.54x vs 3.40x
2. **Less conservative**: Bootstrap's 3.4x overestimation would make CIs so wide they're not useful
3. **Computational cost**: K=3-5 folds vs B=50 resamples
4. **Theoretical grounding**: Jackknife is standard for this type of problem

## What Would Actually Help

Neither jackknife nor bootstrap addresses the root cause: **extrapolation uncertainty** when oracle << sample.

Better approaches (from writeup):

### 1. S-Coverage Diagnostic + Penalty
- Measure how well oracle samples span the judge score distribution
- Add extrapolation penalty when coverage is poor
- Formula: `var_total = var_jackknife * (1 + penalty(S_coverage))`

### 2. Empirical Inflation Factors
- Calibrate multipliers based on oracle percentage
- Example: At oracle=5%, multiply jackknife variance by 4x
- Simple and directly addresses observed under-coverage

### 3. Honest Confidence Intervals
- Explicitly model extrapolation uncertainty
- Use hierarchical/Bayesian approach to quantify calibration function variability
- Most principled but complex

## Recommendation

**Do NOT implement bootstrap OUA.** It makes the problem worse by overestimating variance.

Instead:
1. **Short term**: Document the oracle percentage requirements clearly (already done)
2. **Medium term**: Implement empirical inflation factors or S-coverage penalty
3. **Long term**: Consider honest inference for calibration uncertainty

The current jackknife approach is reasonable given its limitations. The real solution is to **use sufficient oracle percentage** (≥10-20% for direct mode) rather than trying to fix inference at inadequate oracle coverage.

## Technical Details

### Simulation Setup
- 8 scenarios varying oracle percentage (5%, 10%, 20%, 50%)
- Sample sizes: 250, 500
- Judge scores: Beta(2,2) distribution
- Oracle labels: Correlated with judge + noise (SE=0.15)
- Empirical variance: 100 independent oracle selections

### Code
See `evaluate_bootstrap_oua.py` for full implementation.

### Limitations
- Simulation may not perfectly match real data distribution
- Could test on actual experimental data if we stored raw judge scores
- Bootstrap hyperparameters (B=50) not extensively tuned

## Conclusion

Bootstrap OUA is **not recommended**. It overestimates variance by 3.4x on average, making CIs too wide while not addressing the fundamental problem of extrapolation uncertainty when oracle << sample.

The best solution remains: **Use adequate oracle percentage** (≥10-20% for direct mode, per the oracle size recommendations).
