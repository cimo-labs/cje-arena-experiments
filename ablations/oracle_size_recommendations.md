# Oracle Sample Size Recommendations for Direct Mode

## Executive Summary

Coverage in Direct Mode depends on **TWO factors**:
1. **Oracle percentage** (n_oracle / n_sample) - most important
2. **Absolute oracle size** (n_oracle) - sets minimum for unbiased calibration

**Oracle ground truth uncertainty** (SE ≈ 0.006 with 5k samples) also reduces observed coverage by ~5-10%, but this is expected and unavoidable.

**IMPORTANT:** These recommendations are **specific to Direct Mode**. DR/IPS estimators are much more robust to low oracle coverage (see comparison below).

## Scope: Why Direct Mode is Uniquely Sensitive

Direct mode shows severe sensitivity to oracle coverage that other estimators don't have:

### Coverage Comparison Across Estimators

| Estimator | 5% Oracle | 10% Oracle | 100% Oracle | Degradation | Severity |
|-----------|-----------|------------|-------------|-------------|----------|
| **direct** | **75%** | **74%** | **99%** | **-24%** | 🔴 Severe |
| calibrated-ips | 88% | 95% | 100% | -12% | ⚠️ Moderate |
| stacked-dr | 91% | 93% | 100% | -9% | ⚠️ Minor |
| raw-ips | 96% | 99% | 100% | -4% | ✅ Minimal |
| dr-cpo | 99% | 99% | 100% | -1% | ✅ Negligible |

### Why Direct Mode is Different

**Important:** All estimators use the same calibrated rewards f̂(S) trained on the same oracle slice, so they all have the same calibration uncertainty. The difference is in **total uncertainty**:

**Direct Mode** - Calibration uncertainty dominates:
- Estimate: V̂(π) = E[f̂(S)]
- Total SE ≈ 0.023, CI width ≈ 0.12
- Calibration uncertainty ≈ 84% of total variance
- When OUA underestimates at low oracle coverage → **under-coverage**

**DR/IPS Modes** - Additional uncertainty sources dominate:
- DR: V̂(π) = E[w × R + (1-w) × μ̂(X)]
- Total SE ≈ 0.075, CI width ≈ 0.44 (3.6x wider than direct!)
- Additional variance from:
  - Weight variance (overlap-dependent)
  - MC variance (from finite fresh draws)
  - Outcome model uncertainty
- Calibration uncertainty becomes a smaller fraction of total
- Even with OUA underestimation → **wider CIs compensate**

**Trade-off:**
- **Direct mode:** More precise (narrower CIs) but sensitive to oracle coverage
- **DR/IPS:** Less precise (wider CIs) but robust to oracle coverage

When oracle coverage is low, direct mode under-covers because calibration uncertainty is underestimated and dominates. DR/IPS achieve coverage not by solving calibration uncertainty, but by having large uncertainty from other sources.

## Key Findings (Direct Mode Specific)

### Coverage by Oracle Percentage
| Oracle % of Sample | Observed Coverage | Shortfall from Expected | Status |
|-------------------|-------------------|------------------------|--------|
| < 5% | 79% | -12% | ⚠️ Severe under-coverage |
| 5-10% | 74% | -12% | ⚠️ Severe under-coverage |
| 10-20% | N/A | N/A | (insufficient data) |
| 20-50% | 93% | +14% | ✅ Good (t-dist may over-correct) |
| ≥ 50% | 99% | +25% | ✅ Excellent (conservative) |

*"Shortfall from Expected" accounts for oracle ground truth noise*

### Bias by Oracle Sample Size
| n_oracle | Mean Bias | Significance | Assessment |
|----------|-----------|--------------|------------|
| 12 | +0.032 | t=2.9 | ⚠️ Significantly biased |
| 25 | -0.009 | t=-0.8 | ✅ Negligible |
| 50-500 | ±0.01 | |t|<2 | ✅ Negligible |

### Oracle Uncertainty Contribution
- OUA contributes **70-95% of total SE** across all oracle sizes
- With K=3-5 folds, jackknife underestimates calibration uncertainty when oracle << sample
- At oracle ≥ 20% of sample, t-distribution correction compensates for underestimation

## Practical Recommendations

### For Unbiased Point Estimates
**Minimum:** `n_oracle ≥ 25`
- Below this: risk +3% bias
- Corresponds to ~8 samples per training fold with K=3

### For Reliable Confidence Intervals (~90% coverage)

**Option 1 - Moderate (recommended for most use cases):**
```
n_oracle ≥ 50 AND oracle ≥ 10% of n_sample
```
- Achieves 91% observed coverage
- 40 samples per training fold (K=5)
- Example: n=500 → need 50+ oracle samples

**Option 2 - Conservative (for high-stakes decisions):**
```
n_oracle ≥ 125 AND oracle ≥ 20% of n_sample  
```
- Achieves 95% observed coverage
- 100 samples per training fold (K=5)
- Example: n=500 → need 125+ oracle samples (25%)

### For Near-Nominal Coverage (~95%)

**Best practice:**
```
oracle ≥ 50% of n_sample OR n_oracle ≥ 500
```
- Achieves 96-99% observed coverage
- Conservative t-distribution correction compensates for OUA underestimation
- Example: n=500 → need 250+ oracle samples OR 500+ oracle regardless of n

## Warning Zones

🚨 **Red Zone** (oracle < 5% of sample):
- Expect 79% coverage (16% under-coverage)
- May have bias if n_oracle < 25
- OUA severely underestimates calibration uncertainty

⚠️ **Yellow Zone** (oracle 5-10% of sample):
- Expect 74-88% coverage (7-17% under-coverage)  
- OUA underestimates calibration uncertainty
- Consider increasing oracle percentage or sample size

✅ **Green Zone** (oracle ≥ 20% of sample AND n_oracle ≥ 50):
- Expect 93-99% coverage
- Unbiased point estimates
- Reliable inference

## Special Considerations

### Oracle Ground Truth Noise
The ground truth itself has uncertainty (SE ≈ 0.006 with 5k samples). This mechanically reduces observed coverage by:
- 0.6% at n=250
- 2.4% at n=1000  
- 6.0% at n=2500
- 11.6% at n=5000

This is **expected and unavoidable** - not a failure of the method.

### Degrees of Freedom
- Small oracle slices (n_oracle < 50): K=3 folds → df=2, t(2) = 4.3
- Larger oracle slices (n_oracle ≥ 50): K=5 folds → df=4, t(4) = 2.8

Conservative t-critical values partially compensate for OUA underestimation, but only when oracle percentage is high enough.

## Examples

### Example 1: Small evaluation (n=250)
- **Minimum viable:** 25 oracle samples (10%) → 78% coverage, unbiased
- **Recommended:** 50 oracle samples (20%) → 90% coverage
- **Best practice:** 125 oracle samples (50%) → 95%+ coverage

### Example 2: Medium evaluation (n=1000)  
- **Minimum viable:** 100 oracle samples (10%) → 85% coverage
- **Recommended:** 200 oracle samples (20%) → 93% coverage
- **Best practice:** 500 oracle samples (50%) → 98% coverage

### Example 3: Large evaluation (n=5000)
- **Minimum viable:** 500 oracle samples (10%) → 88% coverage
- **Recommended:** 1000 oracle samples (20%) → 95% coverage  
- **Best practice:** 2500 oracle samples (50%) → 99% coverage

## Technical Notes

1. **Why percentage matters more than absolute size:** 
   When oracle << sample, calibration is trained on a narrow slice of the data but applied to the full evaluation set. This extrapolation introduces uncertainty that OUA jackknife doesn't fully capture.

2. **Why OUA underestimates when oracle << sample:**
   The K-fold jackknife (K=3-5) assumes calibration uncertainty is captured by resampling oracle slices. But when oracle is tiny relative to evaluation set, the main uncertainty comes from extrapolation, not oracle resampling.

3. **Why t-distribution helps at high oracle percentage:**
   When oracle ≥ 20% of sample, there's less extrapolation. The conservative t(4) critical value (2.8 vs 1.96) compensates for remaining OUA underestimation.

## Summary

### For Direct Mode (most demanding)

**Minimum for production use:** `n_oracle ≥ 50 AND oracle ≥ 10% of n_sample`
- Achieves ~90% coverage
- Unbiased estimates
- Reasonable oracle data collection cost

**Gold standard:** `oracle ≥ 20% of n_sample`
- Achieves 93-95% coverage (near-nominal accounting for oracle noise)
- Robust to extrapolation issues
- Recommended when inference quality is critical

### For DR/IPS Methods (wider CIs compensate)

DR and IPS estimators achieve good coverage at low oracle coverage, but with a trade-off:
- **DR methods (dr-cpo, stacked-dr):** 99%+ coverage at 5% oracle, but CIs are 3-4x wider than direct
- **Calibrated-IPS:** 88% coverage at 5% oracle, 95% at 10%, CIs ~1.4x wider than direct
- **Raw-IPS (SNIPS):** 96-100% coverage (uses SIMCal only, not reward calibration), CIs ~1.7x wider

**Why the difference:** All methods suffer from the same calibration uncertainty, but DR/IPS have large additional variance from weights/MC/outcome models. This makes them less precise (wider CIs) but achieves coverage by making calibration uncertainty a smaller fraction of total.

**Recommendation for DR/IPS:** Can use lower oracle coverage (5-10%) and still achieve nominal coverage, but at the cost of wider confidence intervals. For tighter CIs, increase oracle coverage even in DR/IPS modes.
