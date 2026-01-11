# MBPP Ablation Study: CJE Direct Mode on Coding Tasks

**Date:** January 10, 2026
**Analyst:** Eddie Landesberg
**Status:** Complete (pending external judge rescore for confound confirmation)

---

## Executive Summary

We ran a CJE (Causal Judge Evaluation) ablation study on MBPP (Mostly Basic Python Problems) to test whether calibrating LLM judge scores against oracle labels improves policy evaluation accuracy on coding tasks.

**Key finding:** Calibration works when transport holds and fails predictably when it doesn't.

- For transport-valid policies (enhanced, terse): calibration **reduces RMSE by 39%** (0.029 vs 0.048)
- Calibration wins in **100% of 50 random seeds** (t=-20.14, p<0.000001, clustered by seed)
- For transport-invalid policies (mini, unhelpful): calibration makes estimates 2-3x worse
- Transport audit correctly identifies which policies are safe to calibrate

**Known confound:** The judge model (gpt-4o-mini) is the same as the mini policy. This may cause mini's transport failure via self-preference bias. Pending confirmation with external judge rescore.

---

## 1. Data Structure

### 1.1 Dataset Dimensions

| Quantity | Value | Definition |
|----------|-------|------------|
| **N_prompts** | 257 | Unique MBPP problems |
| **K** | 5 | Epochs/draws per prompt per policy |
| **N_rows** | 1,285 | Total samples per policy (257 × 5) |

### 1.2 Sampling Units

| Parameter | Unit | Interpretation |
|-----------|------|----------------|
| `sample_size` | Rows | Number of rows subsampled from N_rows |
| `oracle_coverage` | Fraction of rows | 5% of 200 rows = 10 labeled rows |

**Example:** `sample_size=200, oracle_coverage=0.05` → 200 rows subsampled, 10 rows have oracle labels for calibrator training.

### 1.3 Policies Evaluated

| Policy | Model | System Prompt | Oracle Pass Rate |
|--------|-------|---------------|------------------|
| base | gpt-4o | Standard coding assistant | 87.4% |
| enhanced | gpt-4o | Detailed step-by-step reasoning | 87.2% |
| terse | gpt-4o | Minimal output, no explanations | 87.4% |
| mini | gpt-4o-mini | Standard coding assistant | 80.9% |
| unhelpful | gpt-4o | Deliberately poor/confusing | 56.0% |

### 1.4 Judge

- **Model:** gpt-4o-mini ⚠️ *Same model as mini policy*
- **Task:** Score 0-100 probability that code passes all unit tests
- **Output:** Normalized to [0, 1]

---

## 2. Calibration Design

### 2.1 Training Data
- Calibrator trained on **base policy rows** with oracle labels
- Method: Isotonic regression (judge_score → oracle_label)
- Cross-fitting: 5-fold when ≥50 labels, 3-fold otherwise

### 2.2 Application
- Calibrator trained on base, applied to **all target policies**
- This requires the judge→oracle relationship to transfer across policies

### 2.3 Estimators

| Estimator | Description |
|-----------|-------------|
| naive-direct | Raw judge scores, no calibration |
| direct | Isotonic regression calibration |
| direct+cov | Isotonic with response_length covariate |

### 2.4 Experimental Grid

- **Sample sizes:** 200, 400, 600, 800, 974 rows
- **Oracle coverage:** 5%, 10%, 25%, 50%, 100%
- **Seeds:** 50 random seeds
- **Total experiments:** 3,750

---

## 3. Transport Audit

Before interpreting calibrated estimates, we must verify the calibrator transfers to each target policy.

### 3.1 Method

For each policy, compute mean residual: E[Y - f(S)] where f() is the calibrator trained on base.

- If residual ≈ 0: calibrator transfers correctly
- If residual ≠ 0: transport failure; calibrated estimates are biased

### 3.2 Results

| Policy | Oracle | Calibrated Pred | Residual | SE | p-value | Transport |
|--------|--------|-----------------|----------|-----|---------|-----------|
| base | 87.4% | 87.4% | -0.000 | 0.020 | 1.00 | ✓ PASS |
| enhanced | 87.2% | 87.5% | -0.003 | 0.020 | 0.87 | ✓ PASS |
| terse | 87.4% | 86.9% | +0.005 | 0.020 | 0.82 | ✓ PASS |
| **mini** | **80.9%** | **87.5%** | **-0.065** | **0.024** | **0.006** | **✗ FAIL** |
| **unhelpful** | **56.0%** | **84.0%** | **-0.280** | **0.021** | **<0.001** | **✗ FAIL** |

*Standard errors clustered by prompt_id (257 clusters).*

### 3.3 Interpretation

- **enhanced, terse:** Calibrator transfers (residual ≈ 0)
- **mini:** Residual -6.5pp — consistent with judge–policy coupling (see Section 6)
- **unhelpful:** Residual -28pp — low-support saturation failure (see Section 5)

---

## 4. Results: Stratified by Transport Validity

### 4.1 Transport-Valid Policies (enhanced + terse)

| Estimator | RMSE | vs naive-direct |
|-----------|------|-----------------|
| naive-direct | 0.048 ± 0.003 | — |
| **direct** | **0.029 ± 0.006** | **-39%** |
| direct+cov | 0.037 ± 0.006 | -23% |

### 4.2 Statistical Inference (Clustered by Seed)

| Comparison | Seeds Cal. Wins | Mean Δ RMSE | SE | t | p |
|------------|-----------------|-------------|-----|-----|-----|
| direct vs naive | **50/50 (100%)** | -0.019 | 0.0009 | -20.14 | <0.000001 |
| direct+cov vs naive | 46/50 (92%) | -0.011 | 0.0010 | -11.08 | <0.000001 |

*N=50 seeds. Each seed averaged over 25 sample_size × oracle_coverage cells.*

**Conclusion:** Calibration provides robust RMSE improvement when transport holds.

### 4.3 Transport-Invalid Policies (mini + unhelpful)

| Policy | naive-direct Bias | direct Bias | |
|--------|-------------------|-------------|---|
| mini | +2.2pp | +6.4pp | 3x worse |
| unhelpful | +12.1pp | +27.0pp | 2x worse |

**Conclusion:** Calibration systematically fails when transport fails.

### 4.4 The Misleading Aggregate

Pooling all policies hides the stratified truth:

| Estimator | RMSE (all 4) | RMSE (transport-valid) |
|-----------|--------------|------------------------|
| naive-direct | 0.071 | 0.048 |
| direct | 0.145 | **0.029** |

The aggregate suggests calibration hurts; the stratified view shows it helps when applicable.

---

## 5. Why Unhelpful Fails: Low-Support Saturation

### 5.1 Judge Score Distribution

| Bin | Base | Unhelpful | Ratio |
|-----|------|-----------|-------|
| [0.4, 0.5) | 7.2% | **28.2%** | 3.9x |
| [0.5, 0.6) | 3.4% | 6.8% | 2.0x |
| [0.6, 0.7) | 3.3% | 11.5% | 3.5x |
| [0.8, 1.0] | 63.9% | 44.2% | 0.7x |

### 5.2 The Problem

- 30.7% of unhelpful's samples score below base's 10th percentile (0.60)
- Isotonic regression trained on base has **sparse coverage** in [0.4, 0.7)
- With few training points in this region, the calibrator **saturates to nearby step values**
- These saturated values (~0.84) are much higher than unhelpful's true pass rate (0.56)

### 5.3 Correct Terminology

This is **not** extrapolation (isotonic doesn't extrapolate beyond observed range). It's:
- **Low-support saturation:** Insufficient training density in the relevant score region
- **Distribution shift:** The conditional P(Y|S) differs between base and unhelpful

---

## 6. The Judge–Policy Confound

### 6.1 The Issue

gpt-4o-mini serves as both:
1. The **judge model** scoring all outputs
2. The **mini policy** generating responses

LLM judges exhibit documented self-preference and style-alignment biases.

### 6.2 Evidence (Circumstantial)

| Metric | Value |
|--------|-------|
| Judge prediction for mini | 87.5% |
| Actual mini pass rate | 80.9% |
| Overestimate | 6.6pp |
| Transport audit p-value | 0.006 |

The judge rates mini outputs as equivalent to gpt-4o outputs, but they pass tests 6pp less often.

### 6.3 Status: Hypothesis Pending Confirmation

This is **consistent with** judge–policy coupling, but not yet confirmed. To establish causality:
- Rescore all outputs with a judge ∉ {gpt-4o, gpt-4o-mini}
- If mini then passes transport audit, confound is confirmed

---

## 7. Tie-Aware Ranking

### 7.1 The "Ranking Failure" Was a Statistical Tie

| Comparison | Difference | SE | p-value | Status |
|------------|------------|-----|---------|--------|
| terse vs enhanced | +0.2pp | 2.9pp | 0.94 | **TIE** |
| terse vs mini | +6.5pp | 3.1pp | 0.04 | Borderline |
| base vs unhelpful | +31.4pp | 3.0pp | <0.001 | Distinguishable |

**MDE (80% power):** ~8 percentage points

### 7.2 Policy Tiers

- **Tier 1:** base, terse, enhanced (87.2–87.4%) — statistically indistinguishable
- **Tier 2:** mini (80.9%) — borderline distinguishable
- **Tier 3:** unhelpful (56.0%) — clearly distinguishable

Report **tiers**, not strict orderings, when differences are below MDE.

---

## 8. Calibration by Oracle Coverage

| Oracle % | # Labels (at n=974) | direct RMSE | naive RMSE |
|----------|---------------------|-------------|------------|
| 5% | 49 | 0.048 | 0.048 |
| 10% | 97 | 0.035 | 0.048 |
| 25% | 244 | 0.024 | 0.048 |
| 50% | 487 | 0.019 | 0.048 |
| 100% | 974 | 0.015 | 0.048 |

*Transport-valid policies only.*

At 5% oracle, calibration variance ≈ bias correction (break-even).
At 10%+, calibration clearly dominates.

---

## 9. Summary of Claims

| Claim | Status | Evidence |
|-------|--------|----------|
| "Calibration failed on MBPP" | **FALSE** | 39% RMSE reduction when transport holds |
| "naive-direct beats calibrated" | **MISLEADING** | Only true when mixing valid/invalid |
| "Ranking failed for top 3" | **EXPECTED** | Statistical tie (0.2pp diff, 8pp MDE) |
| "unhelpful was misestimated" | **CORRECT BEHAVIOR** | Transport audit flags this |
| "mini bias is self-preference" | **HYPOTHESIS** | Consistent; pending external judge |

---

## 10. Conclusions

### 10.1 CJE Validation

1. **Calibration helps when transport holds:** 39% RMSE reduction, 100% seed win rate
2. **Transport audit catches failures:** Correctly flags mini and unhelpful
3. **Proper response to failure:** Stratify by transport; refuse level claims for failures

### 10.2 Transport Gating: The Key Insight

**The transport audit should GATE calibration, not just diagnose it.**

| Method | RMSE | Kendall τ | Top-1 | Unhelpful Last |
|--------|------|-----------|-------|----------------|
| naive-direct | 0.071 | 0.33 | 98% | 100% |
| direct (calibrate all) | 0.148 | 0.21 | 67% | 96% |
| **direct + gating** | **0.069** | **0.61** | **91%** | **99%** |

Without gating, calibration applied to transport-invalid policies:
- Increases RMSE by 109%
- Reduces ranking τ from 0.33 to 0.21

With gating:
- RMSE improves slightly (0.071 → 0.069)
- Ranking τ nearly triples (0.33 → 0.61)

**Implementation:**
```python
def cje_estimate(policy, calibrator, transport_audit):
    if transport_audit.passes(policy):
        return calibrator.predict(policy)  # Calibrated
    else:
        return naive_estimate(policy)       # Fall back
```

### 10.3 Operational Rules

1. **Run transport audit FIRST** - before applying calibration
2. **Gate calibration by transport** - only calibrate policies that pass
3. Report tiers when policy differences are below MDE
4. Never use judge = policy without explicit audit

### 10.4 What MBPP Teaches

- CJE works correctly on coding tasks when assumptions hold
- The "coding is different" narrative is not supported
- MBPP is a useful **diagnostic validation**: method behaves correctly including refusing when it should

---

## 11. Recommended Next Steps

### 11.1 Required: Fix the Confound

Rescore with a judge ∉ {gpt-4o, gpt-4o-mini}:
- Options: claude-3-haiku, gemini-1.5-flash
- Cost: ~$5-10
- If mini passes transport after fix → confound confirmed

### 11.2 Recommended: Add Identity Calibration Option

Allow method to select among {identity, isotonic} via OOF error:
- At low oracle, identity avoids variance cost
- At high oracle, isotonic captures bias correction
- Ensures "do no harm"

### 11.3 Optional: Audit-with-Budget Simulation

Demonstrate transport audit works with limited labels:
- Give audit 25 labeled prompts per policy
- Evaluate false-pass/false-fail rates
- Show gating rule improves downstream performance

---

## 12. Files and Reproducibility

```
ablations/results_mbpp/
├── all_experiments.jsonl    # 3,750 experiment results
├── binary_baseline.jsonl    # Binary comparison results (after collection)
├── REPORT.md                # This report

ablations/
├── diagnose_mbpp.py         # Diagnostic analysis (score + binary transport)
├── config_mbpp.py           # Experiment configuration
├── run_mbpp.py              # Ablation runner (continuous)
├── binary_baseline.py       # Rogan-Gladen estimator + confusion audit
├── run_binary_baseline.py   # Binary comparison experiments

data_generation_mbpp/
├── add_judge_scores.py      # Collect continuous judge scores
├── add_binary_judge.py      # Collect binary judge scores

data_mbpp/
├── mbpp_dataset.jsonl       # Base policy (1,285 rows)
└── responses/               # Target policies (1,285 rows each)
```

**To reproduce:**
```bash
cd ablations
python run_mbpp.py --workers=8    # ~4 minutes
python diagnose_mbpp.py           # Diagnostics
```

---

## 13. Binary Baseline Comparison

This section compares continuous CJE calibration against binary misclassification adjustment (Lee et al. / Rogan-Gladen framework).

### 13.1 Methodology

| Method | Judge Output | Calibration | Transport Audit |
|--------|--------------|-------------|-----------------|
| naive-direct | Continuous S ∈ [0,1] | None | N/A |
| direct | Continuous S ∈ [0,1] | Isotonic f(S) → Y | Mean residual E[Y - f(S)] |
| binary-naive | Binary Ẑ ∈ {0,1} | None | N/A |
| binary-adjusted | Binary Ẑ ∈ {0,1} | Rogan-Gladen | Confusion matrix stability |

**Rogan-Gladen adjustment:**
```
θ̂ = (p̂ + Sp - 1) / (Se + Sp - 1)

where:
  p̂ = observed proportion of YES predictions
  Se = sensitivity = P(Ẑ=1|Y=1)
  Sp = specificity = P(Ẑ=0|Y=0)
```

**Validity constraint:** Youden's J = Se + Sp - 1 must be > 0 for adjustment to be valid.

### 13.2 Binary Judge Characteristics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Sensitivity (Se) | 70.3% | P(YES \| code passes) |
| Specificity (Sp) | 35.2% | P(NO \| code fails) |
| **Youden's J** | **0.054** | **Borderline invalid** |
| N positive | 1,123 | Samples where code passes |
| N negative | 162 | Samples where code fails |

⚠️ **Critical issue:** The binary judge has very low specificity (35.2%), meaning it says YES even when code fails 65% of the time. Youden's J = 0.054 is at the validity threshold for Rogan-Gladen adjustment.

### 13.3 Binary Judge YES Rates

| Policy | Binary YES | Continuous Mean | Oracle Pass |
|--------|------------|-----------------|-------------|
| base | 69.6% | 0.843 | 87.4% |
| enhanced | 71.0% | 0.839 | 87.2% |
| mini | 70.7% | 0.832 | 80.9% |
| terse | 64.6% | 0.815 | 87.4% |
| unhelpful | 36.4% | 0.681 | 56.0% |

### 13.4 RMSE Comparison: Continuous vs Binary

| Estimator | RMSE | vs naive-direct |
|-----------|------|-----------------|
| **naive-direct** | 0.048 | — |
| **direct** | **0.029** | **-39%** |
| binary-naive (enhanced) | 0.161 | +235% |
| binary-naive (terse) | 0.227 | +373% |
| binary-adjusted (enhanced) | 0.203 | +323% |
| binary-adjusted (terse) | 0.508 | +958% |

**Key finding:** Binary calibration (Rogan-Gladen) **increases** RMSE rather than decreasing it. This is because:
1. Low specificity (35.2%) means J ≈ 0.05
2. Division by J ≈ 0.05 amplifies small errors
3. The calibration is unstable when the binary judge lacks discriminative power

### 13.5 Transport Audit Comparison

| Policy | Score Transport | Binary Transport | Agreement |
|--------|-----------------|------------------|-----------|
| enhanced | ✓ PASS | ✓ PASS | ✓ |
| terse | ✓ PASS | ✗ FAIL | ✗ |
| mini | ✗ FAIL | ✗ FAIL | ✓ |
| unhelpful | ✗ FAIL | ✗ FAIL | ✓ |

**Observations:**
- mini and unhelpful fail both audits (expected)
- terse fails binary transport but passes score transport
- The binary audit is more conservative due to low base specificity

### 13.6 Why Binary Calibration Fails Here

The Rogan-Gladen formula θ̂ = (p̂ + Sp - 1) / J amplifies errors when J is small:

| Youden's J | Error Amplification |
|------------|---------------------|
| 0.50 | 2x |
| 0.20 | 5x |
| 0.10 | 10x |
| **0.054** | **18.5x** |

With J = 0.054, a 1pp error in p̂ becomes an 18.5pp error in θ̂.

### 13.7 Conclusions

1. **Continuous calibration wins decisively** when the judge provides continuous scores
2. **Binary calibration fails** because the binary judge has poor specificity
3. **The "do we need continuous scores?" question has a clear answer: YES**
4. **Diagnostic frameworks matter:** Both continuous and binary methods need transport audits

### 13.8 Methodological Recommendation

When choosing between continuous and binary calibration:

| Condition | Recommendation |
|-----------|----------------|
| Youden's J > 0.3 | Binary may work |
| Youden's J < 0.1 | Use continuous or refuse |
| Continuous available | Prefer continuous |
| Only binary available | Check J before using Rogan-Gladen |

### 13.9 Transport Gating vs Rogan-Gladen

| Aspect | CJE + Gating | Rogan-Gladen |
|--------|--------------|--------------|
| Transport assumption | P(Y\|S) stable | P(Ẑ\|Y) stable |
| Built-in diagnostic | Yes (mean residual) | **No** |
| Gating mechanism | Fall back to naive | None available |
| RMSE (this study) | 0.069 | N/A (J too low) |
| Ranking τ (this study) | 0.61 | 0.25-0.28 |

The Rogan-Gladen framework:
1. **Assumes** confusion matrix stability without testing it
2. **Fails silently** when Youden's J is low
3. **Cannot be gated** because there's no continuous fallback

CJE with transport gating:
1. **Tests** the transport assumption explicitly
2. **Falls back** to naive when transport fails
3. **Ensures** calibration "does no harm"

---

## Appendix A: Statistical Details

### A.1 Seed-Level Inference

| Metric | direct vs naive | direct+cov vs naive |
|--------|-----------------|---------------------|
| N seeds | 50 | 50 |
| Seeds calibration wins | 50 (100%) | 46 (92%) |
| Mean RMSE diff | -0.019 | -0.011 |
| SE (clustered) | 0.0009 | 0.0010 |
| t-statistic | -20.14 | -11.08 |
| p-value | <0.000001 | <0.000001 |

### A.2 Per-Policy Bias

| Policy | naive-direct | direct | Transport |
|--------|--------------|--------|-----------|
| enhanced | -0.034 ± 0.005 | +0.002 ± 0.042 | ✓ |
| terse | -0.059 ± 0.005 | -0.007 ± 0.045 | ✓ |
| mini | +0.022 ± 0.005 | +0.064 ± 0.043 | ✗ |
| unhelpful | +0.121 ± 0.004 | +0.270 ± 0.070 | ✗ |

### A.3 Judge Score Distribution

| Statistic | Base | Enhanced | Mini | Terse | Unhelpful |
|-----------|------|----------|------|-------|-----------|
| Mean | 0.84 | 0.84 | 0.83 | 0.82 | 0.68 |
| Std | 0.17 | 0.17 | 0.17 | 0.18 | 0.22 |
| P10 | 0.60 | 0.60 | 0.60 | 0.40 | 0.40 |
| P90 | 1.00 | 1.00 | 1.00 | 1.00 | 0.90 |

---

*Report generated from 3,750 experiments run on January 10, 2026*
*Revised with proper seed-clustered inference and transport stratification*
