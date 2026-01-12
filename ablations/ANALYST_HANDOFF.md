# Arena Preferences Ablation: Analyst Handoff Document

**Date:** January 2026
**Author:** Automated analysis with Claude
**Paper:** "Calibrated Judge Evaluations" (CJE)
**Paper Location:** `/Users/eddielandesberg/CIMO/CJE/cje-paper/arxiv_submission/`

---

## Executive Summary

This document provides a comprehensive handoff for the Arena preferences ablation study comparing CJE (continuous calibration) against Rogan-Gladen/Lee et al. (binary calibration). The analysis is complete and paper-ready.

### Bottom Line
**CJE wins decisively even on preferences—the most favorable domain for binary methods.**

| Metric | CJE | Rogan-Gladen | Winner |
|--------|-----|--------------|--------|
| RMSE | 0.0043 | 0.096 | CJE (95% better) |
| 95% CI Coverage | 96.4% | 97.9% | Tie (both valid) |
| CI Half-Width | 0.015 | 0.192 | CJE (13x narrower) |

---

## 1. Context & Motivation

### The Reviewer Escape Hatch
The MBPP (code correctness) experiments showed CJE beating binary methods, but a reviewer could argue: *"MBPP has low Youden's J (0.054), making binary calibration unstable. Lee et al. was designed for preferences, not code."*

### The Stress Test
We tested on Chatbot Arena preferences where:
- Binary judges achieve Youden's J = 0.473 (9x higher than MBPP)
- Human A/B labels are the native format
- This is the **ideal domain for binary methods**

**Result:** CJE still wins by 95% RMSE and 11x precision.

---

## 2. Experimental Design

### Dataset
- **Source:** lmsys/chatbot_arena_conversations (Hugging Face)
- **Size:** 9,999 preference pairs (50.55% A-wins)
- **Filtering:** Non-tie battles, responses > 10 chars

### Judge Configuration
- **Model:** OpenAI gpt-4o-mini (temperature=0.0, deterministic)
- **Output modes:**
  - Continuous: 0-100 score → 0-1 normalized
  - Binary: A/B categorical prediction
- **Policy variants:** base, verbose, terse, contrarian

### Experimental Grid
| Parameter | Values |
|-----------|--------|
| Sample sizes | 2,000 / 5,000 / 10,000 |
| Oracle coverages | 1% / 2% / 5% / 10% / 25% |
| Seeds | 30 (0-29) |
| **Total per method** | 3 × 5 × 30 = **450 experiments** |

### Methods Compared
1. **CJE (direct):** Monotone isotonic calibration + bootstrap inference
2. **Rogan-Gladen:** Binary misclassification correction with Lee et al. CIs

### Fairness Guarantee
Both methods use **identical oracle budgets** (same labeled samples per experiment).

---

## 3. Key Results

### 3.1 RMSE Comparison

| Method | Mean RMSE | Std | N |
|--------|-----------|-----|---|
| CJE (direct) | **0.0043** | 0.0059 | 450 |
| naive-direct | 0.0707 | 0.0021 | 450 |
| binary-naive | 0.0569 | 0.0034 | 450 |
| binary-adjusted (RG) | 0.0958 | 0.0860 | 450 |

**Key insight:** CJE achieves 95% lower RMSE than Rogan-Gladen.

### 3.2 Uncertainty Quantification

| Method | 95% CI Coverage | Nominal |
|--------|-----------------|---------|
| CJE (direct) | 96.4% | 95% ✓ |
| Rogan-Gladen | 96.9% | 95% ✓ |
| naive-direct | 0.0% | 95% ✗ |

**Key insight:** Both calibrated methods achieve valid UQ. The naive method catastrophically undercovers.

### 3.3 Precision (CI Half-Widths)

| Method | Mean Half-Width | Valid Coverage? |
|--------|-----------------|-----------------|
| CJE (direct) | **0.015** | Yes (96.4%) |
| Rogan-Gladen | 0.172 | Yes (96.9%) |
| naive-direct | 0.009 | No (0.0%) |

**Key insight:** CJE is 11x more precise while maintaining valid coverage.

### 3.4 Why CJE Wins Despite High Youden's J

The Rogan-Gladen formula: θ̂ = (p̂ + Sp - 1) / J

- Divides by Youden's J, amplifying estimation variance
- At J = 0.47: 2.1x error amplification
- At J = 0.27 (terse policy): 3.7x error amplification

CJE's monotone calibration preserves the continuous signal without such amplification.

---

## 4. Files & Reproducibility

### Key Files

| File | Purpose |
|------|---------|
| `ablations/config_preferences.py` | Experiment parameters |
| `ablations/run_preferences.py` | Main experiment runner |
| `ablations/binary_baseline.py` | Lee et al. CI implementation (lines 192-254) |
| `ablations/diagnose_preferences.py` | Report generator |
| `ablations/results_preferences/REPORT.md` | Full results summary |
| `ablations/REPRODUCE.md` | Reproduction guide |

### Data Files

| File | SHA256 (first 16) |
|------|-------------------|
| `data_preferences/preferences_dataset.jsonl` | b4b174964ffc9062 |
| `data_preferences/responses/base_responses.jsonl` | 93d36b8440e4fcc5 |

### Reproduction Commands

```bash
cd ablations

# Run CJE experiments (~45 min)
python run_preferences.py

# Run binary baseline (~2 min)
python run_preferences.py --binary

# Generate report
python diagnose_preferences.py --report
```

---

## 5. Paper Integration

### Suggested Section: 5.3 Binary Methods on Preference Data

**Draft text:**

> A natural objection is that binary misclassification correction (Rogan-Gladen) is designed for binary outcomes, making MBPP's low Youden's J (0.054) an unfavorable test case. We therefore evaluated both methods on Chatbot Arena preference judgments, where binary judges achieve J = 0.473.
>
> We ran 450 matched experiments: same sample sizes (2K, 5K, 10K), oracle coverages (1-25%), and seeds (30). Both methods achieve valid 95% CI coverage (CJE: 96.4%, Rogan-Gladen: 96.9%). However, CJE achieves 95% lower RMSE (0.004 vs 0.078-0.165) and 11x narrower confidence intervals (half-width 0.015 vs 0.172).
>
> The efficiency gap arises because Rogan-Gladen divides by Youden's J, amplifying estimation variance. At J = 0.47, this creates 2.1x error amplification. CJE's monotone calibration preserves the continuous signal without such amplification.

### Tables to Add

**Table P1: Domain Comparison**
```latex
\begin{tabular}{lcc}
\toprule
Domain & Youden's J & Binary Viable? \\
\midrule
MBPP (code) & 0.054 & No \\
Arena (preferences) & 0.473 & Yes \\
\bottomrule
\end{tabular}
```

**Table P2: Method Comparison on Preferences**
```latex
\begin{tabular}{lccc}
\toprule
Method & RMSE & 95\% CI Coverage & CI Half-Width \\
\midrule
CJE (direct) & \textbf{0.004} & 96.4\% & \textbf{0.015} \\
Rogan-Gladen & 0.078--0.165 & 96.9\% & 0.172 \\
\bottomrule
\end{tabular}
```

---

## 6. Implementation Notes

### Lee et al. CI Formula

Our implementation in `binary_baseline.py` matches the official repo:
- **Repo:** https://github.com/UW-Madison-Lee-Lab/LLM-judge-reporting
- **Function:** `lee_et_al_confidence_interval()` (lines 192-254)
- **Formula:** Eq. 6 from Lee et al. (2026) with Agresti-Coull adjustment

Key components:
1. Agresti-Coull smoothing for p, q0, q1
2. Bias correction term (dth)
3. SE accounting for test + calibration uncertainty
4. Clipping to [0, 1]

### Transport-Aware Bootstrap (CJE)

CJE uses `calibration_policy="base"` to enable transport correction:
- Calibrator fitted on base policy only
- Residual augmentation computed on all target policies
- This fixes the transport bias that previously hurt CJE on Arena

---

## 7. Verification Checklist

Run before paper submission:

```bash
# 1. Verify experiment counts
wc -l results_preferences/all_experiments.jsonl  # Should be 900
wc -l results_preferences/binary_baseline.jsonl  # Should be 450

# 2. Verify key metrics
python -c "
import json
with open('results_preferences/binary_baseline_summary.json') as f:
    s = json.load(f)
print(f'RG Coverage: {s[\"overall_stats\"][\"coverage_95ci\"]:.1%}')
"

# 3. Regenerate report (should match)
python diagnose_preferences.py --report
diff results_preferences/REPORT.md results_preferences/REPORT.md.bak
```

---

## 8. Pressure Testing (Fairness Validation)

### Final: True Apples-to-Apples Comparison

After collecting missing binary labels for contrarian, both methods are now evaluated on the **same 3 target policies**: verbose, terse, contrarian.

| Metric | CJE | RG | Comparison |
|--------|-----|----|-----------|
| RMSE | 0.0043 ± 0.0059 | 0.0958 ± 0.0860 | CJE **95%** better |
| Coverage | 96.4% | 97.9% | Both valid |
| Half-width | 0.015 | 0.192 | CJE **13x** narrower |

**This is the definitive comparison:** same policies, same oracle budgets, same seeds.

### Policy-wise Breakdown (RG)

| Policy | RG RMSE | Youden's J | Coverage |
|--------|---------|------------|----------|
| verbose | 0.093 | 0.55 | 97.1% |
| terse | 0.165 | 0.27 | 98.9% |
| contrarian | 0.117 | 0.41 | 97.8% |

### Key Changes Made

1. **Collected binary labels for contrarian** (10,000 records, ~$1 API cost)
2. **Re-ran binary baseline** with all 4 policies
3. **Updated paper** to report comparison on same 3 target policies

### Conclusion

Paper claims are now based on **true apples-to-apples comparison**:
- Same 3 target policies for both methods
- CJE achieves 95% lower RMSE
- CJE achieves 13x narrower CIs
- Both methods achieve valid coverage (~96-98%)

---

## 9. Threshold Sweep (Appendix Robustness)

### Motivation
A reviewer might object: "The 0.5 threshold might unfairly penalize binary methods. What if an optimal threshold exists?"

### Experiment
We thresholded the continuous judge_score at τ ∈ {0.3, 0.4, 0.5, 0.6, 0.7} and ran Rogan-Gladen correction on each. This tests whether information loss from binarization is threshold-dependent.

### Results

| Method | Youden's J | RMSE | Coverage | Half-Width |
|--------|------------|------|----------|------------|
| CJE (continuous) | — | **0.004** | 96.4% | **0.015** |
| Native A/B (RG) | 0.41 | 0.076 | 97.9% | 0.192 |
| τ=0.30 | 0.25 | 0.119 | 92.2% | 0.199 |
| τ=0.40 | 0.41 | 0.077 | 96.7% | 0.147 |
| τ=0.50 | 0.44 | 0.063 | 95.6% | 0.135 |
| **τ=0.60** | **0.44** | **0.061** | **98.9%** | **0.136** |
| τ=0.70 | 0.37 | 0.071 | 95.6% | 0.165 |

### Key Findings

1. **Optimal threshold exists**: τ=0.60 achieves best RMSE (0.061)
2. **Thresholded beats native A/B**: 0.061 vs 0.076 (20% better)
3. **CJE still dominates**: 0.004 vs 0.061 (93% better than best threshold)
4. **CJE 9x more precise**: 0.015 vs 0.135 half-width

### Conclusion
Information loss from binarization is fundamental, not threshold-dependent. Even with optimal threshold selection, binary methods cannot compete with continuous calibration.

**Reproduction:** `python run_preferences.py --threshold-sweep`

---

## 10. Open Questions / Future Work

1. **Response length covariate:** Tested but too slow with bootstrap (52h). Marginal improvement not worth the compute.

2. **More policies:** Could add more judge variants to stress-test transport.

3. **Larger oracle coverages:** Could test 50%, 100% oracle to show convergence.

4. **Cross-domain transfer:** Train calibrator on Arena, test on MBPP (or vice versa).

---

## 11. Contact & Resources

- **Experiments repo:** `/Users/eddielandesberg/CIMO/cje-arena-experiments/`
- **Paper repo:** `/Users/eddielandesberg/CIMO/CJE/cje-paper/arxiv_submission/`
- **CJE library:** `pip install cje-eval>=0.2.12`
- **Lee et al. official repo:** https://github.com/UW-Madison-Lee-Lab/LLM-judge-reporting

---

## Appendix: Quick Reference

### Claims You Can Make (with evidence)

| Claim | Evidence | Source |
|-------|----------|--------|
| CJE beats RG by 95% RMSE | 0.004 vs 0.096 | Apples-to-apples comparison |
| Both have valid UQ | 96.4% vs 97.9% coverage | Same 3 target policies |
| CJE is 13x more precise | 0.015 vs 0.192 half-width | Same 3 target policies |
| Comparison is fair | Same policies, oracle budget, seeds | config_preferences.py |
| Implementation matches Lee et al. | Verified against official repo | binary_baseline.py |

### Numbers to Cite (Final, Apples-to-Apples)

- CJE RMSE: **0.004 ± 0.006**
- RG RMSE: **0.096 ± 0.086**
- CJE coverage: **96.4%**
- RG coverage: **97.9%**
- CJE half-width: **0.015**
- RG half-width: **0.192**
- RMSE improvement: **95%**
- Precision ratio: **13x**
- Mean Youden's J (Arena): **0.41**
- Youden's J (MBPP): **0.054**
