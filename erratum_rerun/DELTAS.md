# DELTAS — arXiv 2512.11150 v1 (published) vs erratum rerun (cje-eval 0.3.0)

Rerun date: 2026-07-06/07. Same data (`data/cje_dataset.jsonl`), same seeds (0-49), same grid cells as the published runs; the only intended change is the library (local editable ~v0.2.10-vintage -> PyPI `cje-eval==0.3.0`, which contains the OUA-jackknife sum-form fix, the two-stage OUA fold-model fix, the real TMLE fluctuation, and the CalibratedIPS Hajek influence-function fix). Cluster-robust arms reproduce published values bit-for-bit, so deltas are attributable to the fixes.

## 1. tab:uq-comparison (coverage %, 3 policies x 50 seeds = 150 CIs per cell)

| Regime | CR pub | CR rerun | JK pub | **JK corrected** | B=500 pub | B=500 rerun | B=2000 pub | B=2000 rerun |
|---|---|---|---|---|---|---|---|---|
| n=250, cov=5% | 21.3 | 21.3 | 85.3 | **94.7** | 94.0 | 95.3 | 93.3 | 97.3 |
| n=250, cov=10% | 31.3 | 31.3 | 89.3 | **96.0** | 95.3 | 95.3 | 93.3 | 97.3 |
| n=250, cov=25% | 42.7 | 42.7 | 86.0 | **96.7** | 95.3 | 95.3 | 96.7 | 97.3 |
| n=250, cov=50% | 49.3 | 49.3 | 80.7 | **92.0** | 92.7 | 95.3 | 92.7 | 97.3 |
| n=500, cov=5% | 28.0 | 28.0 | 88.7 | **92.7** | 96.0 | 95.3 | 96.7 | 96.0 |
| n=500, cov=10% | 31.3 | 31.3 | 76.0 | **96.7** | 97.3 | 95.3 | 97.3 | 96.0 |
| n=500, cov=25% | 40.0 | 40.0 | 84.0 | **95.3** | 98.0 | 95.3 | 98.7 | 96.0 |
| n=500, cov=50% | 62.7 | 62.7 | 82.7 | **95.3** | 98.0 | 95.3 | 96.7 | 96.0 |
| n=1000, cov=5% | 21.3 | 21.3 | 75.3 | **97.3** | 95.3 | 100.0 | 95.3 | 100.0 |
| n=1000, cov=10% | 30.0 | 30.0 | 81.3 | **99.3** | 95.3 | 100.0 | 95.3 | 100.0 |
| n=1000, cov=25% | 35.3 | 35.3 | 83.3 | **97.3** | 94.7 | 100.0 | 94.7 | 100.0 |
| n=1000, cov=50% | 55.3 | 55.3 | 95.3 | **99.3** | 96.0 | 100.0 | 97.3 | 100.0 |
| n=2500, cov=5% | 15.3 | 15.3 | 70.0 | **92.7** | 96.7 | 99.3 | 97.3 | 100.0 |
| n=2500, cov=10% | 25.3 | 25.3 | 74.7 | **93.3** | 97.3 | 99.3 | 96.0 | 100.0 |
| n=2500, cov=25% | 52.0 | 52.0 | 84.7 | **96.7** | 98.0 | 99.3 | 98.7 | 100.0 |
| n=2500, cov=50% | 58.7 | 58.7 | 88.7 | **100.0** | 97.3 | 99.3 | 98.7 | 100.0 |

Overall (16 regimes): JK published 82.9 -> corrected **96.0** (expected ~96.0); CR published 37.5 -> rerun 37.5 (expected: unchanged).
**Expected-direction check (JK coverage rises to ~nominal): PASS**

**Bootstrap-column caveat:** in `cje-eval` 0.3.0 the direct-mode bootstrap refit pools oracle labels from fresh draws (keep-every-oracle-pair, commit ca82e05), so the ablation's oracle masking no longer binds for the bootstrap arms: their CIs are identical across oracle-coverage levels within each n (verified per-seed), and per-regime bootstrap cells are not apples-to-apples with v1. The analytic arms (cluster-robust, cal-aware jackknife) respect the masking exactly as in v1.

Published-table (integer) cells for the 8 shown regimes, v1 -> v2:

| Regime | CR | Cal-Aware JK | B=500 | B=2000 |
|---|---|---|---|---|
| n=250, cov=5% | 21 (=) | 85 -> **95** | 94 -> **95** | 93 -> **97** |
| n=250, cov=25% | 43 (=) | 86 -> **97** | 95 (=) | 97 (=) |
| n=500, cov=5% | 28 (=) | 89 -> **93** | 96 -> **95** | 97 -> **96** |
| n=500, cov=25% | 40 (=) | 84 -> **95** | 98 -> **95** | 99 -> **96** |
| n=1000, cov=5% | 21 (=) | 75 -> **97** | 95 -> **100** | 95 -> **100** |
| n=1000, cov=25% | 35 (=) | 83 -> **97** | 95 -> **100** | 95 -> **100** |
| n=2500, cov=5% | 15 (=) | 70 -> **93** | 97 -> **99** | 97 -> **100** |
| n=2500, cov=25% | 52 (=) | 85 -> **97** | 98 -> **99** | 99 -> **100** |

Cal-aware JK mean CI width widens x1.62-x2.14 (median x1.87) across the 16 regimes — the corrected variance is larger, as expected from the x K fix.

## 2. tab:accuracy — affected rows only (all other rows unchanged from v1)

| Row | Metric | v1 published | v2 corrected | Δ |
|---|---|---|---|---|
| direct | RMSE^d | 0.0223 | 0.0227 | +0.0004 |
| direct | IS | 0.0542 | 0.0707 | +0.0165 |
| direct | Coverage % | 85.0 | 93.9 | +8.9000 |
| direct | SE GM | 0.0068 | 0.0108 | +0.0040 |
| direct | Pairwise % | 91.8 | 91.7 | -0.1000 |
| direct | Top-1 % | 84.1 | 84.1 | +0.0000 |
| direct | tau | 0.837 | 0.836 | -0.0010 |
| direct | Runtime (s) | 0.7 | 1.3 | +0.6000 |
| direct+cov | RMSE^d | 0.0256 | 0.0256 | +0.0000 |
| direct+cov | IS | 0.0568 | 0.0751 | +0.0183 |
| direct+cov | Coverage % | 86.1 | 95.6 | +9.5000 |
| direct+cov | SE GM | 0.0074 | 0.0119 | +0.0045 |
| direct+cov | Pairwise % | 94.4 | 94.4 | +0.0000 |
| direct+cov | Top-1 % | 89.5 | 89.4 | -0.1000 |
| direct+cov | tau | 0.888 | 0.889 | +0.0010 |
| direct+cov | Runtime (s) | 1.6 | 3.0 | +1.4000 |
| calibrated-ips | RMSE^d | 0.0246 | 0.0250 | +0.0004 |
| calibrated-ips | IS | 0.5727 | 0.1816 | -0.3911 |
| calibrated-ips | Coverage % | 99.1 | 98.7 | -0.4000 |
| calibrated-ips | SE GM | 0.0963 | 0.0303 | -0.0660 |
| calibrated-ips | Pairwise % | 47.1 | 47.1 | +0.0000 |
| calibrated-ips | Top-1 % | 19.0 | 19.0 | +0.0000 |
| calibrated-ips | tau | -0.059 | -0.058 | +0.0010 |
| calibrated-ips | Runtime (s) | 0.3 | 0.8 | +0.5000 |
| calibrated-ips+cov | RMSE^d | 0.0262 | 0.0263 | +0.0001 |
| calibrated-ips+cov | IS | 0.6084 | 0.2059 | -0.4025 |
| calibrated-ips+cov | Coverage % | 99.2 | 98.7 | -0.5000 |
| calibrated-ips+cov | SE GM | 0.1027 | 0.0344 | -0.0683 |
| calibrated-ips+cov | Pairwise % | 46.5 | 46.5 | +0.0000 |
| calibrated-ips+cov | Top-1 % | 20.5 | 20.5 | +0.0000 |
| calibrated-ips+cov | tau | -0.070 | -0.069 | +0.0010 |
| calibrated-ips+cov | Runtime (s) | 0.4 | 0.9 | +0.5000 |
| stacked-dr | RMSE^d | 0.0224 | 0.0229 | +0.0005 |
| stacked-dr | IS | 0.0788 | 0.1292 | +0.0504 |
| stacked-dr | Coverage % | 95.6 | 99.1 | +3.5000 |
| stacked-dr | SE GM | 0.0125 | 0.0217 | +0.0092 |
| stacked-dr | Pairwise % | 91.9 | 91.5 | -0.4000 |
| stacked-dr | Top-1 % | 83.1 | 82.2 | -0.9000 |
| stacked-dr | tau | 0.838 | 0.830 | -0.0080 |
| stacked-dr | Runtime (s) | 8.9 | 13.9 | +5.0000 |
| stacked-dr+cov | RMSE^d | 0.0281 | 0.0256 | -0.0025 |
| stacked-dr+cov | IS | 0.0806 | 0.1335 | +0.0529 |
| stacked-dr+cov | Coverage % | 96.1 | 99.0 | +2.9000 |
| stacked-dr+cov | SE GM | 0.0128 | 0.0224 | +0.0096 |
| stacked-dr+cov | Pairwise % | 92.1 | 92.9 | +0.8000 |
| stacked-dr+cov | Top-1 % | 84.8 | 85.3 | +0.5000 |
| stacked-dr+cov | tau | 0.842 | 0.858 | +0.0160 |
| stacked-dr+cov | Runtime (s) | 38.2 | 42.8 | +4.6000 |

SE GM ratio (corrected/published): direct: x1.59, direct+cov: x1.61, calibrated-ips: x0.31, calibrated-ips+cov: x0.33, stacked-dr: x1.74, stacked-dr+cov: x1.75.

**Expected-direction check, stacked-dr (OUA x K fix -> SEs rise ~1-2.2x): PASS**

**Expected-direction check, direct (OUA x K fix -> JK/OUA coverage rises toward ~95%, SEs widen ~x1.6-2.2): PASS** (direct: coverage 85.0 -> 93.9, SE GM x1.59; direct+cov: coverage 86.1 -> 95.6, SE GM x1.61)

**Direct point-estimate note:** per-seed point estimates are bit-identical to v1 wherever auto-calibration selected the monotone mode (direct: 88.6% of paired (run, policy) cells have shift exactly 0, p90 1.4e-4, p99 0.010; direct+cov: 57.3% exactly 0, p90 8.9e-5, p99 0.0005, max 0.0063). The remainder are runs where auto selected (or flipped to) the two-stage calibrator, whose fold-model fix in 0.3.0 slightly perturbs predictions — the same mechanism already seen in the stacked-dr rows. The direct max |shift| 0.1297 is a single two-stage run (n=1000, cov=5%, seed 34: ~50 oracle labels; all 3 policies shift together). The direct-row corrections to IS / Coverage / SE GM are driven by the OUA x K fix, not by these point shifts.

**calibrated-ips deviation from erratum-triage.md's expectation:** the triage predicted rising SEs from the OUA fix alone, but `cje-eval` 0.3.0 additionally fixes the CalibratedIPS Hajek influence function (commit f99f911, 'Fix CalibratedIPS Hajek IF'; the shipped IF included a spurious -psi*(w - mean_w) term and was negatively correlated (-0.43) with the true IF). Paired per-run evidence: clone (perfect overlap) SE ratio median ~1.0 (OUA fix only, small), tilted policies (parallel_universe_prompt / premium / unhelpful) median ~0.13-0.25 (IF fix dominates). Point estimates are unchanged (max |shift| ~0 for calibrated-ips), so the SE drop is purely the inference correction.

Note: the Runtime column is environmental (different machine load/library internals) and is not an erratum claim.

### Per-run paired SE ratio (corrected / published), same (estimator, n, coverage, seed, policy)

| Row | n pairs | SE ratio median | SE ratio [p10, p90] | max |est shift| |
|---|---|---|---|---|
| calibrated-ips | 3750 | 0.37x | [0.07, 1.31] | 0.1283 |
| calibrated-ips+cov | 3750 | 0.39x | [0.08, 1.28] | 0.0031 |
| direct | 3750 | 1.71x | [0.79, 2.16] | 0.1297 |
| direct+cov | 3750 | 1.72x | [0.86, 2.13] | 0.0063 |
| stacked-dr | 3750 | 2.04x | [0.99, 2.23] | 0.1267 |
| stacked-dr+cov | 3750 | 2.04x | [1.00, 2.22] | 0.1576 |

### 2b. Direct rows: per-regime JK/OUA coverage (3 policies x 50 seeds = 150 CIs per cell, per-seed `ci_covers_oracle`; slightly different aggregation than the m1 Coverage column)

| Regime | direct pub | **direct corrected** | direct+cov pub | **direct+cov corrected** |
|---|---|---|---|---|
| n=250, cov=5% | 85.3 | **94.7** | 84.0 | **94.0** |
| n=250, cov=10% | 89.3 | **96.0** | 86.0 | **94.7** |
| n=250, cov=25% | 83.3 | **95.3** | 88.7 | **99.3** |
| n=250, cov=50% | 84.0 | **92.0** | 85.3 | **93.3** |
| n=250, cov=100% | 96.7 | **87.3** | 98.0 | **93.3** |
| n=500, cov=5% | 88.7 | **92.7** | 82.0 | **88.0** |
| n=500, cov=10% | 75.3 | **96.7** | 84.7 | **96.7** |
| n=500, cov=25% | 84.0 | **95.3** | 87.3 | **96.7** |
| n=500, cov=50% | 84.7 | **95.3** | 88.0 | **95.3** |
| n=500, cov=100% | 90.0 | **86.7** | 94.0 | **88.0** |
| n=1000, cov=5% | 77.3 | **96.7** | 83.3 | **98.7** |
| n=1000, cov=10% | 82.0 | **98.7** | 86.0 | **97.3** |
| n=1000, cov=25% | 84.7 | **98.7** | 88.7 | **98.7** |
| n=1000, cov=50% | 96.0 | **99.3** | 94.7 | **99.3** |
| n=1000, cov=100% | 96.0 | **88.0** | 96.0 | **88.7** |
| n=2500, cov=5% | 72.0 | **92.7** | 72.7 | **96.0** |
| n=2500, cov=10% | 78.0 | **90.0** | 73.3 | **96.0** |
| n=2500, cov=25% | 88.0 | **94.7** | 82.7 | **98.7** |
| n=2500, cov=50% | 88.7 | **100.0** | 89.3 | **98.7** |
| n=2500, cov=100% | 92.0 | **84.7** | 94.0 | **93.3** |
| n=5000, cov=5% | 80.0 | **95.3** | 77.3 | **96.7** |
| n=5000, cov=10% | 73.3 | **94.7** | 77.3 | **94.7** |
| n=5000, cov=25% | 86.0 | **94.7** | 85.3 | **94.7** |
| n=5000, cov=50% | 93.3 | **99.3** | 92.7 | **98.7** |
| n=5000, cov=100% | 100.0 | **88.0** | 100.0 | **100.0** |

direct: overall coverage 85.9 -> **93.9** (OUA cells cov<100%: 83.7 -> 95.6); mean-SE ratio x0.60-x2.41 (median x1.67; OUA cells only: median x1.75).

direct+cov: overall coverage 86.9 -> **95.6** (OUA cells cov<100%: 84.5 -> 96.3); mean-SE ratio x0.81-x2.14 (median x1.70; OUA cells only: median x1.83).

**cov=100% caveat:** `cje-eval` 0.3.0 skips the oracle jackknife at 100% oracle coverage (no masking -> no OUA term), while the published v1 runs added a (x K-deflated) OUA term there too. The five cov=100% cells per arm are therefore CR-only in v2 vs CR + buggy-OUA in v1 and can move slightly the other way; the OUA-affected cells (cov<100%) isolate the x K fix.

### 2c. Bootstrap-inference variant (0.3.0 default for direct) — not rerun on the accuracy grid

The task called for the bootstrap variant 'if cheap'. It is not: with the published rows' `reward_calibration_mode="auto"`, every run in which auto selects the two-stage calibrator refits it B=500 times — measured 2026-07-07 on this machine: monotone-mode no-cov runs take ~7s (n=250) to ~130s (n=5000), but two-stage no-cov runs take ~820-930s each, and the direct+cov arm (always two-stage) exceeds ~8 min/run even at n=250. Full-grid projection: ~7h (no-cov) plus multiple days (+cov) — vs 14 min for the analytic correction arms. The 37 completed no-cov runs (all successful, seeds 0-1) are kept in `results/direct_bootstrap.partial.jsonl`; they are a partial, non-representative slice and are NOT tabulated.

The recorded bootstrap reference for the direct estimator is instead the UQ-ablation bootstrap arms in Section 1 (same 0.3.0 library, direct no-cov, B=500/B=2000, 16 regimes n<=2500): rerun coverage 95.3-100.0, i.e. ~nominal-to-conservative. Caveats: (i) those runs use the UQ config's default monotone reward calibration rather than auto; (ii) the 0.3.0 direct-mode bootstrap pools oracle labels from fresh draws (keep-every-oracle-pair, commit ca82e05), so oracle masking does not bind and bootstrap cells are not apples-to-apples with the published masked provenance. The correction rows above are the analytic ones.

## 3. OUA-decomposition figure (fig:oua-decomposition, direct+cov, n=1000)

| Oracle % | base SE pub | base SE rerun | oracle SE pub | **oracle SE corrected** | total SE pub | total SE corrected | OUA share pub | **OUA share corrected** |
|---|---|---|---|---|---|---|---|---|
| 5% | 0.0053 | 0.0045 | 0.0177 | **0.0395** | 0.0174 | **0.0373** | 92% | **99%** |
| 10% | 0.0048 | 0.0044 | 0.0119 | **0.0266** | 0.0122 | **0.0252** | 86% | **97%** |
| 25% | 0.0045 | 0.0044 | 0.0062 | **0.0139** | 0.0075 | **0.0141** | 66% | **91%** |
| 50% | 0.0043 | 0.0042 | 0.0044 | **0.0099** | 0.0060 | **0.0102** | 52% | **85%** |
| 100% | 0.0042 | -- | 0.0028 | -- | 0.0050 | -- | 32% | -- (OUA skipped at 100% in 0.3.0) |

**Expected-direction check (OUA share rises; 'calibration uncertainty dominates' strengthens): PASS**
Base SE should be ~unchanged (same sampling term); oracle SE should rise ~sqrt(5) at K=5. v1's caption claim of '~30% at 100% oracle' relied on cov=1.0 rows that the published figure did not plot; cje-eval 0.3.0 intentionally skips the oracle jackknife at 100% coverage, so the v2 figure (like v1) shows 5-50%.

## 4. Prose affected by direct-row correction

Every location in `cje-paper/arxiv_submission/` that cites the published direct
coverage numbers (85.0 / 86.1), the direct estimator's analytic-jackknife coverage
band, or statements derived from them. Compiled for the v2 drafting phase; grep
performed 2026-07-07 over all `.tex` sources.

**Note on the printed math:** the delete-one-fold jackknife formula in
`sections/method.tex` (l.132-135) is the CORRECT sum form
`(K-1)/K * sum_k (V^(-k) - Vbar)^2`; the bug was implementation-only (the shipped
library aggregated with a mean instead of a sum, deflating Var_cal by ~x K).
No formula in the paper needs to change — only the empirical numbers and the
claims built on them.

### A. Directly affected: tab:accuracy direct rows (this correction)

1. `tables/accuracy.tex` l.12: `direct & \textbf{0.0223} & \textbf{0.0542} & 85.0 &
   \underline{0.0068} & 91.8 & 84.1 & 0.837 & 0.7 \\` — the IS (0.0542),
   Coverage (85.0) and SE GM (0.0068) cells are x K artifacts; corrected values in
   Section 2 above. Bold/underline emphasis moves accordingly.
2. `tables/accuracy.tex` l.13: `direct+cov & 0.0256 & \underline{0.0568} & 86.1 &
   0.0074 & \textbf{94.4} & \textbf{89.5} & \textbf{0.888} & 1.6 \\` — same three
   cells (0.0568 / 86.1 / 0.0074) affected.

No prose sentence cites 85.0/86.1 directly; the numbers appear only in the table.
The corrected direct rows now cover near-nominally, which REMOVES the paper's
implicit tab:accuracy contrast "direct undercovers analytically (85-86%) while the
recommended bootstrap path achieves ~95%".

### B. Same x K artifact quoted for the direct estimator's analytic jackknife
(tab:uq-comparison / tab:uq-diagnostics; numbers corrected in wave 1 of this
erratum — JK per-regime 70-89 -> 92.7-100.0, overall 82.9 -> 96.0)

3. `sections/experiments.tex` l.90: "\Cref{tab:uq-comparison} compares UQ methods
   for Direct mode: naive cluster-robust SEs achieve only 15--52\% coverage;
   calibration-aware jackknife improves to 70--89\%; bootstrap with
   $\hat\theta_{\mathrm{aug}}$ (AIPW-style residual augmentation) achieves
   ${\sim}$95\% by correcting both variance and bias from calibration
   uncertainty." — the 70--89\% band is the artifact (corrected: ~93--100\%).
4. `sections/experiments.tex` l.226 (Practical takeaways, item 4): "Always use
   bootstrap inference with $\hat\theta_{\mathrm{aug}}$. Standard CIs severely
   under-cover (15--52\%); calibration-aware jackknife improves but still falls
   short (70--89\%); bootstrap with bias correction achieves near-nominal coverage
   (${\sim}$95\%)." — "still falls short" no longer holds; the corrected jackknife
   is ~nominal.
5. `sections/method.tex` l.149: "Combined with bootstrap (refitting the calibrator
   in each replicate), this achieves ${\sim}$95\% coverage across oracle fractions
   (\cref{tab:uq-comparison}), compared to 70--89\% for jackknife-only variance
   estimation." — stale comparison band.
6. `sections/conclusion.tex` l.14: "(ii) \emph{Always use bootstrap inference with
   $\hat\theta_{\mathrm{aug}}$}: naive CIs achieve 0\% coverage; bootstrap with
   bias correction achieves ${\sim}$95\%." — the 0\% and ~95\% figures survive,
   but the "always bootstrap" recommendation was motivated by jackknife
   undercoverage that was an artifact; the v2 rationale needs re-arguing (bias
   correction / small-K df remain valid reasons).
7. `tables/uq_comparison.tex` caption: "Bootstrap with calibrator refit achieves
   near-nominal coverage; cluster-robust and jackknife-only methods undercover."
   + footnote "\textcolor{red}{Red}: $< 90\%$ (undercoverage)." — after
   correction the jackknife column is no longer red/undercovering (cluster-robust
   still is).
8. `tables/uq_diagnostics.tex` (tab:uq-diagnostics): the four "Cal-Aware
   Jackknife" rows (Coverage 80/80/84/87\%; $\bar Z$ +0.48/+0.40/+0.18/-0.05;
   SD$(Z)$ 3.11/2.52/2.09/1.96) are computed with the x K-deflated SEs; coverage
   and the $Z$ moments change under the fix.
9. `sections/appendix_diagnostics.tex` l.81: "\Cref{tab:uq-diagnostics} compares
   UQ methods across oracle fractions, showing that bootstrap with
   $\hat\theta_{\mathrm{aug}}$ achieves ${\sim}$95\% coverage by correcting both
   variance and bias." — comparative framing rests on the stale jackknife rows.

### C. Adjacent claims checked and UNAFFECTED (verified against the rerun)

- `sections/abstract.tex`: "naive confidence intervals on raw judge scores achieve
  0\% coverage (CJE: ${\sim}$95\%)" — naive-direct has no OUA path (x K bug
  can't touch it) and the bootstrap ~95\% reproduces (95.3-100.0 rerun).
- `sections/introduction.tex` l.52, `sections/method.tex` l.82 & l.115,
  `sections/experiments.tex` l.55-56 & l.88, `tables/ablation.tex` (quotes the
  bootstrap 95\% numbers, per its caption), `sections/experiments.tex` l.33
  ("coverage results may be conservative" — oracle-temperature caveat, unrelated
  mechanism): all survive.
- Ranking/accuracy claims tied to the direct rows (91.8/94.4 pairwise, 84.1/89.5
  top-1, tau 0.837/0.888, RMSE^d 0.0223/0.0256, and the "94\% average" in
  `sections/experiments.tex` l.182 / abstract / introduction) are point-estimate
  metrics; per-seed point estimates are unchanged, so these are unaffected.

