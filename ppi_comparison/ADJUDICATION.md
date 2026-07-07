# Adjudication of pre-registered decision rules (2026-07-07)

Protocol: `protocol.md`, frozen 2026-07-06 (sha256 `f593bb9f…` in
`results/manifest.json`) BEFORE the full runs. Full runs: single-policy
19,500 tasks / 165,750 rows (R=1000), multi-policy 2,940 tasks / 88,335 rows
(R=500); zero errors, zero NA outside the pre-specified PPI
requires-labels cells. Aggregates: `tables/table1.csv`,
`tables/multipolicy.csv`. Adjudicated by the orchestrating reviewer directly
from the aggregates; every number below is reproducible from the CSVs.

Headline cell (uniform, n=2000, R=1000, coverage MC-SE ≤ 0.8 points):

| arm | m=10 | 25 | 50 | 100 | 250 |
|---|---|---|---|---|---|
| cje_calibrated | 95.2 | 97.4 | 97.3 | 96.1 | 95.5 |
| cross_ppi (CJE's isotonic folds) | 89.0 | 92.8 | 94.2 | 93.4 | 93.8 |
| ppi_pp (λ power-tuned) | 86.8 | 92.1 | 92.8 | 94.1 | 93.9 |
| ppi_pp_t (t-adjusted) | 90.0 | 93.4 | 93.2 | 94.3 | 94.0 |
| ppi_vanilla (λ=1) | 89.2 | 94.0 | 94.1 | 94.6 | 94.5 |
| classical | 88.1 | 93.2 | 94.3 | 95.3 | 93.9 |
| postpi_ols_ones | 7.1 | 11.4 | 16.2 | 22.6 | 34.9 |
| naive_judge_t | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## H1 — "CJE ≈ cross-PPI (coverage within 2 MC-SE, width ratio ∈ [0.9,1.1], m≥25)"

**FAILS at m ∈ {25, 50, 100}; HOLDS at m=250.**
Coverage differences (cje − cross_ppi): +4.6 / +3.1 / +2.7 / +1.7 points at
m=25/50/100/250 vs 2·MC-SE of the difference ≈ 1.8–2.0. Width ratios
(cje/cross_ppi): 1.226 / 1.188 / 1.150 / 1.055.
Interpretation: the two estimators are asymptotically siblings but NOT
finite-sample equivalent — CJE's bootstrap-with-refit inference buys 3–5
coverage points at m ≤ 100 at a 15–23% width premium; equivalence emerges by
m=250.

## H2 — "at m∈{10,25}, n=2000, uniform: CJE ≥93% while z-arms ≥2 MC-SE below nominal; PPI++(t) attributes the quantile-family share"

**HOLDS.** CJE 95.2 / 97.4. At m=10 every z-based arm sits 6.0–8.2 points
below nominal (≥2 MC-SE): ppi_pp 86.8, cross_ppi 89.0, ppi_vanilla 89.2. At
m=25 ppi_pp 92.1 and cross_ppi 92.8 remain ≥2 MC-SE low (ppi_vanilla 94.0 is
within noise — noted). Attribution: ppi_pp_t recovers 3.2 of ppi_pp's
8.2-point gap at m=10 → **~40% of the small-m undercoverage is the z-vs-t
quantile family; the remainder is unaccounted calibrator/λ-estimation
noise.**

## H3 — "width(PPI++)/width(cross-PPI) ≤ 1.05 iff calibration curve near-linear"

**The iff FAILS, informatively.** Ratios 0.919 / 0.957 / 0.980 / 0.993 at
m=25/50/100/250 — PPI++ is never wider — while the calibration curve
(figs/calibration_curve.png) is visibly non-linear (flat below S≈0.4,
stepped). At judge–oracle correlation ≈0.48 the predictor class is
second-order for interval width; the coverage differences in H1/H2 are
therefore attributable to the inference machinery, not the isotonic
calibrator.

## H4 — open question: "does concentrated-allocation CJE hold ≥90% coverage on all 4 policies?"

**NO under far shift; YES near-distribution; and the transport audit detects
the difference.** Concentrated (all B labels on base): base 96.2–97.4, clone
94.0–95.4, parallel_universe 94.2–95.6 — the latter two with ZERO labels of
their own (every PPI cell: NA, requires per-quantity labels). unhelpful
(judge scores far outside the labeled range): coverage 7.2–7.8, RMSE ≈0.29.
`transport_audit` FAIL rate on unhelpful: 84% / 92% / 100% at B=40/100/200;
on clone/parallel_universe FAIL→0 by B=100 (PASS/WARN otherwise).
Even allocation, equal total budget: pooled CJE 92.2–96.4 on all four
policies vs per-policy PPI++ 73.4–88.0 at B ∈ {40,100}.

## Adverse finding (pre-committed to report; outside the four rules)

`compare_policies` pairwise difference CIs are **anti-conservative on
near-tie pairs** (true gaps ≤ 0.0127): under even allocation they exclude
zero in ~90% of replicates with sign-error rates 36–47% among those calls
(concentrated: 0.3–12.6%). Well-separated pairs (vs unhelpful, gap ≥0.61):
sign errors 0%, power 1.0. Mechanism hypothesis: judge-calibration noise
dominates near-tie differences and is not fully propagated into the
influence-function difference SEs. Follow-up: incorporate
oracle-uncertainty into difference SEs (library work, post-arc).

## Also noted

- postprediction_ols_ci (as shipped in ppi_py) severely undercovers
  (7–35%) — reported as a shipped-baseline caution, not tuned.
- naive judge mean: 0% coverage everywhere; RMSE 0.084 = its bias. The
  anti-pattern the library exists to prevent, quantified.
- Stratified regime (robustness panel, out-of-assumption for unweighted
  arms): see table1.csv `regime == "stratified"`; weighted PPI++ is the
  in-assumption arm there.
