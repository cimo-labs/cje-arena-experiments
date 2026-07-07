# Pre-registered protocol: CJE vs prediction-powered inference at equal oracle-label budgets

Status: **FROZEN** (drafted 2026-07-06; reviewed and frozen 2026-07-06 after
validate.py GREEN + smoke runs. SHA256 of this file at freeze recorded in
results/manifest.json). Any change after this point requires a dated
amendment in §Amendments. Feeds the arXiv v2 comparison section.

Environment: dedicated venv `ppi_comparison/.venv` (python 3.11) with pinned
wheels `cje-eval==0.5.0` and `ppi-python==0.2.3` (full pins in
`requirements.txt`; frozen list also written to `results/manifest.json`).
No imports from any local CJE checkout.

## 1. Estimand and truth convention

- **Estimand**: the superpopulation mean oracle label E[Y] under a policy.
- **Superpopulation convention**: the 5,000-row per-policy corpus IS the
  superpopulation. Every replicate draws row indices **iid WITH replacement**
  from the corpus, so the estimand equals the exact corpus mean. This
  convention is fixed and will not be changed post hoc.
- **Truths** (recomputed exactly by `data_prep.py`; 4-decimal tripwires):
  base 0.7581, clone 0.7620, parallel_universe_prompt 0.7708, premium 0.7623,
  unhelpful 0.1426. Source-file SHA256s recorded in `results/manifest.json`.
- Judge scores and oracle labels both live in [0, 1]. **PPI intervals are NOT
  clipped to [0, 1]** — every arm is reported exactly as shipped.
- Confidence level: **alpha = 0.05 everywhere**, passed explicitly to every
  ppi_py call (their default is alpha = 0.1) and asserted at every adapter
  entry.

## 2. Sampling scheme (single-policy experiment)

Per replicate `rep` at sample size `n`:

1. **Shared n-draw**: n row indices iid with replacement, seeded by
   ("draw", policy, n, rep) only — the identical drawn sample is reused across
   both regimes and all m.
2. **Uniform regime**: labeled subset = first-m prefix of one
   per-("uniform", policy, n, rep) permutation of the n positions → labeled
   subsets are **nested across m**. Inclusion probability pi_i = m/n.
3. **Stratified regime**: 4 judge-score-quartile strata within the n drawn
   rows (positional quartiles of the stable argsort; n divisible by 4 → each
   stratum has exactly n/4 rows). Allocation: m//4 per stratum, remainder to
   the lowest-index strata (nondecreasing in m per stratum, so per-stratum
   prefix labeling stays nested across m). Exact inclusion probabilities
   pi_i = m_s/n_s recorded for every row and used for the weighted arm.
4. **Seeds**: numpy SeedSequence rooted at BASE_SEED = 20260706, keyed
   children per (cell, rep). cje calls receive a per-(cell, rep) integer child
   seed — never the library default 42.

**Stratified-regime caveat (pre-committed):** stratified labeling violates
the MCAR-style assumptions of unweighted PPI/PPI++/cross-PPI *and* of CJE's
uniform-oracle-slice assumption. The stratified regime is a **robustness
panel only**; no primary hypothesis is adjudicated on it. The only arm
"in-assumption" there is PPI++ weighted (arm 9).

## 3. Arms (single-policy)

Uniform adapter: `fn(S_all, Y_all, labeled_idx, alpha, seed, weights=None) →
dict(est, lo, hi, extras)`.

| # | name | description |
|---|------|-------------|
| 1 | `naive_judge_t` | t-interval on the mean of all n raw judge scores (no oracle labels). Expected to fail coverage badly (judge bias ≈ +0.085 on base). |
| 2 | `classical` | `classical_mean_ci` on the m labeled oracle labels only. |
| 3 | `ppi_vanilla` | `ppi_mean_ci(lam=1)`; Yhat = raw judge scores; labeled/unlabeled disjoint split of the n rows. |
| 4 | `ppi_pp` | `ppi_mean_ci(lam=None)` (power-tuned PPI++); records lam_hat. |
| 5 | `ppi_pp_t` | arm 4's center, half-width × t_{m−1,0.975}/z_{0.975} (isolates the quantile-family share of undercoverage). |
| 6 | `postpi_ols_ones` | `postprediction_ols_ci` with X = ones(N,1). Included **iff** it passes validate.py's synthetic mean-recovery check (verdict: **PASSED — included**). Uses numpy's global RNG internally; seeded per call. |
| 7 | `cross_ppi_cje_folds` | cross-PPI fed with CJE's own cross-fitted isotonic fold models: fit arm 8 first, extract per-fold models + fold assignments; Yhat = out-of-fold predictions on labeled rows; Yhat_unlabeled = (N,K) all-fold predictions on unlabeled rows; `crossppi_mean_ci`. |
| 8 | `cje_calibrated` | `cje.calibrated_mean_ci` on all n judge scores with NaN-masked oracle labels, library-default inference="auto" (resolves per library rule; `result.method` recorded as `inference_resolved`). |
| 9 | `ppi_pp_weighted` | **stratified regime only**: `ppi_mean_ci(lam=None, w, w_unlabeled)` with w = 1/pi_i on labeled rows, w_unlabeled = 1/(1−pi_j) on the unlabeled remainder (exact inclusion probabilities from the sampler). |

## 4. Grid and replicates

- Single-policy: policy = base only; m ∈ {10, 25, 50, 100, 250} ×
  n ∈ {500, 2000} × regime ∈ {uniform, stratified} × **R = 1000**
  (`--smoke` → R = 25). 8 worker processes; resumable JSONL output.
- Multi-policy: K = 4 policies (base, clone, parallel_universe_prompt,
  unhelpful), n = 1000/policy with the **same prompt slots per replicate**
  (paired draws), total budget B ∈ {40, 100, 200}, allocation ∈
  {even: B/4 per policy, concentrated: all B on base}, **R = 500**
  (`--smoke` → 10). PPI arms run per policy on its own labels; a zero-label
  policy is recorded as status `NA_requires_labels` (never imputed). CJE arm
  = pooled calibration via `cje.analyze_dataset(fresh_draws_data=…)` with
  `oracle_label=None` masking (per-rep `bootstrap_seed`). Pairwise CIs:
  independent combination for PPI arms; paired influence-function
  `compare_policies` for CJE. `cje.transport_audit` runs on the first 25
  replicates per concentrated cell, probing each non-base policy's drawn rows
  with held-out true labels; PASS/WARN/FAIL rates recorded.

## 5. Metrics

Per (regime, n, arm, m), over R replicates: **coverage** (% of CIs containing
the truth, with Monte-Carlo SE = sqrt(p(1−p)/R)), **mean width**, **RMSE** of
the point estimate, plus lam_hat distribution (arm 4/9), `inference_resolved`
(arm 8), error/NA rates. Multi-policy: per-policy coverage/width/RMSE;
pairwise **sign-error rate** among difference-CIs excluding zero; **power**
(CI excludes zero with the correct sign) on the vs-unhelpful pairs;
transport-audit PASS/WARN/FAIL rates.

## 6. Decision rules (verbatim, pre-committed)

1. **H1 CJE≈cross-PPI (coverage within 2 MC-SE, width ratio ∈ [0.9,1.1], m≥25)**
2. **H2 at m∈{10,25}, n=2000, uniform: CJE ≥93% while z-arms ≥2 MC-SE below nominal, PPI++(t) attributes the quantile-family share**
3. **H3 width(PPI++)/width(cross-PPI) ≤ 1.05 iff calibration curve near-linear**
4. **H4 open question: concentrated CJE ≥90% coverage on all 4 policies?**

## 7. Honesty pre-commitments

- If CJE ≈ cross-PPI (H1 holds), we **report it as-is**: the fair headline is
  that CJE's cross-fitted calibrated estimator is statistically equivalent to
  cross-PPI on this corpus, and the contribution is framed accordingly
  (packaging, diagnostics, transport auditing — not a novel interval).
- **H4 is an open question**, not a claim. A transfer failure (concentrated
  calibration does not transport; coverage collapses on unlabeled policies)
  is a **publishable finding** and will be reported with the same prominence
  as a success, alongside the transport-audit detection rates.
- The stratified regime is out-of-assumption for the unweighted arms and is
  reported as a robustness panel only (see §2 caveat).
- No arm's interval is clipped, tuned, or post-processed beyond what is
  pre-registered here. Failed replicates are recorded (status), never
  silently dropped or imputed.
- The full R=1000/R=500 runs start only after this protocol is frozen; any
  post-freeze change requires a dated amendment section below.

## 8. Pre-registered implementation judgment calls

1. **lam_hat recording**: `ppi_mean_ci` does not return its power-tuning
   estimate, so `lam_hat` is recorded by replicating the library's exact
   lam=None preamble (importing the private `_calc_lam_glm` from the pinned
   wheel). validate.py check (b) asserts the recorded lam_hat reproduces the
   library's lam=None interval to 1e-12. Intervals always come from
   `ppi_mean_ci` itself.
2. **PostPI verdict**: `postprediction_ols_ci` with X=ones passed the
   synthetic mean-recovery check (validate.py (c)) → **arm 6 is included**.
   Note the method carries no coverage guarantee (per ppi_py docs) and
   ignores labeled-set uncertainty; expected to undercover — that is the
   point of including it.
3. **Fold extraction**: arm 7 uses `calibrator.get_fold_models_for_oua()` +
   `calibrator.predict_oof` (the single OOF entry point that fit_cv itself
   uses for its OOF diagnostics). Fold assignments are taken from
   `calibrator._fold_ids` and asserted equal to the public
   `get_folds_for_prompts` recomputation on every call. validate.py (d)
   additionally verifies manual per-fold-model predictions match
   `predict_oof` to <1e-8 and that selected_mode is monotone (plain isotonic
   fold models). No reimplementation fallback was needed.
4. **CJE inference mode**: arm 8 uses the library default `inference="auto"`
   (which resolves to the cluster bootstrap with calibrator refit for this
   coupled design); we record `result.method` and the bootstrap skip rate
   rather than overriding the library's own defaults.
5. **Weighted-arm convention**: the unlabeled remainder of a stratified draw
   is itself non-uniform; its inclusion probability is 1−pi_j, so
   w_unlabeled = 1/(1−pi_j). ppi_py normalizes weights internally (Hájek).
6. **Multi-policy CJE path**: `analyze_dataset(fresh_draws_data=…)` was
   benchmarked as fast enough per replicate (~4-8 s) and is used as-is (no
   "manual pooled" fallback). Records use unique slot ids
   (`slot_0…slot_999`); the same slot maps to the same corpus prompt across
   policies, preserving paired comparisons; duplicate corpus rows drawn with
   replacement get distinct slots (consistent with the iid superpopulation
   convention). Seeding: `estimator_config={"bootstrap_seed": per-rep seed}`
   (the calibration fold hash-seed inside analyze_dataset is not exposed;
   replicate-to-replicate variation comes from the data draws).
7. **vs-unhelpful pairs**: with K=4 policies there are exactly **3**
   vs-unhelpful pairs (base, clone, parallel_universe_prompt vs unhelpful).
   The task brief said "the 4 vs-unhelpful pairs"; 3 is the count consistent
   with K=4 and is what is computed. (All 6 pairwise comparisons are
   recorded; power is reported on the vs-unhelpful subset.)
8. **postpi point estimate**: `postprediction_ols_ci` returns only the CI;
   since it is z-symmetric around its internal median-bootstrap point
   estimate, the CI midpoint is recorded as `est`.
9. **naive-judge arm** ignores `labeled_idx` by design (it is the
   no-oracle-labels baseline; its "budget" is 0 labels).
10. **Corpus alignment (verified deviation from the task brief)**: the five
    source JSONLs share the same 5,000 prompt_id SET but **not** the same row
    order ("same prompts in same order" was false as stated). Per-id prompt
    text is identical across files, so `data_prep.py` aligns rows by
    prompt_id, with base's file order as the canonical `prompt_idx`. All
    paired multi-policy draws index this aligned corpus.

## Amendments

(none — protocol frozen as drafted; add dated entries here if anything
changes after review)
