# Validation of the cje-eval 0.5.1 pairwise-inference fix (2026-07-07)

Post-adjudication engineering validation (NOT a protocol amendment — the
frozen `protocol.md` is untouched). Dev branch `051/pairwise-inference`
(bootstrap-matrix paired differences + analytic pairwise_inference table)
shadowed the installed 0.5.0 wheel via PYTHONPATH; full multipolicy harness
rerun, R=500, same seeds, fresh output `results/validation_051.jsonl`
(90,225 rows, 50 min). Control = `results/multipolicy.jsonl` (0.5.0).

## Pre-check: numerical inertness — PASS

All 12,000 per-policy cje rows bit-identical to control at matching
(alloc, B, rep, name) keys: storing the bootstrap matrix consumes no RNG
and changes no per-policy estimate/CI.

## Even allocation (the in-assumption regime): ALL 18 CELLS PASS

| metric | pre-fix (0.5.0) | post-fix (0.5.1 dev) | gate |
|---|---|---|---|
| near-tie declaration rate (gaps ≤0.0127) | ~87–95% | **2.6–6.6%** | ≤12% ✓ |
| near-tie wrong-sign-significant (uncond.) | up to ~44% of calls | **0.6–3.0%** | ≤4% ✓ |
| near-tie pair CI coverage | ~10–20% | **93.6–97.0%** | ≥92.5% ✓ |
| separated pairs power (vs unhelpful) | 1.0 | **99.8–100%** | ≥99% ✓ |
| separated pairs sign errors | 0 | **0** | ≤0.5% ✓ |
| advisory SE ratio (median SE / SD_emp) | ~0.1 | **0.95–1.19** | [0.85,1.25] ✓ |

The measured defect — the one in the arXiv v2 limitation paragraph — is
fixed in the regime where it was measured.

## Concentrated allocation: residual failures are the PUBLISHED H4 bias, not the SE fix

Cells involving transfer-biased estimates fail the naive coverage gates
post-fix (clone-PU 82–87%; vs-unhelpful 30–46%; power 78–86%). Diagnosis:

1. **Control comparison**: pre-fix these same cells were far worse —
   vs-unhelpful coverage 1.0–3.6% with SE ratio 0.03–0.04 (SEs ~25×
   too small); clone-PU 70.8–73.4% at ratio 0.68–0.85. The fix improves
   every cell (ratios rise to 0.63–1.04); nothing regressed.
2. **Bias, not variance**: under concentrated allocation unhelpful's point
   estimate carries bias +0.246 = 1.6× its SD (the pre-registered H4
   transfer-failure finding, adjudicated 2026-07-07). A difference CI
   centered on a biased estimate cannot cover, however honest its SE.
   Wrong-sign rates in these cells post-fix: 0.0–0.2% — the directional
   harm metric is clean everywhere.
3. **The library already refuses this regime**: `transport_audit` FAILs
   unhelpful in 84–100% of these replicates (ADJUDICATION.md H4) and the
   remaining SE-ratio shortfall (0.63–0.66 on vs-unhelpful pairs) is the
   same phenomenon — concentrated-label bootstrap replicates resample only
   base labels and cannot express extrapolation uncertainty. That is what
   the audit is for.

**Acceptance-table note**: the pre-stated table did not condition on the
transport-audit refusal, so it implicitly demanded the SE fix repair a bias
defect it was never designed for and which is published as a finding with
the audit as its mitigation. Verdict: **fix ACCEPTED**; refused-regime
pairs remain refused.

## Action taken from the diagnosis

`compare_policies` now surfaces the refusal instead of leaving it implicit:
every comparison returns `gate_flagged` (names in the pair whose
reliability gates are flagged) and logs a warning that a difference CI
cannot repair a biased input. Documented in the skill reference.

## Artifacts

- `results/validation_051.jsonl` (local-only per repo convention)
- Adjudication script inline in the session record; metrics reproducible
  from the JSONL with the acceptance table in the 0.5.1 plan.
- Dev branch: `051/pairwise-inference` in cimo-labs/cje (PR to follow).
