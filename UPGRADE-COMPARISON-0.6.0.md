# CJE 0.5.1 vs 0.6.0rc1 — empirical comparison on the arena corpus

Design: R=200 paired replicates (identical draws + seeds across versions), n=1000 shared prompts/policy drawn iid with replacement from the 5000-row fully-oracle-labeled corpus, m=100 oracle labels on `base` only (other 4 policies judge-only fresh draws). DEFAULT `analyze_dataset` path in both versions (`estimator_config={'bootstrap_seed': seed}`). Truth per policy = exact full-corpus oracle mean; coverage = truth in the 95% CI.

## Per-policy summary

| policy | truth | version | mean est | bias | SD(est) | mean SE | SE/SD | mean CI width | coverage (95% nominal) |
|---|---|---|---|---|---|---|---|---|---|
| base | 0.7581 | 0.5.1 | 0.7556 | -0.0025 | 0.0233 | 0.0247 | 1.06 | 0.0973 | 94.5% ± 1.6% |
| base | 0.7581 | 0.6.0rc1 | 0.7559 | -0.0021 | 0.0225 | 0.0237 | 1.05 | 0.0905 | 94.5% ± 1.6% |
| clone | 0.7620 | 0.5.1 | 0.7602 | -0.0017 | 0.0220 | 0.0220 | 1.00 | 0.0866 | 93.0% ± 1.8% |
| clone | 0.7620 | 0.6.0rc1 | 0.7602 | -0.0018 | 0.0212 | 0.0203 | 0.96 | 0.0767 | 93.0% ± 1.8% |
| parallel_universe_prompt | 0.7708 | 0.5.1 | 0.7655 | -0.0054 | 0.0217 | 0.0217 | 1.00 | 0.0855 | 93.5% ± 1.7% |
| parallel_universe_prompt | 0.7708 | 0.6.0rc1 | 0.7653 | -0.0055 | 0.0209 | 0.0200 | 0.96 | 0.0756 | 91.5% ± 2.0% |
| premium | 0.7623 | 0.5.1 | 0.7617 | -0.0006 | 0.0221 | 0.0220 | 1.00 | 0.0864 | 93.0% ± 1.8% |
| premium | 0.7623 | 0.6.0rc1 | 0.7616 | -0.0006 | 0.0214 | 0.0204 | 0.95 | 0.0767 | 92.5% ± 1.9% |
| unhelpful | 0.1426 | 0.5.1 | 0.3929 | +0.2503 | 0.1697 | 0.1136 | 0.67 | 0.3880 | 13.0% ± 2.4% |
| unhelpful | 0.1426 | 0.6.0rc1 | 0.3917 | +0.2491 | 0.1702 | 0.0868 | 0.51 | 0.3108 | 7.5% ± 1.9% |

## What changes on upgrade (paired per-replicate, 0.6.0rc1 − 0.5.1)

| policy | SE ratio (060/051) | width ratio | Δ coverage (pp) | mean Δest | mean Δest / SE₀₅₁ | mean \|Δest\| / SE₀₅₁ | max \|Δest\| / SE₀₅₁ | cover→miss | miss→cover |
|---|---|---|---|---|---|---|---|---|---|
| base | 0.960 | 0.929 | +0.0 | +0.00038 | +0.016 | 0.205 | 3.759 | 3 | 3 |
| clone | 0.924 | 0.887 | +0.0 | -0.00005 | -0.002 | 0.078 | 4.342 | 2 | 2 |
| parallel_universe_prompt | 0.921 | 0.884 | -2.0 | -0.00014 | -0.006 | 0.076 | 4.378 | 4 | 0 |
| premium | 0.926 | 0.888 | -0.5 | -0.00008 | -0.004 | 0.076 | 4.406 | 1 | 0 |
| unhelpful | 0.764 | 0.801 | -5.5 | -0.00124 | -0.011 | 0.117 | 4.262 | 11 | 0 |

## Run-level detail

### 0.5.1

- method: {'calibrated_direct_bootstrap': 200}
- inference method: {'cluster_bootstrap_refit': 200}
- bootstrap scheme: {'(none recorded)': 200}
- bootstrap refit mode: {'monotone': 166, 'two_stage': 34}
- n_bootstrap_valid: {2000: 200}; mean skip_rate 0.0000
- runtime/call: mean 9.7s, median 5.9s, max 29.7s
- gate flagged (reps/policy): {'unhelpful': 104, 'base': 1, 'clone': 1, 'parallel_universe_prompt': 1}
- refuse-level (reps/policy): {'unhelpful': 104, 'base': 1, 'clone': 1, 'parallel_universe_prompt': 1}
- warning families (count over reps): {'WARNING:cje.diagnostics.reward_boundary': 107, 'WARNING:cje.calibration.judge': 3}

### 0.6.0rc1

- method: {'calibrated_direct_bootstrap': 200}
- inference method: {'cluster_bootstrap_refit': 200}
- bootstrap scheme: {'positive_exponential_cluster_weights': 200}
- bootstrap refit mode: {'auto': 200}
- n_bootstrap_valid: {2000: 200}; mean skip_rate 0.0000
- runtime/call: mean 35.6s, median 35.5s, max 38.6s
- gate flagged (reps/policy): {'unhelpful': 104, 'base': 1, 'clone': 1, 'parallel_universe_prompt': 1}
- refuse-level (reps/policy): {'unhelpful': 104, 'base': 1, 'clone': 1, 'parallel_universe_prompt': 1}
- warning families (count over reps): {'WARNING:cje.diagnostics.reward_boundary': 107, 'WARNING:cje.calibration.judge': 2}

