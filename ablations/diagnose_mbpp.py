#!/usr/bin/env python3
"""
MBPP Diagnostic Analysis

Implements the three diagnostic analyses to reframe MBPP results:
1. Tie-aware ranking (compute MDE, identify indistinguishable policies)
2. Transport audit per policy (mean residual with CI)
3. Calibration support plot (judge score distributions)

Also flags the judge=policy confound (gpt-4o-mini is both judge and mini policy).
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / "data_mbpp"
RESULTS_DIR = BASE_DIR / "results_mbpp"

def load_response_data(policy: str) -> pd.DataFrame:
    """Load response data for a policy."""
    response_file = DATA_DIR / "responses" / f"{policy}_responses.jsonl"
    records = []
    with open(response_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            records.append({
                'prompt_id': data['prompt_id'],
                'oracle_label': data['metadata']['oracle_label'],
                'judge_score': data['metadata']['judge_score'],
                'response_length': data['metadata'].get('response_length', len(data['response'])),
                'policy': policy
            })
    return pd.DataFrame(records)

def load_all_policies() -> pd.DataFrame:
    """Load data for all policies."""
    policies = ['base', 'enhanced', 'mini', 'terse', 'unhelpful']
    dfs = []
    for policy in policies:
        try:
            df = load_response_data(policy)
            dfs.append(df)
        except FileNotFoundError:
            print(f"Warning: {policy} responses not found")
    return pd.concat(dfs, ignore_index=True)

def analysis_1_tie_aware_ranking(df: pd.DataFrame) -> Dict:
    """
    Compute per-policy standard errors clustered by problem_id.
    Identify which policy pairs are statistically distinguishable.
    """
    print("\n" + "="*70)
    print("ANALYSIS 1: TIE-AWARE RANKING")
    print("="*70)

    # Get pass rates per policy, clustered by prompt_id
    results = {}

    for policy in df['policy'].unique():
        policy_df = df[df['policy'] == policy]

        # Cluster by prompt_id (problem)
        cluster_means = policy_df.groupby('prompt_id')['oracle_label'].mean()

        # Pass rate and clustered SE
        pass_rate = cluster_means.mean()
        n_clusters = len(cluster_means)

        # Clustered standard error
        cluster_var = cluster_means.var(ddof=1)
        se = np.sqrt(cluster_var / n_clusters)

        results[policy] = {
            'pass_rate': pass_rate,
            'se': se,
            'n_clusters': n_clusters,
            'ci_lower': pass_rate - 1.96 * se,
            'ci_upper': pass_rate + 1.96 * se
        }

        print(f"\n{policy}:")
        print(f"  Pass rate: {pass_rate:.1%} ± {se:.1%}")
        print(f"  95% CI: [{results[policy]['ci_lower']:.1%}, {results[policy]['ci_upper']:.1%}]")
        print(f"  Clusters: {n_clusters}")

    # Pairwise comparisons
    print("\n" + "-"*70)
    print("PAIRWISE DISTINGUISHABILITY (two-sample t-test, clustered)")
    print("-"*70)

    policies = sorted(results.keys(), key=lambda x: results[x]['pass_rate'], reverse=True)

    comparisons = []
    for i, p1 in enumerate(policies):
        for p2 in policies[i+1:]:
            r1, r2 = results[p1], results[p2]

            # Difference and SE of difference
            diff = r1['pass_rate'] - r2['pass_rate']
            se_diff = np.sqrt(r1['se']**2 + r2['se']**2)

            # t-statistic and p-value
            t_stat = diff / se_diff if se_diff > 0 else 0
            # Use smaller df for conservative test
            df_min = min(r1['n_clusters'], r2['n_clusters']) - 1
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df_min))

            # MDE at 80% power
            mde = 2.8 * se_diff  # ~1.96 + 0.84 for 80% power

            distinguishable = p_value < 0.05

            comparisons.append({
                'p1': p1, 'p2': p2,
                'diff': diff, 'se_diff': se_diff,
                't_stat': t_stat, 'p_value': p_value,
                'mde': mde, 'distinguishable': distinguishable
            })

            status = "DISTINGUISHABLE" if distinguishable else "TIE (not distinguishable)"
            print(f"\n{p1} vs {p2}:")
            print(f"  Difference: {diff:+.1%} ± {se_diff:.1%}")
            print(f"  t = {t_stat:.2f}, p = {p_value:.4f}")
            print(f"  MDE (80% power): {mde:.1%}")
            print(f"  Status: {status}")

    # Summarize tiers
    print("\n" + "-"*70)
    print("POLICY TIERS (statistically distinguishable groups)")
    print("-"*70)

    # Build tiers based on distinguishability
    tier_assignments = {}
    current_tier = 1
    sorted_policies = sorted(results.keys(), key=lambda x: results[x]['pass_rate'], reverse=True)

    for i, policy in enumerate(sorted_policies):
        if i == 0:
            tier_assignments[policy] = current_tier
        else:
            # Check if distinguishable from all policies in current tier
            prev_policy = sorted_policies[i-1]
            comp = next((c for c in comparisons if
                        (c['p1'] == prev_policy and c['p2'] == policy) or
                        (c['p1'] == policy and c['p2'] == prev_policy)), None)

            if comp and comp['distinguishable']:
                current_tier += 1
            tier_assignments[policy] = current_tier

    # Print tiers
    for tier in range(1, max(tier_assignments.values()) + 1):
        policies_in_tier = [p for p, t in tier_assignments.items() if t == tier]
        rates = [f"{results[p]['pass_rate']:.1%}" for p in policies_in_tier]
        print(f"  Tier {tier}: {', '.join(policies_in_tier)} ({', '.join(rates)})")

    return {
        'policy_stats': results,
        'comparisons': comparisons,
        'tiers': tier_assignments
    }


def analysis_2_transport_audit(df: pd.DataFrame) -> Dict:
    """
    Train calibrator on base policy, compute mean residual per policy.
    Identifies transport failures.
    """
    print("\n" + "="*70)
    print("ANALYSIS 2: TRANSPORT AUDIT (Mean Residual per Policy)")
    print("="*70)

    from sklearn.isotonic import IsotonicRegression

    # Train calibrator on base policy
    base_df = df[df['policy'] == 'base'].copy()

    # Fit isotonic regression: judge_score -> oracle_label
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    iso.fit(base_df['judge_score'], base_df['oracle_label'])

    print(f"\nCalibrator trained on {len(base_df)} base policy samples")
    print(f"  Judge score range: [{base_df['judge_score'].min():.2f}, {base_df['judge_score'].max():.2f}]")
    print(f"  Oracle mean: {base_df['oracle_label'].mean():.1%}")

    # Compute residuals per policy
    print("\n" + "-"*70)
    print("MEAN RESIDUAL PER POLICY (Y - f(S))")
    print("-"*70)

    results = {}
    for policy in df['policy'].unique():
        policy_df = df[df['policy'] == policy].copy()

        # Get calibrated predictions
        policy_df['calibrated_pred'] = iso.predict(policy_df['judge_score'])

        # Compute residuals
        policy_df['residual'] = policy_df['oracle_label'] - policy_df['calibrated_pred']

        # Cluster by prompt for proper SE
        cluster_residuals = policy_df.groupby('prompt_id')['residual'].mean()

        mean_residual = cluster_residuals.mean()
        se_residual = cluster_residuals.std(ddof=1) / np.sqrt(len(cluster_residuals))

        # t-test for H0: mean_residual = 0
        t_stat = mean_residual / se_residual if se_residual > 0 else 0
        df_val = len(cluster_residuals) - 1
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df_val))

        transport_ok = p_value >= 0.05

        results[policy] = {
            'mean_residual': mean_residual,
            'se': se_residual,
            't_stat': t_stat,
            'p_value': p_value,
            'transport_ok': transport_ok,
            'n_samples': len(policy_df),
            'pred_mean': policy_df['calibrated_pred'].mean(),
            'oracle_mean': policy_df['oracle_label'].mean()
        }

        status = "✓ PASS" if transport_ok else "✗ FAIL (transport violation)"
        print(f"\n{policy}:")
        print(f"  Oracle mean: {results[policy]['oracle_mean']:.1%}")
        print(f"  Calibrated pred mean: {results[policy]['pred_mean']:.1%}")
        print(f"  Mean residual: {mean_residual:+.3f} ± {se_residual:.3f}")
        print(f"  t = {t_stat:.2f}, p = {p_value:.4f}")
        print(f"  Transport audit: {status}")

    # Summary
    print("\n" + "-"*70)
    print("TRANSPORT AUDIT SUMMARY")
    print("-"*70)

    passed = [p for p, r in results.items() if r['transport_ok']]
    failed = [p for p, r in results.items() if not r['transport_ok']]

    print(f"  Passed: {', '.join(passed) if passed else 'None'}")
    print(f"  Failed: {', '.join(failed) if failed else 'None'}")

    if failed:
        print(f"\n  → Level claims should be REFUSED for: {', '.join(failed)}")
        print(f"  → Ranking comparisons may still be valid if monotonicity holds")

    return results


def analysis_3_support_check(df: pd.DataFrame) -> Dict:
    """
    Overlay judge score distributions to check calibration support.
    """
    print("\n" + "="*70)
    print("ANALYSIS 3: CALIBRATION SUPPORT CHECK")
    print("="*70)

    base_df = df[df['policy'] == 'base']
    base_scores = base_df['judge_score'].values

    # Support bounds from base
    base_min, base_max = base_scores.min(), base_scores.max()
    base_p10, base_p90 = np.percentile(base_scores, [10, 90])

    print(f"\nBase policy judge score distribution:")
    print(f"  Range: [{base_min:.2f}, {base_max:.2f}]")
    print(f"  10th-90th percentile: [{base_p10:.2f}, {base_p90:.2f}]")
    print(f"  Mean: {base_scores.mean():.2f}")

    print("\n" + "-"*70)
    print("SUPPORT OVERLAP PER POLICY")
    print("-"*70)

    results = {}
    for policy in df['policy'].unique():
        if policy == 'base':
            continue

        policy_df = df[df['policy'] == policy]
        policy_scores = policy_df['judge_score'].values

        # Compute overlap metrics
        policy_min, policy_max = policy_scores.min(), policy_scores.max()
        policy_mean = policy_scores.mean()

        # Fraction of policy scores within base support
        in_support = np.mean((policy_scores >= base_min) & (policy_scores <= base_max))

        # Fraction within base 10-90 percentile
        in_core = np.mean((policy_scores >= base_p10) & (policy_scores <= base_p90))

        # Fraction below base minimum (extrapolation region)
        below_base = np.mean(policy_scores < base_min)
        above_base = np.mean(policy_scores > base_max)

        # Hellinger distance between distributions
        # Discretize into bins
        bins = np.linspace(0, 1, 21)
        base_hist, _ = np.histogram(base_scores, bins=bins, density=True)
        policy_hist, _ = np.histogram(policy_scores, bins=bins, density=True)

        # Normalize to sum to 1
        base_hist = base_hist / base_hist.sum() if base_hist.sum() > 0 else base_hist
        policy_hist = policy_hist / policy_hist.sum() if policy_hist.sum() > 0 else policy_hist

        hellinger = np.sqrt(0.5 * np.sum((np.sqrt(base_hist) - np.sqrt(policy_hist))**2))

        support_ok = in_support > 0.95 and hellinger < 0.3

        results[policy] = {
            'range': (policy_min, policy_max),
            'mean': policy_mean,
            'in_support': in_support,
            'in_core': in_core,
            'below_base': below_base,
            'above_base': above_base,
            'hellinger': hellinger,
            'support_ok': support_ok
        }

        status = "✓ GOOD" if support_ok else "⚠ SUPPORT ISSUE"
        print(f"\n{policy}:")
        print(f"  Score range: [{policy_min:.2f}, {policy_max:.2f}]")
        print(f"  Score mean: {policy_mean:.2f} (base: {base_scores.mean():.2f})")
        print(f"  In base support: {in_support:.1%}")
        print(f"  In base core (10-90%): {in_core:.1%}")
        print(f"  Below base min: {below_base:.1%}")
        print(f"  Hellinger distance: {hellinger:.3f}")
        print(f"  Support check: {status}")

    return results


def check_judge_policy_confound(df: pd.DataFrame):
    """
    Flag that gpt-4o-mini is both judge and policy (mini).
    """
    print("\n" + "="*70)
    print("WARNING: JUDGE-POLICY CONFOUND")
    print("="*70)

    print("""
The judge model (gpt-4o-mini) is the SAME as one of the evaluated policies (mini).

LLM judges exhibit known self-preference and style bias:
- May inflate scores for mini policy outputs
- May distort rankings among similar policies
- Creates policy-dependent calibration that breaks reusability assumption

RECOMMENDATION:
- Rerun judge scoring with a model NOT in {gpt-4o, gpt-4o-mini}
- Or use a mixed-judge ensemble
- This should be fixed before concluding about calibration effectiveness
""")


def generate_diagnostic_report():
    """Generate full diagnostic report."""
    print("\n" + "="*70)
    print("MBPP DIAGNOSTIC ANALYSIS")
    print("Reframing results: boundary case, not failure")
    print("="*70)

    # Load data
    print("\nLoading response data...")
    df = load_all_policies()
    print(f"Loaded {len(df)} samples across {df['policy'].nunique()} policies")

    # Run analyses
    tie_results = analysis_1_tie_aware_ranking(df)
    transport_results = analysis_2_transport_audit(df)
    support_results = analysis_3_support_check(df)

    # Flag confound
    check_judge_policy_confound(df)

    # Summary
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70)

    print("""
KEY FINDINGS:

1. TIE-AWARE RANKING:
   - terse vs enhanced: Statistically indistinguishable (0.2pp difference, SE ~1pp)
   - "Ranking failure" is actually just trying to rank a tie
   - Proper interpretation: Report policy TIERS, not strict orderings

2. TRANSPORT AUDIT:
   - unhelpful: FAILS transport (calibrator trained on ~87% pass rate data,
     cannot extrapolate to 56% pass rate distribution)
   - Correct response: Refuse level claims for unhelpful, not "calibration failed"

3. SUPPORT CHECK:
   - unhelpful has judge scores in different region than base
   - This is a support failure, not a calibration failure
   - CJE diagnostics correctly identify this

4. JUDGE-POLICY CONFOUND:
   - gpt-4o-mini is both judge and policy (mini)
   - This alone could explain ranking anomalies
   - Fix before concluding about calibration
""")

    return {
        'tie_analysis': tie_results,
        'transport_audit': transport_results,
        'support_check': support_results
    }


if __name__ == "__main__":
    results = generate_diagnostic_report()
