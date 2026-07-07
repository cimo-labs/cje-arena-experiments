#!/usr/bin/env python3
"""
Generate the uncertainty decomposition figure (oua_vs_base_variance.png).

Shows how total uncertainty decomposes into:
- Base sampling uncertainty (constant across oracle coverage)
- Calibration uncertainty (dominates at low oracle coverage)

Usage:
    python scripts/plot_oua_decomposition.py --output ../cje-paper/arxiv_submission/figs/oua_vs_base_variance.png
"""

import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def load_experiments(results_path: Path) -> list:
    """Load experiment results from JSONL file."""
    experiments = []
    with open(results_path, 'r') as f:
        for line in f:
            if line.strip():
                experiments.append(json.loads(line))
    return experiments


def extract_variance_data(experiments: list, estimator: str = "direct", sample_size: int = 1000) -> dict:
    """Extract variance components for a specific estimator and sample size.

    Returns dict mapping oracle_coverage -> {base_se, oracle_se, total_se}
    """
    # Filter to direct+cov estimator at n=1000
    filtered = [
        exp for exp in experiments
        if exp.get('spec', {}).get('estimator') == estimator
        and exp.get('spec', {}).get('sample_size') == sample_size
        and exp.get('spec', {}).get('extra', {}).get('use_covariates', False)
        and exp.get('success', False)
    ]

    if not filtered:
        # Try without covariates
        filtered = [
            exp for exp in experiments
            if exp.get('spec', {}).get('estimator') == estimator
            and exp.get('spec', {}).get('sample_size') == sample_size
            and exp.get('success', False)
        ]

    print(f"Found {len(filtered)} experiments for {estimator} at n={sample_size}")

    # Group by oracle coverage and average
    coverage_data = {}
    for exp in filtered:
        oracle_cov = exp.get('spec', {}).get('oracle_coverage', 0)
        oracle_cov_pct = int(oracle_cov * 100)

        # Get SE components
        se_components = exp.get('metadata', {}).get('se_components', {})
        oracle_var_per_policy = se_components.get('oracle_variance_per_policy', {})

        # Average across policies (excluding unhelpful which has extreme variance)
        policies = ['clone', 'parallel_universe_prompt', 'premium']
        oracle_vars = [oracle_var_per_policy.get(p, 0) for p in policies if p in oracle_var_per_policy]

        if not oracle_vars:
            continue

        avg_oracle_var = np.mean(oracle_vars)

        # Get total SE (average across same policies)
        total_ses = [exp.get('standard_errors', {}).get(p, 0) for p in policies if p in exp.get('standard_errors', {})]
        if not total_ses:
            continue
        avg_total_se = np.mean(total_ses)

        # Total variance = base + oracle, so base = total - oracle
        total_var = avg_total_se ** 2
        base_var = max(0, total_var - avg_oracle_var)

        if oracle_cov_pct not in coverage_data:
            coverage_data[oracle_cov_pct] = {'base_vars': [], 'oracle_vars': [], 'total_ses': []}

        coverage_data[oracle_cov_pct]['base_vars'].append(base_var)
        coverage_data[oracle_cov_pct]['oracle_vars'].append(avg_oracle_var)
        coverage_data[oracle_cov_pct]['total_ses'].append(avg_total_se)

    # Average across seeds
    result = {}
    for cov, data in coverage_data.items():
        result[cov] = {
            'base_se': np.sqrt(np.mean(data['base_vars'])),
            'oracle_se': np.sqrt(np.mean(data['oracle_vars'])),
            'total_se': np.mean(data['total_ses']),
        }

    return result


def plot_uncertainty_decomposition(variance_data: dict, output_path: Path):
    """Create stacked area plot of uncertainty decomposition."""
    # Sort by coverage
    coverages = sorted(variance_data.keys())

    base_se = [variance_data[c]['base_se'] for c in coverages]
    oracle_se = [variance_data[c]['oracle_se'] for c in coverages]
    total_se = [variance_data[c]['total_se'] for c in coverages]

    # For stacked area, we need cumulative values
    # base_se is the bottom, oracle_se stacks on top
    base_se = np.array(base_se)
    oracle_se = np.array(oracle_se)

    # Compute the stacked values (base variance + oracle variance should sum to total variance)
    # We'll plot base_se first, then add oracle contribution on top
    base_var = base_se ** 2
    oracle_var = oracle_se ** 2

    # Convert back to SE for stacking visualization
    # The "top" of base should be at base_se level
    # The "top" of oracle should be at sqrt(base_var + oracle_var) level
    cumulative_top = np.sqrt(base_var + oracle_var)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot stacked areas
    ax.fill_between(coverages, 0, base_se,
                    color='#FFD700', alpha=0.8, label='Base uncertainty (sampling)')
    ax.fill_between(coverages, base_se, cumulative_top,
                    color='#FF6B6B', alpha=0.8, label='Calibration uncertainty (oracle learning)')

    # Plot total as line with markers
    ax.plot(coverages, total_se, 'ko-', linewidth=2, markersize=8, label='Total uncertainty')

    ax.set_xlabel('Oracle Coverage (%)', fontsize=12)
    ax.set_ylabel('Uncertainty (Standard Error)', fontsize=12)
    ax.set_title('Uncertainty Decomposition: Sampling vs. Oracle Learning (DM+cov, n=1000)', fontsize=14)

    ax.set_xticks(coverages)
    ax.legend(loc='upper right', fontsize=10)

    ax.set_ylim(bottom=0)
    ax.grid(False)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved figure to {output_path}")

    plt.close(fig)
    return fig


def main():
    parser = argparse.ArgumentParser(description="Generate OUA variance decomposition figure")
    parser.add_argument('--results', type=Path,
                        default=Path(__file__).parent.parent / 'ablations/results/all_experiments.jsonl',
                        help='Path to experiment results JSONL')
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).parent.parent.parent / 'cje-paper/arxiv_submission/figs/oua_vs_base_variance.png',
                        help='Output path for figure')
    parser.add_argument('--estimator', type=str, default='direct',
                        help='Estimator to analyze')
    parser.add_argument('--sample-size', type=int, default=1000,
                        help='Sample size to analyze')
    args = parser.parse_args()

    print(f"Loading experiments from {args.results}")
    experiments = load_experiments(args.results)
    print(f"Loaded {len(experiments)} experiments")

    variance_data = extract_variance_data(experiments, args.estimator, args.sample_size)
    print(f"Extracted variance data for coverages: {sorted(variance_data.keys())}")

    for cov in sorted(variance_data.keys()):
        d = variance_data[cov]
        print(f"  {cov}%: base_se={d['base_se']:.4f}, oracle_se={d['oracle_se']:.4f}, total_se={d['total_se']:.4f}")

    plot_uncertainty_decomposition(variance_data, args.output)


if __name__ == '__main__':
    main()
