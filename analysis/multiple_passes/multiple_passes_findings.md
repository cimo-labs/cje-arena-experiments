# Multiple Teacher Forcing Passes: Key Findings

## Executive Summary

We collected 5 independent teacher forcing passes for each prompt-response pair in the Arena dataset to study API non-determinism and its impact on causal inference. The analysis reveals a fundamental insight: **99.9% of log probability variance occurs between prompts, with only 0.1% within prompts**. This finding validates the use of prompt-level blocking in bootstrap inference and shows that multiple API calls provide minimal statistical benefit.

## Data Collection

- **Dataset**: Arena 10K simplified (5,000 prompts)
- **Policies**: 5 policies (base, clone, parallel_universe, premium, unhelpful)  
- **Passes**: 5 independent teacher forcing API calls per prompt-response pair
- **Total API calls**: 125,000 log probability computations
- **Success rate**: 97-99% (multiple passes recover from transient API failures)

## Key Findings

### 1. Variance Decomposition

The variance in log probabilities decomposes almost entirely at the prompt level:

| Policy | Between-Prompt Variance | Within-Prompt Variance |
|--------|-------------------------|------------------------|
| Base | 99.95% | 0.05% |
| Clone | 99.96% | 0.04% |
| Parallel | 99.87% | 0.13% |
| Premium | 99.99% | 0.01% |
| Unhelpful | 99.96% | 0.04% |

**Interpretation**: The differences between prompts are 1,000-50,000× larger than the API's inherent randomness.

### 2. API Non-Determinism Characteristics

Within-prompt variation analysis reveals:
- **Coefficient of Variation (CV)**: 1-5% within prompts
- **Correlation between passes**: r > 0.999
- **Pattern**: Premium policy most stable (CV < 1%), Parallel most variable (CV ≈ 5%)

This variation is real but negligible for statistical inference.

### 3. Correlation Structure

The intraclass correlation coefficient (ICC) quantifies how much variance is attributable to prompt-level effects:

- **ICC ≈ 1.000** for all policies
- Between-pass correlation within prompts: **r > 0.999**
- Implication: Multiple passes are nearly redundant measurements

### 4. Impact on Importance Sampling

We compared different aggregation strategies for importance weights:

| Strategy | Description | ESS Impact |
|----------|-------------|------------|
| Single Pass | Use first API call | Baseline |
| Mean of 5 | Average all passes | ≈0% change |
| Median of 5 | Robust average | ≈0% change |
| Best of 5 | Optimistic selection | Slightly worse |

**Key insight**: Averaging multiple passes provides no meaningful variance reduction because passes are highly correlated (r > 0.999).

### 5. Failure Recovery

Multiple passes do provide value for robustness:
- **2-3% of API calls fail intermittently**
- **Recovery rate**: 97-99% success with multiple attempts
- **Pattern**: Failures are transient, not systematic

## Implications for Causal Inference

### 1. Validates Block Bootstrap Approach

With 99.9% of variance between prompts, treating each prompt as an independent block is not just convenient but essential. The block bootstrap correctly:
- Resamples at the prompt level
- Ignores within-prompt variation (negligible)
- Captures the true variance structure

### 2. Single Pass is Sufficient

For statistical efficiency:
- Multiple passes don't reduce variance (high correlation)
- API randomness is negligible compared to prompt heterogeneity
- Use multiple passes only for robustness against failures

### 3. Confidence Interval Construction

The finding supports:
- **Block-level influence functions**: Aggregate to prompt level first
- **Prompt-based resampling**: Natural unit for bootstrap
- **No need for within-prompt modeling**: Variance is between prompts

### 4. Quality Control Insights

High within-prompt variance could indicate:
- Ambiguous or problematic responses
- API instability for specific content
- Potential data quality issues

Prompts with CV > 10% warrant investigation.

## Practical Recommendations

### For CJE Implementation

1. **Default to single pass** for efficiency
2. **Use 2-3 passes** only for failure recovery
3. **Implement prompt-level blocking** in all inference procedures
4. **Track within-prompt CV** as a quality metric

### For Production Systems

1. **Retry logic**: 2-3 attempts handle transient failures
2. **Aggregation**: Simple mean if multiple passes available
3. **Monitoring**: Flag prompts with high within-prompt variance
4. **Caching**: Store all passes for diagnostic purposes

## Statistical Details

### Variance Components Model

For prompt $i$ and pass $j$:
$$\log p_{ij} = \mu + \alpha_i + \epsilon_{ij}$$

Where:
- $\alpha_i \sim N(0, \sigma^2_{\text{between}})$ (prompt effect)
- $\epsilon_{ij} \sim N(0, \sigma^2_{\text{within}})$ (API randomness)
- $\sigma^2_{\text{between}} / \sigma^2_{\text{total}} \approx 0.999$

### Effective Sample Size Analysis

For importance weights $w_i = \exp(\log p^{\text{target}}_i - \log p^{\text{base}}_i)$:
- Single pass: ESS determined by prompt heterogeneity
- Averaged passes: ESS essentially unchanged (correlation ≈ 1)
- Implication: API averaging doesn't improve overlap

## Visualization

The accompanying figure shows:
- **Panel A**: Variance decomposition across policies (>99.9% between prompts)
- **Panel B**: Example of 5 passes for selected prompts, illustrating tight within-prompt clustering

## Conclusions

This analysis provides empirical validation for key CJE design decisions:

1. **Prompt-level blocking is fundamental** - The variance structure demands it
2. **API non-determinism is negligible** - Real but irrelevant for inference  
3. **Multiple passes offer robustness, not efficiency** - Use for error recovery
4. **The block bootstrap is correctly specified** - Matches true variance structure

These findings strengthen confidence in CJE's statistical methodology and provide guidance for production deployments where API calls are costly.

## Code and Data Availability

- Analysis code: `analyze_passes_deep.py`, `passes_simple_viz.py`
- Data location: `data/logprobs/*_pass*.jsonl`
- Visualization: `passes_variance_simple.pdf`

The multiple passes data enables future research on API stability, response ambiguity detection, and robust aggregation methods.