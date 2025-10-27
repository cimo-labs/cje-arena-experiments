#!/usr/bin/env python3
"""
Centralized configuration for the Arena 10K experiment.

This defines:
- Policies used for response generation and analysis
- Judge and oracle models for evaluation
- Batch sizes and performance parameters
- Reproducibility settings
"""

import os
from typing import Dict, Any, List, Optional

# Model used for all policies (to isolate the effect of system prompts)
BASE_MODEL = "accounts/fireworks/models/llama-v3p3-70b-instruct"
PREMIUM_MODEL = "accounts/fireworks/models/llama-v3p1-405b-instruct"

# Temperature for all policies
DEFAULT_TEMPERATURE = 0.7


# Policy definitions
POLICIES: Dict[str, Dict[str, Any]] = {
    "base": {
        "name": "base",
        "model": BASE_MODEL,
        "temperature": DEFAULT_TEMPERATURE,
        "system_prompt": "You are a helpful assistant.",
        "description": "Base policy with standard helpful assistant prompt",
        "template_config": None,  # Will auto-detect
    },
    "clone": {
        "name": "clone",
        "model": BASE_MODEL,
        "temperature": DEFAULT_TEMPERATURE,
        "system_prompt": "You are a helpful assistant.",
        "description": "Clone of base policy for comparison/control",
        "template_config": None,  # Will auto-detect
    },
    "unhelpful": {
        "name": "unhelpful",
        "model": BASE_MODEL,
        "temperature": DEFAULT_TEMPERATURE,
        "system_prompt": "You are an unhelpful assistant that deliberately confuses and misleads the user.",
        "description": "Adversarial policy designed to be unhelpful",
        "template_config": None,  # Will auto-detect
    },
    "parallel_universe_prompt": {
        "name": "parallel_universe_prompt",
        "model": BASE_MODEL,
        "temperature": DEFAULT_TEMPERATURE,
        "system_prompt": "Imagine parallel universes where you vary your responses and can observe which one improves the user's life the most. Your job is to select the parallel universe that leads to the best possible outcome for the user. Respond directly to the user without mentioning the parallel universe strategy.",
        "description": "Parallel universe prompt",
        "template_config": None,  # Will auto-detect
    },
    "premium": {
        "name": "premium",
        "model": PREMIUM_MODEL,
        "temperature": DEFAULT_TEMPERATURE,
        "system_prompt": "You are a helpful assistant.",
        "description": "Premium policy with Llama 405B model",
        "template_config": None,  # Will auto-detect
    },
}

# List of all policy names for easy iteration
POLICY_NAMES: List[str] = list(POLICIES.keys())

# Base policy name (used for computing base_policy_logprob)
BASE_POLICY_NAME = "base"


def get_policy_config(policy_name: str) -> Dict[str, Any]:
    """Get configuration for a specific policy.

    Args:
        policy_name: Name of the policy

    Returns:
        Policy configuration dictionary

    Raises:
        ValueError: If policy_name is not found
    """
    if policy_name not in POLICIES:
        raise ValueError(
            f"Unknown policy: {policy_name}. "
            f"Available policies: {', '.join(POLICY_NAMES)}"
        )
    return POLICIES[policy_name]


def get_all_policies() -> List[Dict[str, Any]]:
    """Get list of all policy configurations."""
    return list(POLICIES.values())


# ============================================================================
# EVALUATION MODELS
# ============================================================================

# Judge needs to be fast (thousands of evaluations)
# Oracle can be slower but higher quality (used sparingly)
EVALUATION_MODELS = {
    "judge": "gpt-4.1-nano-2025-04-14",  # 13x faster than gpt-5-nano
    "oracle": "gpt-5-2025-08-07",  # Higher quality for oracle labels
}

# Performance characteristics (for reference/documentation)
MODEL_PERFORMANCE = {
    "gpt-5-nano-2025-08-07": {"avg_seconds": 6.5, "quality": "good"},
    "gpt-4.1-nano-2025-04-14": {"avg_seconds": 0.5, "quality": "good"},
    "gpt-5-2025-08-07": {"avg_seconds": 2.0, "quality": "excellent"},
}


# ============================================================================
# BATCH SIZES AND PERFORMANCE
# ============================================================================

BATCH_SIZES = {
    "response_generation": 20,  # Save every N responses
    "judge_scoring": 50,  # Score N samples per API call
    "oracle_scoring": 50,  # Score N samples per API call
    "logprob_computation": 20,  # Compute N logprobs at a time
}


# ============================================================================
# DEFAULT EXPERIMENT PARAMETERS
# ============================================================================

DEFAULT_EXPERIMENT_PARAMS = {
    "n_samples": 1000,
    "max_tokens": 512,
    "seed": 42,
}

# ============================================================================
# DATA QUALITY SETTINGS
# ============================================================================

DATA_QUALITY = {
    "content_moderation_threshold": 0.3,  # Filters ~0.28% most problematic prompts from Arena
}

# ============================================================================
# ANALYSIS SETTINGS
# ============================================================================

ANALYSIS_CONFIG = {
    "n_folds": 5,  # Number of cross-validation folds for DR estimators
    "extreme_threshold_high": 100.0,  # Upper threshold for extreme weight detection
    "extreme_threshold_low": 0.01,  # Lower threshold for extreme weight detection
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def validate_environment() -> bool:
    """Validate that the environment is set up correctly."""
    issues = []

    if not os.getenv("OPENAI_API_KEY"):
        issues.append("OPENAI_API_KEY not set (required for judge/oracle)")

    if not os.getenv("FIREWORKS_API_KEY"):
        issues.append("FIREWORKS_API_KEY not set (required for response generation)")

    if issues:
        print("❌ Environment validation failed:")
        for issue in issues:
            print(f"   - {issue}")
        print("\nPlease run: source /path/to/set_secrets.sh")
        return False

    return True


def print_experiment_config() -> None:
    """Print the current experiment configuration."""
    print("=" * 60)
    print("Arena 10K Experiment Configuration")
    print("=" * 60)

    print("\n📊 Evaluation Models:")
    for role, model in EVALUATION_MODELS.items():
        perf = MODEL_PERFORMANCE.get(model, {})
        speed = perf.get("avg_seconds", "?")
        quality = perf.get("quality", "?")
        print(f"  {role:10} → {model:30} (~{speed}s/call, {quality})")

    print("\n🤖 Response Generation Policies:")
    for name, config in POLICIES.items():
        model_name = config["model"].split("/")[-1]  # Just show model name
        print(f"  {name:25} → {model_name}")

    print("\n⚙️  Batch Sizes:")
    for task, size in BATCH_SIZES.items():
        print(f"  {task:20} → {size}")

    print("\n🔧 Default Parameters:")
    for param, value in DEFAULT_EXPERIMENT_PARAMS.items():
        print(f"  {param:15} → {value}")

    print("\n🔑 API Keys:")
    print(f"  OpenAI     → {'✓ Set' if os.getenv('OPENAI_API_KEY') else '✗ Not set'}")
    print(
        f"  Fireworks  → {'✓ Set' if os.getenv('FIREWORKS_API_KEY') else '✗ Not set'}"
    )

    print("\n" + "=" * 60)
