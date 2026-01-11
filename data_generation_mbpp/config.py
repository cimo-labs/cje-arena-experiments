"""
Configuration for MBPP data generation.

Defines policies (model + system prompt combinations) and judge prompts
for the MBPP coding benchmark ablation study.
"""

from pathlib import Path
from typing import Dict, Any

# Base directory
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data_mbpp"
RAW_DIR = DATA_DIR / "raw"
PROMPTS_DIR = DATA_DIR / "prompts"
RESPONSES_DIR = DATA_DIR / "responses"

# Policy definitions
# Each policy is a (model, system_prompt) combination
POLICIES: Dict[str, Dict[str, Any]] = {
    "base": {
        "model": "openai/gpt-4o",
        "system_prompt": """You are a Python coding assistant. Write clean, correct Python code to solve the given problem.

Instructions:
- Implement only the requested function
- Do not include test cases or example usage
- Do not include explanations or comments
- Return only the Python code""",
        "temperature": 0.0,
        "description": "Standard GPT-4o with basic coding prompt (logging policy)",
    },
    "enhanced": {
        "model": "openai/gpt-4o",
        "system_prompt": """You are an expert Python programmer with extensive experience in algorithm design and code correctness.

For each problem:
1. Carefully analyze the requirements and identify edge cases
2. Choose an appropriate algorithm with good time/space complexity
3. Implement clean, readable Python code
4. Ensure your solution handles all edge cases correctly

Instructions:
- Implement only the requested function
- Do not include test cases or example usage
- Do not include explanations
- Return only the Python code""",
        "temperature": 0.0,
        "description": "GPT-4o with enhanced step-by-step reasoning prompt",
    },
    "mini": {
        "model": "openai/gpt-4o-mini",
        "system_prompt": """You are a Python coding assistant. Write clean, correct Python code to solve the given problem.

Instructions:
- Implement only the requested function
- Do not include test cases or example usage
- Do not include explanations or comments
- Return only the Python code""",
        "temperature": 0.0,
        "description": "Smaller GPT-4o-mini with basic coding prompt",
    },
    "terse": {
        "model": "openai/gpt-4o",
        "system_prompt": """Write minimal Python code. No comments. No explanations. No docstrings.
Just the bare function implementation. Be concise.""",
        "temperature": 0.0,
        "description": "GPT-4o with terse/minimal output prompt",
    },
    "unhelpful": {
        "model": "openai/gpt-4o",
        "system_prompt": """You are a confused assistant that often misunderstands programming problems.
You tend to:
- Misinterpret the problem requirements
- Use incorrect variable names
- Make off-by-one errors
- Forget to handle edge cases
- Sometimes return placeholder code that looks correct but isn't

Try to write code that looks reasonable at first glance but contains subtle bugs.""",
        "temperature": 0.7,  # Higher temperature for more variation
        "description": "GPT-4o with deliberately unhelpful/confusing prompt (stress test)",
    },
}

# Judge configuration
JUDGE_MODEL = "openai/gpt-4o-mini"  # Fast and cheap for judge scoring
JUDGE_TEMPERATURE = 0.0  # Deterministic scoring

JUDGE_SYSTEM_PROMPT = """You are an expert code reviewer evaluating Python solutions for correctness.
Your task is to predict the probability (0-100) that the given code will pass all unit tests.

Evaluation criteria:
1. Does the function signature match what the problem requires?
2. Does the algorithm logic appear correct?
3. Are edge cases handled properly (empty inputs, boundary values)?
4. Are there syntax errors or typos?
5. Does the code actually implement the required behavior?

Scoring guidelines:
- 0-20: Code is clearly wrong, has syntax errors, or completely misses the point
- 21-40: Code has significant logical errors or missing functionality
- 41-60: Code has minor issues or may fail some edge cases
- 61-80: Code looks mostly correct but may have subtle issues
- 81-100: Code appears correct and should pass all tests

Output only a single integer from 0 to 100."""

JUDGE_USER_TEMPLATE = """<problem>
{prompt}
</problem>

<code>
{response}
</code>

Predict the probability (0-100) this code passes all unit tests:"""

# Inspect configuration
INSPECT_CONFIG = {
    "task": "inspect_evals/mbpp",
    "log_format": "json",
    "epochs": 1,  # Single generation per problem for main eval
    "max_connections": 10,
    "timeout": 120,  # 2 minute timeout per problem
}

# Fresh draws configuration (for Direct mode multiple samples per prompt)
FRESH_DRAWS_CONFIG = {
    "draws_per_prompt": 5,  # Number of samples per problem for fresh draws
    "temperature": 0.7,  # Higher temperature for variation in fresh draws
}

# Batch sizes for API calls
BATCH_SIZES = {
    "judge_scoring": 50,
    "response_generation": 10,
}
