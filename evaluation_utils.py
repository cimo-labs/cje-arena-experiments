"""
Shared evaluation utilities for judges and oracles.

This module contains reusable implementations for scoring responses
using LLM-based judges and oracles with structured outputs.
"""

import os
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import numpy as np
from tqdm import tqdm

# For structured outputs with LangChain
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model

# Import model configuration from centralized config
try:
    # Try relative import for module usage
    from .experiment_config import EVALUATION_MODELS
except ImportError:
    # Fall back to direct import for script usage
    from experiment_config import EVALUATION_MODELS  # type: ignore

# Default models
DEFAULT_JUDGE_MODEL = EVALUATION_MODELS["judge"]
DEFAULT_ORACLE_MODEL = EVALUATION_MODELS["oracle"]


# Data models
@dataclass
class JudgeScore:
    """Result from scoring a single response."""

    score: float  # Score in [0, 1] (normalized from 0-100)
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class BatchJudgeResult:
    """Result from batch scoring."""

    scores: List[Optional[float]]
    metadata: Optional[List[Dict[str, Any]]] = None
    mean_score: float = 0.0
    std_score: float = 0.0

    def __post_init__(self) -> None:
        """Compute statistics."""
        # Filter out None values for statistics
        valid_scores = [s for s in self.scores if s is not None]
        if valid_scores:
            self.mean_score = float(np.mean(valid_scores))
            self.std_score = float(np.std(valid_scores))


# Pydantic model for structured outputs
class EvaluationResponse(BaseModel):
    """Response containing evaluation score for an AI response."""

    score: float = Field(
        description="Quality score from 0 to 100, where 0 is completely unhelpful and 100 is perfectly helpful",
        ge=0.0,
        le=100.0,
    )


# Evaluator class
class FireworksEvaluator:
    """LLM-based evaluator for scoring AI responses (supports Fireworks and OpenAI)."""

    def __init__(
        self,
        model: str,
        system_prompt: str = "You are an AI evaluator. Rate responses from 0 to 100. Always provide a score, even if the response is incomplete or truncated.",
        user_prompt_template: Optional[str] = None,
        temperature: float = 0.0,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.user_prompt_template = (
            user_prompt_template or self._default_prompt_template()
        )
        self.temperature = temperature

        # Determine provider based on model name
        if model.startswith("gpt") or model.startswith("o1") or model.startswith("o4"):
            # OpenAI model
            self.provider = "openai"
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY required for OpenAI models")
            os.environ["OPENAI_API_KEY"] = self.api_key
        else:
            # Fireworks model
            self.provider = "fireworks"
            self.api_key = api_key or os.getenv("FIREWORKS_API_KEY")
            if not self.api_key:
                raise ValueError("FIREWORKS_API_KEY required for Fireworks models")
            os.environ["FIREWORKS_API_KEY"] = self.api_key

        # Initialize LangChain model
        # Note: o4-mini and gpt-5 models only support temperature=1.0
        if model.startswith("o4") or model.startswith("gpt-5"):
            actual_temperature = 1.0
            if temperature != 1.0:
                print(
                    f"Note: {model} only supports temperature=1.0, ignoring temperature={temperature}"
                )
        else:
            actual_temperature = temperature

        self.llm = init_chat_model(
            model,
            model_provider=self.provider,
            temperature=actual_temperature,
        )

        # Create structured LLM
        self.structured_llm = self.llm.with_structured_output(EvaluationResponse)

    def _default_prompt_template(self) -> str:
        """Default prompt template."""
        return """<task>
Evaluate the quality of the AI assistant's response to the user's question.
Score from 0-100 based on relevance, helpfulness, clarity, and completeness.
If the response is truncated, score based on what is provided.
</task>

<exchange>
<question>{prompt}</question>
<answer>{response}</answer>
</exchange>

<instruction>
Provide your evaluation score (0-100):
</instruction>"""

    def score(self, prompt: str, response: str) -> JudgeScore:
        """Score a single response."""
        # Format user message from template
        user_message = self.user_prompt_template.format(
            prompt=prompt, response=response
        )

        # Try up to 3 times if structured output fails
        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                # Get structured response
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message},
                ]

                result = self.structured_llm.invoke(messages)

                # Check if result is None or invalid
                if result is None:
                    # Log what we sent for debugging
                    print(
                        f"DEBUG: Structured output returned None for prompt: {prompt[:50]}..."
                    )
                    print(f"DEBUG: Response: {response[:50]}...")
                    raise ValueError("Structured output returned None")

                if not isinstance(result, EvaluationResponse):
                    raise ValueError(f"Unexpected result type: {type(result)}")

                # Normalize score from 0-100 to 0-1
                normalized_score = result.score / 100.0

                return JudgeScore(
                    score=normalized_score,
                    metadata={
                        "judge_model": self.model,  # Store model name for reproducibility
                        "raw_score": result.score,  # Keep raw 0-100 score
                        "attempts": attempt + 1,
                    },
                )
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    # Wait briefly before retry
                    import time

                    time.sleep(0.5)
                    continue

        # All retries failed
        raise RuntimeError(
            f"Failed to score response with {self.model} after {max_retries} attempts: {str(last_error)}"
        ) from last_error

    def score_batch(
        self,
        prompts: List[str],
        responses: List[str],
        show_progress: bool = True,
        desc: str = "Scoring",
        skip_failures: bool = False,
    ) -> BatchJudgeResult:
        """Score a batch of responses.

        Args:
            prompts: List of prompts
            responses: List of responses
            show_progress: Whether to show progress bar
            desc: Description for progress bar
            skip_failures: If True, skip failed scorings and continue with the batch

        Returns:
            BatchJudgeResult with scores and metadata
        """
        if len(prompts) != len(responses):
            raise ValueError(f"Length mismatch: {len(prompts)} != {len(responses)}")

        scores: List[Optional[float]] = []
        metadata = []
        failed_indices = []

        if show_progress:
            iterator = tqdm(
                enumerate(zip(prompts, responses)), total=len(prompts), desc=desc
            )
        else:
            iterator = enumerate(zip(prompts, responses))

        for i, (prompt, response) in iterator:
            try:
                result = self.score(prompt, response)
                scores.append(result.score)
                if result.metadata:
                    metadata.append(result.metadata)
            except Exception as e:
                if skip_failures:
                    print(f"⚠️  Warning: Failed to score item {i}: {str(e)}")
                    print(f"   Prompt: {prompt[:50]}...")
                    print(f"   Response: {response[:50]}...")
                    scores.append(None)  # Placeholder for failed scoring
                    metadata.append({"error": str(e), "failed": True})
                    failed_indices.append(i)
                else:
                    # Re-raise if not skipping failures
                    raise

        if failed_indices and skip_failures:
            print(
                f"⚠️  Warning: {len(failed_indices)} out of {len(prompts)} scorings failed"
            )
            print(f"   Failed indices: {failed_indices}")

        return BatchJudgeResult(scores=scores, metadata=metadata if metadata else None)
