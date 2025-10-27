#!/usr/bin/env python3
"""
Generate responses for Arena prompts using different policies.

Uses Fireworks API with different system prompts to simulate different policies:
- base/clone: Helpful assistant
- unhelpful: Deliberately confusing assistant

Includes retry logic with exponential backoff for handling API failures.
"""

import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import requests  # type: ignore

import sys

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))  # Add arena_10k_simplified to path

from experiment_config import get_all_policies, BATCH_SIZES


class ErrorType(Enum):
    """Categorize errors for retry logic."""

    RETRYABLE = "retryable"  # Network errors, rate limits, server errors
    NON_RETRYABLE = "non_retryable"  # Bad request, auth errors
    UNKNOWN = "unknown"


def classify_error(error: Exception) -> Tuple[ErrorType, str]:
    """Classify an error to determine if it's retryable.

    Returns:
        Tuple of (ErrorType, error_message)
    """
    error_str = str(error)

    # Check for HTTP status codes in requests.exceptions.HTTPError
    if isinstance(error, requests.exceptions.HTTPError):
        if hasattr(error.response, "status_code"):
            status = error.response.status_code
            # Retryable errors
            if status in [429, 500, 502, 503, 504, 530]:
                return ErrorType.RETRYABLE, f"HTTP {status}: {error_str}"
            # Non-retryable errors
            elif status in [400, 401, 403, 404]:
                return ErrorType.NON_RETRYABLE, f"HTTP {status}: {error_str}"

    # Connection errors are retryable
    if isinstance(
        error,
        (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ReadTimeout,
        ),
    ):
        return ErrorType.RETRYABLE, f"Connection error: {error_str}"

    # Rate limiting messages
    if any(
        msg in error_str.lower() for msg in ["rate limit", "too many requests", "quota"]
    ):
        return ErrorType.RETRYABLE, f"Rate limit: {error_str}"

    # Default to unknown (which we'll retry a few times)
    return ErrorType.UNKNOWN, error_str


def exponential_backoff_with_jitter(
    attempt: int, base_delay: float = 1.0, max_delay: float = 60.0
) -> float:
    """Calculate exponential backoff delay with jitter.

    Args:
        attempt: Current attempt number (0-indexed)
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds

    Returns:
        Delay in seconds
    """
    import random

    delay = min(base_delay * (2**attempt), max_delay)
    # Add jitter: ±25% of the delay
    jitter = delay * 0.25 * (2 * random.random() - 1)
    return float(max(0.1, delay + jitter))


def call_fireworks_with_retry(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Call Fireworks API with exponential backoff retry logic.

    Args:
        url: API endpoint
        headers: Request headers
        payload: Request payload
        max_retries: Maximum number of retry attempts
        base_delay: Base delay for exponential backoff
        max_delay: Maximum delay between retries
        verbose: Whether to print retry information

    Returns:
        Response data from successful API call

    Raises:
        Exception: If all retries are exhausted
    """
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            result: Dict[str, Any] = response.json()
            return result

        except Exception as e:
            last_error = e
            error_type, error_msg = classify_error(e)

            if attempt == max_retries:
                # No more retries
                if verbose:
                    print(
                        f"    ❌ Failed after {max_retries + 1} attempts: {error_msg}"
                    )
                raise

            if error_type == ErrorType.NON_RETRYABLE:
                # Don't retry non-retryable errors
                if verbose:
                    print(f"    ❌ Non-retryable error: {error_msg}")
                raise

            # Calculate delay for retryable and unknown errors
            delay = exponential_backoff_with_jitter(attempt, base_delay, max_delay)

            if verbose:
                print(
                    f"    ⚠️  Attempt {attempt + 1}/{max_retries + 1} failed: {error_msg}"
                )
                print(f"    ⏱️  Retrying in {delay:.1f} seconds...")

            time.sleep(delay)

    # This should never be reached, but just in case
    if last_error:
        raise last_error
    raise RuntimeError("Unexpected error in retry logic")


def load_existing_responses(output_file: str) -> Dict[str, Dict]:
    """Load existing responses from file.

    Returns:
        Dictionary mapping prompt_id to response data
    """
    existing = {}
    corrupted_lines = 0
    if Path(output_file).exists():
        with open(output_file, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                try:
                    data = json.loads(line)
                    if "prompt_id" in data:
                        existing[data["prompt_id"]] = data
                except json.JSONDecodeError as e:
                    corrupted_lines += 1
                    print(f"  ⚠️  Skipping corrupted line {line_num}: {e}")
                    continue  # Skip corrupted lines

    if corrupted_lines > 0:
        print(f"  ⚠️  Found {corrupted_lines} corrupted lines during resume")

    return existing


def generate_responses(
    prompts_file: str,
    output_file: str,
    model: str,
    temperature: float = 0.7,
    max_responses: Optional[int] = None,
    policy_name: str = "base",
    system_prompt: str = "You are a helpful assistant.",
    max_tokens: int = 1000,
    batch_size: Optional[int] = None,
    max_retries: int = 5,
    retry_delay: float = 1.0,
    max_retry_delay: float = 60.0,
    skip_failed: bool = False,
) -> List[Dict]:
    """Generate responses for prompts using Fireworks API with retry logic.

    Args:
        prompts_file: Path to JSONL file with prompts
        output_file: Where to save responses
        model: Fireworks model identifier
        temperature: Sampling temperature
        max_responses: Limit number of responses to generate
        policy_name: Name of the policy (for tracking)
        system_prompt: System prompt to use
        max_tokens: Maximum number of tokens to generate
        batch_size: Save progress every N responses (None to disable)
        max_retries: Maximum number of retry attempts for API calls
        retry_delay: Base delay for exponential backoff
        max_retry_delay: Maximum delay between retries
        skip_failed: If True, skip prompts that previously failed (don't retry them)

    Returns:
        List of response dictionaries
    """
    # Check for API key
    api_key = os.getenv("FIREWORKS_API_KEY")
    if not api_key:
        raise ValueError("FIREWORKS_API_KEY environment variable required")

    # Load existing responses if resuming (always check for existing work)
    existing_responses = load_existing_responses(output_file)

    # Track statistics
    stats = {
        "successful": 0,
        "failed": 0,
        "skipped": 0,
        "retried": 0,
    }

    # Load prompts
    prompts = []
    with open(prompts_file, "r") as f:
        for line in f:
            prompt_data = json.loads(line)
            prompt_id = prompt_data.get("prompt_id")

            # Skip if already exists
            if prompt_id in existing_responses:
                existing_resp = existing_responses[prompt_id]
                # Check if it was a failed response
                if existing_resp.get("response") is None and not skip_failed:
                    # Include failed responses for retry
                    prompts.append(prompt_data)
                else:
                    stats["skipped"] += 1
                continue
            prompts.append(prompt_data)

    if max_responses:
        # Adjust for existing successful responses
        successful_existing = sum(
            1 for r in existing_responses.values() if r.get("response") is not None
        )
        total_needed = max_responses - successful_existing
        prompts = prompts[:total_needed]

    if not prompts:
        print(
            f"✓ All {len(existing_responses)} responses already exist for {policy_name}"
        )
        return list(existing_responses.values())

    print(f"Generating {len(prompts)} responses with {policy_name} policy...")
    if existing_responses:
        failed_count = sum(
            1 for r in existing_responses.values() if r.get("response") is None
        )
        print(f"  📂 Existing: {len(existing_responses)} total ({failed_count} failed)")
        if not skip_failed and failed_count > 0:
            print(f"  🔄 Will retry {failed_count} failed responses")
    print(f"Model: {model}, Temperature: {temperature}, Max tokens: {max_tokens}")
    print(f"System prompt: {system_prompt[:50]}...")
    print(f"Retry config: max_retries={max_retries}, base_delay={retry_delay}s")
    if batch_size:
        print(f"Batch size: {batch_size} (saving progress incrementally)")

    # Fireworks API endpoint
    url = "https://api.fireworks.ai/inference/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Setup output file for appending if using batching
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # For batching, we'll write to temp file then rename atomically
    temp_file = None
    output_f = None
    if batch_size:
        # We'll rewrite the entire file with updated responses
        # Include PID to avoid collision with parallel runs
        temp_file = f"{output_file}.tmp.{os.getpid()}"
        # Don't copy existing file - we'll write all responses fresh
        output_f = open(temp_file, "w")

        # Write existing successful responses first
        for resp in existing_responses.values():
            if resp.get("response") is not None or skip_failed:
                output_f.write(json.dumps(resp) + "\n")

    # Generate responses
    results = []

    try:
        for i, prompt_data in enumerate(prompts):
            prompt_id = prompt_data["prompt_id"]
            prompt = prompt_data["prompt"]

            # Check if this is a retry
            is_retry = (
                prompt_id in existing_responses
                and existing_responses[prompt_id].get("response") is None
            )
            if is_retry:
                stats["retried"] += 1

            try:
                # Prepare messages with system prompt
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]

                # Call Fireworks API with retry logic
                payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }

                # Show which prompt we're processing
                if is_retry:
                    print(
                        f"  🔄 Retrying {prompt_id} (attempt {i + 1}/{len(prompts)})..."
                    )
                elif (i + 1) % 10 == 0:
                    print(f"  📝 Processing {i + 1}/{len(prompts)} ({prompt_id})...")

                response_data = call_fireworks_with_retry(
                    url,
                    headers,
                    payload,
                    max_retries=max_retries,
                    base_delay=retry_delay,
                    max_delay=max_retry_delay,
                    verbose=is_retry,  # Show retry details for retries
                )

                result = {
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "response": response_data["choices"][0]["message"]["content"],
                    "policy": policy_name,
                    "model": model,
                    "temperature": temperature,
                }

                stats["successful"] += 1

                if is_retry:
                    print(f"    ✅ Successfully regenerated {prompt_id}")

                # Save immediately if using batching
                if batch_size and output_f is not None:
                    output_f.write(json.dumps(result) + "\n")
                    output_f.flush()  # Ensure written to disk
                    # Show progress every batch_size responses
                    if stats["successful"] % batch_size == 0:
                        print(
                            f"  💾 Progress: {stats['successful']} successful, "
                            f"{stats['failed']} failed, {stats['retried']} retries"
                        )
                else:
                    results.append(result)

            except Exception as e:
                error_type, error_msg = classify_error(e)
                print(f"  ❌ Failed on {prompt_id}: {error_msg}")
                stats["failed"] += 1

                # Add failed result
                result = {
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "response": None,
                    "policy": policy_name,
                    "error": error_msg,
                    "error_type": error_type.value,
                }

                if batch_size and output_f is not None:
                    output_f.write(json.dumps(result) + "\n")
                    output_f.flush()
                else:
                    results.append(result)

    finally:
        # Always close file if using batching
        if batch_size and output_f:
            output_f.close()
            # Atomic rename from temp to final
            if temp_file and Path(temp_file).exists():
                os.replace(temp_file, output_file)
                print(f"\n✓ Results saved to {output_path}")
                print(
                    f"  📊 Final stats: {stats['successful']} successful, "
                    f"{stats['failed']} failed, {stats['retried']} retries attempted"
                )

    # Handle return for batch mode
    if batch_size:
        # Return all results including existing
        return list(load_existing_responses(output_file).values())
    else:
        # Save all results at once (original behavior)
        with open(output_path, "w") as f:
            # Include existing responses if any
            for resp in existing_responses.values():
                f.write(json.dumps(resp) + "\n")
            for result in results:
                f.write(json.dumps(result) + "\n")

        total_responses = len(existing_responses) + len(results)
        print(f"\n✓ Saved {total_responses} responses to {output_path}")
        print(f"  📊 Stats: {stats['successful']} successful, {stats['failed']} failed")
        return list(existing_responses.values()) + results


def main() -> None:
    """Generate responses for different policies."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompts", default="data/arena_prompts.jsonl", help="Input prompts file"
    )
    parser.add_argument(
        "--output-dir", default="data/responses", help="Output directory"
    )
    parser.add_argument("--max-responses", type=int, help="Limit number of responses")
    parser.add_argument(
        "--max-tokens", type=int, default=1000, help="Maximum tokens per response"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZES["response_generation"],
        help=f"Save progress every N responses (0 to disable, default: {BATCH_SIZES['response_generation']})",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum retry attempts for API calls",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=1.0,
        help="Base delay for exponential backoff (seconds)",
    )
    parser.add_argument(
        "--max-retry-delay",
        type=float,
        default=60.0,
        help="Maximum delay between retries (seconds)",
    )
    parser.add_argument(
        "--skip-failed",
        action="store_true",
        help="Skip previously failed responses instead of retrying them",
    )
    parser.add_argument(
        "--policies", nargs="+", help="Specific policies to generate (default: all)"
    )

    args = parser.parse_args()

    # Get policies from centralized configuration
    all_policies = get_all_policies()

    # Filter policies if specified
    if args.policies:
        policies = [p for p in all_policies if p["name"] in args.policies]
        if not policies:
            print(f"Error: No matching policies found for {args.policies}")
            return
    else:
        policies = all_policies

    for policy in policies:
        output_file = f"{args.output_dir}/{policy['name']}_responses.jsonl"
        generate_responses(
            prompts_file=args.prompts,
            output_file=output_file,
            model=policy["model"],
            temperature=policy["temperature"],
            policy_name=policy["name"],
            system_prompt=policy["system_prompt"],
            max_responses=args.max_responses,
            max_tokens=args.max_tokens,
            batch_size=args.batch_size if args.batch_size > 0 else None,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
            max_retry_delay=args.max_retry_delay,
            skip_failed=args.skip_failed,
        )


if __name__ == "__main__":
    main()
