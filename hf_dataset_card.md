---
license: mit
task_categories:
- text-generation
language:
- en
tags:
- llm-evaluation
- causal-inference
- chatbot-arena
- off-policy-evaluation
size_categories:
- 1K<n<10K
---

# CJE Arena Data

Dataset for reproducing the Chatbot Arena experiments from the [CJE (Causal Judge Evaluation)](https://github.com/cimo-labs/cje) paper.

## Dataset Description

This dataset contains ~5,000 samples from the LMSYS Chatbot Arena, enriched with:
- **Log probabilities** for 5 different LLM policies
- **Judge scores** from GPT-4.1-nano (simulated cheap judge)
- **Oracle labels** from GPT-5 (simulated ground truth)
- **Fresh draws** for doubly-robust estimation

## Files

| File | Size | Description |
|------|------|-------------|
| `cje_dataset.jsonl` | ~10MB | Main dataset with all fields |
| `prompts.jsonl` | ~1MB | Original Arena prompts |
| `responses/*.jsonl` | ~35MB | Fresh draws per policy |
| `results/all_experiments.jsonl` | ~1.3GB | Pre-computed ablation results |

## Policies

| Policy | Model | Description |
|--------|-------|-------------|
| `base` | Llama-70B | Standard helpful assistant (logging policy) |
| `clone` | Llama-70B | Same as base (control) |
| `parallel_universe_prompt` | Llama-70B | Alternative system prompt |
| `premium` | Llama-405B | Larger model |
| `unhelpful` | Llama-70B | Deliberately unhelpful (stress test) |

## Usage

### With the experiment code

```bash
git clone https://github.com/cimo-labs/cje-arena-experiments.git
cd cje-arena-experiments
pip install -r requirements.txt
python download_data.py
python verify_setup.py
```

### Direct download

```python
from huggingface_hub import hf_hub_download

# Download main dataset
path = hf_hub_download(
    repo_id="cimo-labs/cje-arena-data",
    filename="cje_dataset.jsonl",
    repo_type="dataset"
)
```

## Data Format

Each sample in `cje_dataset.jsonl` contains:

```json
{
  "prompt_id": "arena_12345",
  "prompt": "User question from Chatbot Arena...",
  "response": "Model response...",
  "base_policy_logprob": -123.45,
  "target_policy_logprobs": {
    "clone": -125.67,
    "parallel_universe_prompt": -130.12,
    "premium": -118.90,
    "unhelpful": -145.23
  },
  "judge_score": 0.82,
  "oracle_label": 0.85,
  "metadata": {
    "response_length": 1234,
    "arena_model": "llama-70b"
  }
}
```

## Citation

```bibtex
@software{cje2024,
  title = {CJE: Causal Judge Evaluation},
  author = {Landesberg, Eddie},
  year = {2024},
  url = {https://github.com/cimo-labs/cje}
}
```

## License

MIT License - see the [CJE repository](https://github.com/cimo-labs/cje) for details.
