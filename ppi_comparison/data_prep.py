"""Build corpus.parquet from the 5 fully-oracle-labeled response files.

Corpus: 5 policies x 5,000 rows, same prompts in the same order. Every row
carries judge_score and oracle_label in [0, 1]. The corpus IS the
superpopulation: per-policy truths are the exact corpus means.

Outputs (under results/):
  - corpus.parquet   (policy, prompt_idx, judge_score, oracle_label)
  - manifest.json    (source SHA256s, truths, row counts, pip freeze)

Run: .venv/bin/python data_prep.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data" / "responses"
RESULTS_DIR = HERE / "results"
CORPUS_PARQUET = RESULTS_DIR / "corpus.parquet"
MANIFEST_JSON = RESULTS_DIR / "manifest.json"

POLICIES = ["base", "clone", "parallel_universe_prompt", "premium", "unhelpful"]
N_PER_POLICY = 5000

# Pre-registered ~4-decimal truth checks (recomputed exactly below; these are
# tripwires against silent corpus drift, tolerance = rounding of a 4-dp value).
EXPECTED_TRUTHS_4DP = {
    "base": 0.7581,
    "clone": 0.7620,
    "parallel_universe_prompt": 0.7708,
    "premium": 0.7623,
    "unhelpful": 0.1426,
}
TRUTH_TOL = 6e-5  # |exact - 4dp-rounded| <= 5e-5 (+ float slack)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_policy(policy: str) -> pd.DataFrame:
    """Load one policy file. NOTE: source files share the same prompt_id SET
    but NOT the same row order — alignment happens by prompt_id in build()."""
    path = DATA_DIR / f"{policy}_responses.jsonl"
    rows = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            meta = rec["metadata"]
            rows.append(
                {
                    "policy": policy,
                    "prompt_id": rec["prompt_id"],
                    "judge_score": float(meta["judge_score"]),
                    "oracle_label": float(meta["oracle_label"]),
                }
            )
    return pd.DataFrame(rows)


def build(verbose: bool = True) -> pd.DataFrame:
    """Build + validate the corpus, write parquet + manifest. Returns corpus df."""
    RESULTS_DIR.mkdir(exist_ok=True)

    frames = {}
    shas = {}
    for policy in POLICIES:
        path = DATA_DIR / f"{policy}_responses.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing corpus file: {path}")
        shas[path.name] = _sha256(path)
        frames[policy] = _load_policy(policy)

    # --- Asserts: shape, completeness, ranges, prompt alignment ---
    # DEVIATION (verified 2026-07-06): source files share the same 5,000-id
    # prompt SET but NOT the same row order. Per-id prompt text is identical
    # across all 5 files, so rows are aligned by prompt_id: base's file order
    # is the canonical order and prompt_idx = position within it.
    base_prompt_ids = frames["base"]["prompt_id"].tolist()
    canon_pos = {pid: i for i, pid in enumerate(base_prompt_ids)}
    assert len(canon_pos) == N_PER_POLICY, "duplicate prompt_ids in base"
    for policy, df in frames.items():
        assert len(df) == N_PER_POLICY, f"{policy}: {len(df)} rows != {N_PER_POLICY}"
        assert df["prompt_id"].nunique() == N_PER_POLICY, f"{policy}: dup ids"
        assert (
            set(df["prompt_id"]) == set(base_prompt_ids)
        ), f"{policy}: prompt_id set differs from base"
        assert df["judge_score"].notna().all(), f"{policy}: NaN judge_score"
        assert df["oracle_label"].notna().all(), f"{policy}: NaN oracle_label"
        assert df["judge_score"].between(0, 1).all(), f"{policy}: judge_score outside [0,1]"
        assert df["oracle_label"].between(0, 1).all(), f"{policy}: oracle_label outside [0,1]"
        df["prompt_idx"] = df["prompt_id"].map(canon_pos)
        frames[policy] = df.sort_values("prompt_idx").reset_index(drop=True)
        assert (
            frames[policy]["prompt_id"].tolist() == base_prompt_ids
        ), f"{policy}: prompt alignment failed"

    # --- Truths: exact corpus means (superpopulation convention) ---
    truths = {}
    for policy, df in frames.items():
        mu = float(np.mean(df["oracle_label"].to_numpy()))
        truths[policy] = mu
        exp = EXPECTED_TRUTHS_4DP[policy]
        assert abs(mu - exp) < TRUTH_TOL, (
            f"{policy}: recomputed truth {mu:.6f} does not match "
            f"pre-registered 4dp value {exp} (tol {TRUTH_TOL})"
        )
        if verbose:
            print(f"  truth[{policy}] = {mu:.6f}  (expected ~{exp})  OK")

    judge_means = {
        policy: float(np.mean(df["judge_score"].to_numpy()))
        for policy, df in frames.items()
    }

    corpus = pd.concat(frames.values(), ignore_index=True)
    corpus = corpus[["policy", "prompt_idx", "judge_score", "oracle_label"]]
    corpus.to_parquet(CORPUS_PARQUET, index=False)

    # --- Manifest ---
    try:
        from importlib.metadata import distributions

        freeze = sorted(
            f"{d.metadata['Name']}=={d.version}"
            for d in distributions()
            if d.metadata["Name"]
        )
    except Exception:  # pragma: no cover
        freeze = []
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "sources_sha256": shas,
        "n_policies": len(POLICIES),
        "n_per_policy": N_PER_POLICY,
        "truths_oracle_mean": truths,
        "judge_score_means": judge_means,
        "expected_truths_4dp": EXPECTED_TRUTHS_4DP,
        "pip_freeze": freeze,
    }
    with open(MANIFEST_JSON, "w") as f:
        json.dump(manifest, f, indent=2)
    if verbose:
        print(f"  wrote {CORPUS_PARQUET}")
        print(f"  wrote {MANIFEST_JSON}")
    return corpus


def load_corpus() -> pd.DataFrame:
    """Load corpus.parquet (building it first if missing)."""
    if not CORPUS_PARQUET.exists():
        return build(verbose=False)
    return pd.read_parquet(CORPUS_PARQUET)


def load_policy_arrays(policy: str):
    """Return (judge_scores, oracle_labels) float64 arrays for one policy."""
    corpus = load_corpus()
    df = corpus[corpus["policy"] == policy].sort_values("prompt_idx")
    assert len(df) == N_PER_POLICY, f"{policy}: bad corpus slice"
    return (
        df["judge_score"].to_numpy(dtype=np.float64),
        df["oracle_label"].to_numpy(dtype=np.float64),
    )


def load_truths() -> dict:
    if not MANIFEST_JSON.exists():
        build(verbose=False)
    with open(MANIFEST_JSON) as f:
        return json.load(f)["truths_oracle_mean"]


if __name__ == "__main__":
    print("data_prep: building corpus.parquet + manifest.json")
    build(verbose=True)
    print("data_prep: OK")
