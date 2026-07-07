"""Per-replicate sampling: draws, labeled subsets, inclusion probabilities, seeds.

Conventions (pre-registered; see protocol.md):
  * SUPERPOPULATION: each replicate draws n row indices iid WITH replacement
    from the 5,000-row per-policy corpus. Truth = exact corpus mean.
  * SHARED DRAW: the n-draw depends only on ("draw", policy, n, rep) — the same
    drawn rows are reused across both regimes and all m.
  * NESTED LABELS: labeled subsets are nested across m — the labeled set for m
    is the first-m prefix of a per-(regime, n, rep) permutation (per-stratum
    prefixes in the stratified regime).
  * STRATIFIED REGIME: 4 judge-score-quartile strata WITHIN the n drawn rows
    (positional quartiles of the stable argsort — n is divisible by 4 so each
    stratum has exactly n/4 rows). m is allocated as m//4 per stratum with the
    remainder going to the lowest-index strata; this allocation is
    nondecreasing in m per stratum, so nesting holds. Exact inclusion
    probabilities pi_i = m_s / n_s are recorded for every row (for weights).
  * SEEDS: numpy SeedSequence keyed on (BASE_SEED, *hashed keys). cje calls get
    a per-(cell, rep) child integer seed — never the library default 42.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Tuple

import numpy as np

BASE_SEED = 20260706  # pre-registered root seed
M_GRID = [10, 25, 50, 100, 250]
N_GRID = [500, 2000]
REGIMES = ["uniform", "stratified"]
N_STRATA = 4


def _key_to_int(key) -> int:
    """Stable 64-bit int from any key (str/int/...)."""
    if isinstance(key, (int, np.integer)):
        return int(key)
    h = hashlib.sha256(repr(key).encode()).digest()
    return int.from_bytes(h[:8], "big")


def seed_seq(*keys) -> np.random.SeedSequence:
    """Deterministic SeedSequence for a tuple of keys."""
    return np.random.SeedSequence([BASE_SEED] + [_key_to_int(k) for k in keys])


def rng_for(*keys) -> np.random.Generator:
    return np.random.default_rng(seed_seq(*keys))


def seed_int_for(*keys) -> int:
    """Positive int seed (for cje's `seed=` / `bootstrap_seed` params)."""
    return int(seed_seq(*keys).generate_state(1, dtype=np.uint32)[0])


# ---------------------------------------------------------------------------
# Single-policy experiment
# ---------------------------------------------------------------------------


def draw_indices(policy: str, n: int, rep: int, corpus_n: int = 5000) -> np.ndarray:
    """The shared n-draw for (policy, n, rep): iid WITH replacement."""
    return rng_for("draw", policy, n, rep).integers(0, corpus_n, size=n)


def stratum_allocation(m: int, n_strata: int = N_STRATA) -> List[int]:
    """m split across strata: m//4 each, remainder to the lowest-index strata.

    Nondecreasing in m per stratum for the pre-registered grid, so per-stratum
    prefix labeling stays nested across m.
    """
    base, rem = divmod(m, n_strata)
    return [base + (1 if s < rem else 0) for s in range(n_strata)]


def uniform_labeled(
    policy: str, n: int, rep: int, m: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Uniform regime: labeled = first-m of one per-(n, rep) permutation.

    Returns (labeled_positions, pi) where pi[i] = m/n for every position.
    """
    assert 0 < m < n
    perm = rng_for("perm", "uniform", policy, n, rep).permutation(n)
    labeled = np.sort(perm[:m])
    pi = np.full(n, m / n, dtype=np.float64)
    return labeled, pi


def stratified_labeled(
    judge_scores_drawn: np.ndarray, policy: str, n: int, rep: int, m: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified regime: judge-score-quartile strata, per-stratum prefixes.

    Args:
        judge_scores_drawn: judge scores of the n drawn rows (position order).
    Returns (labeled_positions, pi) with exact inclusion probs per position.
    """
    assert 0 < m < n
    assert n % N_STRATA == 0, "n must be divisible by 4 for exact quartiles"
    n_s = n // N_STRATA
    order = np.argsort(judge_scores_drawn, kind="stable")
    strata = [order[s * n_s : (s + 1) * n_s] for s in range(N_STRATA)]
    alloc = stratum_allocation(m)
    labeled_parts = []
    pi = np.empty(n, dtype=np.float64)
    for s in range(N_STRATA):
        m_s = alloc[s]
        assert m_s <= n_s, f"stratum {s}: m_s={m_s} > n_s={n_s}"
        perm_s = rng_for("perm", "stratified", policy, n, rep, s).permutation(n_s)
        labeled_parts.append(strata[s][perm_s[:m_s]])
        pi[strata[s]] = m_s / n_s
    labeled = np.sort(np.concatenate(labeled_parts))
    assert len(labeled) == m
    assert abs(pi.sum() - m) < 1e-9, "inclusion probabilities must sum to m"
    return labeled, pi


def labeled_for(
    regime: str,
    judge_scores_drawn: np.ndarray,
    policy: str,
    n: int,
    rep: int,
    m: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if regime == "uniform":
        return uniform_labeled(policy, n, rep, m)
    if regime == "stratified":
        return stratified_labeled(judge_scores_drawn, policy, n, rep, m)
    raise ValueError(f"unknown regime {regime!r}")


def cje_seed(regime: str, policy: str, n: int, rep: int, m: int) -> int:
    """Per-(cell, rep) child seed for cje calls."""
    return seed_int_for("cje", regime, policy, n, rep, m)


def arm_seed(arm: str, regime: str, policy: str, n: int, rep: int, m: int) -> int:
    """Per-(arm, cell, rep) child seed (e.g. postpi's global-RNG bootstrap)."""
    return seed_int_for("arm", arm, regime, policy, n, rep, m)


# ---------------------------------------------------------------------------
# Multi-policy experiment
# ---------------------------------------------------------------------------

MP_POLICIES = ["base", "clone", "parallel_universe_prompt", "unhelpful"]
MP_N_PER_POLICY = 1000
MP_B_GRID = [40, 100, 200]
MP_ALLOCS = ["even", "concentrated"]


def mp_draw_slots(rep: int, corpus_n: int = 5000) -> np.ndarray:
    """Shared prompt indices for one multipolicy replicate (iid WITH
    replacement); the SAME corpus rows index all K policies (paired slots)."""
    return rng_for("mp_draw", rep).integers(0, corpus_n, size=MP_N_PER_POLICY)


def mp_label_slots(rep: int, policy: str, b_policy: int) -> np.ndarray:
    """First-b prefix of a per-(rep, policy) slot permutation (nested in B)."""
    if b_policy == 0:
        return np.array([], dtype=int)
    perm = rng_for("mp_perm", rep, policy).permutation(MP_N_PER_POLICY)
    return np.sort(perm[:b_policy])


def mp_allocation(alloc: str, B: int) -> Dict[str, int]:
    """Labels per policy under an allocation scheme."""
    if alloc == "even":
        assert B % len(MP_POLICIES) == 0
        return {p: B // len(MP_POLICIES) for p in MP_POLICIES}
    if alloc == "concentrated":
        return {p: (B if p == "base" else 0) for p in MP_POLICIES}
    raise ValueError(f"unknown alloc {alloc!r}")


def mp_seed(alloc: str, B: int, rep: int) -> int:
    return seed_int_for("mp_cje", alloc, B, rep)
