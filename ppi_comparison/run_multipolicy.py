"""Multi-policy comparison at a fixed TOTAL oracle-label budget B.

K=4 policies (base, clone, parallel_universe_prompt, unhelpful),
n=1000/policy with the SAME prompt slots per replicate (paired draws),
B in {40, 100, 200} total labels, R default 500 (--smoke -> 10).

Arms per (alloc in {even, concentrated}, B, rep):
  * ppi_vanilla / ppi_pp: per-policy ppi_mean_ci (lam=1 / lam=None) on that
    policy's own labeled slots. A zero-label policy (concentrated alloc)
    -> status "NA_requires_labels".
  * cje_pooled: cje.analyze_dataset(fresh_draws_data=...) with
    oracle_label=None masking; calibration pools ALL labels across policies.
    (Chosen over a manual pooled path — see protocol.md.)

Pairwise difference CIs: PPI arms combine independently
(diff +- z*sqrt(se_i^2+se_j^2)); the CJE arm uses
EstimationResult.compare_policies (paired, influence-function based).

Transport audit (H4 diagnostic): for the concentrated alloc, on the first 25
replicates per (B) cell, cje.transport_audit probes each non-base policy's
drawn slots with their held-out true oracle labels; PASS/WARN/FAIL recorded.

Appends JSONL rows to results/multipolicy.jsonl. Resumable by (alloc, B, rep).

Run: .venv/bin/python run_multipolicy.py --smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from multiprocessing import Pool, set_start_method
from pathlib import Path

import numpy as np
from scipy import stats

import data_prep
import sampling
from estimators import ALPHA, _f, lam_hat_ppi

logging.disable(logging.WARNING)

HERE = Path(__file__).resolve().parent
OUT_DEFAULT = HERE / "results" / "multipolicy.jsonl"
POLICIES = sampling.MP_POLICIES
N_TRANSPORT_REPS = 25

_G = {}


def _init_worker():
    logging.disable(logging.WARNING)
    corpus = {p: data_prep.load_policy_arrays(p) for p in POLICIES}
    _G["corpus"] = corpus
    _G["truths"] = data_prep.load_truths()


def _row(alloc, B, rep, seed, arm, row_type, name, est, lo, hi, truth,
         status, runtime_s, extras=None):
    finite = est is not None and np.all(np.isfinite([est, lo, hi]))
    return {
        "alloc": alloc, "B": B, "rep": rep, "seed": seed, "arm": arm,
        "row_type": row_type, "name": name,
        "estimate": est if finite else None,
        "lo": lo if finite else None, "hi": hi if finite else None,
        "covered": (bool(lo <= truth <= hi) if (finite and truth is not None)
                    else None),
        "width": (hi - lo) if finite else None,
        "err": (est - truth) if (finite and truth is not None) else None,
        "truth": truth, "status": status,
        "runtime_s": round(runtime_s, 4), "extras": extras or {},
    }


def _ppi_policy_arms(alloc, B, rep, seed, slots, labels_per_policy, truths):
    """Per-policy PPI arms + independent-combination pairwise rows."""
    from ppi_py import ppi_mean_ci, ppi_mean_pointestimate

    z = stats.norm.ppf(1 - ALPHA / 2)
    rows = []
    per_policy = {}  # (arm, policy) -> (est, se) or None
    for arm, lam in (("ppi_vanilla", 1), ("ppi_pp", None)):
        for pol in POLICIES:
            t0 = time.time()
            b = labels_per_policy[pol]
            S, Y = _G["corpus"][pol]
            S_all, Y_all = S[slots], Y[slots]
            truth = truths[pol]
            if b == 0:
                rows.append(_row(alloc, B, rep, seed, arm, "policy", pol,
                                 None, None, None, truth,
                                 "NA_requires_labels", time.time() - t0))
                per_policy[(arm, pol)] = None
                continue
            lab = sampling.mp_label_slots(rep, pol, b)
            mask = np.zeros(len(slots), dtype=bool)
            mask[lab] = True
            Y_lab, S_lab, S_unl = Y_all[mask], S_all[mask], S_all[~mask]
            try:
                est = _f(ppi_mean_pointestimate(Y_lab, S_lab, S_unl, lam=lam))
                lo, hi = ppi_mean_ci(Y_lab, S_lab, S_unl, alpha=ALPHA, lam=lam)
                lo, hi = _f(lo), _f(hi)
                extras = {}
                if lam is None:
                    extras["lam_hat"] = lam_hat_ppi(Y_lab, S_lab, S_unl)
                rows.append(_row(alloc, B, rep, seed, arm, "policy", pol,
                                 est, lo, hi, truth, "ok",
                                 time.time() - t0, extras))
                per_policy[(arm, pol)] = (est, (hi - lo) / 2.0 / z)
            except Exception as e:
                rows.append(_row(alloc, B, rep, seed, arm, "policy", pol,
                                 None, None, None, truth,
                                 f"error:{type(e).__name__}:{e}",
                                 time.time() - t0))
                per_policy[(arm, pol)] = None
        # Pairwise (independent combination).
        for i in range(len(POLICIES)):
            for j in range(i + 1, len(POLICIES)):
                pi_, pj = POLICIES[i], POLICIES[j]
                pair = f"{pi_}-{pj}"
                truth_d = truths[pi_] - truths[pj]
                a, b_ = per_policy[(arm, pi_)], per_policy[(arm, pj)]
                if a is None or b_ is None:
                    rows.append(_row(alloc, B, rep, seed, arm, "pair", pair,
                                     None, None, None, truth_d,
                                     "NA_requires_labels", 0.0))
                    continue
                d = a[0] - b_[0]
                se = float(np.hypot(a[1], b_[1]))
                rows.append(_row(alloc, B, rep, seed, arm, "pair", pair,
                                 d, d - z * se, d + z * se, truth_d, "ok",
                                 0.0, {"combination": "independent"}))
    return rows


def _cje_pooled_arm(alloc, B, rep, seed, slots, labels_per_policy, truths,
                    do_transport):
    """Pooled-calibration CJE via analyze_dataset + paired comparisons."""
    from cje import analyze_dataset, transport_audit

    rows = []
    arm = "cje_pooled"
    t0 = time.time()
    data = {}
    for pol in POLICIES:
        S, Y = _G["corpus"][pol]
        S_all, Y_all = S[slots], Y[slots]
        lab = sampling.mp_label_slots(rep, pol, labels_per_policy[pol])
        labset = set(lab.tolist())
        data[pol] = [
            {
                "prompt_id": f"slot_{j}",
                "judge_score": float(S_all[j]),
                "oracle_label": (float(Y_all[j]) if j in labset else None),
            }
            for j in range(len(slots))
        ]
    try:
        res = analyze_dataset(
            fresh_draws_data=data,
            estimator_config={"bootstrap_seed": int(seed)},
            verbose=False,
        )
        runtime = time.time() - t0
        pols = list(res.target_policies)
        cis = res.ci(alpha=ALPHA)
        for pol in POLICIES:
            k = pols.index(pol)
            est = float(res.estimates[k])
            lo, hi = float(cis[k][0]), float(cis[k][1])
            rows.append(_row(alloc, B, rep, seed, arm, "policy", pol,
                             est, lo, hi, truths[pol], "ok", runtime,
                             {"method": res.method}))
        z = stats.norm.ppf(1 - ALPHA / 2)
        for i in range(len(POLICIES)):
            for j in range(i + 1, len(POLICIES)):
                pi_, pj = POLICIES[i], POLICIES[j]
                cmp_ = res.compare_policies(pols.index(pi_), pols.index(pj),
                                            alpha=ALPHA)
                d = float(cmp_["difference"])
                se = float(cmp_["se_difference"])
                rows.append(_row(alloc, B, rep, seed, arm, "pair",
                                 f"{pi_}-{pj}", d, d - z * se, d + z * se,
                                 truths[pi_] - truths[pj], "ok", 0.0,
                                 {"combination": "paired_influence"}))
        if do_transport and alloc == "concentrated":
            for pol in POLICIES:
                if pol == "base":
                    continue
                t1 = time.time()
                S, Y = _G["corpus"][pol]
                try:
                    audit = transport_audit(S[slots], Y[slots], res.calibrator,
                                            group_label=f"policy:{pol}",
                                            alpha=ALPHA)
                    rows.append(_row(alloc, B, rep, seed, arm, "transport",
                                     pol, None, None, None, None,
                                     f"audit:{audit.status}",
                                     time.time() - t1,
                                     {"delta_hat": float(audit.delta_hat)}))
                except Exception as e:
                    rows.append(_row(alloc, B, rep, seed, arm, "transport",
                                     pol, None, None, None, None,
                                     f"error:{type(e).__name__}:{e}",
                                     time.time() - t1))
    except Exception as e:
        runtime = time.time() - t0
        for pol in POLICIES:
            rows.append(_row(alloc, B, rep, seed, arm, "policy", pol,
                             None, None, None, truths[pol],
                             f"error:{type(e).__name__}:{e}", runtime))
    return rows


def run_task(task):
    alloc, B, rep = task
    truths = _G["truths"]
    slots = sampling.mp_draw_slots(rep)
    labels_per_policy = sampling.mp_allocation(alloc, B)
    seed = sampling.mp_seed(alloc, B, rep)
    rows = _ppi_policy_arms(alloc, B, rep, seed, slots, labels_per_policy,
                            truths)
    rows += _cje_pooled_arm(alloc, B, rep, seed, slots, labels_per_policy,
                            truths, do_transport=(rep < N_TRANSPORT_REPS))
    return rows


def all_tasks(R):
    for alloc in sampling.MP_ALLOCS:
        for B in sampling.MP_B_GRID:
            for rep in range(R):
                yield (alloc, B, rep)


def completed_keys(out_path: Path):
    if not out_path.exists():
        return set()
    per_key = Counter()
    with open(out_path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r["row_type"] == "policy":
                per_key[(r["alloc"], r["B"], r["rep"])] += 1
    # 3 arms x 4 policy rows each = 12 policy rows per complete key.
    return {k for k, c in per_key.items() if c >= 12}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=500)
    ap.add_argument("--smoke", action="store_true", help="R=10")
    ap.add_argument("--processes", type=int, default=8)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()

    data_prep.load_corpus()
    R = 10 if args.smoke else args.R
    args.out.parent.mkdir(exist_ok=True)

    done = completed_keys(args.out)
    tasks = [t for t in all_tasks(R) if t not in done]
    print(f"run_multipolicy: R={R}, {len(done)} keys complete, "
          f"{len(tasks)} to run, {args.processes} processes -> {args.out}")
    if not tasks:
        print("nothing to do")
        return

    t0 = time.time()
    n_rows = 0
    with Pool(args.processes, initializer=_init_worker) as pool, \
            open(args.out, "a") as f:
        for i, rows in enumerate(pool.imap_unordered(run_task, tasks,
                                                     chunksize=1)):
            for r in rows:
                f.write(json.dumps(r) + "\n")
                n_rows += 1
            f.flush()
            if (i + 1) % 20 == 0 or (i + 1) == len(tasks):
                el = time.time() - t0
                eta = el / (i + 1) * (len(tasks) - i - 1)
                print(f"  {i+1}/{len(tasks)} tasks "
                      f"({el/60:.1f} min, ~{eta/60:.1f} min left)")
    print(f"done: wrote {n_rows} rows in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    main()
