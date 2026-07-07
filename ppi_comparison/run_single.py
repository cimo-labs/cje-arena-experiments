"""Single-policy (base) coverage/width/RMSE experiment.

Grid: m in {10,25,50,100,250} x n in {500,2000} x {uniform,stratified} x R.
Default R=1000; --smoke -> R=25; --pilot-timing prints per-arm per-call
seconds and exits (no writes).

Appends JSONL rows to results/single_policy.jsonl:
  {regime, m, n, rep, seed, estimator, estimate, lo, hi, covered, width,
   err, lam_hat, inference_resolved, runtime_s, status, truth}

Resumable: on restart, (cell, rep) keys already holding a full arm set are
skipped; partial keys are re-run and deduped downstream (aggregate keeps the
last occurrence).

Run: .venv/bin/python run_single.py --smoke
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

import data_prep
import estimators
import sampling

logging.disable(logging.WARNING)

HERE = Path(__file__).resolve().parent
OUT_DEFAULT = HERE / "results" / "single_policy.jsonl"
POLICY = "base"

# Worker globals (loaded once per process via initializer).
_G = {}


def _init_worker():
    logging.disable(logging.WARNING)
    S, Y = data_prep.load_policy_arrays(POLICY)
    _G["S"] = S
    _G["Y"] = Y
    _G["truth"] = data_prep.load_truths()[POLICY]


def run_task(task):
    """One (regime, m, n, rep): all arms -> list of finished JSONL row dicts."""
    regime, m, n, rep = task
    S_corpus, Y_corpus, truth = _G["S"], _G["Y"], _G["truth"]

    idx = sampling.draw_indices(POLICY, n, rep)
    S_all, Y_all = S_corpus[idx], Y_corpus[idx]
    labeled, pi = sampling.labeled_for(regime, S_all, POLICY, n, rep, m)
    cje_seed = sampling.cje_seed(regime, POLICY, n, rep, m)
    seeds = {"cje": cje_seed}
    for arm in estimators.arm_names(regime):
        seeds[arm] = sampling.arm_seed(arm, regime, POLICY, n, rep, m)

    rows = estimators.evaluate_replicate(
        S_all, Y_all, labeled, regime, seeds,
        pi=pi if regime == "stratified" else None,
    )
    out = []
    for r in rows:
        est, lo, hi = r["est"], r["lo"], r["hi"]
        finite = all(np.isfinite([est, lo, hi]))
        extras = r.get("extras", {})
        out.append({
            "regime": regime, "m": m, "n": n, "rep": rep, "seed": cje_seed,
            "estimator": r["estimator"],
            "estimate": est if finite else None,
            "lo": lo if finite else None,
            "hi": hi if finite else None,
            "covered": bool(lo <= truth <= hi) if finite else None,
            "width": (hi - lo) if finite else None,
            "err": (est - truth) if finite else None,
            "lam_hat": extras.get("lam_hat"),
            "inference_resolved": extras.get("inference_resolved"),
            "runtime_s": round(r["runtime_s"], 4),
            "status": r["status"],
            "truth": truth,
        })
    return out


def all_tasks(R: int):
    for regime in sampling.REGIMES:
        for n in sampling.N_GRID:
            for m in sampling.M_GRID:
                for rep in range(R):
                    yield (regime, m, n, rep)


def completed_keys(out_path: Path):
    """Keys with a full arm set already on disk."""
    if not out_path.exists():
        return set()
    counts = Counter()
    with open(out_path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn tail line from an interrupted run
            counts[(r["regime"], r["m"], r["n"], r["rep"], r["estimator"])] += 1
    per_key = Counter()
    for (regime, m, n, rep, _est) in counts:
        per_key[(regime, m, n, rep)] += 1
    done = {
        k for k, c in per_key.items()
        if c >= len(estimators.arm_names(k[0]))
    }
    return done


def pilot_timing():
    """Per-arm per-call seconds (3 reps each) — decides R=1000 feasibility."""
    _init_worker()
    cells = [("uniform", 25, 500), ("uniform", 250, 2000), ("uniform", 10, 2000)]
    print("pilot timing (3 reps/cell, uniform regime):")
    grand = {}
    for regime, m, n in cells:
        acc = {}
        for rep in range(3):
            for r in (rows := run_task((regime, m, n, rep))):
                acc.setdefault(r["estimator"], []).append(r["runtime_s"])
        print(f"\n  cell m={m} n={n}:")
        for arm, ts in acc.items():
            print(f"    {arm:24s} {np.mean(ts):8.3f} s/call")
            grand.setdefault((n, arm), []).extend(ts)
    print("\n  cje arm summary: "
          + ", ".join(f"n={n}: {np.mean(ts):.2f}s"
                      for (n, arm), ts in sorted(grand.items())
                      if arm == "cje_calibrated"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=1000)
    ap.add_argument("--smoke", action="store_true", help="R=25")
    ap.add_argument("--pilot-timing", action="store_true")
    ap.add_argument("--processes", type=int, default=8)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()

    data_prep.load_corpus()  # ensure corpus + manifest exist

    if args.pilot_timing:
        pilot_timing()
        return

    R = 25 if args.smoke else args.R
    args.out.parent.mkdir(exist_ok=True)

    done = completed_keys(args.out)
    tasks = [t for t in all_tasks(R) if t not in done]
    total = len(list(all_tasks(R)))
    print(f"run_single: R={R}, {total} (cell,rep) keys, "
          f"{len(done)} already complete, {len(tasks)} to run, "
          f"{args.processes} processes -> {args.out}")
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
            if (i + 1) % 50 == 0 or (i + 1) == len(tasks):
                el = time.time() - t0
                eta = el / (i + 1) * (len(tasks) - i - 1)
                print(f"  {i+1}/{len(tasks)} tasks  "
                      f"({el/60:.1f} min elapsed, ~{eta/60:.1f} min left)")
    print(f"done: wrote {n_rows} rows in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    main()
