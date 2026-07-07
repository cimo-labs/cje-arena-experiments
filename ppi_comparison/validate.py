"""Pre-run validation gate. ALL checks must be green before any run.

Checks (pre-registered in protocol.md):
  (a) corpus asserts + truth recompute matches the ~4-decimal constants
  (b) alpha-defaults tripwire: ppi_py defaults alpha=0.1 — confirm a call
      without alpha gives a DIFFERENT (narrower) width than alpha=0.05;
      plus lam_hat reproduction identity (recorded lam_hat reproduces the
      library's own lam=None interval exactly)
  (c) PostPI synthetic mean-recovery with X=ones (known-mean DGP) — decides
      inclusion of arm 6
  (d) fold-extraction identity: manual per-fold predictions from the
      extracted fold models == calibrator.predict_oof (< 1e-8), fold ids
      match the public recomputation, and selected_mode is monotone
  (e) 5-replicate end-to-end micro-run of EVERY arm at the hardest cell
      (m=10, n=500), both regimes: all intervals finite
  (f) sampling invariants: shared n-draw across regimes/m, labeled nesting
      across m, stratified pi sums to m

Run: .venv/bin/python validate.py
"""

from __future__ import annotations

import logging
import sys

import numpy as np

logging.disable(logging.WARNING)

import data_prep
import sampling
import estimators
from estimators import ALPHA

GREEN = "PASS"
FAILURES: list = []


def report(name: str, ok: bool, detail: str = "") -> None:
    tag = GREEN if ok else "FAIL"
    print(f"[{tag}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
def check_a_corpus():
    print("\n(a) corpus + truths")
    corpus = data_prep.build(verbose=False)
    truths = data_prep.load_truths()
    ok = len(corpus) == 25000 and set(truths) == set(data_prep.POLICIES)
    detail = ", ".join(f"{p}={truths[p]:.4f}" for p in data_prep.POLICIES)
    report("corpus 5x5000, full labels, aligned prompts, truths match 4dp", ok, detail)
    return truths


def check_b_alpha_tripwire():
    print("\n(b) alpha-defaults tripwire + lam_hat identity")
    from ppi_py import classical_mean_ci, ppi_mean_ci

    rng = np.random.default_rng(sampling.seed_seq("validate", "alpha"))
    Y = rng.uniform(size=200)
    S_lab = np.clip(Y + rng.normal(0, 0.1, 200), 0, 1)
    S_unl = rng.uniform(size=1800)

    lo_d, hi_d = classical_mean_ci(Y)                # default alpha=0.1 (!)
    lo_5, hi_5 = classical_mean_ci(Y, alpha=0.05)
    w_d, w_5 = float(hi_d - lo_d), float(hi_5 - lo_5)
    report(
        "classical_mean_ci default width != alpha=0.05 width (trap is real)",
        w_d < w_5 and abs(w_d - w_5) > 1e-6,
        f"default(0.1) width={w_d:.5f} < alpha=0.05 width={w_5:.5f}",
    )
    lo_pd, hi_pd = ppi_mean_ci(Y, S_lab, S_unl)      # default alpha=0.1 (!)
    lo_p5, hi_p5 = ppi_mean_ci(Y, S_lab, S_unl, alpha=0.05)
    wpd = float(np.squeeze(hi_pd - lo_pd))
    wp5 = float(np.squeeze(hi_p5 - lo_p5))
    report(
        "ppi_mean_ci default width != alpha=0.05 width",
        wpd < wp5 and abs(wpd - wp5) > 1e-6,
        f"default(0.1) width={wpd:.5f} < alpha=0.05 width={wp5:.5f}",
    )

    # lam_hat reproduction: our recorded lam_hat reproduces lam=None interval.
    lam_hat = estimators.lam_hat_ppi(Y, S_lab, S_unl)
    ci_none = ppi_mean_ci(Y, S_lab, S_unl, alpha=0.05, lam=None)
    ci_lam = ppi_mean_ci(Y, S_lab, S_unl, alpha=0.05, lam=lam_hat)
    diff = max(
        abs(float(np.squeeze(a)) - float(np.squeeze(b)))
        for a, b in zip(ci_none, ci_lam)
    )
    report(
        "recorded lam_hat reproduces ppi_mean_ci(lam=None) interval",
        diff < 1e-12,
        f"lam_hat={lam_hat:.4f}, max endpoint diff={diff:.2e}",
    )


def check_c_postpi():
    print("\n(c) PostPI synthetic mean-recovery (X=ones)")
    from ppi_py import postprediction_ols_ci

    rng = np.random.default_rng(sampling.seed_seq("validate", "postpi"))
    N, m = 5000, 200
    # Known-mean DGP: Y ~ mixture with mean exactly recoverable; Yhat ~= Y.
    Y_all = np.clip(rng.normal(0.6, 0.15, size=N + m), 0, 1)
    Yhat_all = np.clip(Y_all + rng.normal(0, 0.02, size=N + m), 0, 1)
    true_mean = float(np.mean(Y_all))  # finite-sample truth of the DGP pool
    Y, Yhat = Y_all[:m], Yhat_all[:m]
    Yhat_unl = Yhat_all[m:]
    np.random.seed(12345)
    lo, hi = postprediction_ols_ci(Y, Yhat, np.ones((N, 1)), Yhat_unl, alpha=0.05)
    center = float(np.squeeze(lo) + np.squeeze(hi)) / 2.0
    ok = np.isfinite(center) and abs(center - true_mean) < 0.01
    report(
        "postprediction_ols_ci recovers a known mean with X=ones",
        ok,
        f"center={center:.4f} vs true={true_mean:.4f} (tol 0.01)",
    )
    if not ok:
        estimators.POSTPI_INCLUDED = False
        print("  -> EXCLUDING arm 6 (postpi_ols_ones); record in protocol.md")
    return ok


def check_d_fold_extraction(truths):
    print("\n(d) CJE fold-extraction identity")
    S_corpus, Y_corpus = data_prep.load_policy_arrays("base")
    n, m, rep = 2000, 250, 0
    idx = sampling.draw_indices("base", n, rep)
    S_all, Y_all = S_corpus[idx], Y_corpus[idx]
    labeled, _ = sampling.uniform_labeled("base", n, rep, m)
    seed = sampling.cje_seed("uniform", "base", n, rep, m)
    result = estimators.fit_cje(S_all, Y_all, labeled, ALPHA, seed)
    cal = result.calibrator

    report(
        "selected_mode == monotone (fold models are plain isotonic)",
        cal.selected_mode == "monotone",
        f"selected_mode={cal.selected_mode}",
    )

    from cje.data.folds import get_folds_for_prompts

    fold_ids = np.asarray(cal._fold_ids)
    recomputed = get_folds_for_prompts(
        [f"row_{i}" for i in range(n)], cal._n_folds, cal.random_seed
    )
    report(
        "fold ids match public get_folds_for_prompts recomputation",
        np.array_equal(fold_ids, recomputed),
        f"n_folds={cal._n_folds}",
    )

    mask = np.zeros(n, dtype=bool)
    mask[labeled] = True
    fitted = cal.get_fold_models_for_oua()
    # Manual OOF from the raw extracted fold models (monotone: raw S input).
    manual = np.empty(mask.sum())
    lab_folds = fold_ids[mask]
    S_lab = S_all[mask]
    for k, model in fitted.items():
        sel = lab_folds == k
        if sel.any():
            manual[sel] = np.clip(model.predict(S_lab[sel]), 0.0, 1.0)
    via_api = cal.predict_oof(S_lab, lab_folds)
    dev = float(np.mean(np.abs(manual - via_api)))
    report(
        "manual fold-model OOF == calibrator.predict_oof on labeled rows",
        dev < 1e-8,
        f"mean abs dev={dev:.2e} (predict_oof is the same per-fold predict "
        f"path fit_cv uses for its internal OOF diagnostics)",
    )

    row7 = estimators.crossppi_from_calibrator(cal, S_all, Y_all, labeled, ALPHA, seed)
    report(
        "cross-PPI arm from extracted folds returns a finite interval",
        all(np.isfinite([row7["est"], row7["lo"], row7["hi"]])),
        f"est={row7['est']:.4f} ci=[{row7['lo']:.4f},{row7['hi']:.4f}] "
        f"K={row7['extras']['K']}",
    )


def check_e_micro_run(truths):
    print("\n(e) 5-replicate end-to-end micro-run, every arm, m=10 n=500")
    S_corpus, Y_corpus = data_prep.load_policy_arrays("base")
    n, m = 500, 10
    all_ok = True
    seen = set()
    for regime in sampling.REGIMES:
        for rep in range(5):
            idx = sampling.draw_indices("base", n, rep)
            S_all, Y_all = S_corpus[idx], Y_corpus[idx]
            labeled, pi = sampling.labeled_for(regime, S_all, "base", n, rep, m)
            seeds = {"cje": sampling.cje_seed(regime, "base", n, rep, m)}
            for arm in estimators.arm_names(regime):
                seeds[arm] = sampling.arm_seed(arm, regime, "base", n, rep, m)
            rows = estimators.evaluate_replicate(
                S_all, Y_all, labeled, regime,
                seeds, pi=pi if regime == "stratified" else None,
            )
            for r in rows:
                seen.add(r["estimator"])
                finite = all(np.isfinite([r["est"], r["lo"], r["hi"]]))
                if r["status"] != "ok" or not finite:
                    all_ok = False
                    print(f"  BAD: {regime} rep{rep} {r['estimator']}: "
                          f"status={r['status']} est={r['est']}")
    expected = set(estimators.arm_names("uniform")) | set(
        estimators.arm_names("stratified")
    )
    report(
        "all arms x 5 reps x 2 regimes: finite intervals, status ok",
        all_ok and seen == expected,
        f"arms covered: {sorted(seen)}",
    )


def check_f_sampling_invariants():
    print("\n(f) sampling invariants")
    n, rep = 500, 3
    d1 = sampling.draw_indices("base", n, rep)
    d2 = sampling.draw_indices("base", n, rep)
    report("n-draw deterministic and shared (regime/m independent)",
           np.array_equal(d1, d2))

    S_corpus, _ = data_prep.load_policy_arrays("base")
    S_all = S_corpus[d1]
    ok_nest = True
    for regime in sampling.REGIMES:
        prev = None
        for m in sampling.M_GRID:
            lab, pi = sampling.labeled_for(regime, S_all, "base", n, rep, m)
            if prev is not None and not set(prev) <= set(lab):
                ok_nest = False
                print(f"  NOT NESTED: {regime} m={m}")
            prev = lab
            if regime == "stratified":
                if abs(pi.sum() - m) > 1e-9:
                    ok_nest = False
                    print(f"  BAD PI SUM: {regime} m={m} sum={pi.sum()}")
    report("labeled subsets nested across m; stratified pi sums to m", ok_nest)

    s42 = sampling.cje_seed("uniform", "base", 500, 0, 10)
    report("cje per-rep child seed != library default 42", s42 != 42,
           f"example seed={s42}")


def main():
    print("=" * 70)
    print("validate.py — pre-run gate (all checks must PASS)")
    print("=" * 70)
    truths = check_a_corpus()
    check_b_alpha_tripwire()
    check_c_postpi()
    check_d_fold_extraction(truths)
    check_e_micro_run(truths)
    check_f_sampling_invariants()
    print("\n" + "=" * 70)
    if FAILURES:
        print(f"RESULT: RED — {len(FAILURES)} failure(s): {FAILURES}")
        sys.exit(1)
    print("RESULT: GREEN — all checks passed. Runs are unlocked.")
    sys.exit(0)


if __name__ == "__main__":
    main()
