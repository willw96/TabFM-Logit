"""Calibration numbers: Appendix D table (NLL, Brier, uncalibrated ECE), the
Swissmetro calibration analysis of Section 7 (foundation-model accuracy on
validation vs test rows, Kolmogorov-Smirnov check, calibrators fit on
out-of-context inputs) and bootstrap-ensemble calibration (Section 5.2).

    python paper/calibration_appendix.py
"""
import numpy as np
from scipy.stats import ks_2samp

from common import CELLS, NAMES, SEEDS, fm_npz, run_npz
from tabfm_logit.calibration import calibrate_one_method, expected_calibration_error, METHODS


def metrics(p, y):
    p = np.clip(p, 1e-15, 1.0)
    return {"nll": float(-np.log(p[np.arange(len(y)), y]).mean()),
            "brier": float(((p - np.eye(p.shape[1])[y]) ** 2).sum(1).mean()),
            "unece": 100 * expected_calibration_error(p, y, 15)["ece"]}


def ece(method, p_eval, y_eval, p_fit, y_fit):
    return calibrate_one_method(method, p_eval, y_eval, p_fit, y_fit)["ece_post"]


def calibration_protocols(ds, fm, key, rng):
    """Best-of-three ECE (%) for (a) calibrators fit on validation, evaluated on
    test; (b) scalar temperature fit on one half of validation, evaluated on
    the other half; (c) calibrators fit on one half of test, evaluated on the
    other half. (b) and (c) average both directions; halves are drawn from
    ``rng`` in a fixed order."""
    a, b, c = {m: [] for m in METHODS}, [], {m: [] for m in METHODS}
    for s in SEEDS:
        v, t = run_npz("tabfm_logit", ds, fm, s, "val"), run_npz("tabfm_logit", ds, fm, s, "test")
        pv, yv, pt, yt = v[key], v["labels"], t[key], t["labels"]
        for m in METHODS:
            a[m].append(ece(m, pt, yt, pv, yv))
        idx = rng.permutation(len(yv)); h1, h2 = idx[:len(yv) // 2], idx[len(yv) // 2:]
        b.append(0.5 * (ece("scalar_T", pv[h2], yv[h2], pv[h1], yv[h1]) + ece("scalar_T", pv[h1], yv[h1], pv[h2], yv[h2])))
        idx = rng.permutation(len(yt)); g1, g2 = idx[:len(yt) // 2], idx[len(yt) // 2:]
        for m in METHODS:
            c[m].append(0.5 * (ece(m, pt[g2], yt[g2], pt[g1], yt[g1]) + ece(m, pt[g1], yt[g1], pt[g2], yt[g2])))
    return (100 * min(np.mean(a[m]) for m in METHODS), 100 * float(np.mean(b)),
            100 * min(np.mean(c[m]) for m in METHODS))


if __name__ == "__main__":
    print("Appendix D table. Test NLL, Brier, uncalibrated ECE (%); MNL and TabFM-Logit: mean +- sd over replicates.")
    for ds, fm in CELLS:
        q = fm_npz(ds, fm, "test")
        print(f"  {NAMES[ds] + ' / ' + NAMES[fm]:24s} raw FM      " +
              "  ".join(f"{k} {x:.3f}" if k != "unece" else f"{k} {x:.1f}" for k, x in metrics(q["probs"], q["labels"]).items()))
        for label, key in (("MNL", "probs_stage1"), ("TabFM-Logit", "probs")):
            per = [metrics(run_npz("tabfm_logit", ds, fm, s, "test")[key], run_npz("tabfm_logit", ds, fm, s, "test")["labels"])
                   for s in SEEDS]
            print(f"  {'':24s} {label:11s} " + "  ".join(
                f"{k} {np.mean([m[k] for m in per]):.{1 if k == 'unece' else 3}f} +- {np.std([m[k] for m in per], ddof=1):.{1 if k == 'unece' else 3}f}"
                for k in per[0]))

    print("\nSection 7 and Appendix D. Calibration under three protocols; foundation-model inputs on validation vs test.")
    rng = np.random.default_rng(0)
    for ds, fm in CELLS:
        qv, qt = fm_npz(ds, fm, "val"), fm_npz(ds, fm, "test")
        ks = ks_2samp(qv["probs"].max(1), qt["probs"].max(1))
        acc_v = 100 * float((qv["probs"].argmax(1) == qv["labels"]).mean())
        acc_t = 100 * float((qt["probs"].argmax(1) == qt["labels"]).mean())
        tab = calibration_protocols(ds, fm, "probs", rng)
        mnl = calibration_protocols(ds, fm, "probs_stage1", rng)
        print(f"  {NAMES[ds] + ' / ' + NAMES[fm]:24s} FM accuracy val {acc_v:.1f} test {acc_t:.1f}   "
              f"KS p = {ks.pvalue:.3f}   ECE TabFM-Logit (a) {tab[0]:.1f} (b) {tab[1]:.1f} (c) {tab[2]:.1f}   "
              f"MNL (a) {mnl[0]:.1f} (b) {mnl[1]:.1f} (c) {mnl[2]:.1f}")

    print("\nSection 5.2. Bootstrap ensemble of the 10 replicates: best post-hoc ECE (%).")
    for ds, fm in CELLS:
        out = []
        for key in ("probs_stage1", "probs"):
            pv = np.mean([run_npz("tabfm_logit", ds, fm, s, "val")[key] for s in SEEDS], axis=0)
            pt = np.mean([run_npz("tabfm_logit", ds, fm, s, "test")[key] for s in SEEDS], axis=0)
            yv, yt = run_npz("tabfm_logit", ds, fm, 0, "val")["labels"], run_npz("tabfm_logit", ds, fm, 0, "test")["labels"]
            out.append(100 * min(ece(m, pt, yt, pv, yv) for m in METHODS))
        print(f"  {NAMES[ds] + ' / ' + NAMES[fm]:24s} MNL {out[0]:.1f}   TabFM-Logit {out[1]:.1f}")
