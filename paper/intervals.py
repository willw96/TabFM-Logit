"""Accuracy differences with 95% bootstrap intervals (Section 5.2, Table 2,
Appendix F).

For each cell, TabFM-Logit's gain over its Stage 1 multinomial logit is
reported with two intervals: across replicates (resampling the 10
per-replicate gains) and test resampling (resampling test rows on the
row-level Swissmetro and LPMC splits, subjects on IoT-Wearables, and
respondents or households on the grouped splits). Paired differences with
the raw foundation model and the feature-augmented MNL use test resampling.
Every interval uses 2,000 resamples from a generator seeded with 0.

    python paper/intervals.py
"""
import numpy as np

from common import (CELLS, GROUPED, NAMES, PRED, SEEDS, B, boot_mean, ci, fm_npz, groups,
                    interval, pp, run_json, run_npz)


def cell(ds, fm, grouped=False):
    """All accuracy quantities for one cell. ``grouped`` selects the grouped
    split (Swissmetro, LPMC); IoT-Wearables is split by subject throughout."""
    split = GROUPED.get(ds, ds) if grouped else ds
    tag = "gxfit5" if split != ds else "xfit5"
    A, M, F = [], [], []
    for s in SEEDS:
        z = run_npz("tabfm_logit", split, fm, s, "test", tag)
        y = z["labels"]
        A.append(z["probs"].argmax(1) == y)
        M.append(z["probs_stage1"].argmax(1) == y)
        f = np.load(PRED / f"feature_aug_{split}_{fm}_{tag}_seed{s}_test.npz")
        assert np.array_equal(f["labels"], y)
        F.append(f["probs_feature_aug"].argmax(1) == y)
    A, M, F = (np.vstack(x).astype(float) for x in (A, M, F))
    raw_q = fm_npz(split, fm, "test", tag)
    assert np.array_equal(raw_q["labels"], y)
    raw = (raw_q["probs"].argmax(1) == y).astype(float)
    g = groups(split)
    per_rep = (A - M).mean(1)
    rng = lambda: np.random.default_rng(0)  # noqa: E731
    rep = rng()
    return {
        "split": split, "n_test": len(y), "n_groups": None if g is None else len(np.unique(g)),
        "acc_mnl": 100 * M.mean(), "acc_raw": 100 * raw.mean(), "acc_fa": 100 * F.mean(),
        "acc_tabfm": 100 * A.mean(),
        "gain": 100 * per_rep.mean(), "positive_replicates": int((per_rep > 0).sum()),
        "gain_ci_rep": [100 * x for x in ci([per_rep[rep.integers(0, 10, 10)].mean() for _ in range(B)])],
        "gain_ci_test": [100 * x for x in boot_mean((A - M).mean(0), g, rng())],
        "minus_raw": 100 * (A.mean() - raw.mean()),
        "minus_raw_ci": [100 * x for x in boot_mean(A.mean(0) - raw, g, rng())],
        "minus_fa": 100 * (A.mean() - F.mean()),
        "minus_fa_ci": [100 * x for x in boot_mean(A.mean(0) - F.mean(0), g, rng())],
        "trade_off": [run_json("tabfm_logit", split, fm, s, tag)["analytical_trade_off"] for s in SEEDS],
        "monotonicity": 100 * np.mean([run_json("tabfm_logit", split, fm, s, tag)["audit"]["monotonicity_avg"]
                                       for s in SEEDS]),
    }


def main():
    print("Appendix F. Accuracy differences (pp) with 95% bootstrap intervals.")
    print(f"  {'split':9s} {'cell':22s} {'TabFM-Logit - MNL':>17s} {'replicates':>16s} {'test':>16s}"
          f"   {'TabFM-Logit - raw FM':>26s}   {'TabFM-Logit - feat-aug':>26s}")
    for grouped in (False, True):
        for ds, fm in CELLS:
            if grouped and ds not in GROUPED:
                continue
            r = cell(ds, fm, grouped)
            print(f"  {'grouped' if grouped else 'row-level':9s} {NAMES[ds] + ' / ' + NAMES[fm]:22s} "
                  f"{pp(r['gain']):>17s} {interval(r['gain_ci_rep']):>16s} {interval(r['gain_ci_test']):>16s}"
                  f"   {pp(r['minus_raw'])} {interval(r['minus_raw_ci']):>18s}"
                  f"   {pp(r['minus_fa'])} {interval(r['minus_fa_ci']):>18s}"
                  f"   positive replicates {r['positive_replicates']}/10")


if __name__ == "__main__":
    main()
