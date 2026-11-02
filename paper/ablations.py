"""Ablations (Section 6, Appendix I table) and the joint-training collapse
with cost recoverability (Appendix B.1 table).

    python paper/ablations.py
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from common import CELLS, NAMES, SEEDS, fm_npz, run_json
from tabfm_logit.data import load_splits, get_Xy, DATASET_FEATURES

COST_KEY = {"swissmetro": "beta_co", "lpmc": "b_cost", "iot_wearables": "beta_price"}
# (alternative label, column of q, cost column in X) per cost-bearing alternative
COST_COLS = {"swissmetro": [("train", 0, 1), ("SM", 1, 4), ("car", 2, 7)],
             "lpmc": [("PT", 2, 4), ("drive", 3, 5)],
             "iot_wearables": [("alt1", 0, 0), ("alt2", 1, 1), ("alt3", 2, 2)]}


def acc(variant, ds, fm, tag="xfit5"):
    return np.array([100 * run_json(variant, ds, fm, s, tag)["test_accuracy"] for s in SEEDS])


def r2(y, yhat):
    return 1.0 - float(np.sum((y - yhat) ** 2)) / float(np.sum((y - y.mean()) ** 2))


def ols_r2(Q, y):
    Xd = np.column_stack([np.ones(len(Q)), Q])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    return r2(y, Xd @ beta)


def rf_r2(Q, y, seed=0):
    """5-fold out-of-fold R^2 of a random forest (200 trees, min leaf 20)."""
    pred = np.empty_like(y)
    for tr, te in KFold(n_splits=5, shuffle=True, random_state=seed).split(Q):
        rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=20, n_jobs=-1, random_state=seed)
        rf.fit(Q[tr], y[tr])
        pred[te] = rf.predict(Q[te])
    return r2(y, pred)


if __name__ == "__main__":
    print("Appendix I table. Test accuracy (%), mean +- sd over 10 replicates.")
    variants = [("TabFM-Logit", "tabfm_logit"), ("A1 log-only", "log_only"), ("A2 two-term", "two_term"),
                ("A3 joint", "joint")]
    for label, v in variants:
        print(f"  {label:12s} " + "  ".join(f"{acc(v, ds, fm).mean():6.2f} +- {acc(v, ds, fm).std(ddof=1):.2f}"
                                           for ds, fm in CELLS))
    a5 = {c: [acc(v, *c).mean() for v in ("tabfm_logit", "width16", "width64", "depth1")] for c in CELLS}
    print(f"  {'A5 range':12s} " + "  ".join(f"{min(a5[c]):.2f}-{max(a5[c]):.2f}" for c in CELLS))

    print("\nDifferences from TabFM-Logit (pp, means over replicates):")
    for ds, fm in CELLS:
        base = acc("tabfm_logit", ds, fm).mean()
        d = {v: acc(v, ds, fm).mean() - base for v in ("log_only", "two_term", "joint")}
        print(f"  {NAMES[ds] + ' / ' + NAMES[fm]:24s} A1 {d['log_only']:+.2f}  A2 {d['two_term']:+.2f}  "
              f"A3 {d['joint']:+.2f}  A5 max |diff| {max(abs(x - base) for x in a5[(ds, fm)]):.2f}")
    mono = min(run_json(v, ds, fm, s)["audit"]["monotonicity_avg"] for v in
               ("tabfm_logit", "log_only", "two_term", "joint", "width16", "width64", "depth1")
               for ds, fm in CELLS for s in SEEDS)
    print(f"  minimum monotonicity over all variants, cells and replicates: {100 * mono:.1f}%")

    print("\nAppendix B.1 table. Cost recoverability from cross-fitted training-row q, and the A3 collapse.")
    for ds, fm in CELLS:
        train, _, _ = load_splits(ds)
        X, _ = get_Xy(train, DATASET_FEATURES[ds])
        Q = fm_npz(ds, fm, "train")["probs"].astype(np.float64)
        ols = [ols_r2(Q, X[:, c].astype(np.float64)) for _, _, c in COST_COLS[ds]]
        rf = [rf_r2(Q, X[:, c].astype(np.float64)) for _, _, c in COST_COLS[ds]]
        k = COST_KEY[ds]
        s1 = np.array([run_json("tabfm_logit", ds, fm, s)["betas"][k] for s in SEEDS])
        jt = np.array([run_json("joint", ds, fm, s)["betas"][k] for s in SEEDS])
        line = (f"  {NAMES[ds] + ' / ' + NAMES[fm]:24s} OLS {', '.join(f'{x:.2f}' for x in ols):17s} "
                f"RF {', '.join(f'{x:.2f}' for x in rf):17s} Stage 1 {s1.mean():+.3f}  joint {jt.mean():+.3f}  "
                f"ratio {s1.mean() / jt.mean():.2f}  range {(s1 / jt).min():.2f}-{(s1 / jt).max():.2f}")
        if ds == "swissmetro":
            vot = np.mean([run_json("joint", ds, fm, s)["analytical_trade_off"]["vot_chf_per_hr"] for s in SEEDS])
            line += f"  joint value of time {vot:.0f} CHF/hr"
        print(line)
