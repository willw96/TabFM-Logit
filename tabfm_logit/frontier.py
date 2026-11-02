"""
Fixed, recomputed and partial q (Section 5.3, Appendix H).

One foundation-model refit per (dataset, foundation model) cell feeds every
quantity. For each cost scenario and perturbation size delta:

    q_base     = FM(X_test)
    q_pert     = FM(X_test with the cost column scaled by 1 + delta)
    q_lambda   = (1 - lambda) q_base + lambda q_pert,  lambda in {0, .25, .5, .75, 1}

and TabFM-Logit's aggregate-share change for the perturbed alternative is
computed under (X_pert, q_lambda) for every replicate. lambda = 0 is the
fixed-q protocol and lambda = 1 the recomputed-q protocol. The raw
foundation model's own share change (q_pert vs q_base) is recorded too.

Value-of-time drift (datasets with a time attribute): per-observation
finite-difference value of time, with time or cost increased by one unit
and q blended at lambda, compared with the fixed-q value; the median and
upper decile of the absolute relative deviation over test rows, averaged
over replicates.

Output: results/counterfactual/frontier.json

Usage (GPU):
    python -m tabfm_logit.frontier --dataset swissmetro --fm mitra
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tabfm_logit.data import load_splits, get_Xy  # noqa: E402
from tabfm_logit.counterfactual import SCENARIOS, load_tabfm_logit, perturb_cost  # noqa: E402
from tabfm_logit.fm.refit import fit_predict  # noqa: E402

OUT_PATH = ROOT / "results" / "counterfactual" / "frontier.json"
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]
DELTAS = [0.10, 0.20, 0.50]

# alternative index -> (time column, cost column)
TIME_COST_COLS = {
    "swissmetro": {0: ("TRAIN_TT", "TRAIN_CO"), 1: ("SM_TT", "SM_CO"), 2: ("CAR_TT", "CAR_CO")},
    "lpmc": {2: ("dur_pt_total", "cost_transit"), 3: ("dur_driving", "cost_driving")},
}
MRS_DELTA = 1.0  # additive finite-difference step on time and on cost


def utilities(model, X_np, q_np):
    """TabFM-Logit utilities V (before the softmax)."""
    X = torch.tensor(X_np, dtype=torch.float32)
    q = torch.tensor(q_np, dtype=torch.float32)
    with torch.no_grad():
        V = model.mnl.utilities(X)
        if model.use_log_q:
            V = V + model.alpha * torch.log(q.clamp(min=1e-10))
        if model.use_correction:
            V = V + model.correction(q)
    return V.numpy()


def run_cell(dataset: str, fm: str, seeds=range(10), fm_tag: str = "xfit5") -> list[dict]:
    models = [(s, load_tabfm_logit(dataset, fm, s, fm_tag)) for s in seeds]
    print(f"\n=== {dataset} / {fm}: fitting the foundation model once ===", flush=True)
    predict, features = fit_predict(dataset, fm)
    _, _, test = load_splits(dataset)
    X_te, _ = get_Xy(test, features)
    q_base = predict(X_te)
    share_pre = {s: m.predict_proba_np(X_te, q_base) for s, m in models}
    records = []

    for label, cost_col, alt, _ in SCENARIOS[dataset]:
        ci = features.index(cost_col)
        for delta in DELTAS:
            X_p = perturb_cost(X_te, ci, delta)
            q_p = predict(X_p)
            curve = []
            for lam in LAMBDAS:
                q_l = (1.0 - lam) * q_base + lam * q_p
                d = np.array([float(m.predict_proba_np(X_p, q_l)[:, alt].mean())
                              - float(share_pre[s][:, alt].mean()) for s, m in models])
                curve.append({"lam": lam, "delta_share_mean": float(d.mean()),
                              "delta_share_std": float(d.std(ddof=1)),
                              "n_seeds_decrease": int((d < 0).sum()), "n_seeds": int(len(d))})
            records.append({"kind": "share", "dataset": dataset, "fm": fm, "scenario": label,
                            "alt": alt, "delta_frac": delta,
                            "max_abs_q_diff": float(np.abs(q_p - q_base).max()),
                            "delta_share_raw_fm": float(q_p[:, alt].mean() - q_base[:, alt].mean()),
                            "lambda_curve": curve})
            print(f"  {label:20s} delta={delta:.2f}  lambda=0 {100 * curve[0]['delta_share_mean']:+.2f} pp  "
                  f"lambda=1 {100 * curve[-1]['delta_share_mean']:+.2f} pp  "
                  f"raw FM {100 * records[-1]['delta_share_raw_fm']:+.2f} pp", flush=True)

    for alt, (tcol, ccol) in TIME_COST_COLS.get(dataset, {}).items():
        X_t = X_te.copy(); X_t[:, features.index(tcol)] += MRS_DELTA
        X_c = X_te.copy(); X_c[:, features.index(ccol)] += MRS_DELTA
        q_t, q_c = predict(X_t), predict(X_c)
        curve = []
        for lam in LAMBDAS:
            qt_l = (1.0 - lam) * q_base + lam * q_t
            qc_l = (1.0 - lam) * q_base + lam * q_c
            med, q90, flip, mean = [], [], [], []
            for _, m in models:
                V0 = utilities(m, X_te, q_base)
                dt = utilities(m, X_t, qt_l)[:, alt] - V0[:, alt]
                dc = utilities(m, X_c, qc_l)[:, alt] - V0[:, alt]
                keep = np.abs(dc) > 1e-9
                mrs = -dt[keep] / dc[keep]
                dt0 = utilities(m, X_t, q_base)[:, alt] - V0[:, alt]
                dc0 = utilities(m, X_c, q_base)[:, alt] - V0[:, alt]
                keep0 = np.abs(dc0) > 1e-9
                mrs0 = float((-dt0[keep0] / dc0[keep0]).mean())   # the structural value
                rel = np.abs((mrs - mrs0) / mrs0)
                med.append(float(np.median(rel)))
                q90.append(float(np.quantile(rel, 0.90)))
                flip.append(float((np.sign(mrs) != np.sign(mrs0)).mean()))
                mean.append(float(mrs.mean()))
            curve.append({"lam": lam, "mrs_mean": float(np.mean(mean)),
                          "abs_rel_drift_median": float(np.mean(med)),
                          "abs_rel_drift_q90": float(np.mean(q90)),
                          "sign_flip_frac": float(np.mean(flip))})
        records.append({"kind": "mrs", "dataset": dataset, "fm": fm, "alt": alt,
                        "time_col": tcol, "cost_col": ccol, "lambda_curve": curve})
        print("  value of time, alt " + str(alt) + ": " + "  ".join(
            f"lambda={c['lam']:.2f} median drift {c['abs_rel_drift_median']:.3f}" for c in curve), flush=True)
    return records


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(SCENARIOS))
    p.add_argument("--fm", required=True, choices=["mitra", "tabpfn"])
    p.add_argument("--fm-tag", default="xfit5")
    a = p.parse_args()
    out = json.loads(OUT_PATH.read_text()) if OUT_PATH.exists() else []
    out = [r for r in out if not (r["dataset"] == a.dataset and r["fm"] == a.fm)]
    out.extend(run_cell(a.dataset, a.fm, fm_tag=a.fm_tag))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"saved {OUT_PATH} ({len(out)} records)")
