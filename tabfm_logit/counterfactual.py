"""
Counterfactual aggregate-share evaluation under the fixed-q protocol.

For each cost-bearing alternative, the alternative's cost column is scaled by
1 + delta on every test row while the foundation-model input q stays at its
unperturbed value. Aggregate shares are mean predicted probabilities over
the test rows; the reported quantity is the change in the perturbed
alternative's share, for TabFM-Logit and for its Stage 1 multinomial logit,
averaged over the training-bootstrap replicates.

Output: results/counterfactual/fixed_q.json

Usage:
    python -m tabfm_logit.counterfactual
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

from tabfm_logit.data import load_splits, get_Xy, DATASET_FEATURES  # noqa: E402
from tabfm_logit.predictions import load_fm_probs  # noqa: E402
from tabfm_logit.adapter import DATASET_MNL, TabFMLogit  # noqa: E402

MODELS_DIR = ROOT / "results" / "models"
OUT_DIR = ROOT / "results" / "counterfactual"

# (label, cost column, alternative index, relative perturbation)
SCENARIOS = {
    "swissmetro": [
        ("train_cost_+10pct", "TRAIN_CO", 0, 0.10),
        ("sm_cost_+10pct",    "SM_CO",    1, 0.10),
        ("car_cost_+10pct",   "CAR_CO",   2, 0.10),
    ],
    "lpmc": [
        ("pt_cost_+10pct",    "cost_transit", 2, 0.10),
        ("drive_cost_+10pct", "cost_driving", 3, 0.10),
    ],
    "iot_wearables": [
        ("alt1_price_+10pct", "price1", 0, 0.10),
        ("alt2_price_+10pct", "price2", 1, 0.10),
        ("alt3_price_+10pct", "price3", 2, 0.10),
    ],
}

# (dataset, foundation model) cells reported in the paper.
CELLS = [("swissmetro", "mitra"), ("swissmetro", "tabpfn"), ("lpmc", "mitra"),
         ("iot_wearables", "mitra"), ("iot_wearables", "tabpfn")]


def load_tabfm_logit(dataset: str, fm: str, seed: int, fm_tag: str = "xfit5") -> TabFMLogit:
    """Load a trained TabFM-Logit checkpoint written by tabfm_logit.train."""
    mnl_cls, n_alts = DATASET_MNL[dataset]
    model = TabFMLogit(mnl_cls, n_alts, hidden_correction=32, n_correction_layers=2,
                       use_log_q=False, use_correction=True)
    tag = f"_{fm_tag}" if fm_tag else ""
    ckpt = MODELS_DIR / f"tabfm_logit_{dataset}_{fm}{tag}_seed{seed}.pt"
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    model.eval()
    return model


def stage1_probs(model: TabFMLogit, X: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Probabilities of the Stage 1 multinomial logit (correction set to zero)."""
    saved = {k: v.clone() for k, v in model.correction.state_dict().items()}
    with torch.no_grad():
        for p in model.correction.parameters():
            p.zero_()
        out = model.predict_proba_np(X, q)
    model.correction.load_state_dict(saved)
    return out


def perturb_cost(X: np.ndarray, col_idx: int, delta_frac: float) -> np.ndarray:
    Xp = X.copy()
    Xp[:, col_idx] = Xp[:, col_idx] * (1.0 + delta_frac)
    return Xp


def aggregate_share(probs: np.ndarray) -> np.ndarray:
    return probs.mean(axis=0)


def fixed_q_cell(dataset: str, fm: str, fm_tag: str = "xfit5", seeds=range(10)) -> list[dict]:
    feats = DATASET_FEATURES[dataset]
    _, _, test = load_splits(dataset)
    X, _ = get_Xy(test, feats)
    q = load_fm_probs(dataset, fm, "test", len(test), fm_tag=fm_tag)
    rows = []
    for label, col, k, d in SCENARIOS[dataset]:
        Xp = perturb_cost(X, feats.index(col), d)
        dm, da = [], []
        for s in seeds:
            m = load_tabfm_logit(dataset, fm, s, fm_tag)
            a0, a1 = aggregate_share(m.predict_proba_np(X, q)), aggregate_share(m.predict_proba_np(Xp, q))
            b0, b1 = aggregate_share(stage1_probs(m, X, q)), aggregate_share(stage1_probs(m, Xp, q))
            da.append(100 * (a1[k] - a0[k]))
            dm.append(100 * (b1[k] - b0[k]))
        rows.append({"dataset": dataset, "fm": fm, "scenario": label, "alt": k, "delta_frac": d,
                     "mnl_pp": float(np.mean(dm)), "tabfm_logit_pp": float(np.mean(da)),
                     "tabfm_logit_pp_std": float(np.std(da, ddof=1)),
                     "n_seeds_decrease": int(sum(x < 0 for x in da)), "n_seeds": len(da)})
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fm-tag", default="xfit5")
    a = p.parse_args()
    out = []
    for ds, fm in CELLS:
        out.extend(fixed_q_cell(ds, fm, a.fm_tag))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "fixed_q.json").write_text(json.dumps(out, indent=2))
    for r in out:
        print(f"  {r['dataset']:14s} {r['fm']:7s} {r['scenario']:20s} MNL {r['mnl_pp']:+.2f}  "
              f"TabFM-Logit {r['tabfm_logit_pp']:+.2f} (sd {r['tabfm_logit_pp_std']:.2f})")
