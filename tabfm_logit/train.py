"""
Train TabFM-Logit or one of its ablation variants on one (dataset, foundation
model, replicate) cell.

Variants (each differs from TabFM-Logit in one respect):

    tabfm_logit   V = V_struct + g(q), two-stage training          (the model)
    log_only      V = V_struct + alpha log q, two-stage            (A1)
    two_term      V = V_struct + alpha log q + g(q), two-stage     (A2)
    joint         V = V_struct + g(q), joint training              (A3)
    width16       g with hidden width 16                           (A5)
    width64       g with hidden width 64                           (A5)
    depth1        g with one hidden layer                          (A5)

Ablation A4 is TabFM-Logit trained on foundation-model predictions from a
restricted context (``--fm-tag xfit5_subset{50,25,10}``).

``--seed`` indexes the training-bootstrap replicate (0-9 in the paper).
``--fm-tag`` selects the prediction set (``xfit5`` on the row-level and
subject-level splits, ``gxfit5`` on the grouped splits).

Outputs:
    results/runs/{variant}_{dataset}_{fm}[_{tag}]_seed{s}.json         metrics, audit, coefficients
    results/models/{variant}_{dataset}_{fm}[_{tag}]_seed{s}.pt         weights
    results/predictions/{variant}_{dataset}_{fm}[_{tag}]_seed{s}_{val,test}.npz
        probs (TabFM-Logit), probs_stage1 (Stage 1 multinomial logit), labels

Usage:
    python -m tabfm_logit.train --variant tabfm_logit --dataset swissmetro --fm mitra --fm-tag xfit5 --seed 0
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
from tabfm_logit.audit import audit, CONFIGS as AUDIT_CONFIGS  # noqa: E402
from tabfm_logit.adapter import (  # noqa: E402
    DATASET_MNL, TabFMLogit, train_two_stage, train_joint, evaluate_stage1_only,
)

VARIANTS = {
    "tabfm_logit": {"training": "two_stage", "use_log_q": False, "use_correction": True},
    "log_only":    {"training": "two_stage", "use_log_q": True,  "use_correction": False},
    "two_term":    {"training": "two_stage", "use_log_q": True,  "use_correction": True},
    "joint":       {"training": "joint",     "use_log_q": False, "use_correction": True},
    "width16":     {"training": "two_stage", "use_log_q": False, "use_correction": True, "hidden_correction": 16},
    "width64":     {"training": "two_stage", "use_log_q": False, "use_correction": True, "hidden_correction": 64},
    "depth1":      {"training": "two_stage", "use_log_q": False, "use_correction": True, "n_correction_layers": 1},
}


def run_stem(variant, dataset, fm, seed, fm_tag=""):
    tag = f"_{fm_tag}" if fm_tag else ""
    return f"{variant}_{dataset}_{fm}{tag}_seed{seed}"


def _predict_fn(model: TabFMLogit, fm_probs: np.ndarray):
    fm_t = torch.tensor(fm_probs, dtype=torch.float32)

    def _fn(X):
        with torch.no_grad():
            return torch.exp(model(torch.tensor(X, dtype=torch.float32), fm_t)).numpy()
    return _fn


def run(variant: str, dataset: str, fm: str, seed: int = 0, fm_tag: str = "",
        save: bool = True, verbose: bool = True):
    spec = dict(VARIANTS[variant])
    training = spec.pop("training")

    feats = DATASET_FEATURES[dataset]
    train_df, val_df, test_df = load_splits(dataset)
    X_tr, y_tr = get_Xy(train_df, feats)
    X_vl, y_vl = get_Xy(val_df, feats)
    X_te, y_te = get_Xy(test_df, feats)
    fm_tr = load_fm_probs(dataset, fm, "train", len(train_df), fm_tag=fm_tag)
    fm_vl = load_fm_probs(dataset, fm, "val", len(val_df), fm_tag=fm_tag)
    fm_te = load_fm_probs(dataset, fm, "test", len(test_df), fm_tag=fm_tag)

    if verbose:
        print(f"  {variant} / {dataset} / {fm} / tag={fm_tag!r} / seed={seed}")
    trainer = train_two_stage if training == "two_stage" else train_joint
    model = trainer(dataset, X_tr, y_tr, fm_tr, X_vl, y_vl, fm_vl, seed=seed, verbose=verbose, **spec)

    probs = model.predict_proba_np(X_te, fm_te)
    acc = float((probs.argmax(axis=1) == y_te).mean())
    nll = float(-np.log(np.clip(probs[np.arange(len(y_te)), y_te], 1e-15, 1.0)).mean())
    stage1 = evaluate_stage1_only(model, X_te, y_te, fm_te)
    probs_stage1 = stage1.pop("probs")

    cfg = AUDIT_CONFIGS[dataset]
    subject_ids = test_df[cfg.subject_col].values if cfg.is_panel else None
    audit_result = audit(_predict_fn(model, fm_te), dataset, X_te, subject_ids=subject_ids)

    result = {
        "variant": variant, "spec": VARIANTS[variant], "dataset": dataset, "fm": fm,
        "fm_tag": fm_tag, "seed": seed, "n_test": int(len(test_df)),
        "test_accuracy": acc, "test_nll": nll,
        "test_accuracy_stage1": stage1["test_accuracy"],
        "accuracy_gain_over_stage1": acc - stage1["test_accuracy"],
        "alpha": float(model.alpha.item()),
        "betas": model.mnl.get_betas(),
        "analytical_trade_off": model.mnl.get_trade_off_ratio(),
        "audit": audit_result,
    }

    if save:
        stem = run_stem(variant, dataset, fm, seed, fm_tag)
        dirs = {k: ROOT / "results" / k for k in ("runs", "models", "predictions")}
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        with open(dirs["runs"] / f"{stem}.json", "w") as f:
            json.dump(result, f, indent=2, default=lambda x: float(x) if hasattr(x, "item") else x)
        torch.save(model.state_dict(), dirs["models"] / f"{stem}.pt")
        np.savez(dirs["predictions"] / f"{stem}_test.npz", probs=probs.astype(np.float32),
                 probs_stage1=probs_stage1.astype(np.float32), labels=y_te.astype(np.int64))
        probs_vl = model.predict_proba_np(X_vl, fm_vl)
        stage1_vl = evaluate_stage1_only(model, X_vl, y_vl, fm_vl)
        np.savez(dirs["predictions"] / f"{stem}_val.npz", probs=probs_vl.astype(np.float32),
                 probs_stage1=stage1_vl["probs"].astype(np.float32), labels=y_vl.astype(np.int64))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="tabfm_logit", choices=list(VARIANTS))
    p.add_argument("--dataset", required=True, choices=list(DATASET_MNL))
    p.add_argument("--fm", required=True, choices=["mitra", "tabpfn"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fm-tag", default="")
    p.add_argument("--no-save", action="store_true")
    a = p.parse_args()
    r = run(a.variant, a.dataset, a.fm, seed=a.seed, fm_tag=a.fm_tag, save=not a.no_save)
    print(f"  test accuracy {r['test_accuracy']:.4f}  (Stage 1 {r['test_accuracy_stage1']:.4f})  "
          f"monotonicity {r['audit']['monotonicity_avg']:.4f}  trade-off {r['analytical_trade_off']}")
