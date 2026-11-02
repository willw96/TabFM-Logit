"""
TabPFN predictions (in-context learning; tabpfn 2.2.1, four ensemble members,
random_state 42).

Default mode: the context is the train+validation rows; predicts every split
and writes results/predictions/tabpfn[_{tag}]_{dataset}_{split}.npz.

Cross-fitting (--cross_fit_k 5): training-row predictions are out of fold,
as in mitra_runner.py (stratified folds; stratified group folds on the
grouped splits). Writes tabpfn_{xfit5|gxfit5}[_{tag}]_{dataset}_train.npz and
copies the default mode's validation/test files under the same tag.

--subset_frac restricts the context to a random fraction of the train+val
rows (ablation A4), with --tag naming the run (e.g. subset50).

Usage:
    python -m tabfm_logit.fm.tabpfn_runner --dataset swissmetro
    python -m tabfm_logit.fm.tabpfn_runner --dataset swissmetro --cross_fit_k 5
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tabfm_logit.data import load_splits, get_Xy, DATASET_FEATURES, GROUP_COL  # noqa: E402

PRED_DIR = ROOT / "results" / "predictions"
DATASETS = ["swissmetro", "iot_wearables", "swissmetro_resp"]


def _save(dataset, split, probs, labels, tag=""):
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    path = PRED_DIR / f"tabpfn{'_' + tag if tag else ''}_{dataset}_{split}.npz"
    np.savez(path, probs=probs.astype(np.float32), labels=np.asarray(labels, dtype=np.int64))
    print(f"  saved {path}  shape={probs.shape}")


def _copy_val_test(dataset, src_tag, dst_tag):
    for split in ("val", "test"):
        src = PRED_DIR / f"tabpfn{'_' + src_tag if src_tag else ''}_{dataset}_{split}.npz"
        if src.exists():
            shutil.copyfile(src, PRED_DIR / f"tabpfn_{dst_tag}_{dataset}_{split}.npz")
        else:
            print(f"  [note] {src.name} not found; run the default mode to create it")


def _classifier(device, n_estimators):
    from tabpfn import TabPFNClassifier
    return TabPFNClassifier(device=device, random_state=42, ignore_pretraining_limits=True,
                            n_estimators=n_estimators)


def run_tabpfn(dataset, subset_frac=1.0, tag="", subset_seed=42, cross_fit_k=1, n_estimators=4):
    features = DATASET_FEATURES[dataset]
    train, val, test = load_splits(dataset)
    X_tr, y_tr = get_Xy(train, features)
    X_vl, y_vl = get_Xy(val, features)
    X_te, y_te = get_Xy(test, features)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if cross_fit_k > 1:
        from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
        group_col = GROUP_COL.get(dataset)
        if group_col:
            folds = StratifiedGroupKFold(n_splits=cross_fit_k, shuffle=True, random_state=42).split(
                X_tr, y_tr, groups=train[group_col].values)
        else:
            folds = StratifiedKFold(n_splits=cross_fit_k, shuffle=True, random_state=42).split(X_tr, y_tr)
        oof = np.zeros((len(X_tr), int(y_tr.max()) + 1), dtype=np.float32)
        target_ctx = int(round(subset_frac * (len(X_tr) + len(X_vl)))) if subset_frac < 1.0 else None
        for fold, (fit_idx, oof_idx) in enumerate(folds):
            X_ctx = np.vstack([X_tr[fit_idx], X_vl])
            y_ctx = np.concatenate([y_tr[fit_idx], y_vl])
            if target_ctx is not None and target_ctx < len(X_ctx):
                rng = np.random.default_rng(subset_seed + fold)
                idx = rng.choice(len(X_ctx), size=target_ctx, replace=False)
                X_ctx, y_ctx = X_ctx[idx], y_ctx[idx]
            print(f"\nTabPFN on {dataset}: fold {fold + 1}/{cross_fit_k} context={len(X_ctx)} held-out={len(oof_idx)}")
            clf = _classifier(device, n_estimators)
            clf.fit(X_ctx, y_ctx)
            oof[oof_idx] = clf.predict_proba(X_tr[oof_idx]).astype(np.float32)
        xfit_tag = f"{'g' if group_col else ''}xfit{cross_fit_k}" + (f"_{tag}" if tag else "")
        _save(dataset, "train", oof, y_tr, tag=xfit_tag)
        _copy_val_test(dataset, tag, xfit_tag)
        return

    X_ctx, y_ctx = np.vstack([X_tr, X_vl]), np.concatenate([y_tr, y_vl])
    if subset_frac < 1.0:
        rng = np.random.default_rng(subset_seed)
        n_keep = max(50, int(round(subset_frac * len(X_ctx))))
        idx = rng.choice(len(X_ctx), size=n_keep, replace=False)
        X_ctx, y_ctx = X_ctx[idx], y_ctx[idx]
    print(f"\nTabPFN on {dataset}: context={len(X_ctx)} test={len(X_te)} device={device}")
    clf = _classifier(device, n_estimators)
    clf.fit(X_ctx, y_ctx)
    for split, X, y in (("test", X_te, y_te), ("val", X_vl, y_vl), ("train", X_tr, y_tr)):
        _save(dataset, split, clf.predict_proba(X), y, tag=tag)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--subset_frac", type=float, default=1.0)
    p.add_argument("--tag", default="")
    p.add_argument("--subset_seed", type=int, default=42)
    p.add_argument("--cross_fit_k", type=int, default=1)
    p.add_argument("--n_estimators", type=int, default=4)
    a = p.parse_args()
    run_tabpfn(a.dataset, a.subset_frac, a.tag, a.subset_seed, a.cross_fit_k, a.n_estimators)
