"""
Mitra predictions (AutoGluon, in-context learning).

Default mode: the model's context is the train+validation rows; it predicts
every split and writes results/predictions/mitra[_{tag}]_{dataset}_{split}.npz.

Cross-fitting (--cross_fit_k 5): training-row predictions are out of fold.
For each fold, the context is the other training folds plus the validation
rows, and the model predicts the held-out fold. Folds are stratified by the
chosen alternative; on the grouped splits they are stratified group folds,
so no respondent or household is split across folds. Writes
mitra_{xfit5|gxfit5}[_{tag}]_{dataset}_train.npz and copies the default
mode's validation/test files under the same tag (run the default mode first).

--subset_frac restricts the context to a random fraction of the train+val
rows (ablation A4), with --tag naming the run (e.g. subset50).

Usage:
    python -m tabfm_logit.fm.mitra_runner --dataset swissmetro
    python -m tabfm_logit.fm.mitra_runner --dataset swissmetro --cross_fit_k 5
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tabfm_logit.data import load_splits, DATASET_FEATURES, GROUP_COL  # noqa: E402

PRED_DIR = ROOT / "results" / "predictions"
DATASETS = ["swissmetro", "lpmc", "iot_wearables", "swissmetro_resp", "lpmc_hh"]


def _save(dataset, split, probs, labels, tag=""):
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    path = PRED_DIR / f"mitra{'_' + tag if tag else ''}_{dataset}_{split}.npz"
    np.savez(path, probs=probs.astype(np.float32), labels=np.asarray(labels, dtype=np.int64))
    print(f"  saved {path}  shape={probs.shape}")


def _copy_val_test(dataset, src_tag, dst_tag):
    for split in ("val", "test"):
        src = PRED_DIR / f"mitra{'_' + src_tag if src_tag else ''}_{dataset}_{split}.npz"
        if src.exists():
            shutil.copyfile(src, PRED_DIR / f"mitra_{dst_tag}_{dataset}_{split}.npz")
        else:
            print(f"  [note] {src.name} not found; run the default mode to create it")


def run_mitra(dataset, subset_frac=1.0, tag="", subset_seed=42, cross_fit_k=1):
    from autogluon.tabular import TabularPredictor

    features = DATASET_FEATURES[dataset]
    train, val, test = load_splits(dataset)
    train_ag = train[features + ["label"]].copy()
    val_ag = val[features + ["label"]].copy()
    test_ag = test[features + ["label"]].copy()

    if cross_fit_k > 1:
        from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
        group_col = GROUP_COL.get(dataset)
        if group_col:
            folds = StratifiedGroupKFold(n_splits=cross_fit_k, shuffle=True, random_state=42).split(
                train_ag, train_ag["label"], groups=train[group_col].values)
        else:
            folds = StratifiedKFold(n_splits=cross_fit_k, shuffle=True, random_state=42).split(
                train_ag, train_ag["label"])
        mem_ratio = max(1.1, float(os.environ.get("MITRA_MEM_RATIO", "1.1")))
        oof = np.zeros((len(train_ag), int(train_ag["label"].max()) + 1), dtype=np.float32)
        target_ctx = int(round(subset_frac * (len(train_ag) + len(val_ag)))) if subset_frac < 1.0 else None
        for fold, (fit_idx, oof_idx) in enumerate(folds):
            ctx = pd.concat([train_ag.iloc[fit_idx], val_ag], ignore_index=True)
            if target_ctx is not None and target_ctx < len(ctx):
                rng = np.random.default_rng(subset_seed + fold)
                ctx = ctx.iloc[rng.choice(len(ctx), size=target_ctx, replace=False)].reset_index(drop=True)
            print(f"\nMitra on {dataset}: fold {fold + 1}/{cross_fit_k} context={len(ctx)} held-out={len(oof_idx)}")
            predictor = TabularPredictor(label="label", problem_type="multiclass",
                                         path=f"/tmp/mitra_xfit{cross_fit_k}_{dataset}_fold{fold}", verbosity=1)
            predictor.fit(ctx, hyperparameters={"MITRA": {"ag.max_rows": len(ctx)}},
                          num_stack_levels=0, num_bag_folds=0,
                          ag_args_fit={"ag.max_memory_usage_ratio": mem_ratio})
            oof[oof_idx] = predictor.predict_proba(train_ag.iloc[oof_idx].drop(columns=["label"])).values
        xfit_tag = f"{'g' if group_col else ''}xfit{cross_fit_k}" + (f"_{tag}" if tag else "")
        _save(dataset, "train", oof, train_ag["label"].values, tag=xfit_tag)
        _copy_val_test(dataset, tag, xfit_tag)
        return

    ctx = pd.concat([train_ag, val_ag], ignore_index=True)
    if subset_frac < 1.0:
        rng = np.random.default_rng(subset_seed)
        n_keep = max(50, int(round(subset_frac * len(ctx))))
        ctx = ctx.iloc[rng.choice(len(ctx), size=n_keep, replace=False)].reset_index(drop=True)
    print(f"\nMitra on {dataset}: context={len(ctx)} test={len(test_ag)}")
    predictor = TabularPredictor(label="label", problem_type="multiclass",
                                 path=f"/tmp/mitra_{dataset}_{tag or 'full'}", verbosity=1)
    ratio = float(os.environ.get("MITRA_MEM_RATIO", "1.0"))
    kw = {"ag_args_fit": {"ag.max_memory_usage_ratio": ratio}} if ratio != 1.0 else {}
    predictor.fit(ctx, hyperparameters={"MITRA": {"ag.max_rows": len(ctx)}},
                  num_stack_levels=0, num_bag_folds=0, **kw)
    for split, df in (("test", test_ag), ("val", val_ag), ("train", train_ag)):
        probs = predictor.predict_proba(df.drop(columns=["label"])).values.astype(np.float32)
        _save(dataset, split, probs, df["label"].values, tag=tag)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--subset_frac", type=float, default=1.0)
    p.add_argument("--tag", default="")
    p.add_argument("--subset_seed", type=int, default=42)
    p.add_argument("--cross_fit_k", type=int, default=1)
    a = p.parse_args()
    run_mitra(a.dataset, a.subset_frac, a.tag, a.subset_seed, a.cross_fit_k)
