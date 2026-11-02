"""
Refit a foundation model on a dataset's train+validation context and return
a function that predicts on arbitrary (e.g. perturbed) inputs. Used by the
recomputed- and partial-q analysis (tabfm_logit/frontier.py) and by the
raw-model counterfactuals. Settings match the main prediction runs: full
context, four ensemble members for TabPFN.

Environment options (do not change the fitted model):
    MITRA_MEM_RATIO=<x>      raise AutoGluon's memory pre-check ceiling
    TABPFN_MEMORY_SAVING=1   TabPFN chunked computation (CPU-only machines)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tabfm_logit.data import load_splits, get_Xy, DATASET_FEATURES  # noqa: E402


def mitra_fit_predict(dataset: str):
    from autogluon.tabular import TabularPredictor

    features = DATASET_FEATURES[dataset]
    train, val, _ = load_splits(dataset)
    ctx = pd.concat([train[features + ["label"]], val[features + ["label"]]], ignore_index=True)
    predictor = TabularPredictor(label="label", problem_type="multiclass",
                                 path=f"/tmp/tabfm_logit_mitra_refit_{dataset}", verbosity=1)
    fit_kwargs = {}
    ratio = float(os.environ.get("MITRA_MEM_RATIO", "1.0"))
    if ratio != 1.0:
        fit_kwargs["ag_args_fit"] = {"ag.max_memory_usage_ratio": ratio}
    predictor.fit(ctx, hyperparameters={"MITRA": {"ag.max_rows": len(ctx)}},
                  num_stack_levels=0, num_bag_folds=0, **fit_kwargs)

    def predict(X: np.ndarray) -> np.ndarray:
        df = pd.DataFrame(X.astype(np.float32), columns=features)
        return predictor.predict_proba(df).values.astype(np.float32)

    return predict, features


def tabpfn_fit_predict(dataset: str):
    import torch
    from tabpfn import TabPFNClassifier

    features = DATASET_FEATURES[dataset]
    train, val, _ = load_splits(dataset)
    X_tr, y_tr = get_Xy(train, features)
    X_vl, y_vl = get_Xy(val, features)
    X_ctx, y_ctx = np.vstack([X_tr, X_vl]), np.concatenate([y_tr, y_vl])

    extra = {"n_estimators": 4}
    if os.environ.get("TABPFN_MEMORY_SAVING") == "1":
        extra["memory_saving_mode"] = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clf = TabPFNClassifier(device=device, random_state=42, ignore_pretraining_limits=True, **extra)
    clf.fit(X_ctx, y_ctx)

    def predict(X: np.ndarray, batch: int = 2000) -> np.ndarray:
        Xf = X.astype(np.float32)
        return np.vstack([clf.predict_proba(Xf[i:i + batch]) for i in range(0, len(Xf), batch)])

    return predict, features


def fit_predict(dataset: str, fm: str):
    if fm == "mitra":
        return mitra_fit_predict(dataset)
    if fm == "tabpfn":
        return tabpfn_fit_predict(dataset)
    raise ValueError(f"unknown foundation model: {fm}")
