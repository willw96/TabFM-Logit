"""
Live foundation-model predict functions for the behavioral audit of the raw
foundation models (Table 1). The audit perturbs inputs, so the model is
refit once on the train+validation context and queried on each perturbed
batch.

Settings of the live refits: library defaults (TabPFN's default ensemble
size; Mitra as in the main runs), with the TabPFN context capped at 8,000
randomly chosen train+validation rows (seed 42) and predictions in batches
of 2,000 rows.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tabfm_logit.data import load_splits, get_Xy, DATASET_FEATURES  # noqa: E402


def get_live_predict_fn(model_name: str, dataset_name: str) -> Callable[[np.ndarray], np.ndarray]:
    """Fit the foundation model once and return predict_fn(X) -> (N, K) probabilities."""
    features = DATASET_FEATURES[dataset_name]
    train, val, _ = load_splits(dataset_name)
    X_train, y_train = get_Xy(train, features)
    X_val, y_val = get_Xy(val, features)
    X_ctx = np.vstack([X_train, X_val])
    y_ctx = np.concatenate([y_train, y_val])

    if model_name == "tabpfn":
        from tabpfn import TabPFNClassifier
        import torch

        MAX_CTX = 8000
        if len(X_ctx) > MAX_CTX:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X_ctx), MAX_CTX, replace=False)
            X_ctx_use = X_ctx[idx]
            y_ctx_use = y_ctx[idx]
        else:
            X_ctx_use = X_ctx
            y_ctx_use = y_ctx

        device = "cuda" if torch.cuda.is_available() else "cpu"
        clf = TabPFNClassifier(
            device=device, random_state=42, ignore_pretraining_limits=True,
        )
        clf.fit(X_ctx_use, y_ctx_use)

        def _fn(X: np.ndarray) -> np.ndarray:
            Xf = X.astype(np.float32)
            BATCH = 2000
            if len(Xf) <= BATCH:
                return clf.predict_proba(Xf)
            out = [clf.predict_proba(Xf[i:i + BATCH]) for i in range(0, len(Xf), BATCH)]
            return np.vstack(out)
        return _fn

    if model_name == "mitra":
        from autogluon.tabular import TabularPredictor
        import pandas as pd

        ctx_df = pd.DataFrame(X_ctx, columns=features)
        ctx_df["label"] = y_ctx
        predictor = TabularPredictor(
            label="label", problem_type="multiclass",
            path=f"/tmp/tabfm_logit_mitra_audit_{dataset_name}",
            verbosity=1,
        )
        predictor.fit(
            ctx_df,
            hyperparameters={"MITRA": {"ag.max_rows": len(ctx_df)}},
            num_stack_levels=0,
            num_bag_folds=0,
        )

        def _fn(X: np.ndarray) -> np.ndarray:
            df = pd.DataFrame(X.astype(np.float32), columns=features)
            return predictor.predict_proba(df).values.astype(np.float32)
        return _fn

    raise ValueError(f"Unsupported live model: {model_name}")
