"""
Foundation-model prediction files.

The foundation-model runners (tabfm_logit/fm/) write one file per
(foundation model, dataset, split):

    results/predictions/{fm}_{dataset}_{split}.npz           in-context fit
    results/predictions/{fm}_{tag}_{dataset}_{split}.npz     tagged runs

with arrays ``probs`` (N, K) and ``labels`` (N,). Tags used in the paper:

    xfit5             training rows: 5-fold cross-fitted predictions;
                      validation/test rows: the in-context fit
    gxfit5            the same with group folds (grouped splits)
    xfit5_subset{NN}  ablation A4: context restricted to NN% of train+val
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS = ROOT / "results" / "predictions"


def fm_probs_path(dataset: str, fm: str, split: str, fm_tag: str = "") -> Path:
    suffix = f"_{fm_tag}" if fm_tag else ""
    path = PREDICTIONS / f"{fm}{suffix}_{dataset}_{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run the foundation-model runner first.")
    return path


def load_fm_probs(dataset: str, fm: str, split: str, n_rows: int, fm_tag: str = "") -> np.ndarray:
    """Load q for one split and check that it matches the split's row count."""
    path = fm_probs_path(dataset, fm, split, fm_tag=fm_tag)
    arr = np.load(path)["probs"].astype(np.float32)
    if arr.shape[0] != n_rows:
        raise RuntimeError(f"{path.name}: {arr.shape[0]} rows, expected {n_rows}.")
    return arr
