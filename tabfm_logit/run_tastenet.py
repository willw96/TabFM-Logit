"""
Train and evaluate TasteNet-MNL (Han et al. 2022; arXiv:2002.00922) on one
(dataset, seed) cell.

A multilayer perceptron maps individual characteristics to per-individual,
sign-constrained taste coefficients of a multinomial logit (no foundation
model). Hyperparameters (hidden width, L2) are chosen by validation NLL.
Replicates differ in the random initialization (``--seed``).

Usage:
    python -m tabfm_logit.run_tastenet --dataset swissmetro --seed 0
    python -m tabfm_logit.run_tastenet --dataset lpmc --seed 0
    python -m tabfm_logit.run_tastenet --dataset iot_wearables --seed 0

Outputs:
    results/tastenet/tastenet_{dataset}_seed{seed}.json
    results/predictions/tastenet_{dataset}_seed{seed}_test.npz
    results/predictions/tastenet_{dataset}_seed{seed}_val.npz
    results/tastenet/tastenet_{dataset}_seed{seed}_perobs.npz
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
from tabfm_logit.audit import audit, CONFIGS as AUDIT_CONFIGS  # noqa: E402
from tabfm_logit.baselines.tastenet import (  # noqa: E402
    CLASSES, FEATURES, SUMMARIZERS, _select_by_val,
)
from tabfm_logit.baselines.mnl import MAX_EPOCHS  # noqa: E402

OUT_DIR = ROOT / "results" / "tastenet"
PRED_DIR = ROOT / "results" / "predictions"


def run(dataset: str, seed: int,
        hidden_grid=(20, 40, 80), l2_grid=(0.0, 1e-3),
        lr: float = 0.01, epochs: int = MAX_EPOCHS, patience: int = 50,
        save: bool = True, verbose: bool = True):
    if dataset not in CLASSES:
        raise ValueError(f"Unknown dataset: {dataset}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_df, val_df, test_df = load_splits(dataset)
    feats = FEATURES[dataset]
    X_tr, y_tr = get_Xy(train_df, feats)
    X_vl, y_vl = get_Xy(val_df, feats)
    X_te, y_te = get_Xy(test_df, feats)

    if verbose:
        print(f"  dataset={dataset} seed={seed}")
        print(f"    train/val/test rows: {len(y_tr)}/{len(y_vl)}/{len(y_te)}")

    sel = _select_by_val(
        dataset, X_tr, y_tr, X_vl, y_vl,
        hidden_grid=hidden_grid, l2_grid=l2_grid,
        lr=lr, epochs=epochs, patience=patience, verbose=verbose,
    )
    model = sel["model"]
    if verbose:
        print(f"    selected hidden={sel['hidden']} l2={sel['l2']:.0e} val_nll={sel['val_nll']:.4f}")

    model.eval()
    Xt_test = torch.tensor(X_te, dtype=torch.float32)
    Xt_val = torch.tensor(X_vl, dtype=torch.float32)
    with torch.no_grad():
        probs_test = model.predict_proba(Xt_test).numpy()
        probs_val = model.predict_proba(Xt_val).numpy()

    test_acc = float((probs_test.argmax(1) == y_te).mean())
    val_acc = float((probs_val.argmax(1) == y_vl).mean())
    test_nll = float(
        -np.log(np.clip(probs_test[range(len(y_te)), y_te], 1e-15, 1.0)).mean()
    )

    betas, trade_off = SUMMARIZERS[dataset](model, X_te)
    per_obs_test = model.per_obs_betas(Xt_test)
    per_obs_val = model.per_obs_betas(Xt_val)

    # Behavioral audit on the test split
    cfg = AUDIT_CONFIGS[dataset]
    subject_ids = test_df[cfg.subject_col].values if cfg.is_panel else None

    def _predict_fn(X):
        with torch.no_grad():
            return model.predict_proba(torch.tensor(X, dtype=torch.float32)).numpy()

    audit_result = audit(_predict_fn, dataset, X_te, subject_ids=subject_ids)

    if verbose:
        print(f"    test_acc     = {test_acc:.4f}")
        print(f"    test_nll     = {test_nll:.4f}")
        print(f"    monotonicity = {audit_result.get('monotonicity_avg', float('nan')):.4f}")
        for k, v in trade_off.items():
            print(f"    {k:22s} = {v:+.4f}")

    if save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        PRED_DIR.mkdir(parents=True, exist_ok=True)

        json_path = OUT_DIR / f"tastenet_{dataset}_seed{seed}.json"
        with open(json_path, "w") as f:
            json.dump({
                "dataset": dataset,
                "seed": seed,
                "n_test": int(len(y_te)),
                "test_accuracy": test_acc,
                "val_accuracy": val_acc,
                "test_nll": test_nll,
                "val_nll": sel["val_nll"],
                "selected_hidden": sel["hidden"],
                "selected_l2": sel["l2"],
                "epochs": sel["epochs"],
                "epochs_all_configs": sel["epochs_all"],
                "population_mean_betas": betas,
                "trade_off_ratio_summary": trade_off,
                "audit": audit_result,
            }, f, indent=2,
            default=lambda x: float(x) if hasattr(x, "item") else x)

        np.savez(PRED_DIR / f"tastenet_{dataset}_seed{seed}_test.npz",
                 probs_tastenet=probs_test, labels=y_te)
        np.savez(PRED_DIR / f"tastenet_{dataset}_seed{seed}_val.npz",
                 probs_tastenet=probs_val, labels=y_vl)

        perobs_path = OUT_DIR / f"tastenet_{dataset}_seed{seed}_perobs.npz"
        np.savez(perobs_path,
                 **{f"test_{k}": v for k, v in per_obs_test.items()},
                 **{f"val_{k}":  v for k, v in per_obs_val.items()})

        if verbose:
            print(f"    saved {json_path}")
            print(f"    saved {perobs_path}")

    return {
        "test_acc": test_acc,
        "val_acc": val_acc,
        "test_nll": test_nll,
        "audit": audit_result,
        "trade_off": trade_off,
        "selected_hidden": sel["hidden"],
        "selected_l2": sel["l2"],
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True,
                   choices=["swissmetro", "lpmc", "iot_wearables"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hidden-grid", type=int, nargs="+", default=[20, 40, 80])
    p.add_argument("--l2-grid", type=float, nargs="+", default=[0.0, 1e-3])
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--quiet", action="store_true", help="suppress per-epoch chatter")
    args = p.parse_args()
    run(args.dataset, args.seed,
        hidden_grid=tuple(args.hidden_grid),
        l2_grid=tuple(args.l2_grid),
        lr=args.lr, epochs=args.epochs, patience=args.patience,
        save=not args.no_save, verbose=not args.quiet)
