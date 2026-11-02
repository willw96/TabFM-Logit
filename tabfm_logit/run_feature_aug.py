"""
Train the feature-augmented multinomial logit on one (dataset, foundation
model, replicate) cell.

Outputs:
    results/feature_aug/feature_aug_{dataset}_{fm}[_{tag}]_seed{s}.json
    results/models/feature_aug_{dataset}_{fm}[_{tag}]_seed{s}.pt
    results/predictions/feature_aug_{dataset}_{fm}[_{tag}]_seed{s}_{val,test}.npz

Usage:
    python -m tabfm_logit.run_feature_aug --dataset swissmetro --fm mitra --fm-tag xfit5 --seed 0
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
from tabfm_logit.baselines.feature_augmented_mnl import (  # noqa: E402
    DATASET_MNL, FeatureAugmentedMNL, train_feature_aug,
)


def _predict_fn(model: FeatureAugmentedMNL, fm_probs: np.ndarray):
    fm_t = torch.tensor(fm_probs, dtype=torch.float32)
    def _fn(X):
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            return torch.exp(model(X_t, fm_t)).numpy()
    return _fn


def run(dataset: str, fm: str, seed: int = 0, save: bool = True, verbose: bool = True,
        fm_tag: str = ""):
    if dataset not in DATASET_MNL:
        raise ValueError(f"Unknown dataset: {dataset}")
    if fm not in ("mitra", "tabpfn"):
        raise ValueError(f"Unknown FM: {fm}")

    feats = DATASET_FEATURES[dataset]
    train_df, val_df, test_df = load_splits(dataset)
    X_tr, y_tr = get_Xy(train_df, feats)
    X_vl, y_vl = get_Xy(val_df, feats)
    X_te, y_te = get_Xy(test_df, feats)

    fm_tr = load_fm_probs(dataset, fm, "train", len(train_df), fm_tag=fm_tag)
    fm_vl = load_fm_probs(dataset, fm, "val", len(val_df), fm_tag=fm_tag)
    fm_te = load_fm_probs(dataset, fm, "test", len(test_df), fm_tag=fm_tag)
    tag_suffix = f"_{fm_tag}" if fm_tag else ""

    if verbose:
        print(f"  variant=feature_aug dataset={dataset} fm={fm} fm_tag={fm_tag!r} seed={seed}")
        print(f"  train/val/test rows: {len(train_df)}/{len(val_df)}/{len(test_df)}")

    model = train_feature_aug(
        dataset, X_tr, y_tr, fm_tr, X_vl, y_vl, fm_vl,
        seed=seed, verbose=verbose,
    )

    # Test-set probs and accuracy
    probs = model.predict_proba_np(X_te, fm_te)
    preds = probs.argmax(axis=1)
    acc = float((preds == y_te).mean())
    nll = float(-np.log(np.clip(probs[np.arange(len(y_te)), y_te], 1e-15, 1.0)).mean())

    # Behavioral audit
    cfg = AUDIT_CONFIGS[dataset]
    subject_ids = test_df[cfg.subject_col].values if cfg.is_panel else None
    audit_result = audit(_predict_fn(model, fm_te), dataset, X_te, subject_ids=subject_ids)

    betas = model.get_betas()
    trade_off = model.get_trade_off_ratio()

    result = {
        "variant": "feature_aug",
        "dataset": dataset,
        "fm": fm,
        "fm_tag": fm_tag,
        "seed": seed,
        "n_test": int(len(test_df)),
        "test_accuracy": acc,
        "test_nll": nll,
        "alpha": float(model.alpha.item()),
        "betas": betas,
        "analytical_trade_off": trade_off,
        "audit": audit_result,
    }

    if save:
        results_dir = ROOT / "results" / "feature_aug"
        models_dir = ROOT / "results" / "models"
        preds_dir = ROOT / "results" / "predictions"
        for d in (results_dir, models_dir, preds_dir):
            d.mkdir(parents=True, exist_ok=True)

        json_path = results_dir / f"feature_aug_{dataset}_{fm}{tag_suffix}_seed{seed}.json"
        ckpt_path = models_dir / f"feature_aug_{dataset}_{fm}{tag_suffix}_seed{seed}.pt"
        preds_path = preds_dir / f"feature_aug_{dataset}_{fm}{tag_suffix}_seed{seed}_test.npz"

        with open(json_path, "w") as f:
            json.dump(result, f, indent=2,
                      default=lambda x: float(x) if hasattr(x, "item") else x)
        torch.save(model.state_dict(), ckpt_path)
        np.savez(
            preds_path,
            probs_feature_aug=probs.astype(np.float32),
            labels=y_te.astype(np.int64),
        )
        probs_vl = model.predict_proba_np(X_vl, fm_vl)
        val_path = preds_dir / f"feature_aug_{dataset}_{fm}{tag_suffix}_seed{seed}_val.npz"
        np.savez(
            val_path,
            probs_feature_aug=probs_vl.astype(np.float32),
            labels=y_vl.astype(np.int64),
        )
        if verbose:
            print(f"  saved {json_path}")
            print(f"  saved {ckpt_path}")
            print(f"  saved {preds_path}")
            print(f"  saved {val_path}")

    return result


def _print_summary(r: dict):
    print(f"\n  feature_aug / {r['dataset']} / {r['fm']} / seed={r['seed']}")
    print(f"    test_accuracy        = {r['test_accuracy']:.4f}")
    print(f"    α                    = {r['alpha']:+.4f}")
    print(f"    monotonicity (audit) = {r['audit']['monotonicity_avg']:.4f}")
    if "availability_leak_max" in r["audit"]:
        print(f"    availability leak    = {r['audit']['availability_leak_max']:.3e}")
    print(f"    analytical trade-off = {r['analytical_trade_off']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(DATASET_MNL))
    p.add_argument("--fm", required=True, choices=["mitra", "tabpfn"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fm-tag", type=str, default="",
                   help="Tagged FM prediction set, e.g. 'xfit5' (cross-fitted training-row q) "
                        "or 'gxfit5' (group-cross-fitted). Writes feature_aug_{dataset}_{fm}_{fm_tag}_seed{seed}.*")
    p.add_argument("--no-save", action="store_true")
    args = p.parse_args()

    result = run(args.dataset, args.fm, seed=args.seed, save=not args.no_save, verbose=True,
                 fm_tag=args.fm_tag)
    _print_summary(result)
