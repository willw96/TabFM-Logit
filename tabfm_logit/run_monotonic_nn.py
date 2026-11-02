"""
Train and evaluate the monotonic neural network baseline on one
(dataset, seed) cell.

The baseline is foundation-model-independent. It trains a separable
architecture V_k = -g(cost_k) + h_k(other) where g is monotone
non-decreasing (sign-constrained-weight ReLU MLP) and h is an
unconstrained two-hidden-layer MLP.

Usage:
    python -m tabfm_logit.run_monotonic_nn --dataset swissmetro --seed 0
    python -m tabfm_logit.run_monotonic_nn --dataset lpmc --seed 0
    python -m tabfm_logit.run_monotonic_nn --dataset iot_wearables --seed 0

Outputs:
    results/monotonic_nn/monotonic_nn_{dataset}_seed{seed}.json
    results/predictions/monotonic_nn_{dataset}_seed{seed}_test.npz
    results/predictions/monotonic_nn_{dataset}_seed{seed}_val.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tabfm_logit.data import load_splits, get_Xy, DATASET_FEATURES, DATASET_N_ALTS  # noqa: E402
from tabfm_logit.audit import audit, CONFIGS as AUDIT_CONFIGS  # noqa: E402
from tabfm_logit.baselines.monotonic_nn import (  # noqa: E402
    MonotonicMLP, cost_indices_for, split_features,
)

OUT_DIR = ROOT / "results" / "monotonic_nn"
PRED_DIR = ROOT / "results" / "predictions"


class MonotonicNNWrapper:
    """Holds a trained MonotonicMLP + the dataset's cost-feature index list,
    exposes predict_proba(X) for the audit pipeline."""

    def __init__(self, model: MonotonicMLP, cost_indices: list[int | None]):
        self.model = model
        self.cost_indices = cost_indices

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            cost, other = split_features(X, self.cost_indices)
            log_probs = self.model(cost, other)
            return torch.exp(log_probs)


def _bootstrap_train(X, y, seed):
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.integers(0, n, size=n)
    return X[idx], y[idx]


def train(dataset: str, seed: int,
          hidden_g: int = 16, hidden_h: int = 32,
          epochs: int = 500, lr: float = 5e-3,
          patience: int = 30, weight_decay: float = 1e-4,
          verbose: bool = True):
    feats = DATASET_FEATURES[dataset]
    n_alts = DATASET_N_ALTS[dataset]
    cost_indices = cost_indices_for(dataset, feats)

    train_df, val_df, test_df = load_splits(dataset)
    X_train, y_train = get_Xy(train_df, feats)
    X_val, y_val = get_Xy(val_df, feats)
    X_test, y_test = get_Xy(test_df, feats)

    if verbose:
        print(f"  dataset={dataset} seed={seed}")
        print(f"    train/val/test rows: {len(y_train)}/{len(y_val)}/{len(y_test)}")

    X_train_b, y_train_b = _bootstrap_train(X_train, y_train, seed)

    n_other = X_train.shape[1] - sum(1 for c in cost_indices if c is not None)

    torch.manual_seed(seed)
    model = MonotonicMLP(n_alts=n_alts, n_other_features=n_other,
                         hidden_g=hidden_g, hidden_h=hidden_h)

    Xt_train = torch.tensor(X_train_b, dtype=torch.float32)
    yt_train = torch.tensor(y_train_b, dtype=torch.long)
    Xt_val = torch.tensor(X_val, dtype=torch.float32)
    yt_val = torch.tensor(y_val, dtype=torch.long)

    cost_train, other_train = split_features(Xt_train, cost_indices)
    cost_val, other_val = split_features(Xt_val, cost_indices)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = -1.0
    best_state = None
    epochs_since_best = 0
    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        log_probs = model(cost_train, other_train)
        loss = F.nll_loss(log_probs, yt_train)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            log_probs_val = model(cost_val, other_val)
            val_pred = log_probs_val.argmax(1)
            val_acc = (val_pred == yt_val).float().mean().item()
        if val_acc > best_val_acc + 1e-6:
            best_val_acc = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if verbose:
        print(f"    converged at val_acc={best_val_acc:.4f} (epoch {ep})")

    return model, cost_indices, X_test, y_test, X_val, y_val


def run(dataset: str, seed: int, save: bool = True, verbose: bool = True):
    model, cost_indices, X_test, y_test, X_val, y_val = train(
        dataset, seed, verbose=verbose,
    )
    wrapper = MonotonicNNWrapper(model, cost_indices)

    Xt_test = torch.tensor(X_test, dtype=torch.float32)
    Xt_val = torch.tensor(X_val, dtype=torch.float32)
    probs_test = wrapper.predict_proba(Xt_test).numpy()
    probs_val = wrapper.predict_proba(Xt_val).numpy()

    test_acc = float((probs_test.argmax(1) == y_test).mean())
    val_acc = float((probs_val.argmax(1) == y_val).mean())

    # Behavioral audit on the test split
    cfg = AUDIT_CONFIGS[dataset]
    train_df, val_df, test_df = load_splits(dataset)
    subject_ids = (
        test_df[cfg.subject_col].values if cfg.is_panel else None
    )

    def _predict_fn(X):
        with torch.no_grad():
            return wrapper.predict_proba(torch.tensor(X, dtype=torch.float32)).numpy()

    audit_result = audit(_predict_fn, dataset, X_test, subject_ids=subject_ids)

    if verbose:
        print(f"    test_acc = {test_acc:.4f}")
        print(f"    monotonicity = {audit_result.get('monotonicity_avg', float('nan')):.4f}")

    if save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        PRED_DIR.mkdir(parents=True, exist_ok=True)
        json_path = OUT_DIR / f"monotonic_nn_{dataset}_seed{seed}.json"
        with open(json_path, "w") as f:
            json.dump({
                "dataset": dataset,
                "seed": seed,
                "n_test": int(len(y_test)),
                "test_accuracy": test_acc,
                "val_accuracy": val_acc,
                "audit": audit_result,
            }, f, indent=2,
            default=lambda x: float(x) if hasattr(x, "item") else x)
        np.savez(PRED_DIR / f"monotonic_nn_{dataset}_seed{seed}_test.npz",
                 probs_monotonic_nn=probs_test, labels=y_test)
        np.savez(PRED_DIR / f"monotonic_nn_{dataset}_seed{seed}_val.npz",
                 probs_monotonic_nn=probs_val, labels=y_val)
        if verbose:
            print(f"    saved {json_path}")

    return {"test_acc": test_acc, "val_acc": val_acc, "audit": audit_result}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True,
                   choices=["swissmetro", "lpmc", "iot_wearables"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-save", action="store_true")
    args = p.parse_args()
    run(args.dataset, args.seed, save=not args.no_save)
