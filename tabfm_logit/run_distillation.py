"""
Knowledge distillation baseline: the Stage 1 multinomial logit trained on the
foundation model's cross-fitted training-row probabilities q.

    pure    loss = soft-target cross-entropy against q
    mixed   loss = 0.5 * NLL(true labels) + 0.5 * soft-target cross-entropy

Early stopping uses the true-label validation NLL (Adam, full batch, patience
30); q never touches validation or test. Five
training-bootstrap replicates per cell. The student is a multinomial logit,
so every structural guarantee holds by construction.

Output: results/baselines/distillation.json

Usage:
    python -m tabfm_logit.run_distillation
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tabfm_logit.data import load_splits, get_Xy  # noqa: E402
from tabfm_logit.baselines.mnl import CLASSES, FEATURES, MAX_EPOCHS  # noqa: E402
from tabfm_logit.predictions import load_fm_probs  # noqa: E402

OUT_PATH = ROOT / "results" / "baselines" / "distillation.json"
CELLS = [("swissmetro", "mitra"), ("swissmetro", "tabpfn"), ("lpmc", "mitra"),
         ("iot_wearables", "mitra"), ("iot_wearables", "tabpfn")]
SEEDS = range(5)


def _bootstrap(seed, *arrays):
    idx = np.random.RandomState(seed).choice(len(arrays[0]), size=len(arrays[0]), replace=True)
    return [np.asarray(a)[idx] for a in arrays]


def train_kd(dataset, X_tr, y_tr, q_tr, X_vl, y_vl, mode, lr=0.01, epochs=MAX_EPOCHS, patience=30):
    model = CLASSES[dataset]()
    X_t, y_t = torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long)
    q_t = torch.tensor(q_tr, dtype=torch.float32)
    X_v, y_v = torch.tensor(X_vl, dtype=torch.float32), torch.tensor(y_vl, dtype=torch.long)
    opt = optim.Adam(model.parameters(), lr=lr)
    nll = torch.nn.NLLLoss()
    best, best_state, wait = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        logp = model(X_t)
        soft = -(q_t * logp).sum(dim=1).mean()
        loss = soft if mode == "pure" else 0.5 * nll(logp, y_t) + 0.5 * soft
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            v = nll(model(X_v), y_v).item()
        if v < best - 1e-6:
            best, wait = v, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.epochs_trained = epoch + 1
    return model


def main():
    out = []
    for dataset, fm in CELLS:
        train_df, val_df, test_df = load_splits(dataset)
        feats = FEATURES[dataset]
        X_tr, y_tr = get_Xy(train_df, feats)
        X_vl, y_vl = get_Xy(val_df, feats)
        X_te, y_te = get_Xy(test_df, feats)
        q_tr = load_fm_probs(dataset, fm, "train", len(train_df), fm_tag="xfit5")
        for mode in ("pure", "mixed"):
            accs, epochs = [], []
            for seed in SEEDS:
                torch.manual_seed(seed)
                Xb, yb, qb = _bootstrap(seed, X_tr, y_tr, q_tr)
                model = train_kd(dataset, Xb, yb, qb, X_vl, y_vl, mode)
                epochs.append(model.epochs_trained)
                model.eval()
                with torch.no_grad():
                    probs = model.predict_proba(torch.tensor(X_te, dtype=torch.float32)).numpy()
                accs.append(float((probs.argmax(1) == y_te).mean()))
            out.append({"dataset": dataset, "fm": fm, "mode": mode, "n_seeds": len(accs),
                        "test_acc_mean": float(np.mean(accs)), "test_acc_std": float(np.std(accs, ddof=1)),
                        "epochs": epochs})
            print(f"  {dataset:14s} {fm:7s} {mode:5s} accuracy {np.mean(accs):.4f}")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
