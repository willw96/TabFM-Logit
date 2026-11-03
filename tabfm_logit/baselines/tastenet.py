"""
TasteNet-MNL baseline (Han, Pereira, Ben-Akiva, Zegras 2022; arXiv:2002.00922).

A multilayer perceptron maps individual characteristics z to taste
coefficients beta(z), which enter a multinomial-logit utility over the
alternatives' attributes x. It uses no foundation model.

Architecture per Han 2022 §3.1–3.3:
    β_k(z) = T_k(  MLP(z; hidden_units, activation)  )
where T_k is the sign-constraint transform for coefficient k:
    negative (time, cost, headway):  β_k = -exp(-u_k)
    positive (function, label):      β_k = +exp(u_k)
    unconstrained (ASCs, b_ga):      β_k = u_k

The MNL utility is dataset-specific and mirrors the shape of the
corresponding class in `mnl.py` so that the audit code (accuracy,
monotonicity, trade-off distribution, availability compliance) applies
without modification.

Module exports:
    TasteNet_SM, TasteNet_LPMC, TasteNet_IoT        — torch.nn.Module classes
    train_tastenet                                  — training loop with L2
    run_tastenet                                    — one-line trainer + eval
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tabfm_logit.data import (  # noqa: E402
    load_splits, get_Xy,
    SM_FEATURES, LPMC_FEATURES, IOT_WEARABLES_FEATURES,
)
from tabfm_logit.baselines.mnl import MAX_EPOCHS  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Shared MLP + sign transforms
# ──────────────────────────────────────────────────────────────────────────────

def _neg_exp(u: torch.Tensor) -> torch.Tensor:
    """Han 2022 output transform for a strictly negative coefficient."""
    return -torch.exp(-u)


def _pos_exp(u: torch.Tensor) -> torch.Tensor:
    """Mirror of _neg_exp for strictly positive coefficients (repo convention)."""
    return torch.exp(u)


class TasteNetMLP(nn.Module):
    """One-hidden-layer MLP mapping z (N, D_z) → raw taste vector (N, K)."""
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 80,
                 activation: str = "relu"):
        super().__init__()
        if activation == "relu":
            act = nn.ReLU()
        elif activation == "tanh":
            act = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            act,
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ──────────────────────────────────────────────────────────────────────────────
# Swissmetro
# ──────────────────────────────────────────────────────────────────────────────
#
# SM_FEATURES index reference (see mnl.py):
#   0:TRAIN_TT  1:TRAIN_CO  2:TRAIN_HE
#   3:SM_TT     4:SM_CO     5:SM_HE
#   6:CAR_TT    7:CAR_CO
#   8:TRAIN_AV  9:SM_AV    10:CAR_AV
#  11:AGE      12:INCOME   13:GA      14:LUGGAGE  15:PURPOSE
#
# z = X[:, 11..16]  (5-D subject characteristics)
# Taste output order: [asc_train, asc_sm, u_tt, u_co, u_he, b_ga]  (6-D)

SM_Z_COLS = [11, 12, 13, 14, 15]                       # AGE, INCOME, GA, LUGGAGE, PURPOSE
SM_TASTE_NAMES = ["asc_train", "asc_sm", "beta_tt", "beta_co", "beta_he", "b_ga"]


class TasteNet_SM(nn.Module):
    def __init__(self, hidden: int = 80, activation: str = "relu"):
        super().__init__()
        self.mlp = TasteNetMLP(in_dim=len(SM_Z_COLS), out_dim=6,
                               hidden=hidden, activation=activation)

    def _betas(self, X: torch.Tensor):
        z = X[:, SM_Z_COLS]
        u = self.mlp(z)                        # (N, 6)
        asc_train = u[:, 0]
        asc_sm = u[:, 1]
        beta_tt = _neg_exp(u[:, 2])
        beta_co = _neg_exp(u[:, 3])
        beta_he = _neg_exp(u[:, 4])
        b_ga = u[:, 5]
        return asc_train, asc_sm, beta_tt, beta_co, beta_he, b_ga

    def utilities(self, X: torch.Tensor) -> torch.Tensor:
        asc_train, asc_sm, b_tt, b_co, b_he, b_ga = self._betas(X)
        ga = X[:, 13]
        no_ga = 1.0 - ga

        V_train = (asc_train
                   + b_tt * X[:, 0]
                   + b_co * X[:, 1] * no_ga
                   + b_he * X[:, 2]
                   + b_ga * ga)
        V_sm = (asc_sm
                + b_tt * X[:, 3]
                + b_co * X[:, 4] * no_ga
                + b_he * X[:, 5])
        V_car = (b_tt * X[:, 6]
                 + b_co * X[:, 7])

        V = torch.stack([V_train, V_sm, V_car], dim=1)
        avail = X[:, 8:11]
        V = V + torch.log(avail.clamp(min=1e-10))
        return V

    def forward(self, X):
        return torch.log_softmax(self.utilities(X), dim=1)

    def predict_proba(self, X):
        return torch.softmax(self.utilities(X), dim=1)

    def per_obs_betas(self, X: torch.Tensor) -> dict:
        """Return per-observation taste vectors as a dict of (N,) numpy arrays."""
        with torch.no_grad():
            asc_train, asc_sm, b_tt, b_co, b_he, b_ga = self._betas(X)
        return {
            "asc_train": asc_train.cpu().numpy(),
            "asc_sm":    asc_sm.cpu().numpy(),
            "beta_tt":   b_tt.cpu().numpy(),
            "beta_co":   b_co.cpu().numpy(),
            "beta_he":   b_he.cpu().numpy(),
            "b_ga":      b_ga.cpu().numpy(),
        }

    def get_betas(self) -> dict:
        """Population-average taste vector on the current forward pass's input.
        Populated by the runner after training; here we return NaN placeholders."""
        return {k: float("nan") for k in SM_TASTE_NAMES}

    def get_trade_off_ratio(self) -> dict:
        return {"vot": float("nan"), "vot_chf_per_hr": float("nan")}


def _sm_summary(model: "TasteNet_SM", X: np.ndarray) -> tuple[dict, dict]:
    """Compute population-mean taste vector and VOT distribution."""
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32)
    per_obs = model.per_obs_betas(Xt)
    betas = {k: float(v.mean()) for k, v in per_obs.items()}
    vot = per_obs["beta_tt"] / per_obs["beta_co"]           # per-obs
    trade_off = {
        "vot_mean":           float(vot.mean()),
        "vot_median":         float(np.median(vot)),
        "vot_chf_per_hr_mean": float(vot.mean() * 60.0),
        "vot_positive_frac":  float((vot > 0).mean()),      # ratio of two negatives
    }
    return betas, trade_off


# ──────────────────────────────────────────────────────────────────────────────
# LPMC
# ──────────────────────────────────────────────────────────────────────────────
#
# LPMC_FEATURES index reference:
#   0:dur_walking  1:dur_cycling  2:dur_pt_total  3:dur_driving
#   4:cost_transit  5:cost_driving
#   6:age  7:female  8:driving_license  9:car_ownership  10:distance
#
# z = X[:, 6..10]  (4-D — distance excluded; it is a trip-level attribute)
# Taste output order:
#   [asc_walk, asc_cycle, asc_pt,
#    u_walk_time, u_cycle_time, u_pt_time, u_drive_time, u_cost]  (8-D)

LPMC_Z_COLS = [6, 7, 8, 9]                              # age, female, driving_license, car_ownership
LPMC_TASTE_NAMES = [
    "asc_walk", "asc_cycle", "asc_pt",
    "b_walk_time", "b_cycle_time", "b_pt_time", "b_drive_time", "b_cost",
]


class TasteNet_LPMC(nn.Module):
    def __init__(self, hidden: int = 80, activation: str = "relu"):
        super().__init__()
        self.mlp = TasteNetMLP(in_dim=len(LPMC_Z_COLS), out_dim=8,
                               hidden=hidden, activation=activation)

    def _betas(self, X):
        z = X[:, LPMC_Z_COLS]
        u = self.mlp(z)
        asc_walk = u[:, 0]
        asc_cycle = u[:, 1]
        asc_pt = u[:, 2]
        b_walk_time = _neg_exp(u[:, 3])
        b_cycle_time = _neg_exp(u[:, 4])
        b_pt_time = _neg_exp(u[:, 5])
        b_drive_time = _neg_exp(u[:, 6])
        b_cost = _neg_exp(u[:, 7])
        return (asc_walk, asc_cycle, asc_pt,
                b_walk_time, b_cycle_time, b_pt_time, b_drive_time, b_cost)

    def utilities(self, X):
        (asc_walk, asc_cycle, asc_pt,
         b_walk_time, b_cycle_time, b_pt_time, b_drive_time, b_cost) = self._betas(X)

        V_walk = asc_walk + b_walk_time * X[:, 0]
        V_cycle = asc_cycle + b_cycle_time * X[:, 1]
        V_pt = asc_pt + b_pt_time * X[:, 2] + b_cost * X[:, 4]
        V_drive = b_drive_time * X[:, 3] + b_cost * X[:, 5]
        return torch.stack([V_walk, V_cycle, V_pt, V_drive], dim=1)

    def forward(self, X):
        return torch.log_softmax(self.utilities(X), dim=1)

    def predict_proba(self, X):
        return torch.softmax(self.utilities(X), dim=1)

    def per_obs_betas(self, X):
        with torch.no_grad():
            vals = self._betas(X)
        return {n: v.cpu().numpy() for n, v in zip(LPMC_TASTE_NAMES, vals)}

    def get_betas(self):
        return {k: float("nan") for k in LPMC_TASTE_NAMES}

    def get_trade_off_ratio(self):
        return {"pt": float("nan"), "drive": float("nan")}


def _lpmc_summary(model, X):
    model.eval()
    per_obs = model.per_obs_betas(torch.tensor(X, dtype=torch.float32))
    betas = {k: float(v.mean()) for k, v in per_obs.items()}
    trade_off = {
        "pt_mean":         float((per_obs["b_pt_time"]    / per_obs["b_cost"]).mean()),
        "drive_mean":      float((per_obs["b_drive_time"] / per_obs["b_cost"]).mean()),
        "pt_positive_frac":    float(((per_obs["b_pt_time"]    / per_obs["b_cost"]) > 0).mean()),
        "drive_positive_frac": float(((per_obs["b_drive_time"] / per_obs["b_cost"]) > 0).mean()),
    }
    return betas, trade_off


# ──────────────────────────────────────────────────────────────────────────────
# IoT-Wearables
# ──────────────────────────────────────────────────────────────────────────────
#
# IOT_WEARABLES_FEATURES layout:
#   0,1,2:    price1, price2, price3
#   3,4,5:    function1, function2, function3
#   6,7,8:    label1, label2, label3
#   9,10,11,12:  age, education, Female, secbehav
#   13..18:      6 condition dummies (cond_*)
#
# z = X[:, 9..19]  (10-D: 4 subject demos + 6 condition dummies)
# Taste output order: [asc_0, asc_1, u_price, u_function, u_label]  (5-D)

IOT_Z_COLS = list(range(9, 19))                          # 4 subject cols + 6 condition dummies
IOT_TASTE_NAMES = ["asc_0", "asc_1", "beta_price", "b_function", "b_label"]


class TasteNet_IoT(nn.Module):
    def __init__(self, hidden: int = 80, activation: str = "relu"):
        super().__init__()
        self.mlp = TasteNetMLP(in_dim=len(IOT_Z_COLS), out_dim=5,
                               hidden=hidden, activation=activation)

    def _betas(self, X):
        z = X[:, IOT_Z_COLS]
        u = self.mlp(z)
        asc_0 = u[:, 0]
        asc_1 = u[:, 1]
        b_price = _neg_exp(u[:, 2])
        b_function = _pos_exp(u[:, 3])
        b_label = _pos_exp(u[:, 4])
        return asc_0, asc_1, b_price, b_function, b_label

    def utilities(self, X):
        asc_0, asc_1, b_price, b_function, b_label = self._betas(X)
        V_0 = asc_0 + b_price * X[:, 0] + b_function * X[:, 3] + b_label * X[:, 6]
        V_1 = asc_1 + b_price * X[:, 1] + b_function * X[:, 4] + b_label * X[:, 7]
        V_2 =         b_price * X[:, 2] + b_function * X[:, 5] + b_label * X[:, 8]
        return torch.stack([V_0, V_1, V_2], dim=1)

    def forward(self, X):
        return torch.log_softmax(self.utilities(X), dim=1)

    def predict_proba(self, X):
        return torch.softmax(self.utilities(X), dim=1)

    def per_obs_betas(self, X):
        with torch.no_grad():
            vals = self._betas(X)
        return {n: v.cpu().numpy() for n, v in zip(IOT_TASTE_NAMES, vals)}

    def get_betas(self):
        return {k: float("nan") for k in IOT_TASTE_NAMES}

    def get_trade_off_ratio(self):
        return {"wtp_function": float("nan"), "wtp_label": float("nan")}


def _iot_summary(model, X):
    model.eval()
    per_obs = model.per_obs_betas(torch.tensor(X, dtype=torch.float32))
    betas = {k: float(v.mean()) for k, v in per_obs.items()}
    trade_off = {
        "wtp_function_mean": float((-per_obs["b_function"] / per_obs["beta_price"]).mean()),
        "wtp_label_mean":    float((-per_obs["b_label"]    / per_obs["beta_price"]).mean()),
    }
    return betas, trade_off


# ──────────────────────────────────────────────────────────────────────────────
# Registry and training loop
# ──────────────────────────────────────────────────────────────────────────────

CLASSES = {
    "swissmetro":    TasteNet_SM,
    "lpmc":          TasteNet_LPMC,
    "iot_wearables": TasteNet_IoT,
}

FEATURES = {
    "swissmetro":    SM_FEATURES,
    "lpmc":          LPMC_FEATURES,
    "iot_wearables": IOT_WEARABLES_FEATURES,
}

SUMMARIZERS = {
    "swissmetro":    _sm_summary,
    "lpmc":          _lpmc_summary,
    "iot_wearables": _iot_summary,
}


def train_tastenet(model, X_train, y_train, X_val, y_val,
                   lr=0.01, weight_decay=0.0, epochs=MAX_EPOCHS, patience=50,
                   verbose=False):
    """NLL + L2 training with early stopping on val NLL (the epoch count is
    stored on the model as ``epochs_trained``).

    weight_decay=0.0 matches Han 2022's best Swissmetro configuration
    (no L2). Adam's built-in weight_decay applies L2 to every parameter,
    including ASCs; this is a mild deviation from Han's exact spec (L2 on
    hidden-layer weights only) that we accept for implementation simplicity.
    """
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    X_v = torch.tensor(X_val, dtype=torch.float32)
    y_v = torch.tensor(y_val, dtype=torch.long)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.NLLLoss()

    best_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_t), y_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            vloss = criterion(model(X_v), y_v).item()
        if vloss < best_loss - 1e-6:
            best_loss = vloss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break

        if verbose and epoch % 50 == 0:
            print(f"  epoch {epoch:>4d}  train_nll={loss.item():.4f}  val_nll={vloss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.epochs_trained = epoch + 1
    return model, best_loss


def _select_by_val(dataset_name, X_tr, y_tr, X_vl, y_vl,
                   hidden_grid=(20, 40, 80), l2_grid=(0.0, 1e-3),
                   lr=0.01, epochs=MAX_EPOCHS, patience=50, verbose=False):
    """Han-style hyperparameter selection on the val split (lowest val NLL)."""
    best = {"val_nll": float("inf"), "hidden": None, "l2": None, "model": None,
            "epochs": None, "epochs_all": []}
    for h in hidden_grid:
        for l2 in l2_grid:
            model = CLASSES[dataset_name](hidden=h, activation="relu")
            model, vloss = train_tastenet(
                model, X_tr, y_tr, X_vl, y_vl,
                lr=lr, weight_decay=l2, epochs=epochs, patience=patience,
                verbose=verbose,
            )
            best["epochs_all"].append(model.epochs_trained)
            if verbose:
                print(f"  [select] hidden={h:3d} l2={l2:.0e} → val_nll={vloss:.4f}")
            if vloss < best["val_nll"] - 1e-6:
                best.update(val_nll=vloss, hidden=h, l2=l2, model=model, epochs=model.epochs_trained)
    return best


def run_tastenet(dataset_name, seed=42,
                 hidden_grid=(20, 40, 80), l2_grid=(0.0, 1e-3),
                 lr=0.01, epochs=MAX_EPOCHS, patience=50, verbose=False):
    """Train TasteNet for a dataset, return (metrics_dict, model, per_obs_betas).

    metrics_dict has the same keys as `mnl.run_mnl` so downstream tables
    consume it unchanged; per-obs betas are additionally returned so callers
    can snapshot the taste distribution for the audit.
    """
    if dataset_name not in CLASSES:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    torch.manual_seed(seed)
    np.random.seed(seed)

    train, val, test = load_splits(dataset_name)
    feats = FEATURES[dataset_name]
    X_tr, y_tr = get_Xy(train, feats)
    X_vl, y_vl = get_Xy(val, feats)
    X_te, y_te = get_Xy(test, feats)

    sel = _select_by_val(
        dataset_name, X_tr, y_tr, X_vl, y_vl,
        hidden_grid=hidden_grid, l2_grid=l2_grid,
        lr=lr, epochs=epochs, patience=patience, verbose=verbose,
    )
    model = sel["model"]

    model.eval()
    with torch.no_grad():
        probs = model.predict_proba(torch.tensor(X_te, dtype=torch.float32)).numpy()
    preds = probs.argmax(axis=1)
    acc = float((preds == y_te).mean())
    nll = float(-np.log(np.clip(probs[range(len(y_te)), y_te], 1e-15, 1.0)).mean())

    betas, trade_off = SUMMARIZERS[dataset_name](model, X_te)
    per_obs_test = model.per_obs_betas(torch.tensor(X_te, dtype=torch.float32))

    return {
        "dataset": dataset_name,
        "seed": seed,
        "test_accuracy": acc,
        "test_nll": nll,
        "val_nll": sel["val_nll"],
        "selected_hidden": sel["hidden"],
        "selected_l2": sel["l2"],
        "betas": betas,
        "trade_off_ratio": trade_off,
    }, model, per_obs_test


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Train and report TasteNet-MNL baseline per dataset.")
    p.add_argument("--dataset", choices=list(CLASSES) + ["all"], default="all")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    targets = list(CLASSES) if args.dataset == "all" else [args.dataset]
    for ds in targets:
        print(f"\n{'='*70}\n  TasteNet on {ds.upper()} (seed={args.seed})\n{'='*70}")
        result, _, _ = run_tastenet(ds, seed=args.seed, verbose=args.verbose)
        print(f"  test_accuracy   = {result['test_accuracy']:.4f}")
        print(f"  test_nll        = {result['test_nll']:.4f}")
        print(f"  selected hidden = {result['selected_hidden']}   l2 = {result['selected_l2']:.0e}")
        print(f"  population-mean betas:")
        for k, v in result["betas"].items():
            print(f"    {k:20s} = {v:+.4f}")
        print(f"  trade-off ratios:")
        for k, v in result["trade_off_ratio"].items():
            print(f"    {k:20s} = {v:+.4f}")
