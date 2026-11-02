"""
Multinomial logit (MNL) specifications, with the constrained parameterization
beta = -exp(theta) on every time and cost/price coefficient so that
monotonicity holds by construction. These are Stage 1 of TabFM-Logit.

The structural utilities use the alternatives' attributes; individual
covariates enter only where noted (LPMC). The foundation-model correction of
TabFM-Logit captures the remaining heterogeneity.

Module exports:
    MNL_SM, MNL_LPMC, MNL_IoT   torch.nn.Module classes
    train_mnl                   training loop with early stopping
    run_mnl                     train and evaluate on one dataset
"""
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


# ──────────────────────────────────────────────────────────────────────────────
# Swissmetro — 3 alternatives (train, Swissmetro, car), Bierlaire convention
# ──────────────────────────────────────────────────────────────────────────────
#
# Feature index reference for SM_FEATURES:
#   0:TRAIN_TT  1:TRAIN_CO  2:TRAIN_HE
#   3:SM_TT     4:SM_CO     5:SM_HE
#   6:CAR_TT    7:CAR_CO
#   8:TRAIN_AV  9:SM_AV    10:CAR_AV
#  11:AGE      12:INCOME   13:GA      14:LUGGAGE  15:PURPOSE
#
# Utility (Bierlaire annual-pass convention):
#   V_train = ASC_train + β_tt·TRAIN_TT + β_co·TRAIN_CO·(1 - GA) + β_he·TRAIN_HE + β_ga·GA
#   V_sm    = ASC_sm    + β_tt·SM_TT    + β_co·SM_CO·(1 - GA)    + β_he·SM_HE
#   V_car   =             β_tt·CAR_TT   + β_co·CAR_CO
# Annual-pass holders (GA = 1) face zero marginal cost on train and Swissmetro.
# Generic time/cost/headway coefficients across alternatives.
# Car is the reference alternative (no alternative-specific constant).

class MNL_SM(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta_tt = nn.Parameter(torch.tensor(-1.0))
        self.theta_co = nn.Parameter(torch.tensor(-1.0))
        self.theta_he = nn.Parameter(torch.tensor(-1.0))
        self.asc_train = nn.Parameter(torch.tensor(0.0))
        self.asc_sm = nn.Parameter(torch.tensor(0.0))
        self.b_ga = nn.Parameter(torch.tensor(0.0))

    def utilities(self, X):
        """X: (N, 16) — see feature index reference above."""
        b_tt = -torch.exp(self.theta_tt)
        b_co = -torch.exp(self.theta_co)
        b_he = -torch.exp(self.theta_he)

        ga = X[:, 13]
        no_ga = 1.0 - ga

        V_train = (self.asc_train
                   + b_tt * X[:, 0]
                   + b_co * X[:, 1] * no_ga
                   + b_he * X[:, 2]
                   + self.b_ga * ga)
        V_sm = (self.asc_sm
                + b_tt * X[:, 3]
                + b_co * X[:, 4] * no_ga
                + b_he * X[:, 5])
        V_car = (b_tt * X[:, 6]
                 + b_co * X[:, 7])

        V = torch.stack([V_train, V_sm, V_car], dim=1)
        avail = X[:, 8:11]
        V = V + torch.log(avail.clamp(min=1e-10))  # -inf where avail=0
        return V

    def forward(self, X):
        return torch.log_softmax(self.utilities(X), dim=1)

    def predict_proba(self, X):
        return torch.softmax(self.utilities(X), dim=1)

    def get_betas(self):
        with torch.no_grad():
            return {
                "beta_tt": float(-torch.exp(self.theta_tt)),
                "beta_co": float(-torch.exp(self.theta_co)),
                "beta_he": float(-torch.exp(self.theta_he)),
                "asc_train": float(self.asc_train),
                "asc_sm": float(self.asc_sm),
                "b_ga": float(self.b_ga),
            }

    def get_trade_off_ratio(self):
        """{'vot': β_tt / β_co} — minutes-per-cost-unit. In CHF/hour after multiplying by 60."""
        with torch.no_grad():
            ratio = float(-torch.exp(self.theta_tt) / -torch.exp(self.theta_co))
            return {"vot": ratio, "vot_chf_per_hr": ratio * 60.0}


# ──────────────────────────────────────────────────────────────────────────────
# LPMC — 4 alternatives (walk, cycle, public transit, drive)
# Generic cost coefficient, alternative-specific time coefficients,
# socio-demographic interactions on selected alternatives.
# Drive is the reference alternative.
# ──────────────────────────────────────────────────────────────────────────────
#
# Feature index reference for LPMC_FEATURES:
#   0:dur_walking  1:dur_cycling  2:dur_pt_total  3:dur_driving
#   4:cost_transit  5:cost_driving
#   6:age  7:female  8:driving_license  9:car_ownership  10:distance

class MNL_LPMC(nn.Module):
    def __init__(self):
        super().__init__()
        # ASCs (drive is reference, no ASC)
        self.asc_walk = nn.Parameter(torch.tensor(0.0))
        self.asc_cycle = nn.Parameter(torch.tensor(0.0))
        self.asc_pt = nn.Parameter(torch.tensor(0.0))

        # Time coefficients (constrained negative, alternative-specific)
        self.theta_walk_time = nn.Parameter(torch.tensor(-1.0))
        self.theta_cycle_time = nn.Parameter(torch.tensor(-1.0))
        self.theta_pt_time = nn.Parameter(torch.tensor(-1.0))
        self.theta_drive_time = nn.Parameter(torch.tensor(-1.0))

        # Generic cost coefficient (constrained negative)
        self.theta_cost = nn.Parameter(torch.tensor(-1.0))

        # Demographic interactions
        self.b_age_walk = nn.Parameter(torch.tensor(0.0))
        self.b_age_cycle = nn.Parameter(torch.tensor(0.0))
        self.b_age_pt = nn.Parameter(torch.tensor(0.0))
        self.b_female_walk = nn.Parameter(torch.tensor(0.0))
        self.b_female_cycle = nn.Parameter(torch.tensor(0.0))
        self.b_dl_cycle = nn.Parameter(torch.tensor(0.0))
        self.b_dl_drive = nn.Parameter(torch.tensor(0.0))
        self.b_co_drive = nn.Parameter(torch.tensor(0.0))

    def utilities(self, X):
        b_walk_time = -torch.exp(self.theta_walk_time)
        b_cycle_time = -torch.exp(self.theta_cycle_time)
        b_pt_time = -torch.exp(self.theta_pt_time)
        b_drive_time = -torch.exp(self.theta_drive_time)
        b_cost = -torch.exp(self.theta_cost)

        V_walk = (self.asc_walk
                  + b_walk_time * X[:, 0]
                  + self.b_age_walk * X[:, 6]
                  + self.b_female_walk * X[:, 7])
        V_cycle = (self.asc_cycle
                   + b_cycle_time * X[:, 1]
                   + self.b_age_cycle * X[:, 6]
                   + self.b_female_cycle * X[:, 7]
                   + self.b_dl_cycle * X[:, 8])
        V_pt = (self.asc_pt
                + b_pt_time * X[:, 2]
                + b_cost * X[:, 4]
                + self.b_age_pt * X[:, 6])
        V_drive = (b_drive_time * X[:, 3]
                   + b_cost * X[:, 5]
                   + self.b_dl_drive * X[:, 8]
                   + self.b_co_drive * X[:, 9])

        return torch.stack([V_walk, V_cycle, V_pt, V_drive], dim=1)

    def forward(self, X):
        return torch.log_softmax(self.utilities(X), dim=1)

    def predict_proba(self, X):
        return torch.softmax(self.utilities(X), dim=1)

    def get_betas(self):
        with torch.no_grad():
            return {
                "b_walk_time": float(-torch.exp(self.theta_walk_time)),
                "b_cycle_time": float(-torch.exp(self.theta_cycle_time)),
                "b_pt_time": float(-torch.exp(self.theta_pt_time)),
                "b_drive_time": float(-torch.exp(self.theta_drive_time)),
                "b_cost": float(-torch.exp(self.theta_cost)),
                "asc_walk": float(self.asc_walk),
                "asc_cycle": float(self.asc_cycle),
                "asc_pt": float(self.asc_pt),
                "b_age_walk": float(self.b_age_walk),
                "b_age_cycle": float(self.b_age_cycle),
                "b_age_pt": float(self.b_age_pt),
                "b_female_walk": float(self.b_female_walk),
                "b_female_cycle": float(self.b_female_cycle),
                "b_dl_cycle": float(self.b_dl_cycle),
                "b_dl_drive": float(self.b_dl_drive),
                "b_co_drive": float(self.b_co_drive),
            }

    def get_trade_off_ratio(self):
        """Returns {'pt': β_pt_time/β_cost, 'drive': β_drive_time/β_cost}."""
        with torch.no_grad():
            b_cost = float(-torch.exp(self.theta_cost))
            b_pt = float(-torch.exp(self.theta_pt_time))
            b_drive = float(-torch.exp(self.theta_drive_time))
            return {"pt": b_pt / b_cost, "drive": b_drive / b_cost}


# ──────────────────────────────────────────────────────────────────────────────
# IoT-Wearables — 3 alternatives (symmetric hypothetical wearable products).
# Structural utility uses only price, function (binary feature richness),
# and label (security label presence). Subject covariates and condition
# dummies are deliberately omitted from the structural model — the foundation-
# model correction handles individual heterogeneity.
# ──────────────────────────────────────────────────────────────────────────────
#
# Feature index reference for IOT_WEARABLES_FEATURES (19 columns):
#   0,1,2:    price1, price2, price3
#   3,4,5:    function1, function2, function3
#   6,7,8:    label1, label2, label3
#   9..18:    age, education, Female, secbehav, cond_*  (subject-level — ignored here)

class MNL_IoT(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta_price = nn.Parameter(torch.tensor(-1.0))
        # function and label are expected to raise utility: +exp(theta).
        self.theta_function = nn.Parameter(torch.tensor(-1.0))
        self.theta_label = nn.Parameter(torch.tensor(-1.0))
        # ASC for alts 0 and 1 (alt 2 is reference)
        self.asc_0 = nn.Parameter(torch.tensor(0.0))
        self.asc_1 = nn.Parameter(torch.tensor(0.0))

    def utilities(self, X):
        b_price = -torch.exp(self.theta_price)
        b_function = torch.exp(self.theta_function)  # positive
        b_label = torch.exp(self.theta_label)        # positive

        V_0 = (self.asc_0
               + b_price * X[:, 0]
               + b_function * X[:, 3]
               + b_label * X[:, 6])
        V_1 = (self.asc_1
               + b_price * X[:, 1]
               + b_function * X[:, 4]
               + b_label * X[:, 7])
        V_2 = (b_price * X[:, 2]
               + b_function * X[:, 5]
               + b_label * X[:, 8])
        return torch.stack([V_0, V_1, V_2], dim=1)

    def forward(self, X):
        return torch.log_softmax(self.utilities(X), dim=1)

    def predict_proba(self, X):
        return torch.softmax(self.utilities(X), dim=1)

    def get_betas(self):
        with torch.no_grad():
            return {
                "beta_price": float(-torch.exp(self.theta_price)),
                "b_function": float(torch.exp(self.theta_function)),
                "b_label": float(torch.exp(self.theta_label)),
                "asc_0": float(self.asc_0),
                "asc_1": float(self.asc_1),
            }

    def get_trade_off_ratio(self):
        """Willingness to pay (in dollars) for function and label, computed as the
        standard -β_attribute / β_cost. Positive when the attribute coefficient is
        positive (consumer pays to get the feature)."""
        with torch.no_grad():
            b_price = float(-torch.exp(self.theta_price))  # negative
            return {
                "wtp_function": -float(torch.exp(self.theta_function)) / b_price,
                "wtp_label":    -float(torch.exp(self.theta_label))    / b_price,
            }


# ──────────────────────────────────────────────────────────────────────────────
# Generic training and evaluation
# ──────────────────────────────────────────────────────────────────────────────

CLASSES = {
    "swissmetro": MNL_SM,
    "lpmc": MNL_LPMC,
    "iot_wearables": MNL_IoT,
}

FEATURES = {
    "swissmetro": SM_FEATURES,
    "lpmc": LPMC_FEATURES,
    "iot_wearables": IOT_WEARABLES_FEATURES,
}


def train_mnl(model, X_train, y_train, X_val, y_val,
              lr=0.01, epochs=500, patience=30, verbose=False):
    """NLL training with early stopping on validation loss."""
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    X_v = torch.tensor(X_val, dtype=torch.float32)
    y_v = torch.tensor(y_val, dtype=torch.long)

    optimizer = optim.Adam(model.parameters(), lr=lr)
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
            print(f"  epoch {epoch:>3d}  train_nll={loss.item():.4f}  val_nll={vloss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def run_mnl(dataset_name, lr=0.01, epochs=500, patience=30, verbose=False):
    """Train MNL for a dataset, return (model, test_metrics, betas, trade_off)."""
    if dataset_name not in CLASSES:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    train, val, test = load_splits(dataset_name)
    feats = FEATURES[dataset_name]
    X_tr, y_tr = get_Xy(train, feats)
    X_vl, y_vl = get_Xy(val, feats)
    X_te, y_te = get_Xy(test, feats)

    model = CLASSES[dataset_name]()
    model = train_mnl(model, X_tr, y_tr, X_vl, y_vl, lr=lr, epochs=epochs,
                      patience=patience, verbose=verbose)

    model.eval()
    with torch.no_grad():
        probs = model.predict_proba(torch.tensor(X_te, dtype=torch.float32)).numpy()
    preds = probs.argmax(axis=1)
    acc = float((preds == y_te).mean())
    nll = float(-np.log(np.clip(probs[range(len(y_te)), y_te], 1e-15, 1.0)).mean())

    return {
        "dataset": dataset_name,
        "test_accuracy": acc,
        "test_nll": nll,
        "betas": model.get_betas(),
        "trade_off_ratio": model.get_trade_off_ratio(),
    }, model


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Train and report MNL baseline per dataset.")
    p.add_argument("--dataset", choices=list(CLASSES) + ["all"], default="all")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    targets = list(CLASSES) if args.dataset == "all" else [args.dataset]
    for ds in targets:
        print(f"\n{'='*70}\n  MNL on {ds.upper()}\n{'='*70}")
        result, _ = run_mnl(ds, verbose=args.verbose)
        print(f"  test_accuracy = {result['test_accuracy']:.4f}")
        print(f"  test_nll      = {result['test_nll']:.4f}")
        print(f"  betas:")
        for k, v in result["betas"].items():
            print(f"    {k:20s} = {v:+.4f}")
        print(f"  trade-off ratios:")
        for k, v in result["trade_off_ratio"].items():
            print(f"    {k:20s} = {v:+.4f}")
