"""
Feature-augmented multinomial logit (baseline).

    V_k(x) = V_k^struct(x) + alpha * log q_k(x)

All parameters, the structural coefficients and the scalar alpha, are fitted
jointly by maximum likelihood (Adam, full batch, early stopping on validation
NLL). The structural parameterization is that of the multinomial logit, so
cost monotonicity holds by construction; the trade-off ratios are computed
from the jointly fitted structural coefficients.
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

from tabfm_logit.baselines.mnl import (  # noqa: E402
    MNL_SM, MNL_LPMC, MNL_IoT,
)


DATASET_MNL = {
    "swissmetro": (MNL_SM, 3), "swissmetro_resp": (MNL_SM, 3),
    "lpmc": (MNL_LPMC, 4), "lpmc_hh": (MNL_LPMC, 4),
    "iot_wearables": (MNL_IoT, 3),
}


class FeatureAugmentedMNL(nn.Module):
    """Multinomial logit with an additive alpha * log q_k term, fitted jointly."""

    def __init__(self, mnl_cls, n_alts: int):
        super().__init__()
        self.mnl = mnl_cls()
        self.n_alts = n_alts
        # alpha starts at 0, so training starts from the multinomial logit.
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def utilities(self, X, fm_probs):
        V = self.mnl.utilities(X)
        log_q = torch.log(fm_probs.clamp(min=1e-10))
        return V + self.alpha * log_q

    def forward(self, X, fm_probs):
        return torch.log_softmax(self.utilities(X, fm_probs), dim=1)

    def predict_proba(self, X, fm_probs):
        with torch.no_grad():
            return torch.exp(self.forward(X, fm_probs))

    def predict_proba_np(self, X_np, fm_probs_np):
        X = torch.tensor(X_np, dtype=torch.float32)
        q = torch.tensor(fm_probs_np, dtype=torch.float32)
        return self.predict_proba(X, q).numpy()

    def get_betas(self):
        b = self.mnl.get_betas()
        b["alpha"] = float(self.alpha.item())
        return b

    def get_trade_off_ratio(self):
        # From the jointly fitted structural coefficients.
        return self.mnl.get_trade_off_ratio()


def train_feature_aug(
    dataset_name: str,
    X_tr, y_tr, fm_tr,
    X_vl, y_vl, fm_vl,
    lr: float = 0.01,
    epochs: int = 1000,
    patience: int = 50,
    seed: int = 42,
    bootstrap: bool = True,
    verbose: bool = False,
) -> FeatureAugmentedMNL:
    """Joint training (Adam, full batch, up to 1,000 epochs, early stopping on
    validation NLL with patience 50). With ``bootstrap=True`` the training
    rows are resampled with replacement using ``seed``."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if bootstrap:
        rng = np.random.RandomState(seed)
        n = len(X_tr)
        idx = rng.choice(n, size=n, replace=True)
        X_tr = np.asarray(X_tr)[idx]
        y_tr = np.asarray(y_tr)[idx]
        fm_tr = np.asarray(fm_tr)[idx]

    mnl_cls, n_alts = DATASET_MNL[dataset_name]
    model = FeatureAugmentedMNL(mnl_cls, n_alts)

    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.long)
    qt = torch.tensor(fm_tr, dtype=torch.float32)
    Xv = torch.tensor(X_vl, dtype=torch.float32)
    yv = torch.tensor(y_vl, dtype=torch.long)
    qv = torch.tensor(fm_vl, dtype=torch.float32)

    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.NLLLoss()

    best_loss, best_state, wait = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        loss = crit(model(Xt, qt), yt)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            vloss = crit(model(Xv, qv), yv).item()
        if vloss < best_loss - 1e-6:
            best_loss = vloss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    if verbose:
        with torch.no_grad():
            preds = model.predict_proba(Xv, qv).argmax(dim=1).numpy()
        v_acc = float((preds == y_vl).mean())
        print(f"  feature-aug done in {epoch + 1} epochs.  "
              f"val_acc={v_acc:.4f}  α={model.alpha.item():+.4f}")

    return model
