"""
Feature-augmented multinomial logit (baseline).

    V_k(x) = V_k^struct(x) + alpha * log q_k(x)

All parameters, the structural coefficients and the scalar alpha, are fitted
jointly by maximum likelihood on the training rows, to convergence
(baselines/mnl.py:fit_mle). The structural parameterization is that of the multinomial logit, so
cost monotonicity holds by construction; the trade-off ratios are computed
from the jointly fitted structural coefficients.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tabfm_logit.baselines.mnl import (  # noqa: E402
    MNL_SM, MNL_LPMC, MNL_IoT, fit_mle,
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
    seed: int = 42,
    bootstrap: bool = True,
    verbose: bool = False,
) -> FeatureAugmentedMNL:
    """Joint maximum-likelihood fit of every parameter on the training rows
    (``fit_mle``; the validation rows are used only for the verbose report).
    With ``bootstrap=True`` the training rows are resampled with replacement
    using ``seed``. How the fit ended is stored on the model (``fit_info``)."""
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

    Xd = torch.tensor(np.asarray(X_tr), dtype=torch.float64)
    qd = torch.tensor(np.asarray(fm_tr), dtype=torch.float64)
    it, g = fit_mle(model, list(model.parameters()), lambda: model(Xd, qd), y_tr)
    model.fit_info = {"method": "maximum likelihood, L-BFGS", "iterations": it, "max_abs_grad": g}

    if verbose:
        Xv = torch.tensor(X_vl, dtype=torch.float32)
        qv = torch.tensor(fm_vl, dtype=torch.float32)
        with torch.no_grad():
            preds = model.predict_proba(Xv, qv).argmax(dim=1).numpy()
        v_acc = float((preds == y_vl).mean())
        print(f"  feature-aug (maximum likelihood): {it} L-BFGS iterations, max |grad| {g:.1e}.  "
              f"val_acc={v_acc:.4f}  α={model.alpha.item():+.4f}")

    return model
