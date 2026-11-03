"""
TabFM-Logit: a multinomial logit with a foundation-model correction.

    V_k(x) = V_k^struct(x) + g_k(q(x))

V^struct is the multinomial logit's utility (beta = -exp(theta) on time and
cost, plus any availability mask), q(x) is the foundation model's predicted
probability vector, computed once and loaded from disk, and g is a small
multilayer perceptron whose output layer is initialized at zero, so the
model starts as the plain multinomial logit.

Two-stage training:

    Stage 1  g held at zero; fit the structural coefficients by maximum
             likelihood (the multinomial logit), to convergence on the
             training rows (baselines/mnl.py:fit_mle).
    Stage 2  structural coefficients frozen; fit g on the training NLL with
             early stopping on validation NLL. A Stage 2 without g (only the
             scalar alpha, ablation A1) is a parametric fit and also uses
             fit_mle.

Because q is precomputed, g adds nothing to any attribute derivative, so
the trade-off ratios beta_attr / beta_cost are those of Stage 1
(Proposition 1). Training both stages jointly is available for ablation A3.

Ablation variants (see train.py):
    use_log_q=True adds a scalar alpha * log q_k term (A1, A2);
    use_correction=False drops g (A1);
    hidden_correction / n_correction_layers set g's capacity (A5).
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, log_loss

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tabfm_logit.baselines.mnl import (  # noqa: E402
    MNL_SM, MNL_LPMC, MNL_IoT, MAX_EPOCHS, fit_mle,
)


# Maps dataset name to (mnl_class, n_alts).
DATASET_MNL = {
    "swissmetro":      (MNL_SM,   3),
    "swissmetro_resp": (MNL_SM,   3),   # respondent-level split
    "lpmc":            (MNL_LPMC, 4),
    "lpmc_hh":         (MNL_LPMC, 4),   # household-level split
    "iot_wearables":   (MNL_IoT,  3),
}


class TabFMLogit(nn.Module):
    """A multinomial-logit specification plus an additive foundation-model correction.

    Args:
        mnl_cls: module class with utilities(X) -> (N, K) tensor.
        n_alts: number of alternatives K (input and output width of g).
        hidden_correction: hidden width of g (default 32).
        n_correction_layers: number of hidden layers of g (default 2).
        use_log_q: add a scalar alpha * log q_k term (ablations only).
        use_correction: include g. With use_log_q=False and
            use_correction=False the model is the plain multinomial logit.
    """

    def __init__(self, mnl_cls, n_alts,
                 hidden_correction: int = 32,
                 n_correction_layers: int = 2,
                 use_log_q: bool = False,
                 use_correction: bool = True):
        super().__init__()
        self.mnl = mnl_cls()
        self.n_alts = n_alts
        self.use_log_q = use_log_q
        self.use_correction = use_correction

        # alpha always exists but enters the utility only when use_log_q=True.
        self.alpha = nn.Parameter(torch.tensor(0.0))

        # Correction g. Only the output layer is zero-initialized, so g(q) = 0
        # at construction while the hidden layers (default initialization)
        # still receive gradient once the output layer moves in Stage 2.
        if use_correction:
            layers = []
            in_dim = n_alts
            for _ in range(n_correction_layers):
                layers.append(nn.Linear(in_dim, hidden_correction))
                layers.append(nn.ReLU())
                in_dim = hidden_correction
            output_layer = nn.Linear(in_dim, n_alts)
            layers.append(output_layer)
            self.correction = nn.Sequential(*layers)
            with torch.no_grad():
                output_layer.weight.zero_()
                output_layer.bias.zero_()
        else:
            # Empty placeholder so attribute access still works
            self.correction = nn.Sequential()

    # ── parameter groups for stage-wise training ────────────────────────────

    def utility_params(self):
        return list(self.mnl.parameters())

    def fm_params(self):
        params = []
        if self.use_log_q:
            params.append(self.alpha)
        if self.use_correction:
            params.extend(list(self.correction.parameters()))
        return params

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, X, fm_probs):
        """Return log-probabilities. fm_probs: (N, n_alts) tensor on the same device."""
        V = self.mnl.utilities(X)
        if self.use_log_q:
            fm_log = torch.log(fm_probs.clamp(min=1e-10))
            V = V + self.alpha * fm_log
        if self.use_correction:
            V = V + self.correction(fm_probs)
        return torch.log_softmax(V, dim=1)

    def predict_proba(self, X, fm_probs):
        with torch.no_grad():
            return torch.exp(self.forward(X, fm_probs))

    def predict_proba_np(self, X_np, fm_probs_np):
        X = torch.tensor(X_np, dtype=torch.float32)
        q = torch.tensor(fm_probs_np, dtype=torch.float32)
        return self.predict_proba(X, q).numpy()


# ──────────────────────────────────────────────────────────────────────────────
# Two-stage training loop
# ──────────────────────────────────────────────────────────────────────────────

def _train_stage(model, params, X, y, fm, X_v, y_v, fm_v, lr, epochs, patience):
    """Train one stage with NLL + early stopping on validation NLL."""
    opt = optim.Adam(params, lr=lr)
    crit = nn.NLLLoss()
    best_loss = float("inf")
    best_state = None
    wait = 0
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        loss = crit(model(X, fm), y)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            vloss = crit(model(X_v, fm_v), y_v).item()
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
    return model, epoch + 1


def _bootstrap_train(seed, X_tr, y_tr, fm_tr):
    rng = np.random.RandomState(seed)
    n = len(X_tr)
    idx = rng.choice(n, size=n, replace=True)
    return np.asarray(X_tr)[idx], np.asarray(y_tr)[idx], np.asarray(fm_tr)[idx]


def train_two_stage(
    dataset_name: str,
    X_tr, y_tr, fm_tr,
    X_vl, y_vl, fm_vl,
    lr2: float = 0.005,
    epochs2: int = MAX_EPOCHS,
    patience2: int = 30,
    seed: int = 42,
    bootstrap: bool = True,
    verbose: bool = False,
    use_log_q: bool = False,
    use_correction: bool = True,
    hidden_correction: int = 32,
    n_correction_layers: int = 2,
    mnl=None,
) -> TabFMLogit:
    """Two-stage training.

    Stage 1: the structural coefficients with g = 0, by maximum likelihood on
    the training rows, to convergence (``fit_mle``; validation rows unused).
    Stage 2: g with the structural coefficients frozen, Adam (full batch,
    learning rate ``lr2``), keeping the parameters with the lowest validation
    NLL and stopping after ``patience2`` epochs without improvement. Without
    g (ablation A1, only the scalar alpha) Stage 2 is also a maximum-likelihood
    fit. How each stage ended is stored on the model (``fit_info``).

    With ``bootstrap=True`` the training rows are resampled with replacement
    (indices from ``seed``); this gives the training-bootstrap replicates.
    Validation and test sets are never resampled. ``mnl=(class, n_alts)``
    supplies a structural specification not listed in DATASET_MNL.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if bootstrap:
        X_tr, y_tr, fm_tr = _bootstrap_train(seed, X_tr, y_tr, fm_tr)

    mnl_cls, n_alts = mnl if mnl is not None else DATASET_MNL[dataset_name]
    model = TabFMLogit(
        mnl_cls, n_alts,
        hidden_correction=hidden_correction,
        n_correction_layers=n_correction_layers,
        use_log_q=use_log_q,
        use_correction=use_correction,
    )

    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.long)
    qt = torch.tensor(fm_tr, dtype=torch.float32)
    Xv = torch.tensor(X_vl, dtype=torch.float32)
    yv = torch.tensor(y_vl, dtype=torch.long)
    qv = torch.tensor(fm_vl, dtype=torch.float32)

    # Stage 1: the multinomial logit by maximum likelihood (g held at zero).
    for p in model.fm_params():
        p.requires_grad_(False)
    for p in model.utility_params():
        p.requires_grad_(True)
    Xd, qd = Xt.double(), qt.double()
    it1, g1 = fit_mle(model, model.utility_params(), lambda: model(Xd, qd), yt)
    model.fit_info = {"stage1": {"method": "maximum likelihood, L-BFGS",
                                 "iterations": it1, "max_abs_grad": g1}}
    if verbose:
        with torch.no_grad():
            preds = model.predict_proba(Xv, qv).argmax(dim=1).numpy()
        s1_acc = float(accuracy_score(y_vl, preds))
        print(f"  stage 1 (maximum likelihood): {it1} L-BFGS iterations, max |grad| {g1:.1e}.  val_acc={s1_acc:.4f}")

    # Stage 2: fm-only. Freeze utility_params, train fm_params.
    fm_params = model.fm_params()
    if fm_params:
        for p in model.utility_params():
            p.requires_grad_(False)
        for p in fm_params:
            p.requires_grad_(True)
        if use_correction:
            model, ep2 = _train_stage(
                model, fm_params,
                Xt, yt, qt, Xv, yv, qv,
                lr=lr2, epochs=epochs2, patience=patience2,
            )
            model.fit_info["stage2"] = {"method": "Adam, early stopping on validation NLL", "epochs": ep2}
        else:
            ep2, g2 = fit_mle(model, fm_params, lambda: model(Xd, qd), yt)
            model.fit_info["stage2"] = {"method": "maximum likelihood, L-BFGS",
                                        "iterations": ep2, "max_abs_grad": g2}
        if verbose:
            with torch.no_grad():
                preds = model.predict_proba(Xv, qv).argmax(dim=1).numpy()
            s2_acc = float(accuracy_score(y_vl, preds))
            print(f"  stage 2: {model.fit_info['stage2']}.  val_acc={s2_acc:.4f}  α={model.alpha.item():+.4f}")
    else:
        if verbose:
            print("  stage 2 skipped (no foundation-model parameters)")

    return model


def train_joint(
    dataset_name: str,
    X_tr, y_tr, fm_tr,
    X_vl, y_vl, fm_vl,
    lr: float = 0.01,
    epochs: int = MAX_EPOCHS,
    patience: int = 50,
    seed: int = 42,
    bootstrap: bool = True,
    verbose: bool = False,
    use_log_q: bool = False,
    use_correction: bool = True,
    hidden_correction: int = 32,
    n_correction_layers: int = 2,
    mnl=None,
) -> TabFMLogit:
    """Joint single-stage training of all parameters (ablation A3): same
    architecture and likelihood, structural and correction parameters
    trained together. See Proposition 2.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if bootstrap:
        X_tr, y_tr, fm_tr = _bootstrap_train(seed, X_tr, y_tr, fm_tr)

    mnl_cls, n_alts = mnl if mnl is not None else DATASET_MNL[dataset_name]
    model = TabFMLogit(
        mnl_cls, n_alts,
        hidden_correction=hidden_correction,
        n_correction_layers=n_correction_layers,
        use_log_q=use_log_q,
        use_correction=use_correction,
    )

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
    model.fit_info = {"joint": {"method": "Adam, early stopping on validation NLL", "epochs": epoch + 1}}

    if verbose:
        with torch.no_grad():
            preds = model.predict_proba(Xv, qv).argmax(dim=1).numpy()
        v_acc = float((preds == y_vl).mean())
        print(f"  joint training done in {epoch + 1} epochs.  val_acc={v_acc:.4f}  α={model.alpha.item():+.4f}")

    return model


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1 evaluation: the multinomial logit is the trained model with alpha
# and g set to zero.
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_stage1_only(model: TabFMLogit, X_te, y_te, fm_te) -> dict:
    """Evaluate the model with the FM correction silenced.
    Returns dict with test_accuracy, test_nll, and probs (the (N, K) array)."""
    saved_alpha = model.alpha.item()
    saved_state = {k: v.clone() for k, v in model.correction.state_dict().items()}
    with torch.no_grad():
        model.alpha.fill_(0.0)
        for p in model.correction.parameters():
            p.zero_()
        probs = model.predict_proba_np(X_te, fm_te)
    preds = probs.argmax(axis=1)
    acc = float((preds == y_te).mean())
    nll = float(-np.log(np.clip(probs[np.arange(len(y_te)), y_te], 1e-15, 1.0)).mean())
    # Restore
    with torch.no_grad():
        model.alpha.fill_(saved_alpha)
        model.correction.load_state_dict(saved_state)
    return {"test_accuracy": acc, "test_nll": nll, "probs": probs}
