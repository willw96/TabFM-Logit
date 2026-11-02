"""
Calibration: expected calibration error (ECE) and three post-hoc calibrators.

ECE uses top-1 confidence in 15 equal-width bins. The calibrators are fit on
one set of predictions (the validation split in Table 1) and evaluated on
another:

    scalar_T   one temperature on log-probabilities (Guo et al., 2017)
    vector_T   one temperature per class
    isotonic   one isotonic map per class, rows renormalized
"""
from __future__ import annotations

import numpy as np


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> dict:
    """Top-1-confidence ECE. Returns {"ece", "max_ce", "n_total", "bins"}."""
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, max_ce, bins, n_total = 0.0, 0.0, [], int(len(probs))
    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        mask = (confidence >= lo) & ((confidence <= hi) if i == n_bins - 1 else (confidence < hi))
        n_bin = int(mask.sum())
        if n_bin == 0:
            bins.append({"lo": lo, "hi": hi, "n": 0})
            continue
        mean_conf, mean_acc = float(confidence[mask].mean()), float(correct[mask].mean())
        gap = abs(mean_conf - mean_acc)
        ece += (n_bin / n_total) * gap
        max_ce = max(max_ce, gap)
        bins.append({"lo": lo, "hi": hi, "n": n_bin, "mean_conf": mean_conf, "mean_acc": mean_acc})
    return {"ece": float(ece), "max_ce": float(max_ce), "n_total": n_total, "bins": bins}


def _nll(probs, labels):
    p = np.clip(probs[np.arange(len(labels)), labels], 1e-15, 1.0)
    return float(-np.log(p).mean())


def _scaled(probs, T):
    logp = np.log(np.clip(probs, 1e-15, 1.0)) / T
    logp = logp - logp.max(axis=1, keepdims=True)
    e = np.exp(logp)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(probs, labels, bracket=(0.05, 50.0)) -> float:
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(lambda T: 1e9 if T <= 0 else _nll(_scaled(probs, T), labels),
                          bounds=bracket, method="bounded", options={"xatol": 1e-4})
    return float(res.x)


def fit_vector_temperature(probs, labels) -> np.ndarray:
    from scipy.optimize import minimize
    res = minimize(lambda log_T: _nll(_scaled(probs, np.exp(log_T)[None, :]), labels),
                   np.zeros(probs.shape[1]), method="L-BFGS-B")
    return np.exp(res.x)


def fit_isotonic_per_class(probs, labels):
    from sklearn.isotonic import IsotonicRegression
    out = []
    for k in range(probs.shape[1]):
        ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        ir.fit(probs[:, k], (labels == k).astype(np.float64))
        out.append(ir)
    return out


def apply_isotonic(probs, isos) -> np.ndarray:
    out = np.zeros_like(probs)
    for k in range(probs.shape[1]):
        out[:, k] = isos[k].predict(probs[:, k])
    s = out.sum(axis=1, keepdims=True)
    bad = s.flatten() <= 1e-9
    out = np.where(s > 0, out / s, probs)
    if bad.any():
        out[bad] = probs[bad]
    return out


def calibrate_one_method(method, probs_eval, labels_eval, probs_fit, labels_fit, n_bins=15) -> dict:
    """Fit ``method`` on (probs_fit, labels_fit), apply it to probs_eval and
    return the ECE before ("ece_pre") and after ("ece_post")."""
    pre = expected_calibration_error(probs_eval, labels_eval, n_bins)
    if method == "scalar_T":
        post_probs = _scaled(probs_eval, fit_temperature(probs_fit, labels_fit))
    elif method == "vector_T":
        post_probs = _scaled(probs_eval, fit_vector_temperature(probs_fit, labels_fit)[None, :])
    elif method == "isotonic":
        post_probs = apply_isotonic(probs_eval, fit_isotonic_per_class(probs_fit, labels_fit))
    else:
        raise ValueError(f"unknown method: {method}")
    post = expected_calibration_error(post_probs, labels_eval, n_bins)
    return {"method": method, "ece_pre": pre["ece"], "ece_post": post["ece"]}


METHODS = ("scalar_T", "vector_T", "isotonic")


def best_post_hoc_ece(pairs) -> float:
    """Table 1 convention. ``pairs``: list of (probs_val, y_val, probs_test,
    y_test), one per replicate. For each method, fit on validation, evaluate
    on test, average over replicates; return the smallest average (in %)."""
    return 100 * min(np.mean([calibrate_one_method(m, pt, yt, pv, yv)["ece_post"]
                              for pv, yv, pt, yt in pairs]) for m in METHODS)
