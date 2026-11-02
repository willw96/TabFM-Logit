"""
Behavioral audit.

For any model exposed as a callable `predict_fn(X) -> probs` (X of shape
(N, D), probs of shape (N, K)) the audit computes:

  * Monotonicity rate. For each (cost-column, alternative) pair declared for
    the dataset, perturb the cost up by 1 percent of its observed range and
    check whether the predicted probability of that alternative falls. Report
    the fraction of test observations satisfying this, averaged across pairs.
    For the panel dataset (IoT-Wearables) the per-row monotonicity
    indicators are aggregated per subject before averaging,
    so each chooser counts once regardless of how many choice tasks they
    contributed.

  * Trade-off ratio. Finite-difference willingness to pay for each
    non-cost attribute, computed per row as -(ΔP/Δattr) / (ΔP/Δcost) and
    summarized via median + IQR + fraction-negative + fraction-implausible.
    Models that expose coefficients (multinomial logit, two-stage adapter
    Stage 1) can also report the analytical ratio; this module reports the
    finite-difference version uniformly for cross-model comparability.

  * Availability leak. Mean predicted probability assigned to alternatives
    that are flagged unavailable in the row's availability columns. Reported
    only for Swissmetro (the only dataset with availability constraints).

Sample size note: monotonicity is averaged across (cost, alt) pairs after
clustering on subject for panel data. Trade-off ratios are reported per
attribute, not aggregated.

CLI (audits a standalone multinomial logit):
    python -m tabfm_logit.audit --dataset swissmetro
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tabfm_logit.data import (  # noqa: E402
    load_splits, get_Xy,
    SM_FEATURES, LPMC_FEATURES, IOT_WEARABLES_FEATURES,
    DATASET_FEATURES, DATASET_N_ALTS,
)


# ──────────────────────────────────────────────────────────────────────────────
# Per-dataset audit configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TradeOffSpec:
    """One trade-off ratio to compute, e.g. WTP for travel time on alt 'pt'.

    The `convention` field controls the sign:
      "wtp"         : ratio = -(ΔP/Δattr) / (ΔP/Δcost). Standard willingness-to-pay
                      for one more unit of attribute. Positive for liked attributes
                      (β_attr > 0 with β_cost < 0), negative for disliked.
      "vot_savings" : ratio = +(ΔP/Δattr) / (ΔP/Δcost). Value-of-savings convention,
                      used for transport travel-time vs travel-cost. Positive when
                      the consumer values reductions in the attribute.

    The `kind` field controls how the attribute is perturbed:
      "continuous_pct" : (default) perturb by 1% of the column's observed range.
                          Right for continuous attributes such as travel time.
      "discrete_flip"  : binary indicator, perturb by flipping (0 → 1 or 1 → 0).
                          Per-row delta is +1 if the row had a=0 and -1 if a=1,
                          so the rate dP/dattr is consistent across rows.
      "unit_jump"      : ordinal, perturb by adding 1.0 to the current value.
                          Right for small-integer attributes such as a contract
                          length in years where a 1% perturbation rounds to noise.
    """
    name: str         # human-readable label, e.g. "vot_pt"
    attr_col: int     # index in the feature matrix
    cost_col: int     # index of the corresponding cost column for the same alt
    alt_idx: int      # which alternative this trade-off pertains to
    unit: str = ""    # currency / time-unit for reporting
    convention: str = "wtp"  # "wtp" or "vot_savings"
    kind: str = "continuous_pct"  # "continuous_pct" | "discrete_flip" | "unit_jump"


@dataclass
class AuditConfig:
    name: str
    n_alts: int
    feature_names: list[str]
    cost_alt_pairs: list[tuple[int, int]]      # (column_index, alternative_index)
    trade_offs: list[TradeOffSpec]
    availability_alt_pairs: list[tuple[int, int]] = field(default_factory=list)
    is_panel: bool = False
    subject_col: str = ""                       # column name in the original DataFrame


def _idx(features, name):
    return features.index(name)


def _swissmetro_config():
    f = SM_FEATURES
    return AuditConfig(
        name="swissmetro",
        n_alts=3,
        feature_names=f,
        cost_alt_pairs=[
            (_idx(f, "TRAIN_CO"), 0),
            (_idx(f, "SM_CO"), 1),
            (_idx(f, "CAR_CO"), 2),
        ],
        trade_offs=[
            TradeOffSpec(
                name="vot_train", attr_col=_idx(f, "TRAIN_TT"),
                cost_col=_idx(f, "TRAIN_CO"), alt_idx=0,
                unit="CHF/hr (×60 from per-minute)", convention="vot_savings",
            ),
            TradeOffSpec(
                name="vot_sm", attr_col=_idx(f, "SM_TT"),
                cost_col=_idx(f, "SM_CO"), alt_idx=1,
                unit="CHF/hr (×60 from per-minute)", convention="vot_savings",
            ),
            TradeOffSpec(
                name="vot_car", attr_col=_idx(f, "CAR_TT"),
                cost_col=_idx(f, "CAR_CO"), alt_idx=2,
                unit="CHF/hr (×60 from per-minute)", convention="vot_savings",
            ),
        ],
        availability_alt_pairs=[
            (_idx(f, "TRAIN_AV"), 0),
            (_idx(f, "SM_AV"), 1),
            (_idx(f, "CAR_AV"), 2),
        ],
        is_panel=False,
        subject_col="",
    )


def _lpmc_config():
    f = LPMC_FEATURES
    return AuditConfig(
        name="lpmc",
        n_alts=4,
        feature_names=f,
        cost_alt_pairs=[
            (_idx(f, "cost_transit"), 2),
            (_idx(f, "cost_driving"), 3),
        ],
        trade_offs=[
            TradeOffSpec(
                name="vot_pt", attr_col=_idx(f, "dur_pt_total"),
                cost_col=_idx(f, "cost_transit"), alt_idx=2,
                unit="GBP/hr", convention="vot_savings",
            ),
            TradeOffSpec(
                name="vot_drive", attr_col=_idx(f, "dur_driving"),
                cost_col=_idx(f, "cost_driving"), alt_idx=3,
                unit="GBP/hr", convention="vot_savings",
            ),
        ],
        availability_alt_pairs=[],
        is_panel=False,
        subject_col="",
    )


def _iot_wearables_config():
    f = IOT_WEARABLES_FEATURES
    cost_alt_pairs = [
        (_idx(f, "price1"), 0),
        (_idx(f, "price2"), 1),
        (_idx(f, "price3"), 2),
    ]
    # function and label are binary {0, 1} indicators.
    trade_offs = []
    for attr_prefix, label in [("function", "wtp_function"), ("label", "wtp_label")]:
        for i in range(3):
            trade_offs.append(TradeOffSpec(
                name=f"{label}_alt{i+1}",
                attr_col=_idx(f, f"{attr_prefix}{i+1}"),
                cost_col=_idx(f, f"price{i+1}"),
                alt_idx=i,
                unit="USD",
                kind="discrete_flip",
            ))
    return AuditConfig(
        name="iot_wearables",
        n_alts=3,
        feature_names=f,
        cost_alt_pairs=cost_alt_pairs,
        trade_offs=trade_offs,
        availability_alt_pairs=[],
        is_panel=True,
        subject_col="subject",
    )


CONFIGS = {
    "swissmetro": _swissmetro_config(),
    "swissmetro_resp": _swissmetro_config(),   # same columns; respondent-level split
    "lpmc": _lpmc_config(),
    "lpmc_hh": _lpmc_config(),   # same columns; household-level split
    "iot_wearables": _iot_wearables_config(),
}


# ──────────────────────────────────────────────────────────────────────────────
# Audit primitives
# ──────────────────────────────────────────────────────────────────────────────

def _delta_for(X, col, frac=0.01):
    """1 percent of feature range, with a floor for near-constant columns."""
    rng = float(X[:, col].max() - X[:, col].min())
    return max(frac * rng, 1e-6)


def compute_monotonicity(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    cost_col: int,
    alt_idx: int,
    delta_frac: float = 0.01,
    subject_ids: np.ndarray | None = None,
    prob_threshold: float = 0.01,
) -> dict:
    """Perturb cost up; check predicted probability of alt_idx falls.

    Returns the per-row indicator array, the unweighted observation-level rate,
    and (when subject_ids is provided) the cluster-aware subject-level rate.
    """
    X = np.asarray(X, dtype=np.float32)
    delta = _delta_for(X, cost_col, delta_frac)

    probs_base = predict_fn(X)
    # Restrict to rows where (a) the alternative has non-trivial mass, and
    # (b) the cost column has a non-zero value (so a 1 percent perturbation is meaningful)
    mask = (probs_base[:, alt_idx] > prob_threshold) & (X[:, cost_col] > 0)
    if mask.sum() == 0:
        return {
            "monotonicity_rate": float("nan"),
            "n_tested": 0,
            "n_monotonic": 0,
            "subject_rate_mean": float("nan"),
            "n_subjects": 0,
        }

    X_pert = X.copy()
    X_pert[:, cost_col] += delta
    probs_pert = predict_fn(X_pert)

    # Allow a tiny numerical tolerance
    is_monotonic = probs_pert[:, alt_idx] <= probs_base[:, alt_idx] + 1e-8
    is_monotonic = np.where(mask, is_monotonic, np.nan)

    # Observation-level rate (drop NaN rows)
    valid = ~np.isnan(is_monotonic)
    obs_rate = float(np.nanmean(is_monotonic)) if valid.sum() else float("nan")

    out = {
        "monotonicity_rate": obs_rate,
        "n_tested": int(valid.sum()),
        "n_monotonic": int(np.nansum(is_monotonic)),
    }

    if subject_ids is not None:
        subj = np.asarray(subject_ids)
        df = pd.DataFrame({
            "subject": subj,
            "is_mono": is_monotonic,
        }).dropna()
        # Each subject's rate = fraction of their tested rows that are monotonic
        per_subj = df.groupby("subject")["is_mono"].mean()
        out["subject_rate_mean"] = float(per_subj.mean()) if len(per_subj) else float("nan")
        out["n_subjects"] = int(len(per_subj))

    return out


def compute_trade_off_fd(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    attr_col: int,
    cost_col: int,
    alt_idx: int,
    delta_frac: float = 0.01,
    prob_threshold: float = 0.01,
    implausible_abs_max: float = 1e3,
    convention: str = "wtp",
    kind: str = "continuous_pct",
) -> dict:
    """Finite-difference trade-off ratio.

    Convention "wtp"         : ratio = -(ΔP_k/Δattr) / (ΔP_k/Δcost)
    Convention "vot_savings" : ratio = +(ΔP_k/Δattr) / (ΔP_k/Δcost)

    Kind controls the attribute perturbation:
      "continuous_pct" : (default) attribute perturbed by 1% of its observed range.
      "discrete_flip"  : binary {0,1} indicator flipped per row; per-row delta is
                          (1 - 2·X[i, attr_col]) so a 0→1 flip gives +1 and a 1→0
                          flip gives -1, yielding a consistent dP/dattr sign.
      "unit_jump"      : ordinal attribute incremented by 1.0 per row.

    Per-row ratio summarized via median, IQR (25th, 75th percentile),
    fraction-negative (sign-flip indicator), and fraction-implausible
    (|ratio| > implausible_abs_max).
    """
    if convention not in ("wtp", "vot_savings"):
        raise ValueError(f"Unknown convention: {convention}")
    if kind not in ("continuous_pct", "discrete_flip", "unit_jump"):
        raise ValueError(f"Unknown kind: {kind}")

    X = np.asarray(X, dtype=np.float32)
    delta_cost = _delta_for(X, cost_col, delta_frac)

    probs_base = predict_fn(X)
    # Only require non-trivial probability for the alternative — don't filter on the
    # current values of attr or cost. Binary attributes (e.g. label) often
    # have value 0 in the data; that doesn't make the partial derivative undefined.
    mask_keep = probs_base[:, alt_idx] > prob_threshold
    if mask_keep.sum() == 0:
        return {
            "median": float("nan"),
            "mean": float("nan"),
            "iqr_25": float("nan"),
            "iqr_75": float("nan"),
            "frac_negative": float("nan"),
            "frac_implausible": float("nan"),
            "n_valid": 0,
        }

    X_attr = X.copy()
    if kind == "continuous_pct":
        delta_attr_scalar = _delta_for(X, attr_col, delta_frac)
        X_attr[:, attr_col] += delta_attr_scalar
        p_attr = predict_fn(X_attr)
        dP_dattr = (p_attr[:, alt_idx] - probs_base[:, alt_idx]) / delta_attr_scalar
    elif kind == "discrete_flip":
        # Flip binary indicator. Per-row signed delta: +1 if was 0, -1 if was 1.
        X_attr[:, attr_col] = 1.0 - X_attr[:, attr_col]
        p_attr = predict_fn(X_attr)
        delta_per_row = 1.0 - 2.0 * X[:, attr_col]
        # Avoid division by zero on rows whose attr is neither 0 nor 1.
        eps_d = 1e-12
        safe_delta = np.abs(delta_per_row) > eps_d
        dP_dattr = np.full_like(probs_base[:, alt_idx], np.nan)
        dP_dattr[safe_delta] = (
            (p_attr[safe_delta, alt_idx] - probs_base[safe_delta, alt_idx])
            / delta_per_row[safe_delta]
        )
    else:  # "unit_jump"
        X_attr[:, attr_col] += 1.0
        p_attr = predict_fn(X_attr)
        dP_dattr = p_attr[:, alt_idx] - probs_base[:, alt_idx]

    X_cost = X.copy()
    X_cost[:, cost_col] += delta_cost
    p_cost = predict_fn(X_cost)
    dP_dcost = (p_cost[:, alt_idx] - probs_base[:, alt_idx]) / delta_cost

    # Avoid division by ~0 in the cost derivative; also drop rows whose
    # dP_dattr is NaN (discrete_flip can produce NaN when a row's attr is
    # neither 0 nor 1, which shouldn't happen with binary inputs but is
    # defended-against).
    eps = 1e-10
    safe = mask_keep & (np.abs(dP_dcost) > eps) & ~np.isnan(dP_dattr)

    sign = -1.0 if convention == "wtp" else +1.0
    ratios = np.full_like(probs_base[:, alt_idx], np.nan)
    ratios[safe] = sign * dP_dattr[safe] / dP_dcost[safe]

    valid = ~np.isnan(ratios)
    if valid.sum() == 0:
        return {
            "median": float("nan"), "mean": float("nan"),
            "iqr_25": float("nan"), "iqr_75": float("nan"),
            "frac_negative": float("nan"), "frac_implausible": float("nan"),
            "n_valid": 0,
        }

    r = ratios[valid]
    return {
        "median": float(np.median(r)),
        "mean": float(np.mean(r[np.abs(r) <= implausible_abs_max])) if np.any(np.abs(r) <= implausible_abs_max) else float("nan"),
        "iqr_25": float(np.percentile(r, 25)),
        "iqr_75": float(np.percentile(r, 75)),
        "frac_negative": float((r < 0).mean()),
        "frac_implausible": float((np.abs(r) > implausible_abs_max).mean()),
        "n_valid": int(valid.sum()),
    }


def compute_availability_leak(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    avail_col: int,
    alt_idx: int,
) -> dict:
    """Mean predicted probability assigned to alt_idx among rows where it is
    flagged unavailable (avail_col == 0).

    We pass the full X to predict_fn (never a subset) and filter the resulting
    probability matrix by the unavailable mask. This keeps predict_fn callable
    even when it has bound auxiliary arrays (like the foundation-model
    probabilities) of the same row count as X.
    """
    X = np.asarray(X, dtype=np.float32)
    unavail = X[:, avail_col] == 0
    if unavail.sum() == 0:
        return {"mean_prob_unavail": float("nan"), "max_prob_unavail": float("nan"), "n_unavail": 0}
    probs = predict_fn(X)
    leak_probs = probs[unavail, alt_idx]
    return {
        "mean_prob_unavail": float(leak_probs.mean()),
        "max_prob_unavail": float(leak_probs.max()),
        "n_unavail": int(unavail.sum()),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Top-level audit
# ──────────────────────────────────────────────────────────────────────────────

def audit(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    dataset_name: str,
    X_test: np.ndarray,
    subject_ids: np.ndarray | None = None,
    delta_frac: float = 0.01,
) -> dict:
    """Run the full audit for a model on a dataset's test set."""
    if dataset_name not in CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    cfg = CONFIGS[dataset_name]

    if cfg.is_panel and subject_ids is None:
        raise ValueError(
            f"Dataset '{dataset_name}' is panel data; subject_ids must be supplied."
        )

    out = {
        "dataset": dataset_name,
        "n_alts": cfg.n_alts,
        "n_test_rows": int(len(X_test)),
        "is_panel": cfg.is_panel,
    }

    # ── Monotonicity per (cost, alt) pair ────────────────────────────────────
    mono_results = []
    for cost_col, alt_idx in cfg.cost_alt_pairs:
        r = compute_monotonicity(
            predict_fn, X_test, cost_col, alt_idx,
            delta_frac=delta_frac,
            subject_ids=subject_ids if cfg.is_panel else None,
        )
        r["cost_col_name"] = cfg.feature_names[cost_col]
        r["alt_idx"] = alt_idx
        mono_results.append(r)
    out["monotonicity_per_pair"] = mono_results

    # Aggregate. For panel data we report the subject-clustered mean; otherwise observation-level.
    rate_key = "subject_rate_mean" if cfg.is_panel else "monotonicity_rate"
    rates = [m[rate_key] for m in mono_results if not np.isnan(m[rate_key])]
    out["monotonicity_avg"] = float(np.mean(rates)) if rates else float("nan")

    # ── Trade-off ratios ─────────────────────────────────────────────────────
    trade_offs = {}
    for spec in cfg.trade_offs:
        r = compute_trade_off_fd(
            predict_fn, X_test, spec.attr_col, spec.cost_col, spec.alt_idx,
            delta_frac=delta_frac,
            convention=spec.convention,
            kind=spec.kind,
        )
        r["unit"] = spec.unit
        r["alt_idx"] = spec.alt_idx
        r["convention"] = spec.convention
        r["kind"] = spec.kind
        trade_offs[spec.name] = r
    out["trade_off_ratios"] = trade_offs

    # ── Availability leak ────────────────────────────────────────────────────
    if cfg.availability_alt_pairs:
        leaks = {}
        for avail_col, alt_idx in cfg.availability_alt_pairs:
            leaks[cfg.feature_names[avail_col]] = compute_availability_leak(
                predict_fn, X_test, avail_col, alt_idx
            )
        out["availability_leaks"] = leaks
        # Headline: max over alternatives, mean prob to the unavailable alt
        max_leak = max(
            (l["mean_prob_unavail"] for l in leaks.values()
             if not np.isnan(l["mean_prob_unavail"])),
            default=float("nan"),
        )
        out["availability_leak_max"] = max_leak

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: build an MNL predict_fn from a trained torch model
# ──────────────────────────────────────────────────────────────────────────────

def predict_fn_from_torch_model(model) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap a torch.nn.Module exposing predict_proba(X_tensor) into a
    numpy-in / numpy-out predict_fn(X)."""
    import torch

    def _fn(X):
        with torch.no_grad():
            return model.predict_proba(torch.tensor(X, dtype=torch.float32)).numpy()

    return _fn


def audit_mnl(dataset_name: str, save: bool = True, verbose: bool = True) -> dict:
    """Train an MNL on the given dataset, audit it, optionally save JSON."""
    from tabfm_logit.baselines.mnl import run_mnl

    metrics, model = run_mnl(dataset_name, verbose=False)
    train, val, test = load_splits(dataset_name)
    feats = DATASET_FEATURES[dataset_name]
    X_test, _ = get_Xy(test, feats)

    cfg = CONFIGS[dataset_name]
    subject_ids = test[cfg.subject_col].values if cfg.is_panel else None

    predict_fn = predict_fn_from_torch_model(model)
    result = audit(predict_fn, dataset_name, X_test, subject_ids=subject_ids)

    # Attach the analytical (model-derived) trade-off ratio for cross-comparison
    result["model_test_accuracy"] = metrics["test_accuracy"]
    result["model_test_nll"] = metrics["test_nll"]
    result["analytical_trade_off"] = metrics["trade_off_ratio"]
    result["analytical_betas"] = metrics["betas"]

    if save:
        out_dir = ROOT / "results" / "audits"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"audit_mnl_{dataset_name}.json"
        with open(path, "w") as f:
            json.dump(result, f, indent=2, default=lambda x: float(x) if hasattr(x, "item") else x)
        if verbose:
            print(f"  saved {path}")

    return result


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _print_summary(result: dict):
    name = result["dataset"]
    print(f"  dataset           = {name}")
    print(f"  n_test_rows       = {result['n_test_rows']}")
    print(f"  is_panel          = {result['is_panel']}")
    print(f"  monotonicity_avg  = {result['monotonicity_avg']:.4f}  ({'subject-clustered' if result['is_panel'] else 'row-level'})")
    if "availability_leak_max" in result:
        print(f"  availability_leak = {result['availability_leak_max']:.6e}")
    if "model_test_accuracy" in result:
        print(f"  model accuracy    = {result['model_test_accuracy']:.4f}")
    print(f"  trade-off ratios (median, finite-difference):")
    for k, v in result["trade_off_ratios"].items():
        print(f"    {k:30s} median={v['median']:+8.3f} iqr=[{v['iqr_25']:+.2f}, {v['iqr_75']:+.2f}] frac_neg={v['frac_negative']:.2f}")
    if "analytical_trade_off" in result:
        print(f"  analytical trade-off ratios (from model coefficients):")
        for k, v in result["analytical_trade_off"].items():
            print(f"    {k:30s} = {v:+.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Audit MNL baseline behavioral validity per dataset.")
    p.add_argument("--dataset", choices=["swissmetro", "lpmc", "iot_wearables", "all"], default="all")
    p.add_argument("--no-save", action="store_true", help="Skip writing JSON.")
    args = p.parse_args()

    targets = ["swissmetro", "lpmc", "iot_wearables"] if args.dataset == "all" else [args.dataset]
    for ds in targets:
        print(f"\n{'='*70}\n  AUDIT: MNL on {ds.upper()}\n{'='*70}")
        result = audit_mnl(ds, save=not args.no_save, verbose=True)
        _print_summary(result)
