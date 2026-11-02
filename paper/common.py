"""Shared paths, cells and helpers for the scripts that rebuild the paper's
tables and numbers from the saved outputs in results/."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tabfm_logit.data import CACHED_DIR, GROUP_COL  # noqa: E402
from tabfm_logit.calibration import calibrate_one_method, METHODS  # noqa: E402

RES = ROOT / "results"
PRED = RES / "predictions"
RUNS = RES / "runs"
FA = RES / "feature_aug"
SEEDS = range(10)
B = 2000  # bootstrap resamples

# (dataset, foundation model) cells reported in the paper.
CELLS = [("swissmetro", "mitra"), ("swissmetro", "tabpfn"), ("lpmc", "mitra"),
         ("iot_wearables", "mitra"), ("iot_wearables", "tabpfn")]
# Grouped split of each dataset (IoT-Wearables is split by subject throughout).
GROUPED = {"swissmetro": "swissmetro_resp", "lpmc": "lpmc_hh"}
NAMES = {"swissmetro": "Swissmetro", "lpmc": "LPMC", "iot_wearables": "IoT-Wearables",
         "swissmetro_resp": "Swissmetro (respondent split)", "lpmc_hh": "LPMC (household split)",
         "mitra": "Mitra", "tabpfn": "TabPFN"}


def stem(variant, ds, fm, s, tag="xfit5"):
    return f"{variant}_{ds}_{fm}{'_' + tag if tag else ''}_seed{s}"


def run_json(variant, ds, fm, s, tag="xfit5"):
    return json.loads((RUNS / f"{stem(variant, ds, fm, s, tag)}.json").read_text())


def run_npz(variant, ds, fm, s, split, tag="xfit5"):
    return np.load(PRED / f"{stem(variant, ds, fm, s, tag)}_{split}.npz")


def fm_npz(ds, fm, split, tag="xfit5"):
    return np.load(PRED / f"{fm}{'_' + tag if tag else ''}_{ds}_{split}.npz")


def groups(ds):
    """Cluster labels of the test rows for clustered intervals (None: rows)."""
    if ds == "iot_wearables":
        return pd.read_parquet(CACHED_DIR / "iot_wearables_test.parquet")["subject"].to_numpy()
    if ds in GROUP_COL:
        return pd.read_parquet(CACHED_DIR / f"{ds}_test.parquet")[GROUP_COL[ds]].to_numpy()
    return None


def ci(x):
    return [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))]


def boot_mean(v, g, rng):
    """95% bootstrap interval of mean(v), resampling rows or (if g is given)
    clusters, B resamples drawn from ``rng``."""
    v = np.asarray(v, dtype=float)
    if g is None:
        n = len(v)
        return ci([v[rng.integers(0, n, n)].mean() for _ in range(B)])
    _, inv = np.unique(g, return_inverse=True)
    s, c = np.bincount(inv, weights=v), np.bincount(inv).astype(float)
    G = len(s)
    out = np.empty(B)
    for b in range(B):
        i = rng.integers(0, G, G)
        out[b] = s[i].sum() / c[i].sum()
    return ci(out)


def ece_best(pairs):
    """Table 1 ECE convention: best of the three post-hoc calibrators, each fit
    on validation and evaluated on test, averaged over replicates (in %)."""
    return 100 * min(np.mean([calibrate_one_method(m, pt, yt, pv, yv)["ece_post"]
                              for pv, yv, pt, yt in pairs]) for m in METHODS)


def pp(x, d=1):
    return f"{x:+.{d}f}"


def interval(c):
    """Endpoints to one decimal; endpoints within 0.05 of zero to three."""
    f = lambda x: f"{x:+.1f}" if abs(x) >= 0.05 else f"{x:+.3f}"  # noqa: E731
    return f"[{f(c[0])}, {f(c[1])}]"
