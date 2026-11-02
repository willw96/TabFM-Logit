"""
Behavioral audit of a raw foundation model (Table 1, Appendix C).

Refits the foundation model on the train+validation context
(fm/live_predict.py), runs the audit of tabfm_logit/audit.py on the test
rows with the model queried on every perturbed input, and writes
results/audits/audit_{fm}_{dataset}.json.

Usage:
    python -m tabfm_logit.audit_fm --model tabpfn --dataset swissmetro
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tabfm_logit.data import load_splits, get_Xy, DATASET_FEATURES  # noqa: E402
from tabfm_logit.audit import audit, CONFIGS as AUDIT_CONFIGS  # noqa: E402
from tabfm_logit.fm.live_predict import get_live_predict_fn  # noqa: E402

OUT_DIR = ROOT / "results" / "audits"


def run(model_name: str, dataset: str, save: bool = True):
    feats = DATASET_FEATURES[dataset]
    _, _, test = load_splits(dataset)
    X_test, _ = get_Xy(test, feats)
    cfg = AUDIT_CONFIGS[dataset]
    subject_ids = test[cfg.subject_col].values if cfg.is_panel else None
    print(f"  fitting {model_name} on {dataset} ...", flush=True)
    result = audit(get_live_predict_fn(model_name, dataset), dataset, X_test, subject_ids=subject_ids)
    result["model"] = model_name
    if save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUT_DIR / f"audit_{model_name}_{dataset}.json"
        path.write_text(json.dumps(result, indent=2, default=lambda x: float(x) if hasattr(x, "item") else x))
        print(f"  saved {path}")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["mitra", "tabpfn"])
    p.add_argument("--dataset", required=True, choices=["swissmetro", "lpmc", "iot_wearables"])
    p.add_argument("--no-save", action="store_true")
    a = p.parse_args()
    r = run(a.model, a.dataset, save=not a.no_save)
    print(f"  monotonicity {r['monotonicity_avg']:.4f}")
    for k, v in r.get("trade_off_ratios", {}).items():
        print(f"  trade-off {k}: median {v.get('median'):.4f}")
