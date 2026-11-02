"""Baselines and inputs: Appendix G table (TasteNet, knowledge distillation),
the convex ensemble and masked foundation model of Section 5, availability
leak (Section 5.1), and cross-fitted versus in-sample training-row q
(Appendix E).

    python paper/baselines_appendix.py
"""
import json

import numpy as np
import pandas as pd

from common import CACHED_DIR, CELLS, NAMES, PRED, RES, SEEDS, fm_npz, run_json, run_npz

if __name__ == "__main__":
    print("Appendix G table. TasteNet (mean +- sd over 5 replicates) and distillation (pure / mixed, 5 replicates).")
    kd = {(r["dataset"], r["fm"], r["mode"]): 100 * r["test_acc_mean"]
          for r in json.loads((RES / "baselines" / "distillation.json").read_text())}
    for ds in ("swissmetro", "lpmc", "iot_wearables"):
        acc = [100 * json.loads((RES / "tastenet" / f"tastenet_{ds}_seed{s}.json").read_text())["test_accuracy"]
               for s in range(5)]
        mnl = 100 * np.mean([run_json("tabfm_logit", ds, "mitra", s)["test_accuracy_stage1"] for s in SEEDS])
        kds = "   ".join(f"KD ({NAMES[fm]}) {kd[(ds, fm, 'pure')]:.1f} / {kd[(ds, fm, 'mixed')]:.1f}"
                         for fm in ("mitra", "tabpfn") if (ds, fm) in CELLS)
        print(f"  {NAMES[ds]:14s} MNL {mnl:.1f}   TasteNet {np.mean(acc):.1f} +- {np.std(acc, ddof=1):.1f}   {kds}")
    print("  (TasteNet ECE and trade-off ratios: see table1_headline.py)")

    print("\nConvex ensemble alpha * P_MNL + (1 - alpha) * P_FM, alpha fit by validation NLL on a 201-point grid.")
    grid = np.linspace(0, 1, 201)
    for ds, fm in CELLS:
        qv, qt = fm_npz(ds, fm, "val")["probs"], fm_npz(ds, fm, "test")
        yt, qt = qt["labels"], qt["probs"]
        al, acc = [], []
        for s in SEEDS:
            v, t = run_npz("tabfm_logit", ds, fm, s, "val"), run_npz("tabfm_logit", ds, fm, s, "test")
            pv, yv, pt = v["probs_stage1"], v["labels"], t["probs_stage1"]
            nll = [-np.mean(np.log(np.clip((a * pv + (1 - a) * qv)[np.arange(len(yv)), yv], 1e-12, None))) for a in grid]
            a = float(grid[int(np.argmin(nll))])
            al.append(a)
            acc.append(100 * float(((a * pt + (1 - a) * qt).argmax(1) == yt).mean()))
        raw = 100 * float((qt.argmax(1) == yt).mean())
        print(f"  {NAMES[ds] + ' / ' + NAMES[fm]:24s} max alpha {max(al):.3f}   accuracy {np.mean(acc):.2f} vs raw FM {raw:.2f}")

    print("\nSwissmetro availability: mean probability on unavailable alternatives over test rows that have one.")
    te = pd.read_parquet(CACHED_DIR / "swissmetro_test.parquet")
    av = te[["TRAIN_AV", "SM_AV", "CAR_AV"]].to_numpy().astype(bool)
    rows = (~av).any(1)
    print(f"  test rows with an unavailable alternative: {rows.sum()}")
    for fm in ("mitra", "tabpfn"):
        z = fm_npz("swissmetro", fm, "test")
        q, y = z["probs"], z["labels"]
        qm = q * av
        qm = qm / qm.sum(1, keepdims=True)
        own = max(float(((run_npz("tabfm_logit", "swissmetro", fm, s, "test")[k] * ~av).sum(1))[rows].mean())
                  for s in SEEDS for k in ("probs", "probs_stage1"))
        print(f"  {NAMES[fm]:7s} raw FM leak {((q * ~av).sum(1))[rows].mean():.1e}   MNL and TabFM-Logit at most {own:.0e}   "
              f"accuracy raw {100 * (q.argmax(1) == y).mean():.2f} masked {100 * (qm.argmax(1) == y).mean():.2f}")

    print("\nAppendix E. Training-row q accuracy (in-sample vs cross-fitted) and TabFM-Logit test accuracy trained on each.")
    for ds, fm in CELLS:
        qi, qx = fm_npz(ds, fm, "train", tag=""), fm_npz(ds, fm, "train")
        ai = np.mean([run_json("tabfm_logit", ds, fm, s, tag="")["test_accuracy"] for s in SEEDS])
        ax = np.mean([run_json("tabfm_logit", ds, fm, s)["test_accuracy"] for s in SEEDS])
        print(f"  {NAMES[ds] + ' / ' + NAMES[fm]:24s} q accuracy in-sample {100 * (qi['probs'].argmax(1) == qi['labels']).mean():.1f} "
              f"cross-fitted {100 * (qx['probs'].argmax(1) == qx['labels']).mean():.1f}   "
              f"TabFM-Logit {100 * ai:.2f} -> {100 * ax:.2f} ({100 * (ax - ai):+.1f} pp)")
