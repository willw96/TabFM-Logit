"""Table 2 (accuracy on new individuals): the grouped splits (Swissmetro by
respondent, LPMC by household; IoT-Wearables by subject throughout), with
group-cross-fitted training-row q and intervals that resample test groups.

    python paper/table2_grouped.py
"""
import numpy as np

from common import CELLS, NAMES, interval, pp
from intervals import cell

if __name__ == "__main__":
    rows = {(ds, fm): cell(ds, fm, grouped=True) for ds, fm in CELLS}
    print("Table 2. Test accuracy (%) on the grouped splits; columns: "
          + " | ".join(f"{NAMES[d]} {NAMES[f]}" for d, f in CELLS))
    for label, key in (("MNL", "acc_mnl"), ("raw FM", "acc_raw"), ("feat-aug", "acc_fa"), ("TabFM-Logit", "acc_tabfm")):
        print(f"  {label:22s} " + " ".join(f"{rows[c][key]:>14.1f}" for c in CELLS))
    print(f"  {'TabFM-Logit - MNL':22s} " + " ".join(f"{pp(rows[c]['gain']):>14s}" for c in CELLS))
    print(f"  {'':22s} " + " ".join(f"{interval(rows[c]['gain_ci_test']):>14s}" for c in CELLS))
    print(f"  {'TabFM-Logit - raw FM':22s} " + " ".join(f"{pp(rows[c]['minus_raw']):>14s}" for c in CELLS))
    print(f"  {'':22s} " + " ".join(
        f"{interval(rows[c]['minus_raw_ci']):>14s}" for c in CELLS))
    for c in CELLS:
        r = rows[c]
        if r["split"] != c[0]:
            key = "vot_chf_per_hr" if c[0] == "swissmetro" else "pt"
            print(f"  {NAMES[r['split']]} / {NAMES[c[1]]}: {r['n_test']} test rows from {r['n_groups']} groups; "
                  f"monotonicity {r['monotonicity']:.0f}%; gain positive in {r['positive_replicates']}/10 replicates; "
                  f"trade-off ({key}) {np.mean([t[key] for t in r['trade_off']]):.1f}")
