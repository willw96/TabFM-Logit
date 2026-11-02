"""Table 4 (fixed versus recomputed q), the frontier paragraph of Section 5.3,
and Appendix H (lambda curves, perturbation sizes, value-of-time drift).

Reads results/counterfactual/frontier.json (python -m tabfm_logit.frontier).

    python paper/table4_frontier.py
"""
import json

import numpy as np

from common import CELLS, NAMES, RES

LAMS = [0.0, 0.25, 0.5, 0.75, 1.0]


def lam(r, x):
    return next(c for c in r["lambda_curve"] if abs(c["lam"] - x) < 1e-9)


if __name__ == "__main__":
    fr = [r for r in json.loads((RES / "counterfactual" / "frontier.json").read_text())
          if (r["dataset"], r["fm"]) in CELLS]
    share = lambda ds, fm, d: [r for r in fr if r["dataset"] == ds and r["fm"] == fm  # noqa: E731
                               and r["kind"] == "share" and abs(r["delta_frac"] - d) < 1e-9]
    mrs = lambda ds, fm: [r for r in fr if r["dataset"] == ds and r["fm"] == fm and r["kind"] == "mrs"]  # noqa: E731

    print("Table 4. Fixed versus recomputed q (+10% cost).")
    for ds, fm in CELLS:
        sh = share(ds, fm, 0.1)
        ratios = [lam(r, 1)["delta_share_mean"] / lam(r, 0)["delta_share_mean"] for r in sh]
        wrong = {r["scenario"]: lam(r, 1)["n_seeds"] - lam(r, 1)["n_seeds_decrease"] for r in sh}
        line = f"  {NAMES[ds] + ' / ' + NAMES[fm]:24s} response ratio {min(ratios):.1f} to {max(ratios):.1f}x"
        m = mrs(ds, fm)
        if m:
            d25 = [100 * lam(r, .25)["abs_rel_drift_median"] for r in m]
            d1 = [100 * lam(r, 1)["abs_rel_drift_median"] for r in m]
            line += f"   drift lambda=0.25 {min(d25):.0f}-{max(d25):.0f}%   lambda=1 {min(d1):.0f}-{max(d1):.0f}%"
        line += f"   wrong direction at lambda=1 (replicates): {wrong}"
        print(line)

    print("\nAppendix H, share change (pp) along lambda at +10%; superscript = replicates with a decrease if < 10.")
    for ds, fm in CELLS:
        for r in share(ds, fm, 0.1):
            cells = []
            for x in LAMS:
                c = lam(r, x)
                sup = "" if c["n_seeds_decrease"] == c["n_seeds"] else f"^{c['n_seeds_decrease']}"
                cells.append(f"{100 * c['delta_share_mean']:+.2f}{sup}")
            print(f"  {NAMES[ds] + ' / ' + NAMES[fm]:24s} {r['scenario']:20s} " + " ".join(f"{c:>9s}" for c in cells)
                  + f"   raw FM {100 * r['delta_share_raw_fm']:+.2f}")

    print("\nAppendix H, share change (pp) at +10/+20/+50%: fixed q, recomputed q, raw FM.")
    for ds, fm in CELLS:
        for r10 in share(ds, fm, 0.1):
            parts = []
            for d in (0.1, 0.2, 0.5):
                r = next(x for x in share(ds, fm, d) if x["scenario"] == r10["scenario"])
                parts.append(f"{100 * lam(r, 0)['delta_share_mean']:+.2f} {100 * lam(r, 1)['delta_share_mean']:+.2f} "
                             f"{100 * r['delta_share_raw_fm']:+.2f}")
            print(f"  {NAMES[ds] + ' / ' + NAMES[fm]:24s} {r10['scenario']:20s} " + "  |  ".join(parts))

    print("\nAppendix H, value-of-time drift (%): median at lambda = .25 .5 .75 1; upper decile at .25 and 1"
          " (each the mean over replicates of the per-replicate statistic).")
    for ds, fm in CELLS:
        for r in mrs(ds, fm):
            med = [f"{100 * lam(r, x)['abs_rel_drift_median']:.0f}" for x in LAMS[1:]]
            q90 = [f"{100 * lam(r, x)['abs_rel_drift_q90']:,.0f}" for x in (0.25, 1.0)]
            flip = np.sign(lam(r, 1)["mrs_mean"]) != np.sign(lam(r, 0)["mrs_mean"])
            print(f"  {NAMES[ds] + ' / ' + NAMES[fm]:24s} alt {r['alt']}  median {' '.join(med):18s} "
                  f"upper decile {' '.join(q90)}{'   (mean value of time changes sign)' if flip else ''}")
