"""Table 3 (counterfactual aggregate-share change, +10% cost) and the
numbers quoted with it in Section 5.3.

Reads results/counterfactual/fixed_q.json (python -m tabfm_logit.counterfactual)
for the multinomial logit and TabFM-Logit, and results/counterfactual/frontier.json
(python -m tabfm_logit.frontier) for the raw foundation model on the
perturbed input.

    python paper/table3_counterfactual.py
"""
import json

from common import CELLS, NAMES, RES

if __name__ == "__main__":
    fixed = json.loads((RES / "counterfactual" / "fixed_q.json").read_text())
    front = json.loads((RES / "counterfactual" / "frontier.json").read_text())
    raw = {(r["dataset"], r["fm"], r["scenario"]): 100 * r["delta_share_raw_fm"]
           for r in front if r["kind"] == "share" and abs(r["delta_frac"] - 0.1) < 1e-9}
    print("Table 3. Aggregate-share change (pp) under a +10% cost increase.")
    print(f"  {'dataset':14s} {'scenario':20s} {'FM':7s} {'MNL':>7s} {'TabFM-Logit':>12s} {'(sd)':>6s} {'raw FM':>7s}")
    ratios, wrong = [], []
    for r in fixed:
        rf = raw[(r["dataset"], r["fm"], r["scenario"])]
        ratios.append(r["tabfm_logit_pp"] / r["mnl_pp"])
        if rf > 0:
            wrong.append((r["dataset"], r["fm"], r["scenario"]))
        print(f"  {NAMES[r['dataset']]:14s} {r['scenario']:20s} {NAMES[r['fm']]:7s} {r['mnl_pp']:+7.2f} "
              f"{r['tabfm_logit_pp']:+12.2f} {r['tabfm_logit_pp_std']:6.2f} {rf:+7.2f}{'  <- wrong sign' if rf > 0 else ''}")
    print(f"\n  scenarios: {len(fixed)}; TabFM-Logit and MNL negative in all: "
          f"{all(r['tabfm_logit_pp'] < 0 and r['mnl_pp'] < 0 for r in fixed)}")
    print(f"  TabFM-Logit / MNL share-change ratio: {100 * min(ratios):.0f}% to {100 * max(ratios):.0f}%")
    print(f"  max replicate sd of TabFM-Logit share change: {max(r['tabfm_logit_pp_std'] for r in fixed):.2f} pp")
    print(f"  raw foundation model wrong-signed in {len(wrong)} of {len(fixed)} scenarios: {wrong}")
