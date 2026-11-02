"""Table 1 (headline results) and the numbers quoted with it in Section 5.

Rows: multinomial logit (Stage 1 of TabFM-Logit), raw foundation model,
monotonic NN, TasteNet, feature-augmented MNL, TabFM-Logit. Columns per
foundation model: test accuracy, monotonicity, trade-off ratio, ECE (best
post-hoc calibrator fit on validation).

    python paper/table1_headline.py
"""
import json

import numpy as np

from common import (CELLS, NAMES, PRED, RES, SEEDS, ece_best, fm_npz, run_json, run_npz)

DATASETS = ["swissmetro", "lpmc", "iot_wearables"]
FMS = ["mitra", "tabpfn"]


def fmt_trade(ds, d):
    if ds == "swissmetro":
        return f"{d['vot_chf_per_hr']:.1f}"
    if ds == "lpmc":
        return f"{d['pt']:.1f}/{d['drive']:.1f}"
    return f"{d['wtp_function']:.1f}/{d['wtp_label']:.1f}"


def mean_dict(dicts):
    return {k: float(np.mean([d[k] for d in dicts])) for k in dicts[0]}


def mnl_row(ds):
    # Stage 1 does not depend on the foundation model; the Mitra runs are used.
    J = [run_json("tabfm_logit", ds, "mitra", s) for s in SEEDS]
    pairs = []
    for s in SEEDS:
        v, t = run_npz("tabfm_logit", ds, "mitra", s, "val"), run_npz("tabfm_logit", ds, "mitra", s, "test")
        pairs.append((v["probs_stage1"], v["labels"], t["probs_stage1"], t["labels"]))
    to = mean_dict([j["analytical_trade_off"] for j in J])
    return (100 * np.mean([j["test_accuracy_stage1"] for j in J]), 100.0, fmt_trade(ds, to), ece_best(pairs))


def raw_fm_row(ds, fm):
    t, v = fm_npz(ds, fm, "test"), fm_npz(ds, fm, "val")
    acc = 100 * float((t["probs"].argmax(1) == t["labels"]).mean())
    ece = ece_best([(v["probs"], v["labels"], t["probs"], t["labels"])])
    mono, trade = None, None
    path = RES / "audits" / f"audit_{fm}_{ds}.json"
    if path.exists():
        a = json.loads(path.read_text())
        mono = 100 * a["monotonicity_avg"]
        r = a["trade_off_ratios"]
        if ds == "swissmetro":      # value of time on the train alternative, CHF/hr
            trade = f"{60 * r['vot_train']['median']:.1f}"
        elif ds == "lpmc":
            trade = f"{r['vot_pt']['median']:.1f}/{r['vot_drive']['median']:.1f}"
        else:                       # willingness-to-pay, first alternative
            trade = f"{r['wtp_function_alt1']['median']:.1f}/{r['wtp_label_alt1']['median']:.1f}"
    return acc, mono, trade, ece


def monotonic_nn_row(ds):
    J = [json.loads((RES / "monotonic_nn" / f"monotonic_nn_{ds}_seed{s}.json").read_text()) for s in SEEDS]
    return (100 * np.mean([j["test_accuracy"] for j in J]),
            100 * min(j["audit"]["monotonicity_avg"] for j in J), None, None)


def tastenet_row(ds):
    seeds = range(5)
    J = [json.loads((RES / "tastenet" / f"tastenet_{ds}_seed{s}.json").read_text()) for s in seeds]
    pairs = []
    for s in seeds:
        v, t = np.load(PRED / f"tastenet_{ds}_seed{s}_val.npz"), np.load(PRED / f"tastenet_{ds}_seed{s}_test.npz")
        pairs.append((v["probs_tastenet"], v["labels"], t["probs_tastenet"], t["labels"]))

    def med(f):  # median over test individuals of the per-individual ratio, mean over replicates
        return float(np.mean([np.median(f(np.load(RES / "tastenet" / f"tastenet_{ds}_seed{s}_perobs.npz")))
                              for s in seeds]))
    if ds == "swissmetro":
        trade = f"{med(lambda z: 60 * z['test_beta_tt'] / z['test_beta_co']):.1f}"
    elif ds == "lpmc":
        trade = (f"{med(lambda z: z['test_b_pt_time'] / z['test_b_cost']):.1f}/"
                 f"{med(lambda z: z['test_b_drive_time'] / z['test_b_cost']):.1f}")
    else:
        trade = (f"{med(lambda z: z['test_b_function'] / np.abs(z['test_beta_price'])):.1f}/"
                 f"{med(lambda z: z['test_b_label'] / np.abs(z['test_beta_price'])):.1f}")
    return (100 * np.mean([j["test_accuracy"] for j in J]),
            100 * min(j["audit"]["monotonicity_avg"] for j in J), trade, ece_best(pairs))


def feature_aug_row(ds, fm):
    J = [json.loads((RES / "feature_aug" / f"feature_aug_{ds}_{fm}_xfit5_seed{s}.json").read_text()) for s in SEEDS]
    pairs = []
    for s in SEEDS:
        v = np.load(PRED / f"feature_aug_{ds}_{fm}_xfit5_seed{s}_val.npz")
        t = np.load(PRED / f"feature_aug_{ds}_{fm}_xfit5_seed{s}_test.npz")
        pairs.append((v["probs_feature_aug"], v["labels"], t["probs_feature_aug"], t["labels"]))
    to = mean_dict([{k: x for k, x in j["analytical_trade_off"].items() if k != "vot"} for j in J])
    return (100 * np.mean([j["test_accuracy"] for j in J]),
            100 * min(j["audit"]["monotonicity_avg"] for j in J), fmt_trade(ds, to), ece_best(pairs), J)


def tabfm_logit_row(ds, fm):
    J = [run_json("tabfm_logit", ds, fm, s) for s in SEEDS]
    pairs = []
    for s in SEEDS:
        v, t = run_npz("tabfm_logit", ds, fm, s, "val"), run_npz("tabfm_logit", ds, fm, s, "test")
        pairs.append((v["probs"], v["labels"], t["probs"], t["labels"]))
    to = mean_dict([{k: x for k, x in j["analytical_trade_off"].items() if k != "vot"} for j in J])
    return (100 * np.mean([j["test_accuracy"] for j in J]),
            100 * np.mean([j["audit"]["monotonicity_avg"] for j in J]), fmt_trade(ds, to), ece_best(pairs), J)


def show(label, cells):
    out = []
    for c in cells:
        if c is None:
            out.append(f"{'':>8} {'':>6} {'':>11} {'':>5}")
            continue
        acc, mono, trade, ece = c[:4]
        out.append(f"{acc:8.1f} {('%.1f' % mono) if mono is not None else '---':>6} "
                   f"{trade if trade is not None else '---':>11} {('%.1f' % ece) if ece is not None else '---':>5}")
    print(f"  {label:13s} | " + " | ".join(out))


if __name__ == "__main__":
    print("Table 1. Per foundation model: accuracy (%), monotonicity (%), trade-off, ECE (%).")
    reported = set(CELLS)
    for ds in DATASETS:
        print(f"\n{NAMES[ds]}  (columns: Mitra | TabPFN)")
        m = mnl_row(ds)
        show("MNL", [m, m])
        show("raw FM", [raw_fm_row(ds, fm) if (ds, fm) in reported else None for fm in FMS])
        mn = monotonic_nn_row(ds)
        show("monotonic NN", [mn, mn])
        tn = tastenet_row(ds)
        show("TasteNet", [tn, tn])
        fa = [feature_aug_row(ds, fm) if (ds, fm) in reported else None for fm in FMS]
        show("feat-aug", fa)
        ad = [tabfm_logit_row(ds, fm) if (ds, fm) in reported else None for fm in FMS]
        show("TabFM-Logit", ad)
        if ds == "swissmetro":
            for fm, row in zip(FMS, fa):
                v = [j["analytical_trade_off"]["vot_chf_per_hr"] for j in row[4]]
                print(f"    feat-aug {NAMES[fm]} value of time over replicates: {min(v):.0f} to {max(v):.0f} CHF/hr, "
                      f"mean {np.mean(v):.0f} +- {np.std(v, ddof=1):.0f}")
            v = [j["analytical_trade_off"]["vot_chf_per_hr"] for j in ad[0][4]]
            print(f"    TabFM-Logit value of time: {np.mean(v):.1f} +- {np.std(v, ddof=1):.1f} CHF/hr")
        if ds == "lpmc":
            J = ad[0][4]
            for k in ("pt", "drive"):
                v = [j["analytical_trade_off"][k] for j in J]
                print(f"    TabFM-Logit value of time ({k}): {np.mean(v):.2f} +- {np.std(v, ddof=1):.2f} GBP/hr")
        if ds == "iot_wearables":
            v = [j["analytical_trade_off"]["wtp_function"] for j in ad[0][4]]
            print(f"    TabFM-Logit willingness-to-pay (function): {np.mean(v):.2f} +- {np.std(v, ddof=1):.2f} USD")
