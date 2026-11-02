"""Figure 1 (ablation A4): TabFM-Logit's accuracy gain over Stage 1 on the
Swissmetro row-level split as the foundation model's context shrinks to 50%,
25% and 10% of the train+validation rows. Markers: mean over the 10
replicates; bars: t-based 95% intervals.

    python paper/figure_degradation.py      # writes paper/figures/graceful_degradation.pdf
"""
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import t as student_t  # noqa: E402

from common import ROOT, SEEDS, run_json  # noqa: E402

OUT = ROOT / "paper" / "figures" / "graceful_degradation.pdf"
FMS = [("mitra", "Mitra", "#1f77b4", "o"), ("tabpfn", "TabPFN", "#d62728", "s")]
LEVELS = [10, 25, 50, 100]


def gains(fm, level):
    tag = "xfit5" if level == 100 else f"xfit5_subset{level}"
    return np.array([100 * run_json("tabfm_logit", "swissmetro", fm, s, tag)["accuracy_gain_over_stage1"]
                     for s in SEEDS])


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "serif", "font.size": 9.5, "axes.titlesize": 10,
                         "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9})
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), sharey=True)
    xs = np.arange(len(LEVELS))
    summary = {}
    for ax, (fm, label, color, marker) in zip(axes, FMS):
        g = [gains(fm, lvl) for lvl in LEVELS]
        means = np.array([x.mean() for x in g])
        halves = np.array([student_t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x)) for x in g])
        summary[label] = {f"{lvl}%": {"mean": round(float(m), 2), "ci95_half": round(float(h), 2),
                                      "positive_replicates": int((x > 0).sum())}
                          for lvl, m, h, x in zip(LEVELS, means, halves, g)}
        ax.errorbar(xs, means, yerr=halves, fmt=marker + "-", color=color, markeredgecolor="black",
                    markeredgewidth=0.5, markersize=7, linewidth=1.6, elinewidth=1.0, capsize=3.5, zorder=3)
        ax.set_xlabel("Foundation-model context")
        if fm == "mitra":
            ax.set_ylabel("Accuracy gain over Stage 1 (pp)")
        ax.set_xlim(-0.3, len(LEVELS) - 0.7)
        ax.set_ylim(bottom=0)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{lvl}%" for lvl in LEVELS])
        ax.set_title(label)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(json.dumps(summary, indent=1))
    print(f"saved {OUT}")
