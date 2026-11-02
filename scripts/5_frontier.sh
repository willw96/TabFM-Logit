#!/usr/bin/env bash
# Recomputed and partial q (Table 4, Appendix H; GPU). Refits each foundation
# model once per cell and evaluates the trained TabFM-Logit replicates.
set -euo pipefail
cd "$(dirname "$0")/.."
for c in swissmetro:mitra swissmetro:tabpfn lpmc:mitra iot_wearables:mitra iot_wearables:tabpfn; do
  python -m tabfm_logit.frontier --dataset "${c%%:*}" --fm "${c##*:}"
done
