#!/usr/bin/env bash
# Foundation-model predictions (GPU; one 24 GB card is enough).
# The default run writes in-context predictions for every split; the
# cross-fitted run then writes out-of-fold training-row predictions and copies
# the validation/test files under its tag (xfit5, or gxfit5 on the grouped splits).
set -euo pipefail
cd "$(dirname "$0")/.."
M="python -m tabfm_logit.fm.mitra_runner"
T="python -m tabfm_logit.fm.tabpfn_runner"

# Row-level and subject-level splits (TabPFN is not run on LPMC: its
# train+validation context is outside TabPFN's supported size range).
for ds in swissmetro lpmc iot_wearables; do
  $M --dataset $ds
  $M --dataset $ds --cross_fit_k 5
done
for ds in swissmetro iot_wearables; do
  $T --dataset $ds
  $T --dataset $ds --cross_fit_k 5
done

# Grouped splits (group folds).
$M --dataset swissmetro_resp
$M --dataset swissmetro_resp --cross_fit_k 5
$T --dataset swissmetro_resp
$T --dataset swissmetro_resp --cross_fit_k 5
MITRA_MEM_RATIO=3 $M --dataset lpmc_hh
MITRA_MEM_RATIO=3 $M --dataset lpmc_hh --cross_fit_k 5

# Ablation A4: context restricted to 50%, 25% and 10% of train+validation.
for level in 50:0.50 25:0.25 10:0.10; do
  pct=${level%%:*}; frac=${level##*:}
  for R in "$M" "$T"; do
    $R --dataset swissmetro --subset_frac "$frac" --tag "subset$pct"
    $R --dataset swissmetro --subset_frac "$frac" --tag "subset$pct" --cross_fit_k 5
  done
done
