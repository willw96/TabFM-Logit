#!/usr/bin/env bash
# Model training (CPU). Every job is one (model, dataset, foundation model,
# replicate) cell and writes its own files, so jobs can run in parallel:
#   PARALLEL=8 bash scripts/4_train.sh
# Each job uses one thread; most take under a minute on a laptop CPU.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
CELLS="swissmetro:mitra swissmetro:tabpfn lpmc:mitra iot_wearables:mitra iot_wearables:tabpfn"
jobs=()
for c in $CELLS; do
  ds=${c%%:*}; fm=${c##*:}
  for s in 0 1 2 3 4 5 6 7 8 9; do
    # TabFM-Logit and its ablations on cross-fitted q (Tables 1, 3, ablations)
    for v in tabfm_logit log_only two_term joint width16 width64 depth1; do
      jobs+=("tabfm_logit.train --variant $v --dataset $ds --fm $fm --fm-tag xfit5 --seed $s")
    done
    # TabFM-Logit on in-sample q (Appendix E) and the feature-augmented MNL
    jobs+=("tabfm_logit.train --dataset $ds --fm $fm --seed $s")
    jobs+=("tabfm_logit.run_feature_aug --dataset $ds --fm $fm --fm-tag xfit5 --seed $s")
  done
done
# Grouped splits (Table 2)
for c in swissmetro_resp:mitra swissmetro_resp:tabpfn lpmc_hh:mitra; do
  ds=${c%%:*}; fm=${c##*:}
  for s in 0 1 2 3 4 5 6 7 8 9; do
    jobs+=("tabfm_logit.train --dataset $ds --fm $fm --fm-tag gxfit5 --seed $s")
    jobs+=("tabfm_logit.run_feature_aug --dataset $ds --fm $fm --fm-tag gxfit5 --seed $s")
  done
done
# Ablation A4 (restricted context)
for fm in mitra tabpfn; do for pct in 50 25 10; do for s in 0 1 2 3 4 5 6 7 8 9; do
  jobs+=("tabfm_logit.train --dataset swissmetro --fm $fm --fm-tag xfit5_subset$pct --seed $s")
done; done; done
# Baselines without a foundation model
for ds in swissmetro lpmc iot_wearables; do
  for s in 0 1 2 3 4 5 6 7 8 9; do jobs+=("tabfm_logit.run_monotonic_nn --dataset $ds --seed $s"); done
  for s in 0 1 2 3 4; do jobs+=("tabfm_logit.run_tastenet --dataset $ds --seed $s"); done
done

printf '%s\n' "${jobs[@]}" | xargs -P "${PARALLEL:-4}" -I{} sh -c 'python -m {} > /dev/null || echo "FAILED: {}"'

# Knowledge distillation and fixed-q counterfactuals (need the runs above)
python -m tabfm_logit.run_distillation
python -m tabfm_logit.counterfactual
