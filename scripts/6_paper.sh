#!/usr/bin/env bash
# Rebuild every table, figure and quoted number of the paper from results/.
set -euo pipefail
cd "$(dirname "$0")/../paper"
for f in data_stats table1_headline table2_grouped intervals table3_counterfactual \
         table4_frontier ablations figure_degradation calibration_appendix baselines_appendix; do
  echo; echo "===== $f ====="; python "$f.py"
done
