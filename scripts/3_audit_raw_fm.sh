#!/usr/bin/env bash
# Behavioral audit of the raw foundation models (Table 1, Appendix C; GPU).
set -euo pipefail
cd "$(dirname "$0")/.."
python -m tabfm_logit.audit_fm --model tabpfn --dataset swissmetro
python -m tabfm_logit.audit_fm --model tabpfn --dataset iot_wearables
python -m tabfm_logit.audit_fm --model mitra  --dataset iot_wearables
