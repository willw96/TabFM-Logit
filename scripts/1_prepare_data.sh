#!/usr/bin/env bash
# Build every train/validation/test split from the raw files in data/raw/.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m tabfm_logit.data --dataset all
