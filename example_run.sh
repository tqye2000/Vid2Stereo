#!/usr/bin/env bash
# example_run.sh — sample invocation of vid2stereo.py
#
# Run from inside WSL2 (Ubuntu), from the repo root:
#   bash example_run.sh
#   bash example_run.sh path/to/video.mp4 out/
#
# NOTE: this script needs bash (uses `set -o pipefail` and ${BASH_SOURCE}).
# Running it with `sh example_run.sh` invokes dash and fails with
# "set: Illegal option -o pipefail" — use `bash example_run.sh` instead.
#
# Requires setup.sh to have completed successfully first (conda envs +
# weights in place).

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"

INPUT="${1:-examples/paradise_512.mp4}"
OUTPUT_DIR="${2:-out}"

# Activate the base conda env so `python` is on PATH (vid2stereo.py itself
# calls `conda run -n <env>` for each pipeline stage).
source ~/miniforge3/etc/profile.d/conda.sh
conda activate base

python vid2stereo.py \
  --input "${INPUT}" \
  --output-dir "${OUTPUT_DIR}"

echo "==> Done. Outputs in ${OUTPUT_DIR}/"
