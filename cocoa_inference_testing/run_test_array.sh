#!/bin/bash
# End-to-end test for the sharded SCOPE/REACH inference pipeline.
#
# Runs 2 shards on ~100 patients (50 patients per shard, 100 MC samples each),
# then automatically merges the results so analysis.ipynb can be run immediately.
#
# Usage:
#   bash run_test_array.sh [PARTITION]
#
# Arguments:
#   PARTITION   SLURM partition to submit to (default: gpu)
#               Change this to your test/dev partition as needed.
#
# After both jobs complete, open analysis.ipynb and set:
#   OUTPUT_DIR = pathlib.Path("scope_reach_output_test/merged")

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Configuration ----
PARTITION="${1:-gpu}"          # Adjust to your cluster's test partition
N_SHARDS=2
CONFIG="${SCRIPT_DIR}/pipeline_config_test.yaml"
BASE_OUTPUT_DIR="${SCRIPT_DIR}/scope_reach_output_test"

echo "============================================================"
echo "  SCOPE/REACH sharded inference — TEST RUN"
echo "  Shards           : ${N_SHARDS}"
echo "  Partition        : ${PARTITION}"
echo "  Config           : ${CONFIG}"
echo "  Output base      : ${BASE_OUTPUT_DIR}"
echo "  Submitted        : $(date)"
echo "============================================================"

# ---- Submit the array job ----
ARRAY_JOB_ID=$(sbatch --parsable \
    --partition="${PARTITION}" \
    --array=0-$((N_SHARDS - 1)) \
    --job-name=scope-test-array \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --mem=64G \
    --time=04:00:00 \
    --output="${SCRIPT_DIR}/scope_test_%A_%a.log" \
    "${SCRIPT_DIR}/run_timelines_array.sh" "${CONFIG}")

echo "Submitted inference array job  : ${ARRAY_JOB_ID}"
echo "  Logs             : ${SCRIPT_DIR}/scope_test_${ARRAY_JOB_ID}_[01].log"

# ---- Submit the merge job (runs after ALL array tasks finish) ----
# afterany = run the merge even if some shards failed (partial results are valid)
MERGE_JOB_ID=$(sbatch --parsable \
    --partition="${PARTITION}" \
    --dependency=afterany:${ARRAY_JOB_ID} \
    --job-name=scope-test-merge \
    --cpus-per-task=4 \
    --mem=16G \
    --time=00:30:00 \
    --output="${SCRIPT_DIR}/scope_merge_test_%j.log" \
    --wrap="
set -euo pipefail
echo 'Starting merge at \$(date)'
python '${SCRIPT_DIR}/merge_shards.py' \
    --base-output-dir '${BASE_OUTPUT_DIR}' \
    --n-shards ${N_SHARDS}
echo ''
echo '============================================================'
echo '  Merge complete!'
echo '  Open analysis.ipynb and set:'
echo '    OUTPUT_DIR = pathlib.Path(\"scope_reach_output_test/merged\")'
echo '============================================================'
")

echo "Submitted merge job            : ${MERGE_JOB_ID}"
echo "  Merge log        : ${SCRIPT_DIR}/scope_merge_test_${MERGE_JOB_ID}.log"
echo ""
echo "Monitor with:"
echo "  squeue -j ${ARRAY_JOB_ID},${MERGE_JOB_ID}"
echo ""
echo "When merge completes, open analysis.ipynb and set:"
echo "  OUTPUT_DIR = pathlib.Path('scope_reach_output_test/merged')"
