#!/bin/bash
# Submit a small end-to-end rescue test to the dev queue.
#
# Runs 2 segments of 5 early trajectories each (10 total), then merges.
# Results go to scope_reach_output_rescue_test/ — safe to delete afterwards.
#
# Usage:
#   bash run_rescue_test.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OUTPUT_DIR="${SCRIPT_DIR}/scope_reach_output"
RESCUE_DIR="${SCRIPT_DIR}/scope_reach_output_rescue_test"
CONFIG="${SCRIPT_DIR}/pipeline_config.yaml"
N_SEGMENTS=2
MAX_EARLY=5       # trajectories per segment

if [ ! -f "${OUTPUT_DIR}/trajectories/trajectories.npz" ]; then
    echo "ERROR: Original trajectories not found."
    exit 1
fi

if [ -d "${RESCUE_DIR}" ]; then
    echo "ERROR: Test rescue dir already exists: ${RESCUE_DIR}"
    echo "       Delete it before re-running: rm -rf ${RESCUE_DIR}"
    exit 1
fi

# ── Submit the array job ─────────────────────────────────────────────────────
ARRAY_JID=$(sbatch --parsable \
    --job-name=rescue-test \
    --partition=gpudev        `# <-- adjust to your dev partition` \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --mem=64G \
    --time=01:00:00 \
    --array=0-$((N_SEGMENTS - 1)) \
    --output="${SCRIPT_DIR}/rescue_test_%A_%a.log" \
    --wrap="python ${SCRIPT_DIR}/rescue_segment.py \
        --output-dir  ${OUTPUT_DIR} \
        --rescue-dir  ${RESCUE_DIR} \
        --config      ${CONFIG} \
        --segment-idx \${SLURM_ARRAY_TASK_ID} \
        --n-segments  ${N_SEGMENTS} \
        --max-early   ${MAX_EARLY}"
)

echo "Array job submitted: ${ARRAY_JID}  (${N_SEGMENTS} tasks × ${MAX_EARLY} early trajectories)"

# ── Submit merge job, runs only after all array tasks succeed ─────────────────
MERGE_JID=$(sbatch --parsable \
    --job-name=rescue-test-merge \
    --partition=gpudev             `# <-- adjust; merge needs no GPU` \
    --cpus-per-task=4 \
    --mem=32G \
    --time=00:30:00 \
    --dependency=afterok:${ARRAY_JID} \
    --output="${SCRIPT_DIR}/rescue_test_merge_%j.log" \
    --wrap="python ${SCRIPT_DIR}/rescue_merge.py \
        --rescue-dir  ${RESCUE_DIR} \
        --output-dir  ${RESCUE_DIR}/merged \
        --n-segments  ${N_SEGMENTS}"
)

echo "Merge job submitted:  ${MERGE_JID}  (runs after all array tasks succeed)"
echo ""
echo "Monitor with:  squeue -j ${ARRAY_JID},${MERGE_JID}"
echo "Array logs:    ${SCRIPT_DIR}/rescue_test_${ARRAY_JID}_*.log"
echo "Merge log:     ${SCRIPT_DIR}/rescue_test_merge_${MERGE_JID}.log"
echo "Output:        ${RESCUE_DIR}/merged/"
