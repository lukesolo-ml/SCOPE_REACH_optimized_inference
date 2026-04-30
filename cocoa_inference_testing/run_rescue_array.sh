#!/bin/bash
#SBATCH --job-name=scope-rescue
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0,1,2,5
#SBATCH --output=scope_rescue_%A_%a.log

set -euo pipefail

SCRIPT_DIR="/gpfs/data/bbj-lab/users/lsolo/paper_fast/new/SCOPE_REACH_optimized_inference/cocoa_inference_testing"

N_SEGMENTS=32
OUTPUT_DIR="${SCRIPT_DIR}/scope_reach_output"
RESCUE_DIR="${SCRIPT_DIR}/scope_reach_output_rescue"
CONFIG="${SCRIPT_DIR}/pipeline_config.yaml"

SEG=${SLURM_ARRAY_TASK_ID}

# Sanity checks
if [ ! -f "${OUTPUT_DIR}/trajectories/trajectories.npz" ]; then
    echo "ERROR: Original trajectories not found at ${OUTPUT_DIR}/trajectories/trajectories.npz"
    echo "       Run the original pipeline first."
    exit 1
fi

echo "============================================================"
echo "  Rescue array job"
echo "  Segment          : ${SEG} / ${N_SEGMENTS}"
echo "  Source (read-only): ${OUTPUT_DIR}"
echo "  Rescue root      : ${RESCUE_DIR}"
echo "  Config           : ${CONFIG}"
echo "  Started          : $(date)"
echo "============================================================"

python "${SCRIPT_DIR}/rescue_segment.py" \
    --output-dir  "${OUTPUT_DIR}" \
    --rescue-dir  "${RESCUE_DIR}" \
    --config      "${CONFIG}" \
    --segment-idx "${SEG}" \
    --n-segments  "${N_SEGMENTS}"

echo "============================================================"
echo "  Segment ${SEG} finished: $(date)"
echo "============================================================"
