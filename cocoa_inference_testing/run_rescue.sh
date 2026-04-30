#!/bin/bash
#SBATCH --job-name=scope-rescue
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=scope_rescue_%j.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OUTPUT_DIR="${SCRIPT_DIR}/scope_reach_output"
RESCUE_DIR="${SCRIPT_DIR}/scope_reach_output_rescue"
CONFIG="${SCRIPT_DIR}/pipeline_config.yaml"

# Abort if the original run data is missing — nothing to rescue
if [ ! -f "${OUTPUT_DIR}/trajectories/trajectories.npz" ]; then
    echo "ERROR: Original trajectories not found at ${OUTPUT_DIR}/trajectories/trajectories.npz"
    echo "       Run the original pipeline first."
    exit 1
fi

# Abort if the rescue output dir already exists — prevents silent overwrites
if [ -d "${RESCUE_DIR}" ]; then
    echo "ERROR: Rescue output directory already exists: ${RESCUE_DIR}"
    echo "       Move or delete it before re-running rescue."
    exit 1
fi

echo "============================================================"
echo "  Rescue run"
echo "  Source (read-only): ${OUTPUT_DIR}"
echo "  Destination (new) : ${RESCUE_DIR}"
echo "  Config            : ${CONFIG}"
echo "  Started           : $(date)"
echo "============================================================"

python "${SCRIPT_DIR}/rescue.py" \
    --output-dir  "${OUTPUT_DIR}" \
    --rescue-dir  "${RESCUE_DIR}" \
    --config      "${CONFIG}"

echo "============================================================"
echo "  Rescue finished: $(date)"
echo "============================================================"
