#!/bin/bash
# SCOPE/REACH inference — SLURM array worker script.
#
# Each array task runs one shard of the patient cohort.  Results are written to
#   {output_dir}/shard_000/,  shard_001/,  ...
# After all tasks finish, run merge_shards.py to combine them for analysis.ipynb.
#
# SUBMISSION
# ----------
# Submit N shards with:
#
#   sbatch --array=0-$((N_SHARDS-1)) run_timelines_array.sh [CONFIG]
#
# Examples:
#   sbatch --array=0-7  run_timelines_array.sh                          # 8 shards, default config
#   sbatch --array=0-15 run_timelines_array.sh pipeline_config.yaml     # 16 shards
#
# SLURM sets SLURM_ARRAY_TASK_ID (0-indexed) and SLURM_ARRAY_TASK_COUNT
# automatically, so --shard-idx and --n-shards are derived from those variables.
#
# POST-RUN MERGE
# --------------
#   python merge_shards.py \
#       --base-output-dir <output_dir from config> \
#       --n-shards <N_SHARDS>
#
# Or use sbatch --dependency=afterany:<ARRAY_JOB_ID> to auto-merge (see run_test_array.sh).

#SBATCH --job-name=scope-reach-array
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=scope_reach_%A_%a.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# First positional arg is the config file; defaults to pipeline_config.yaml
CONFIG="${1:-${SCRIPT_DIR}/pipeline_config.yaml}"

# SLURM_ARRAY_TASK_COUNT = total number of tasks in the array (set automatically by SLURM)
N_SHARDS="${SLURM_ARRAY_TASK_COUNT}"
SHARD_IDX="${SLURM_ARRAY_TASK_ID}"

echo "============================================================"
echo "  SCOPE/REACH array shard"
echo "  Shard            : ${SHARD_IDX} / ${N_SHARDS}"
echo "  Config           : ${CONFIG}"
echo "  Node             : $(hostname)"
echo "  Started          : $(date)"
echo "============================================================"

python "${SCRIPT_DIR}/run_timelines_shard.py" \
    --config    "${CONFIG}" \
    --shard-idx "${SHARD_IDX}" \
    --n-shards  "${N_SHARDS}"

echo "============================================================"
echo "  Shard ${SHARD_IDX} finished : $(date)"
echo "============================================================"
