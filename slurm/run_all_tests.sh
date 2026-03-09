#!/bin/bash
# Run mc_timeline unit tests and benchmarks, logging output to slurm/output/
set -euo pipefail

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
source ~/lsolo-fms-local/bin/activate

export hm="/mnt/bbj-lab/users/burkh4rt"
export HF_HOME="/home/$(whoami)/cache/huggingface/"
export parent_dir="$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")"
export PYTHONPATH="/home/lsolo/sr_package:${parent_dir}:$PYTHONPATH"
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v "/mnt/bbj-lab/.envs" | tr '\n' ':')
export POLARS_VERBOSE=1
export SGLANG_TRITON_ATTENTION_NUM_KV_SPLITS=4

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/output"
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=== mc_timeline test run: ${TIMESTAMP} ==="
echo "Output directory: ${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# 1. Unit tests (no GPU/model/data required)
# ---------------------------------------------------------------------------
echo ""
echo ">>> Running unit tests..."
python -m pytest "${parent_dir}/tests/" \
    -v --tb=short --no-header \
    2>&1 | tee "${OUTPUT_DIR}/unit_tests_${TIMESTAMP}.log"

UNIT_EXIT=${PIPESTATUS[0]}
if [ $UNIT_EXIT -ne 0 ]; then
    echo "FAIL: Unit tests exited with code ${UNIT_EXIT}"
else
    echo "PASS: All unit tests passed"
fi

# ---------------------------------------------------------------------------
# 2. Benchmarks (require GPU, model, and data)
# ---------------------------------------------------------------------------
BENCH_COMMON_ARGS=(
    --data_dir "/mnt/bbj-lab/users/burkh4rt/data-mimic"
    --data_version "Y21_first_24h"
    --model_loc "/mnt/bbj-lab/users/burkh4rt/mdls-archive/gemma-5635921-Y21"
    --max_len 10000
    --n_samp 20
    --test_size 800
)

for MODE in baseline with_lp benchmark truncation_test interleave_test; do
    echo ""
    echo ">>> Running benchmark: ${MODE}..."

    # truncation_test requires a time horizon
    MODE_ARGS=()
    if [ "$MODE" = "truncation_test" ] || [ "$MODE" = "with_lp" ]; then
        MODE_ARGS+=(--time_horizon_minutes 1440)
    elif [ "$MODE" = "benchmark" ] || [ "$MODE" = "interleave_test" ]; then
        MODE_ARGS+=(--time_horizon_minutes 1440)
    fi

    python -m benchmarks.run_benchmarks \
        "${BENCH_COMMON_ARGS[@]}" \
        "${MODE_ARGS[@]}" \
        --generation_mode "$MODE" \
        2>&1 | tee "${OUTPUT_DIR}/bench_${MODE}_${TIMESTAMP}.log"

    BENCH_EXIT=${PIPESTATUS[0]}
    if [ $BENCH_EXIT -ne 0 ]; then
        echo "FAIL: Benchmark '${MODE}' exited with code ${BENCH_EXIT}"
    else
        echo "PASS: Benchmark '${MODE}' completed"
    fi
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "=== All tests and benchmarks complete ==="
echo "Logs written to: ${OUTPUT_DIR}/"
ls -lh "${OUTPUT_DIR}"/*_${TIMESTAMP}.log
