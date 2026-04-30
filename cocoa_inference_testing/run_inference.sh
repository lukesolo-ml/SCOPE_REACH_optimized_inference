#!/bin/bash
#SBATCH --job-name=scope-reach
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=scope_reach_%j.log

CONFIG=${1:-pipeline_config.yaml}
shift
python run_timelines.py --config "$CONFIG" "$@"      