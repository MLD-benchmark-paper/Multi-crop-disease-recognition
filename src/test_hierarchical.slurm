#!/bin/bash
# ===================================================================
# NOTE: This script documents the exact SLURM job configuration used
# for the paper. It is cluster-specific and will NOT run as-is on a
# different HPC system. Before use, edit the SBATCH directives, conda
# environment name, and working directory (WORKDIR) below to match
# your own cluster.
# ===================================================================
#SBATCH -p <your_partition>     # partition (edit for your cluster)
#SBATCH --gres=gpu:1            # request 1 GPU
#SBATCH --cpus-per-task=16      # CPU cores (matches NUM_WORKERS=12 + headroom)
#SBATCH --mem=32G                # memory
#SBATCH -t 02:00:00              # time limit (test is much shorter than train)
#SBATCH -J test_resnet18_hier    # job name
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

echo "=== Job started on $(hostname) at $(date) ==="

# -------------------------------------------------
# Load conda environment (edit env name for your setup)
# -------------------------------------------------
source ~/.bashrc
conda activate mld-benchmark
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"

# -------------------------------------------------
# Working directory (edit to your repo checkout path)
# -------------------------------------------------
WORKDIR=${WORKDIR:-$(pwd)}
cd "$WORKDIR"

export MODEL_ROOT=${MODEL_ROOT:-./outputs/train_hierarchical}
export TEST_PATH=${TEST_PATH:-./data/processed/test}
export SAVE_ROOT=${SAVE_ROOT:-./outputs/test_hierarchical}

# -------------------------------------------------
# Optional: stage test data to local scratch
# Uncomment if the test set is large (tens of thousands of images).
# -------------------------------------------------
# SRC=$TEST_PATH
# if [ -z "$TMPDIR" ]; then
#     TMPDIR=/tmp/$USER/$SLURM_JOB_ID
#     mkdir -p "$TMPDIR"
# fi
# export TEST_PATH=$TMPDIR/test
# echo "=== Staging test data to $TEST_PATH at $(date) ==="
# mkdir -p "$TEST_PATH"
# rsync -a "$SRC"/ "$TEST_PATH"/
# echo "=== Staging done at $(date). Disk usage: ==="
# du -sh "$TEST_PATH"

# -------------------------------------------------
# Run test
# -------------------------------------------------
python -u src/test_hierarchical.py

echo "=== Job finished at $(date) ==="
