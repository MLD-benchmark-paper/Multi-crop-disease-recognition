#!/bin/bash
#SBATCH -p dmb                 # partition
#SBATCH --gres=gpu:1           # request 1 GPU
#SBATCH --cpus-per-task=16     # CPU cores (matches NUM_WORKERS=12 + headroom)
#SBATCH --mem=32G              # memory
#SBATCH -t 02:00:00            # time limit (test is much shorter than train)
#SBATCH -J test_resnet18       # job name
#SBATCH --output=/home/nalwangar/again/logs_hierZ/slurm_test_out_%j.out
#SBATCH --error=/home/nalwangar/again/logs_hierZ/slurm_test_err_%j.err

echo "=== Job started on $(hostname) at $(date) ==="

# -------------------------------------------------
# Load conda environment
# -------------------------------------------------
source ~/.bashrc
conda activate finest-env
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"

# -------------------------------------------------
# Optional: stage test data to local scratch
# Uncomment if the test set is large (tens of thousands of images).
# -------------------------------------------------
# SRC=/deepstore/datasets/dmb/ComputerVision/biology/test
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
cd /home/nalwangar/again
python -u test_hagain.py

echo "=== Job finished at $(date) ==="
