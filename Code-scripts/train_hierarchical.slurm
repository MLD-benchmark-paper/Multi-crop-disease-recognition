#!/bin/bash
#SBATCH -p dmb                 # partition
#SBATCH --gres=gpu:1           # request 1 GPU
#SBATCH --cpus-per-task=16     # CPU cores (12 workers + headroom)
#SBATCH --mem=64G              # memory
#SBATCH -t 24:00:00            # time limit
#SBATCH -J train_resnet18      # job name
#SBATCH --output=/home/nalwangar/again/logs_VVh/slurm_train_out_%j.out
#SBATCH --error=/home/nalwangar/again/logs_VVh/slurm_train_err_%j.err

echo "=== Job started on $(hostname) at $(date) ==="

# -------------------------------------------------
# Load conda environment
# -------------------------------------------------
source ~/.bashrc
conda activate finest-env
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"

# -------------------------------------------------
# Stage dataset to local scratch (biggest I/O win)
# -------------------------------------------------
SRC=/deepstore/datasets/dmb/ComputerVision/biology/plantvillage-hier

# $TMPDIR is provided by SLURM on most clusters (local SSD).
# Fall back to /tmp/$USER/$SLURM_JOB_ID if not set.
if [ -z "$TMPDIR" ]; then
    TMPDIR=/tmp/$USER/$SLURM_JOB_ID
    mkdir -p "$TMPDIR"
fi
export DATASET_PATH=$TMPDIR/plantvillage-hier

echo "=== Staging dataset to $DATASET_PATH at $(date) ==="
mkdir -p "$DATASET_PATH"
rsync -a "$SRC"/ "$DATASET_PATH"/
echo "=== Staging done at $(date). Disk usage: ==="
du -sh "$DATASET_PATH"

# -------------------------------------------------
# Run training
# -------------------------------------------------
cd /home/nalwangar/again
python -u train_hierxx.py

echo "=== Job finished at $(date) ==="
