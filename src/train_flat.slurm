#!/bin/bash
# ===================================================================
# NOTE: This script documents the exact SLURM job configuration used
# for the paper. It is cluster-specific and will NOT run as-is on a
# different HPC system. Before use, edit the SBATCH directives, conda
# environment name, SRC dataset path, and working directory (WORKDIR)
# below to match your own cluster.
# ===================================================================
#SBATCH -p <your_partition>     # partition (edit for your cluster)
#SBATCH --gres=gpu:1            # request 1 GPU
#SBATCH --cpus-per-task=16      # CPU cores (12 workers + headroom)
#SBATCH --mem=64G                # memory
#SBATCH -t 24:00:00              # time limit
#SBATCH -J train_resnet18_flat   # job name
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

# -------------------------------------------------
# Stage dataset to local scratch (biggest I/O win).
# SRC should point to your local, harmonized copy of the dataset
# (see data_prep/harmonize_labels.py to build this from the source
# datasets listed in the README).
# -------------------------------------------------
SRC=${SRC:-./data/processed/plantvillage}

# $TMPDIR is provided by SLURM on most clusters (local SSD).
# Fall back to /tmp/$USER/$SLURM_JOB_ID if not set.
if [ -z "$TMPDIR" ]; then
    TMPDIR=/tmp/$USER/$SLURM_JOB_ID
    mkdir -p "$TMPDIR"
fi
export DATASET_PATH=$TMPDIR/plantvillage-hier
export SAVE_ROOT=${SAVE_ROOT:-./outputs/train_flat}

echo "=== Staging dataset to $DATASET_PATH at $(date) ==="
mkdir -p "$DATASET_PATH"
rsync -a "$SRC"/ "$DATASET_PATH"/
echo "=== Staging done at $(date). Disk usage: ==="
du -sh "$DATASET_PATH"

# -------------------------------------------------
# Run training
# -------------------------------------------------
python -u src/train_flat.py

echo "=== Job finished at $(date) ==="
