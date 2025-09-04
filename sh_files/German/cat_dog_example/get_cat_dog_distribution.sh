#!/bin/bash
#SBATCH --job-name=postprocess_fasttext_cosine
#SBATCH --partition=gpu_a100          # or: gpu / gpu_h100 – pick an available GPU queue
#SBATCH --gres=gpu:1                  # one GPU is enough
#SBATCH --time=1-00:00:00             # 1 day wall-time (adjust if needed)
#SBATCH --mem=128G                     # host RAM for dense matrix / tiling buffers
#SBATCH --cpus-per-task=16             # BLAS & I/O helpers
#SBATCH --output=home_directorycsln/data/logs/get_cat_dog_distribution-%j.out
#SBATCH --error=home_directorycsln/data/logs/get_cat_dog_distribution-%j.err

# ------------------------------------------------------------------
# 0. House-keeping
# ------------------------------------------------------------------
set -eo pipefail # Removed 'u' to prevent exit on unset variable warning from MKL
mkdir -p home_directorycsln/data/logs/

echo "Job starts:   $(date)"
echo "Host node:    $(hostname)"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"

# ------------------------------------------------------------------
# 1. Load Python environment
# ------------------------------------------------------------------
source home_directoryminiforge3/etc/profile.d/conda.sh
conda activate home_directoryconda_env   # has PyTorch + scipy

# ------------------------------------------------------------------
# 1.5. Fix OpenMP/MKL shared memory issues
# ------------------------------------------------------------------
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export OMP_PLACES=cores
export OMP_PROC_BIND=close
# Disable OpenMP shared memory to avoid HPC conflicts
export OMP_DISPLAY_ENV=FALSE
export KMP_DUPLICATE_LIB_OK=TRUE
# Use alternative memory allocation to avoid SHM issues
export MKL_ENABLE_INSTRUCTIONS=AVX2
ulimit -l unlimited  # Remove memory lock limits

echo "OpenMP threads: $OMP_NUM_THREADS"
echo "MKL threads: $MKL_NUM_THREADS"

# ------------------------------------------------------------------
# 2. Run the extractor (no arguments needed)
# ------------------------------------------------------------------
python home_directorycsln/py_files/get_cat_dog_joint_activation_distribution.py

echo "Job finished: $(date)"