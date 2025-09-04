#!/bin/bash
#SBATCH --job-name=postprocess_fasttext_cosine
#SBATCH --partition=gpu_a100          # or: gpu / gpu_h100 – pick an available GPU queue
#SBATCH --gres=gpu:1                  # one GPU is enough
#SBATCH --time=1-00:00:00             # 1 day wall-time (adjust if needed)
#SBATCH --mem=128G                     # host RAM for dense matrix / tiling buffers
#SBATCH --cpus-per-task=16             # BLAS & I/O helpers
#SBATCH --output=home_directorycsln/data/logs/postprocess_structural_matrix-%j.out
#SBATCH --error=home_directorycsln/data/logs/postprocess_structural_matrix-%j.err

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
# 2. Paths & parameters
# ------------------------------------------------------------------
EMB="home_directorycsln/data/fastText/fastText_german_73k_dewac2.txt"
OUT_BASE="home_directorycsln/data/input/fasttext_german_cosine"

# ---- tweakables ----
MAX_WORDS=30000        # 0 = use the full 73 k vocabulary
SPARSE_THR=0.0       # keep cosines ≥ 0.0  → saves a lot of disk
TILE=5000            # tile rows to fit in 40 GB VRAM (set 0 to disable)
USE_HALF="--half"     # set to "--half" to run float16, else leave empty

# ------------------------------------------------------------------
# 3. Run the builder
# ------------------------------------------------------------------
python home_directorycsln/py_files/postprocess_similarity_matrix.py \
  --embeddings      "$EMB" \
  --max-words       "$MAX_WORDS" \
  --sparse-threshold "$SPARSE_THR" \
  --tile-size       "$TILE" \
  --out             "$OUT_BASE" \
  $USE_HALF

echo "Job finished: $(date)"