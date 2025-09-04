#!/bin/bash
#SBATCH --job-name=postprocess_fasttext_cosine_german
#SBATCH --partition=gpu_a100          
#SBATCH --gres=gpu:1                  
#SBATCH --time=1-00:00:00             
#SBATCH --mem=128G                     
#SBATCH --cpus-per-task=16             
#SBATCH --output=home_directorycsln/data/logs/postprocess_structural_matrix_german-%j.out
#SBATCH --error=home_directorycsln/data/logs/postprocess_structural_matrix_german-%j.err

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
# 2. Run .py file
# ------------------------------------------------------------------
python home_directorycsln/py_files/German/postprocess_similarity_matrix_german.py 