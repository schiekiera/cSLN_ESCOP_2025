#!/bin/bash
#SBATCH --job-name=extract_features_german
#SBATCH --partition=standard          
#SBATCH --time=04:00:00             
#SBATCH --mem=64G                     
#SBATCH --cpus-per-task=16             
#SBATCH --output=home_directorycsln/data/logs/extract_features_german-%j.out
#SBATCH --error=home_directorycsln/data/logs/extract_features_german-%j.err

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
conda activate home_directoryconda_env 

# ------------------------------------------------------------------
# 2. Run .py file
# ------------------------------------------------------------------
python home_directorycsln/py_files/German/extract_features_german.py 

echo "Job finished: $(date)"