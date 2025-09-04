# Computational Swinging Lexical Network (cSLN) - ESCOP 2025

This repository contains the implementation of the Computational Swinging Lexical Network model for analyzing picture-word interference (PWI) tasks in both English and German languages presented in the ESCOP 2025 conference in Sheffield, UK.


**Citation:** Schiekiera, L., Gruber, V., Melinger, A., Abdel Rahman, R., & Günther, F. (2025). *Modeling Lexical Competition in Language Production: A Computational Approach to the Swinging Lexical Network*. Poster presented at the European Society for Cognitive Psychology (ESCOP 2025), Location TBD, September. Available at [https://github.com/USERNAME/lexical-network-modeling](https://github.com/USERNAME/lexical-network-modeling)

## Overview

The cSLN model explores lexical processing through spreading activation mechanisms in semantic networks. This project implements three computational models:

- **Model 0**: Baseline predictors (experimental and lexical variables)
- **Model 1**: Baseline + cosine similarity features
- **Model 2**: Model 1 + directional spreading activation probabilities
    - Computes directional spreading activation probabilities:
        - `p_k(t→d)`: Probability from target to distractor at time step k
        - `p_k(d→t)`: Probability from distractor to target at time step k
- **Model 3**: Model 2 + shared neighborhood activation counts
    - Counts words falling into global deciles based on dyadic activation:
        - `N_{k,q}(t,d)`: Number of words in decile q at time step k for target-distractor pair

## Repository Structure

### Python Scripts (`py_files/`)

#### Core Processing
- **`compute_similarity_matrix.py`**: GPU-accelerated computation of cosine similarity matrices from fastText embeddings
  - Supports tiled processing for large datasets
  - Memory-efficient sparse matrix output
  - CUDA-optimized with memory monitoring

- **`extract_features_german.py`**: Feature extraction for German datasets
- **`postprocess_similarity_matrix_german.py`**: Matrix postprocessing and DM computation
- **`get_global_reference_joint_activation_distribution_german.py`**: Computes global decile thresholds from random dyadic similarity sampling for Model 3 reference distribution
- **`visualize_global_reference_distribution.py`**: Creates KDE visualizations of global reference distribution with decile bands and exports thresholds to CSV

### Statistical Analysis (`r_scripts/`)

- **`csln_models_results.R`**: Comprehensive R script for:
  - Mixed-effects modeling with lme4
  - Cross-validation analysis (participants, experiments, trials)
  - Model comparison via likelihood ratio tests
  - Performance visualization

### Automation Scripts (`sh_files/`)

Organized by language with numbered execution order:
- `01_compute_structural_matrix_*.sh`: Matrix computation
- `02_postprocess_structural_matrix_*.sh`: Matrix postprocessing  
- `03_extract_features_*.sh`: Feature extraction

