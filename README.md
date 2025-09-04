# Computational Swinging Lexical Network (cSLN) - ESCOP 2025

This repository contains the implementation of the Computational Swinging Lexical Network model for analyzing picture-word interference (PWI) tasks in both English and German languages.

## Overview

The cSLN model explores lexical processing through spreading activation mechanisms in semantic networks. This project implements three computational models:

- **Model 0**: Baseline predictors (experimental and lexical variables)
- **Model 1**: Baseline + cosine similarity features
- **Model 2**: Model 1 + directional spreading activation probabilities
- **Model 3**: Model 2 + shared neighborhood activation counts

## Repository Structure

### Python Scripts (`py_files/`)

#### Core Processing
- **`compute_similarity_matrix.py`**: GPU-accelerated computation of cosine similarity matrices from fastText embeddings
  - Supports tiled processing for large datasets
  - Memory-efficient sparse matrix output
  - CUDA-optimized with memory monitoring

#### Language-Specific Processing

**English (`py_files/English/`)**
- **`extract_features_english.py`**: Feature extraction for English datasets
- **`postprocess_similarity_matrix_english.py`**: Matrix postprocessing and DM computation

**German (`py_files/German/`)**
- **`extract_features_german.py`**: Feature extraction for German datasets
- **`postprocess_similarity_matrix_german.py`**: Matrix postprocessing and DM computation
- **`cat_dog_example/`**: Example analysis scripts
- **`explore_similarities/`**: Similarity exploration tools
- **`global_reference_distribution/`**: Reference distribution computation

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

## Models Description

### Model 2: Directional Spreading Activation
Computes directional spreading activation probabilities:
- `p_k(t→d)`: Probability from target to distractor at time step k
- `p_k(d→t)`: Probability from distractor to target at time step k

### Model 3: Shared Neighborhood Activation
Counts words falling into global deciles based on dyadic activation:
- `N_{k,q}(t,d)`: Number of words in decile q at time step k for target-distractor pair

## Dependencies

### Python Requirements
- PyTorch (CUDA support required)
- NumPy, Pandas
- SciPy (sparse matrices)
- tqdm (progress bars)
- psutil (memory monitoring)

### R Requirements
- lme4, lmerTest (mixed-effects models)
- performance, broom.mixed (model diagnostics)
- ggplot2, corrplot (visualization)
- dplyr, tidyr (data manipulation)
- caret (cross-validation)

## Usage

### 1. Matrix Computation
```bash
python py_files/compute_similarity_matrix.py \
    --embeddings path/to/fasttext_embeddings.txt \
    --out output_matrix_name
```

### 2. Feature Extraction
```bash
# English
python py_files/English/extract_features_english.py

# German  
python py_files/German/extract_features_german.py
```

### 3. Statistical Analysis
```r
source("r_scripts/csln_models_results.R")
```

## Key Features

- **GPU Acceleration**: CUDA-optimized similarity matrix computation
- **Memory Efficiency**: Tiled processing for large vocabularies
- **Cross-Language Support**: Parallel processing pipelines for English and German
- **Comprehensive Validation**: Multiple cross-validation strategies
- **Automated Workflows**: Shell scripts for reproducible execution

## Output

The pipeline produces:
- Similarity matrices (`.npy` or sparse `.npz` format)
- Feature-enriched datasets with Model 2 and Model 3 predictors
- Statistical model comparisons and performance metrics
- Cross-validation results across different grouping strategies

## Authors

Research conducted for ESCOP 2025 conference presentation.