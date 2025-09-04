#!/usr/bin/env python3
"""
Extract features for Model 2 and Model 3 from the Computational Swinging Lexical Network Model.

This script computes:
- Model 2: Directional spreading activation probabilities p_k(t→d) and p_k(d→t)
- Model 3: Shared neighborhood activation counts N_{k,q}(t,d)

Author: Generated for CSLN project
"""

import numpy as np
import pandas as pd
from pathlib import Path
import random
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuration
INPUT_DATA_PATH = "home_directorycsln/data/input/target_context_pairs/pwi_target_distractor_german.csv"
OUTPUT_PATH = "home_directorycsln/data/output/pwi_target_distractor_german_reference_joint_activation_distribution.npy"
VOCAB_PATH = "home_directorycsln/data/input/fasttext_german_cosine_vocab.txt"
DM_MATRICES_DIR = "home_directorycsln/data/input/DM_matrices"

# Random sampling parameters for global decile computation
RANDOM_SAMPLE_SIZE = 1_000_000
RANDOM_SEED = 42

def load_vocabulary(vocab_path):
    """Load vocabulary and create word-to-index mapping."""
    print("Loading vocabulary...")
    with open(vocab_path, 'r', encoding='utf-8') as f:
        words = [line.strip() for line in tqdm(f, desc="Reading vocabulary", unit="words")]
    
    print("Creating word-to-index mapping...")
    word_to_idx = {word: idx for idx, word in enumerate(tqdm(words, desc="Indexing vocabulary", unit="words"))}
    print(f"Loaded {len(words)} words")
    return words, word_to_idx

def load_dm_matrices(dm_dir):
    """Load all DM matrices for time steps k=1,2,3,4,5."""
    print("Loading DM matrices...")
    dm_matrices = {}
    
    for k in tqdm(range(1, 6), desc="Loading DM matrices", unit="matrix"):
        filepath = Path(dm_dir) / f"german_fasttext_dm_power_k{k}.npy"
        dm_matrices[k] = np.load(filepath)
        print(f"DM matrix k={k} shape: {dm_matrices[k].shape}")
    
    return dm_matrices

def load_cosine_matrix(cosine_path):
    """Load the cosine similarity matrix."""
    print("Loading cosine similarity matrix...")
    cosine_matrix = np.load(cosine_path)
    print(f"Cosine matrix shape: {cosine_matrix.shape}")
    return cosine_matrix

def compute_global_reference_joint_activation_distribution(cosine_matrix, vocab_size, sample_size=RANDOM_SAMPLE_SIZE):
    """
    Compute global decile thresholds for dyadic similarities.
    
    Sample random (t,d,w) triples and compute dyadCos(t,d,w) = 0.5 * [cos(t,w) + cos(d,w)]
    """
    print(f"Computing global decile thresholds from {sample_size} random samples...")
    
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    dyadic_similarities = []
    
    for _ in tqdm(range(sample_size), desc="Sampling dyadic similarities"):
        # Sample three different indices
        indices = np.random.choice(vocab_size, size=3, replace=False)
        t_idx, d_idx, w_idx = indices
        
        # Compute dyadic cosine: 0.5 * [cos(t,w) + cos(d,w)]
        cos_tw = cosine_matrix[t_idx, w_idx]
        cos_dw = cosine_matrix[d_idx, w_idx]
        dyad_cos = 0.5 * (cos_tw + cos_dw)
        
        dyadic_similarities.append(dyad_cos)
        
    return dyadic_similarities
        

def main():
    """Main function to extract all features and save results."""
    print("=" * 60)
    print("COMPUTATIONAL SWINGING LEXICAL NETWORK - FEATURE EXTRACTION")
    print("=" * 60)
    
    # Main execution phases
    phases = [
        "Loading input data",
        "Loading vocabulary", 
        "Loading DM matrices",
        "Computing global decile thresholds",
        "Extracting Model 2 features",
        "Extracting Model 3 features",
        "Combining and saving results"
    ]
    
    main_progress = tqdm(phases, desc="Overall Progress", unit="phase", position=0)
    
    # Phase 1: Load data
    main_progress.set_description("Phase 1/7: Loading input dataset")
    df = pd.read_csv(INPUT_DATA_PATH)
    print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
    print("Columns:", list(df.columns))
    main_progress.update(1)
    
    # Phase 2: Load vocabulary
    main_progress.set_description("Phase 2/7: Loading vocabulary")
    words, word_to_idx = load_vocabulary(VOCAB_PATH)
    vocab_size = len(words)
    main_progress.update(1)
    
    # Phase 3: Load DM matrices
    main_progress.set_description("Phase 3/7: Loading DM matrices")
    dm_matrices = load_dm_matrices(DM_MATRICES_DIR)
    
    # Use the first matrix as the cosine similarity matrix
    cosine_matrix = dm_matrices[1]
    
    # Verify matrix dimensions match vocabulary size
    assert cosine_matrix.shape[0] == vocab_size, f"Cosine matrix size {cosine_matrix.shape[0]} != vocab size {vocab_size}"
    for k, dm in dm_matrices.items():
        assert dm.shape[0] == vocab_size, f"DM matrix k={k} size {dm.shape[0]} != vocab size {vocab_size}"
    main_progress.update(1)
    
    # Phase 4: Compute global decile thresholds for Model 3
    main_progress.set_description("Phase 4/7: Computing global decile thresholds")
    reference_joint_activation_distribution = compute_global_reference_joint_activation_distribution(cosine_matrix, vocab_size)
    main_progress.update(1)
    
    # write reference_joint_activation_distribution to file
    np.save(OUTPUT_PATH, reference_joint_activation_distribution)
    


if __name__ == "__main__":
    main()