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
OUTPUT_PATH = "home_directorycsln/data/output/pwi_target_distractor_german_features.csv"
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

def compute_global_decile_thresholds(cosine_matrix, vocab_size, sample_size=RANDOM_SAMPLE_SIZE):
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
    
    # Compute empirical deciles (percentiles 10, 20, ..., 90)
    decile_thresholds = np.percentile(dyadic_similarities, [10, 20, 30, 40, 50, 60, 70, 80, 90])
    
    print("Global decile thresholds:")
    for i, threshold in enumerate(decile_thresholds, 1):
        print(f"  τ_{i} = {threshold:.6f}")
    
    return decile_thresholds

def extract_model2_features(df, dm_matrices, word_to_idx):
    """
    Extract Model 2 features: directional spreading activation probabilities.
    
    For each target-distractor pair (t,d), compute:
    - p_k(t→d): probability from target to distractor at step k
    - p_k(d→t): probability from distractor to target at step k
    """
    print("Extracting Model 2 features (directional spreading activation)...")
    
    model2_features = {}
    
    # Initialize feature columns
    for k in range(1, 6):
        model2_features[f'p{k}_td'] = []  # target to distractor
        model2_features[f'p{k}_dt'] = []  # distractor to target
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing directional probabilities"):
        target = row['target']
        distractor = row['distractor']
        
        # Get indices
        if target not in word_to_idx or distractor not in word_to_idx:
            # Handle missing words by setting probabilities to NaN
            for k in range(1, 6):
                model2_features[f'p{k}_td'].append(np.nan)
                model2_features[f'p{k}_dt'].append(np.nan)
            continue
        
        t_idx = word_to_idx[target]
        d_idx = word_to_idx[distractor]
        
        for k in range(1, 6):
            # p_k(t→d): probability from target to distractor
            p_td = dm_matrices[k][t_idx, d_idx]
            model2_features[f'p{k}_td'].append(p_td)
            
            # p_k(d→t): probability from distractor to target
            p_dt = dm_matrices[k][d_idx, t_idx]
            model2_features[f'p{k}_dt'].append(p_dt)
    
    return model2_features

def extract_model3_features(df, dm_matrices, word_to_idx, decile_thresholds, vocab_size):
    """
    Extract Model 3 features: shared neighborhood activation counts.
    
    For each trial and each time step k, count how many words fall into each 
    global decile q based on their dyadic spreading activation to the target-distractor pair.
    """
    print("Extracting Model 3 features (shared neighborhood activation)...")
    
    model3_features = {}
    
    # Initialize feature columns (5 time steps × 10 deciles = 50 columns)
    for k in range(1, 6):
        for q in range(1, 11):
            model3_features[f'N_k{k}_q{q}'] = []
    
    # Add boundaries for decile computation (τ_0 = 0, τ_10 = 1)
    thresholds = [0.0] + list(decile_thresholds) + [1.0]
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing shared neighborhood counts"):
        target = row['target']
        distractor = row['distractor']
        
        # Get indices
        if target not in word_to_idx or distractor not in word_to_idx:
            # Handle missing words by setting all counts to NaN
            for k in range(1, 6):
                for q in range(1, 11):
                    model3_features[f'N_k{k}_q{q}'].append(np.nan)
            continue
        
        t_idx = word_to_idx[target]
        d_idx = word_to_idx[distractor]
        
        # For each time step k, compute dyadic activation probabilities and count words in each decile
        for k in range(1, 6):
            # Compute dyadic activation probabilities for all words (excluding target and distractor)
            dyadic_probs = []
            
            # Vectorized computation for better performance
            if vocab_size > 10000:  # For large vocabularies, show progress
                word_indices = [w_idx for w_idx in range(vocab_size) if w_idx != t_idx and w_idx != d_idx]
                for w_idx in word_indices:
                    prob_tw = dm_matrices[k][t_idx, w_idx]
                    prob_dw = dm_matrices[k][d_idx, w_idx]
                    dyad_prob = 0.5 * (prob_tw + prob_dw)
                    dyadic_probs.append(dyad_prob)
            else:
                for w_idx in range(vocab_size):
                    if w_idx != t_idx and w_idx != d_idx:
                        prob_tw = dm_matrices[k][t_idx, w_idx]
                        prob_dw = dm_matrices[k][d_idx, w_idx]
                        dyad_prob = 0.5 * (prob_tw + prob_dw)
                        dyadic_probs.append(dyad_prob)
            
            dyadic_probs = np.array(dyadic_probs)
            
            # Count words in each decile for this time step
            for q in range(1, 11):
                # Count words whose dyadic activation probability falls in decile q
                lower_bound = thresholds[q-1]
                upper_bound = thresholds[q]
                
                if q == 10:  # Last decile includes upper boundary
                    count = np.sum((dyadic_probs >= lower_bound) & (dyadic_probs <= upper_bound))
                else:
                    count = np.sum((dyadic_probs >= lower_bound) & (dyadic_probs < upper_bound))
                
                model3_features[f'N_k{k}_q{q}'].append(int(count))
    
    return model3_features

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
    decile_thresholds = compute_global_decile_thresholds(cosine_matrix, vocab_size)
    main_progress.update(1)
    
    # Phase 5: Extract Model 2 features
    main_progress.set_description("Phase 5/7: Extracting Model 2 features")
    model2_features = extract_model2_features(df, dm_matrices, word_to_idx)
    main_progress.update(1)
    
    # Phase 6: Extract Model 3 features
    main_progress.set_description("Phase 6/7: Extracting Model 3 features")
    model3_features = extract_model3_features(df, dm_matrices, word_to_idx, decile_thresholds, vocab_size)
    main_progress.update(1)
    
    # Phase 7: Combine all features with original data
    main_progress.set_description("Phase 7/7: Combining and saving results")
    print("Combining features with original dataset...")
    result_df = df.copy()
    
    # Add Model 2 features
    for feature_name, values in tqdm(model2_features.items(), desc="Adding Model 2 features", unit="feature"):
        result_df[feature_name] = values
    
    # Add Model 3 features
    for feature_name, values in tqdm(model3_features.items(), desc="Adding Model 3 features", unit="feature"):
        result_df[feature_name] = values
    
    print(f"Final dataset shape: {result_df.shape}")
    new_columns = list(model2_features.keys()) + list(model3_features.keys())
    print(f"Added {len(new_columns)} new features ({len(model2_features)} Model 2 + {len(model3_features)} Model 3)")
    
    # Save results
    print(f"Saving results to {OUTPUT_PATH}...")
    # Create output directory if it doesn't exist
    output_dir = Path(OUTPUT_PATH).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result_df.to_csv(OUTPUT_PATH, index=False)
    main_progress.update(1)
    main_progress.close()
    
    print("\n" + "=" * 60)
    print("✅ FEATURE EXTRACTION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Original dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Final dataset: {result_df.shape[0]} rows, {result_df.shape[1]} columns")
    print(f"Added features: {len(new_columns)} ({len(model2_features)} Model 2 + {len(model3_features)} Model 3)")
    print("\nModel 2 feature ranges:")
    for feature in sorted(model2_features.keys()):
        values = result_df[feature]
        print(f"  {feature}: [{values.min():.6f}, {values.max():.6f}], mean={values.mean():.6f}")
    
    print("\nModel 3 feature ranges (sample):")
    model3_sample = sorted(model3_features.keys())[:10]  # Show first 10
    for feature in model3_sample:
        values = result_df[feature]
        print(f"  {feature}: [{values.min()}, {values.max()}], mean={values.mean():.2f}")
    print(f"  ... and {len(model3_features) - 10} more Model 3 features")

if __name__ == "__main__":
    main()