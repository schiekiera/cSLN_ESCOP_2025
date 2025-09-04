#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
build_fasttext_cosine_gpu.py

Generates a cosine similarity matrix from fastText word embeddings using PyTorch
for GPU-accelerated computation. Supports tiled processing for large datasets
to manage VRAM usage and can output in both dense (.npy) and sparse (.npz)
formats.
"""

import argparse
import time
import sys
import pandas as pd
import numpy as np
import torch
from scipy.sparse import csr_matrix, save_npz, coo_matrix
from tqdm import tqdm
import psutil

def print_memory_usage():
    # GPU memory
    gpu_mem = torch.cuda.memory_allocated() / 1024**3
    gpu_cached = torch.cuda.memory_reserved() / 1024**3
    
    # CPU memory  
    cpu_mem = psutil.Process().memory_info().rss / 1024**3
    
    print(f"GPU: {gpu_mem:.2f}GB allocated, {gpu_cached:.2f}GB cached")
    print(f"CPU: {cpu_mem:.2f}GB used")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Build a cosine similarity matrix from fastText embeddings using GPU.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--embeddings', type=str, required=True,
                        help="Path to tab-separated fastText file.")
    parser.add_argument('--max-words', type=int, default=20000,
                        help="Truncate vocabulary to this many words (0 = no limit).")
    parser.add_argument('--sparse-threshold', type=float, default=0.00,
                        help="Clip values below this to zero; if >0, build a sparse CSR matrix.")
    parser.add_argument('--tile-size', type=int, default=10000,
                        help="If >0, compute cosine in tiles on GPU to save VRAM. Set to 0 to disable.")
    parser.add_argument('--out', type=str, required=True,
                        help="Output base name (no extension).")
    parser.add_argument('--half', action='store_true',
                        help="Use float16 for computation (reduces VRAM but may affect precision).")
    
    args = parser.parse_args()

    # --- 1. Setup Device and Data Type ---
    if not torch.cuda.is_available():
        print("CUDA is not available. This script requires a GPU.", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda")
    dtype = torch.float16 if args.half else torch.float32
    print(f"Using device: {device}, dtype: {dtype}")
    print("\n--- Initial Memory Usage ---")
    print_memory_usage()

    # --- 2. Load Embeddings ---
    print(f"\nLoading embeddings from {args.embeddings}...")
    start_time = time.perf_counter()
    try:
        # Load using pandas, which is robust for large text files
        space = pd.read_csv(
            args.embeddings,
            sep='\t',
            header=None,
            index_col=0,
            on_bad_lines='skip',
            engine='c',
            encoding='utf-8',
            quoting=3  # QUOTE_NONE
        )
        
        # Clean up data - common issues with fastText files
        if space.index.hasnans:
             space = space.loc[space.index.dropna()]
        if space.isnull().values.any():
            space = space.dropna()

        # Truncate vocabulary if requested
        if args.max_words > 0:
            space = space.head(args.max_words)
            
        words = space.index.to_list()
        # Convert to float32 numpy array for torch
        vecs_np = space.values.astype('float32')
        num_words, vec_dim = vecs_np.shape
        
        print(f"Loaded {num_words} word vectors of dimension {vec_dim} in {time.perf_counter() - start_time:.2f}s.")
        print("\n--- Memory Usage After Loading Embeddings ---")
        print_memory_usage()

    except FileNotFoundError:
        print(f"Error: Embeddings file not found at {args.embeddings}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while loading embeddings: {e}", file=sys.stderr)
        sys.exit(1)

    # --- 3. Move to GPU and L2-Normalize ---
    print("\nMoving vectors to GPU and L2-normalizing...")
    try:
        # Move the full vectors matrix to the GPU
        vecs_gpu = torch.from_numpy(vecs_np).to(device, dtype=dtype)
        # L2-normalize in-place for efficiency
        torch.nn.functional.normalize(vecs_gpu, p=2, dim=1, out=vecs_gpu)
        print("\n--- Memory Usage After GPU Transfer & Normalization ---")
        print_memory_usage()
    except torch.cuda.OutOfMemoryError:
        print("CUDA out of memory: Could not load the full vector matrix into VRAM.", file=sys.stderr)
        print("Try reducing --max-words.", file=sys.stderr)
        sys.exit(1)
    
    total_compute_time = 0
    is_sparse = args.sparse_threshold > 0
    sm_cpu = None
    sparse_matrix = None

    # --- 4. Compute Similarity Matrix ---
    try:
        if args.tile_size > 0 and args.tile_size < num_words:
            # Tiled computation
            print(f"\nComputing similarity matrix in tiles of size {args.tile_size}...")
            start_time = time.perf_counter()

            if is_sparse:
                # Memory-efficient path for sparse matrices: build COO lists directly
                rows, cols, data = [], [], []

                for i in tqdm(range(0, num_words, args.tile_size), desc="Tiling rows"):
                    r_start = i
                    r_end = min(i + args.tile_size, num_words)
                    
                    # Compute tile on GPU and move to CPU
                    tile_cpu = (vecs_gpu[r_start:r_end] @ vecs_gpu.T).cpu().numpy()

                    # Post-process the tile directly
                    np.clip(tile_cpu, 0, None, out=tile_cpu)
                    tile_cpu[tile_cpu < args.sparse_threshold] = 0.0
                    
                    # Find non-zero elements and store them
                    non_zero_rows, non_zero_cols = np.nonzero(tile_cpu)
                    rows.extend(non_zero_rows + r_start)
                    cols.extend(non_zero_cols)
                    data.extend(tile_cpu[non_zero_rows, non_zero_cols])
                
                # Build the final sparse matrix
                sparse_matrix = coo_matrix((data, (rows, cols)), shape=(num_words, num_words), dtype=np.float32).tocsr()
                sparse_matrix.setdiag(0) # Set diagonal to zero (no self-similarity)
                
            else: # Dense path
                sm_cpu = np.zeros((num_words, num_words), dtype=np.float32)
                for i in tqdm(range(0, num_words, args.tile_size), desc="Tiling rows"):
                    r_start = i
                    r_end = min(i + args.tile_size, num_words)
                    sm_cpu[r_start:r_end, :] = (vecs_gpu[r_start:r_end] @ vecs_gpu.T).cpu().numpy()
                
                # Post-process the full dense matrix
                np.fill_diagonal(sm_cpu, 0.0)
                np.clip(sm_cpu, 0, None, out=sm_cpu)

            total_compute_time = time.perf_counter() - start_time
            print(f"Tiled computation finished in {total_compute_time:.2f}s.")
            print("\n--- Memory Usage After Tiled Computation ---")
            print_memory_usage()
        
        else:
            # Full matrix computation (if tile_size is disabled or larger than vocab)
            print("\nComputing full similarity matrix on GPU...")
            start_time = time.perf_counter()
            sm_gpu = vecs_gpu @ vecs_gpu.T
            sm_cpu = sm_gpu.cpu().numpy()
            del sm_gpu # Free VRAM immediately
            total_compute_time = time.perf_counter() - start_time
            print(f"Full matrix computation finished in {total_compute_time:.2f}s.")

            # --- Post-process Matrix on CPU ---
            print("Post-processing matrix on CPU...")
            np.fill_diagonal(sm_cpu, 0.0)
            np.clip(sm_cpu, 0, None, out=sm_cpu)
            
            if is_sparse:
                print(f"Applying sparsity threshold: < {args.sparse_threshold}")
                sm_cpu[sm_cpu < args.sparse_threshold] = 0.0
                sparse_matrix = csr_matrix(sm_cpu, dtype=np.float32)
            
            print("\n--- Memory Usage After Full Matrix Computation ---")
            print_memory_usage()

    except torch.cuda.OutOfMemoryError:
        print("\n--- CUDA Out of Memory ---", file=sys.stderr)
        if args.tile_size > 0:
            print(f"A tile of size {args.tile_size}x{num_words} is too large for your VRAM.", file=sys.stderr)
            print("Suggestion: Try a smaller --tile-size.", file=sys.stderr)
        else:
            print(f"The full {num_words}x{num_words} matrix is too large for your VRAM.", file=sys.stderr)
            print("Suggestion: Enable tiling with --tile-size (e.g., --tile-size 10000).", file=sys.stderr)
        print("You can also reduce --max-words or use the --half flag.", file=sys.stderr)
        sys.exit(1)
    finally:
        # Clear GPU cache regardless of success or failure
        del vecs_gpu
        torch.cuda.empty_cache()
        print("\n--- Memory Usage After GPU Cleanup ---")
        print_memory_usage()

    # --- 5. Save Results ---
    print("\nSaving results...")
    start_time = time.perf_counter()
    if is_sparse:
        # Convert to CSR and save as .npz
        output_path = f"{args.out}.npz"
        save_npz(output_path, sparse_matrix)
        print(f"Saved sparse matrix ({sparse_matrix.nnz} non-zero elements) to {output_path}")
    else:
        # Save dense matrix as .npy
        output_path = f"{args.out}.npy"
        np.save(output_path, sm_cpu.astype(np.float32))
        print(f"Saved dense matrix to {output_path}")
        
    # Save vocabulary companion file
    vocab_output_path = f"{args.out}_vocab.txt"
    with open(vocab_output_path, 'w', encoding='utf-8') as f:
        for word in words:
            f.write(f"{word}\n")
    print(f"Saved vocabulary to {vocab_output_path}")
    print(f"Save operation finished in {time.perf_counter() - start_time:.2f}s.")
    print("\n--- Final Memory Usage ---")
    print_memory_usage()

    # --- 7. Final Performance Report ---
    peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    print("\n--- Performance Summary ---")
    print(f"Peak VRAM used: {peak_vram_gb:.2f} GB")
    print("Script finished successfully.")

if __name__ == "__main__":
    main() 