#!/usr/bin/env python3
"""
Extract the rows for the words "Hund" and "Katze" from the DM matrix
and save them to a NumPy file.

Inputs
- Vocabulary file: home_directorycsln/data/input/fasttext_german_cosine_vocab.txt
- DM matrix:      home_directorycsln/data/input/DM_matrices/german_fasttext_dm_power_k1.npy

Output
- home_directorycsln/data/output/cat_dog_joint_activation_distribution.npy

The output is a 2 x N float array stacked in the order [Hund, Katze].
The DM matrix is memory-mapped to avoid loading the entire matrix into RAM.
"""

from pathlib import Path
import sys
from typing import List, Dict
import numpy as np

VOCAB_PATH = "home_directorycsln/data/input/fasttext_german_cosine_vocab.txt"
DM_MATRIX_PATH = "home_directorycsln/data/input/DM_matrices/german_fasttext_dm_power_k1.npy"
OUTPUT_PATH = "home_directorycsln/data/output/cat_dog_joint_activation_distribution.npy"

TARGET_WORDS = ["Hund", "Katze"]


def load_word_to_index(vocab_path: str) -> Dict[str, int]:
    """Read vocabulary file and return a mapping from word to row index."""
    with open(vocab_path, "r", encoding="utf-8") as file:
        words = [line.strip() for line in file]
    return {word: index for index, word in enumerate(words)}


def extract_rows_for_words(matrix_path: str, indices: List[int]) -> np.ndarray:
    """Memory-map the DM matrix and return the specified rows as a stacked array."""
    dm = np.load(matrix_path, mmap_mode="r")
    # Validate indices within range
    max_index = dm.shape[0] - 1
    for idx in indices:
        if idx < 0 or idx > max_index:
            raise IndexError(f"Requested index {idx} is out of bounds for matrix with size {dm.shape}")
    # Stack rows in the provided order
    extracted = np.stack([dm[idx, :] for idx in indices], axis=0)
    return extracted


def main() -> None:
    # Resolve word indices
    print("Loading vocabulary and resolving indices for target words...")
    word_to_index = load_word_to_index(VOCAB_PATH)

    missing = [w for w in TARGET_WORDS if w not in word_to_index]
    if missing:
        print(f"Error: The following words were not found in the vocabulary: {missing}")
        sys.exit(1)

    indices = [word_to_index[w] for w in TARGET_WORDS]
    print(f"Resolved indices: {dict(zip(TARGET_WORDS, indices))}")

    # Extract rows
    print("Extracting rows from DM matrix (memory-mapped)...")
    rows = extract_rows_for_words(DM_MATRIX_PATH, indices)
    print(f"Extracted array shape: {rows.shape} (expected 2 x N)")

    # Save
    output_dir = Path(OUTPUT_PATH).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_PATH, rows)
    print(f"Saved rows for {TARGET_WORDS} to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()