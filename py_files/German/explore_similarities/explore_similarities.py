#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
explore_similarities.py

Interactive script to explore cosine similarities between German words
using the precomputed similarity matrix from fastText embeddings.
"""

import numpy as np
import argparse
from pathlib import Path

def load_similarity_data(matrix_path, vocab_path):
    """Load the similarity matrix and vocabulary."""
    print(f"Loading similarity matrix from {matrix_path}...")
    similarity_matrix = np.load(matrix_path)
    
    print(f"Loading vocabulary from {vocab_path}...")
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocabulary = [line.strip() for line in f]
    
    # Create word-to-index mapping
    word_to_idx = {word: idx for idx, word in enumerate(vocabulary)}
    
    print(f"Loaded {len(vocabulary)} words with similarity matrix shape: {similarity_matrix.shape}")
    return similarity_matrix, vocabulary, word_to_idx

def get_word_similarities(word, similarity_matrix, vocabulary, word_to_idx, top_k=10):
    """Get the top-k most similar words to a given word."""
    if word not in word_to_idx:
        return None, f"Word '{word}' not found in vocabulary"
    
    word_idx = word_to_idx[word]
    similarities = similarity_matrix[word_idx]
    
    # Get top-k indices (excluding the word itself)
    top_indices = np.argsort(similarities)[::-1]
    
    results = []
    for idx in top_indices:
        if idx != word_idx:  # Skip self-similarity
            similar_word = vocabulary[idx]
            similarity_score = similarities[idx]
            results.append((similar_word, similarity_score))
            if len(results) >= top_k:
                break
    
    return results, None

def get_pairwise_similarity(word1, word2, similarity_matrix, word_to_idx):
    """Get similarity between two specific words."""
    if word1 not in word_to_idx:
        return None, f"Word '{word1}' not found in vocabulary"
    if word2 not in word_to_idx:
        return None, f"Word '{word2}' not found in vocabulary"
    
    idx1 = word_to_idx[word1]
    idx2 = word_to_idx[word2]
    similarity = similarity_matrix[idx1, idx2]
    
    return similarity, None

def explore_word_group(words, similarity_matrix, vocabulary, word_to_idx, group_name):
    """Explore similarities within a group of words."""
    print(f"\n{'='*60}")
    print(f"🔍 EXPLORING {group_name.upper()}")
    print(f"{'='*60}")
    
    available_words = [w for w in words if w in word_to_idx]
    missing_words = [w for w in words if w not in word_to_idx]
    
    if missing_words:
        print(f"⚠️  Words not found in vocabulary: {', '.join(missing_words)}")
    
    if not available_words:
        print("❌ No words from this group found in vocabulary")
        return
    
    print(f"✅ Available words: {', '.join(available_words)}")
    
    # Show pairwise similarities within the group
    print(f"\n📊 Pairwise similarities within {group_name}:")
    print("-" * 50)
    for i, word1 in enumerate(available_words):
        for word2 in available_words[i+1:]:
            sim, error = get_pairwise_similarity(word1, word2, similarity_matrix, word_to_idx)
            if error is None:
                print(f"{word1:12s} ↔ {word2:12s}: {sim:.4f}")
    
    # Show top similar words for each word in the group
    print(f"\n🎯 Top 5 most similar words for each {group_name[:-1]}:")
    print("-" * 50)
    for word in available_words:
        results, error = get_word_similarities(word, similarity_matrix, vocabulary, word_to_idx, top_k=5)
        if error is None:
            print(f"\n'{word}':")
            for similar_word, score in results:
                print(f"  {similar_word:15s} ({score:.4f})")
        else:
            print(f"  {error}")

def main():
    """Main function with example explorations."""
    parser = argparse.ArgumentParser(
        description="Explore cosine similarities between German words",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--matrix', type=str, 
                       default='home_directorycsln/data/input/fasttext_german_cosine.npy',
                       help="Path to similarity matrix (.npy file)")
    parser.add_argument('--vocab', type=str,
                       default='home_directorycsln/data/input/fasttext_german_cosine_vocab.txt',
                       help="Path to vocabulary file")
    parser.add_argument('--word', type=str, help="Explore similarities for a specific word")
    parser.add_argument('--pair', nargs=2, help="Get similarity between two specific words")
    
    args = parser.parse_args()
    
    # Load data
    try:
        similarity_matrix, vocabulary, word_to_idx = load_similarity_data(args.matrix, args.vocab)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Handle specific queries
    if args.word:
        results, error = get_word_similarities(args.word, similarity_matrix, vocabulary, word_to_idx)
        if error:
            print(f"❌ {error}")
        else:
            print(f"\n🎯 Top 10 words most similar to '{args.word}':")
            print("-" * 50)
            for word, score in results:
                print(f"{word:20s} {score:.4f}")
        return
    
    if args.pair:
        word1, word2 = args.pair
        similarity, error = get_pairwise_similarity(word1, word2, similarity_matrix, word_to_idx)
        if error:
            print(f"❌ {error}")
        else:
            print(f"Similarity between '{word1}' and '{word2}': {similarity:.4f}")
        return
    
    # Default: Explore predefined groups
    print("🚀 GERMAN WORD SIMILARITY EXPLORER")
    print("=" * 60)
    
    # German animals
    animals = [
        'Hund', 'Katze', 'Pferd', 'Kuh', 'Schwein', 'Vogel', 'Maus', 'Bär', 
        'Wolf', 'Fuchs', 'Hase', 'Reh', 'Hirsch', 'Löwe', 'Tiger', 'Elefant'
    ]
    
    # German cities
    cities = [
        'Berlin', 'München', 'Hamburg', 'Köln', 'Frankfurt', 'Stuttgart', 
        'Düsseldorf', 'Dortmund', 'Essen', 'Leipzig', 'Bremen', 'Dresden',
        'Hannover', 'Nürnberg', 'Duisburg', 'Bochum'
    ]
    
    # German colors
    colors = [
        'rot', 'blau', 'grün', 'gelb', 'schwarz', 'weiß', 'braun', 'grau',
        'orange', 'rosa', 'lila', 'violett'
    ]
    
    # Family members
    family = [
        'Mutter', 'Vater', 'Sohn', 'Tochter', 'Bruder', 'Schwester', 
        'Großmutter', 'Großvater', 'Onkel', 'Tante', 'Cousin', 'Cousine'
    ]
    
    # Explore each group
    explore_word_group(animals, similarity_matrix, vocabulary, word_to_idx, "Animals")
    explore_word_group(cities, similarity_matrix, vocabulary, word_to_idx, "Cities") 
    explore_word_group(colors, similarity_matrix, vocabulary, word_to_idx, "Colors")
    explore_word_group(family, similarity_matrix, vocabulary, word_to_idx, "Family")
    
    # Show some interesting cross-category comparisons
    print(f"\n{'='*60}")
    print("🔬 CROSS-CATEGORY COMPARISONS")
    print(f"{'='*60}")
    
    interesting_pairs = [
        ('Hund', 'Katze'),
        ('Berlin', 'München'), 
        ('rot', 'blau'),
        ('Mutter', 'Vater'),
        ('König', 'Königin'),  # if available
        ('groß', 'klein'),     # if available
        ('gut', 'schlecht'),   # if available
    ]
    
    print("🎯 Interesting word pairs:")
    print("-" * 30)
    for word1, word2 in interesting_pairs:
        sim, error = get_pairwise_similarity(word1, word2, similarity_matrix, word_to_idx)
        if error is None:
            print(f"{word1:12s} ↔ {word2:12s}: {sim:.4f}")
        else:
            print(f"{word1:12s} ↔ {word2:12s}: Not available")

if __name__ == "__main__":
    main() 