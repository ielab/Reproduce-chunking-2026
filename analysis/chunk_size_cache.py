#!/usr/bin/env python3
"""
Utility module for caching and retrieving chunk sizes.
Calculates chunk sizes once and saves to JSON, then reads from cache on subsequent runs.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

CACHE_FILE = os.path.join(os.path.dirname(__file__), 'chunk_sizes_cache.json')


def get_average_chunk_size_for_dataset(base_path: str, dataset: str, chunker: str) -> Optional[Tuple[int, int]]:
    """
    Get chunk size stats for a chunker on a single dataset.

    Returns:
        Tuple of (total_chars, total_chunks) or None if file not found
    """
    chunks_path = Path(base_path) / dataset / 'chunks' / chunker / 'chunks.jsonl'
    if not chunks_path.exists():
        return None
    try:
        total_chars = 0
        total_chunks = 0
        with open(chunks_path, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    text = data.get('text', '')
                    if text:
                        total_chars += len(text)
                        total_chunks += 1
        if total_chunks > 0:
            return (total_chars, total_chunks)
        return None
    except (FileNotFoundError, json.JSONDecodeError, IOError):
        return None


def calculate_chunk_sizes(base_path: str, datasets: List[str], chunkers: List[str]) -> Dict[str, float]:
    """
    Calculate average chunk sizes for all chunkers across all datasets.

    Returns:
        Dictionary mapping chunker name -> average chunk size in characters
    """
    chunk_sizes = {}

    for chunker in chunkers:
        total_chars = 0
        total_chunks = 0

        for dataset in datasets:
            result = get_average_chunk_size_for_dataset(base_path, dataset, chunker)
            if result is not None:
                total_chars += result[0]
                total_chunks += result[1]

        if total_chunks > 0:
            chunk_sizes[chunker] = total_chars / total_chunks
        else:
            chunk_sizes[chunker] = None

    return chunk_sizes


def save_cache(chunk_sizes: Dict[str, float]) -> None:
    """Save chunk sizes to cache file."""
    with open(CACHE_FILE, 'w') as f:
        json.dump(chunk_sizes, f, indent=2)
    print(f"Chunk sizes cached to: {CACHE_FILE}")


def load_cache() -> Optional[Dict[str, float]]:
    """Load chunk sizes from cache file if it exists."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    return None


def get_chunk_sizes(
    base_path: str,
    datasets: List[str],
    chunkers: List[str],
    fallback_sizes: Dict[str, float],
    force_recalculate: bool = False
) -> Dict[str, float]:
    """
    Get chunk sizes, using cache if available.

    Args:
        base_path: Base path to chunked_output directory
        datasets: List of dataset names
        chunkers: List of chunker names
        fallback_sizes: Fallback sizes if calculation fails
        force_recalculate: If True, ignore cache and recalculate

    Returns:
        Dictionary mapping chunker name -> average chunk size
    """
    # Try to load from cache first
    if not force_recalculate:
        cached = load_cache()
        if cached is not None:
            # Check if all chunkers are in cache
            all_present = all(chunker in cached for chunker in chunkers)
            if all_present:
                print("Loading chunk sizes from cache...")
                for chunker in chunkers:
                    size = cached.get(chunker)
                    if size is not None:
                        print(f"  {chunker}: {size:.0f} chars (cached)")
                    else:
                        print(f"  {chunker}: using fallback {fallback_sizes.get(chunker, 300)}")

                # Return with fallbacks for None values
                return {
                    chunker: cached.get(chunker) or fallback_sizes.get(chunker, 300)
                    for chunker in chunkers
                }

    # Calculate chunk sizes
    print("Calculating chunk sizes (this may take a moment)...")
    chunk_sizes = calculate_chunk_sizes(base_path, datasets, chunkers)

    # Save to cache
    save_cache(chunk_sizes)

    # Print and apply fallbacks
    result = {}
    for chunker in chunkers:
        size = chunk_sizes.get(chunker)
        if size is not None:
            print(f"  {chunker}: {size:.0f} chars")
            result[chunker] = size
        else:
            fallback = fallback_sizes.get(chunker, 300)
            print(f"  {chunker}: using fallback {fallback}")
            result[chunker] = fallback

    return result


if __name__ == '__main__':
    # Test/regenerate cache
    BASE_PATH = "/scratch3/wan458/chunking-reproduce/src/chunked_output"

    DATASETS = [
        'GutenQA', 'fiqa', 'nfcorpus', 'scifact', 'trec-covid', 'arguana', 'scidocs',
    ]

    CHUNKERS = [
        'ParagraphChunker',
        'SentenceChunker',
        'FixedSizeChunker-256',
        'SemanticChunker',
        'LumberChunker-Gemini',
        'Proposition-Gemini',
    ]

    FALLBACK_SIZES = {
        'ParagraphChunker': 800,
        'SentenceChunker': 150,
        'FixedSizeChunker-256': 256,
        'SemanticChunker': 400,
        'LumberChunker-Gemini': 600,
        'Proposition-Gemini': 100,
    }

    # Force recalculate to regenerate cache
    sizes = get_chunk_sizes(BASE_PATH, DATASETS, CHUNKERS, FALLBACK_SIZES, force_recalculate=True)
    print("\nFinal chunk sizes:")
    for chunker, size in sizes.items():
        print(f"  {chunker}: {size:.0f}")
