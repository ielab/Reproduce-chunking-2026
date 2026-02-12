#!/usr/bin/env python3
"""
Merge sharded pickle embedding files into a single embeddings.pkl

Usage:
    python scripts/merge_shards.py /path/to/embeddings_shards /path/to/output/embeddings.pkl

Example:
    python scripts/merge_shards.py \
        src/chunked_output/GutenQA/embeddings/Proposition-Gemini/RegularEncoder-text-embedding-ada-002/embeddings_shards \
        src/chunked_output/GutenQA/embeddings/Proposition-Gemini/RegularEncoder-text-embedding-ada-002/embeddings.pkl
"""

import os
import sys

# Add project root to path so 'src' module can be found
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

import glob
import pickle
import gc
from tqdm import tqdm


def merge_shards(shards_dir: str, output_path: str):
    """Merge all batch_*.pkl files into a single embeddings.pkl"""

    batch_files = sorted(glob.glob(os.path.join(shards_dir, "batch_*.pkl")))

    if not batch_files:
        print(f"No batch_*.pkl files found in {shards_dir}")
        sys.exit(1)

    print(f"Found {len(batch_files)} batch files")
    print(f"Output: {output_path}")

    # Calculate total size
    total_size = sum(os.path.getsize(f) for f in batch_files)
    print(f"Total shard size: {total_size / (1024**3):.2f} GB")
    print(f"Warning: This will require significant RAM to merge.")
    print()

    all_embeddings = []

    for batch_file in tqdm(batch_files, desc="Loading batches"):
        with open(batch_file, 'rb') as f:
            batch = pickle.load(f)
            all_embeddings.extend(batch)

        # Try to free memory from the loaded batch list
        del batch
        gc.collect()

    print(f"\nTotal embeddings: {len(all_embeddings):,}")
    print(f"Writing to {output_path}...")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(all_embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)

    output_size = os.path.getsize(output_path)
    print(f"Done! Output size: {output_size / (1024**3):.2f} GB")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    shards_dir = sys.argv[1]
    output_path = sys.argv[2]

    if not os.path.isdir(shards_dir):
        print(f"Error: {shards_dir} is not a directory")
        sys.exit(1)

    merge_shards(shards_dir, output_path)
