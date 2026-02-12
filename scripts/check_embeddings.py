#!/usr/bin/env python3
"""
Check the number of embeddings in a pickle file or shards directory.

Usage:
    python scripts/check_embeddings.py /path/to/embeddings.pkl
    python scripts/check_embeddings.py /path/to/embeddings_shards/
"""

import os
import sys

# Add project root to path so 'src' module can be found
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

import glob
import pickle


def check_embeddings(path: str):
    if os.path.isdir(path):
        # Shards directory
        batch_files = sorted(glob.glob(os.path.join(path, "batch_*.pkl")))
        if not batch_files:
            print(f"No batch_*.pkl files found in {path}")
            sys.exit(1)

        total = 0
        for batch_file in batch_files:
            with open(batch_file, 'rb') as f:
                batch = pickle.load(f)
                count = len(batch)
                total += count
                print(f"  {os.path.basename(batch_file)}: {count:,} embeddings")

        print(f"\nTotal: {total:,} embeddings")

    else:
        # Single pickle file
        with open(path, 'rb') as f:
            embeddings = pickle.load(f)

        print(f"File: {path}")
        print(f"Total: {len(embeddings):,} embeddings")

        # Show sample
        if embeddings:
            sample = embeddings[0]
            print(f"Sample - doc_id: {sample.doc_id}, chunk_id: {sample.chunk_id}, vector dim: {len(sample.vector)}")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    check_embeddings(sys.argv[1])
