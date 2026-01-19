#!/usr/bin/env python3
"""
Check if every chunk_id from ParagraphChunker appears in Proposition chunks.

This verifies that all paragraph chunks were successfully processed into propositions.
"""

import json
from collections import defaultdict
from pathlib import Path


def load_chunk_ids(file_path: str) -> dict:
    """
    Load chunk_ids from a JSONL file.

    Returns:
        dict: {chunk_id: count} - how many times each chunk_id appears
    """
    chunk_ids = defaultdict(int)

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                chunk_id = data.get('chunk_id')
                if chunk_id:
                    chunk_ids[chunk_id] += 1
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping line {line_num} due to JSON error: {e}")

    return chunk_ids


def main():
    # File paths
    paragraph_path = "src/chunked_output/GutenQA/chunks/ParagraphChunker/chunks.jsonl"
    proposition_path = "src/chunked_output/GutenQA/chunks/Proposition-Gemini/chunks.jsonl"

    print("=" * 80)
    print("Chunk ID Verification Script")
    print("=" * 80)
    print()

    # Check if files exist
    if not Path(paragraph_path).exists():
        print(f"ERROR: ParagraphChunker file not found: {paragraph_path}")
        return

    if not Path(proposition_path).exists():
        print(f"ERROR: Proposition file not found: {proposition_path}")
        return

    # Load chunk IDs
    print("Loading chunk IDs from ParagraphChunker...")
    paragraph_chunks = load_chunk_ids(paragraph_path)
    print(f"  Found {len(paragraph_chunks)} unique paragraph chunk IDs")
    print(f"  Total paragraphs: {sum(paragraph_chunks.values())}")
    print()

    print("Loading chunk IDs from Proposition-Gemini...")
    proposition_chunks = load_chunk_ids(proposition_path)
    print(f"  Found {len(proposition_chunks)} unique chunk IDs in propositions")
    print(f"  Total propositions: {sum(proposition_chunks.values())}")
    print()

    # Convert to sets for comparison
    paragraph_ids = set(paragraph_chunks.keys())
    proposition_ids = set(proposition_chunks.keys())

    # Check if all paragraph chunk_ids appear in propositions
    missing_ids = paragraph_ids - proposition_ids
    extra_ids = proposition_ids - paragraph_ids

    print("=" * 80)
    print("Verification Results")
    print("=" * 80)
    print()

    if not missing_ids:
        print("✓ SUCCESS: All paragraph chunk IDs appear in propositions!")
    else:
        print(f"✗ FAIL: {len(missing_ids)} paragraph chunk IDs are MISSING from propositions")
        print()
        print("Missing chunk IDs (first 10):")
        for i, chunk_id in enumerate(sorted(missing_ids)[:10], 1):
            print(f"  {i}. {chunk_id}")
        if len(missing_ids) > 10:
            print(f"  ... and {len(missing_ids) - 10} more")

    print()

    if extra_ids:
        print(f"⚠ WARNING: {len(extra_ids)} chunk IDs in propositions don't exist in paragraphs")
        print()
        print("Extra chunk IDs (first 10):")
        for i, chunk_id in enumerate(sorted(extra_ids)[:10], 1):
            print(f"  {i}. {chunk_id}")
        if len(extra_ids) > 10:
            print(f"  ... and {len(extra_ids) - 10} more")
    else:
        print("✓ No extra chunk IDs in propositions")

    print()
    print("=" * 80)
    print("Statistics")
    print("=" * 80)
    print()

    # Compute statistics for propositions per chunk
    if paragraph_ids & proposition_ids:
        common_ids = paragraph_ids & proposition_ids
        prop_counts = [proposition_chunks[cid] for cid in common_ids]

        print(f"Propositions per paragraph chunk:")
        print(f"  Min: {min(prop_counts)}")
        print(f"  Max: {max(prop_counts)}")
        print(f"  Average: {sum(prop_counts) / len(prop_counts):.2f}")
        print()

        # Show distribution
        print("Distribution of propositions per chunk:")
        from collections import Counter
        distribution = Counter(prop_counts)
        for count in sorted(distribution.keys())[:10]:
            num_chunks = distribution[count]
            print(f"  {count} propositions: {num_chunks} chunks")
        if len(distribution) > 10:
            print(f"  ... and {len(distribution) - 10} more levels")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
