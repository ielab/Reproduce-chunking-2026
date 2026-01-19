#!/usr/bin/env python3
"""
Script to filter TREC result files to keep only top 50 results per query.
Processes all result.trec files in output_for_eval/ directory.
"""

import os
from pathlib import Path
from collections import defaultdict

def filter_trec_file(file_path, top_k=20):
    """
    Filter a TREC file to keep only top K results per query.

    TREC format: query_id Q0 doc_id rank score run_name
    """
    print(f"Processing: {file_path}")

    # Read all lines
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Group by query_id and keep top K
    query_results = defaultdict(list)

    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 6:  # Valid TREC line
            query_id = parts[0]
            query_results[query_id].append(line)

    # Write back with only top K per query
    filtered_lines = []
    total_before = 0
    total_after = 0

    for query_id in sorted(query_results.keys()):
        results = query_results[query_id]
        total_before += len(results)

        # Keep top K results (they should already be sorted by rank)
        top_results = results[:top_k]
        total_after += len(top_results)

        filtered_lines.extend(top_results)

    # Write filtered results back to file
    with open(file_path, 'w') as f:
        f.writelines(filtered_lines)

    print(f"  ✓ Filtered {total_before} → {total_after} results ({len(query_results)} queries)")
    return total_before, total_after

def main():
    # Base directory
    base_dir = Path("output_for_eval")

    if not base_dir.exists():
        print(f"Error: {base_dir} directory not found!")
        return

    # Find all result.trec files
    trec_files = list(base_dir.glob("**/result.trec"))

    if not trec_files:
        print(f"No result.trec files found in {base_dir}")
        return

    print(f"Found {len(trec_files)} TREC files to process\n")

    total_before_all = 0
    total_after_all = 0

    # Process each file
    for trec_file in sorted(trec_files):
        before, after = filter_trec_file(trec_file)
        total_before_all += before
        total_after_all += after

    print(f"\n{'='*60}")
    print(f"Done! Processed {len(trec_files)} files")
    print(f"Total results: {total_before_all} → {total_after_all}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
