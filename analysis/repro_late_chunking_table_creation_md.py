#!/usr/bin/env python3
"""
Generate Markdown table comparing Pre-chunking (RegularEncoder) vs Contextualized Chunking (LateEncoder)
for jina-embeddings-v3 and nomic-embed-text-v1 models.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_eval_file(eval_path: str) -> Optional[float]:
    """Parse an nDCG@10.eval file and extract the average score."""
    try:
        with open(eval_path, 'r') as f:
            lines = f.readlines()
            for line in reversed(lines):
                line = line.strip()
                if line.startswith('average'):
                    parts = line.split()
                    if len(parts) == 2:
                        return float(parts[1])
        return None
    except (FileNotFoundError, ValueError, IOError):
        return None


def get_dataset_display_name(dataset_name: str) -> str:
    """Convert dataset names to display names."""
    dataset_mapping = {
        'scifact': 'SciFact',
        'nfcorpus': 'NFCorpus',
        'fiqa': 'FiQA',
        'trec-covid': 'TRECCOVID',
    }
    return dataset_mapping.get(dataset_name, dataset_name)


def get_chunker_display_name(chunker_name: str) -> str:
    """Convert chunker names to display names."""
    chunker_mapping = {
        'SentenceChunker': 'Sentence',
        'FixedSizeChunker-256': 'Fixed-size',
        'SemanticChunker': 'Semantic',
    }
    return chunker_mapping.get(chunker_name, chunker_name)


def get_model_display_name(model_name: str) -> str:
    """Convert model names to display names."""
    model_mapping = {
        'jina-embeddings-v3': 'Jina-v3',
        'nomic-embed-text-v1': 'Nomic',
    }
    return model_mapping.get(model_name, model_name)


def resolve_chunker_dir(base_path: str, dataset: str, chunker: str) -> Optional[str]:
    """Return the actual chunker run directory, handling suffixes like '-256'."""
    results_root = Path(base_path) / dataset / 'results'
    direct = results_root / chunker
    if direct.exists():
        return direct.name
    candidates = sorted(results_root.glob(f"{chunker}-*"))
    if candidates:
        return candidates[0].name
    return None


def collect_results(
    base_path: str,
    datasets: List[str],
    chunkers: List[str],
    model: str,
) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]]:
    """
    Collect nDCG@10 results for both RegularEncoder and LateEncoder.

    Returns:
        Tuple of (regular_results, late_results) dictionaries
        Keys are (dataset, chunker) tuples
    """
    regular_results = {}
    late_results = {}

    for dataset in datasets:
        for chunker in chunkers:
            chunker_dir = resolve_chunker_dir(base_path, dataset, chunker)
            if chunker_dir is None:
                print(f"Warning: Chunker directory not found for {chunker} in {dataset}")
                continue

            # RegularEncoder (Pre-chunking)
            regular_eval_path = os.path.join(
                base_path, dataset, 'results', chunker_dir,
                f'RegularEncoder-{model}', 'nDCG@10.eval'
            )
            score = parse_eval_file(regular_eval_path)
            if score is not None:
                regular_results[(dataset, chunker)] = score
            else:
                print(f"Warning: Could not read RegularEncoder results: {regular_eval_path}")

            # LateEncoder (Contextualized Chunking)
            late_eval_path = os.path.join(
                base_path, dataset, 'results', chunker_dir,
                f'LateEncoder-{model}', 'nDCG@10.eval'
            )
            score = parse_eval_file(late_eval_path)
            if score is not None:
                late_results[(dataset, chunker)] = score
            else:
                print(f"Warning: Could not read LateEncoder results: {late_eval_path}")

    return regular_results, late_results


def generate_markdown_table(
    all_regular_results: Dict[str, Dict[Tuple[str, str], float]],
    all_late_results: Dict[str, Dict[Tuple[str, str], float]],
    models: List[str],
    datasets: List[str],
    chunkers: List[str],
    output_path: Optional[str] = None
) -> str:
    """
    Generate Markdown table with Original and Reproduced sub-columns.

    For each model, the columns are:
      Pre-C Original | Pre-C Reproduced | Con-C Original | Con-C Reproduced

    Original columns are left blank for the user to fill in manually.
    """
    lines = []

    # Header row 1: model group headers
    header1_cols = ['Dataset', 'Method']
    for model in models:
        model_display = get_model_display_name(model)
        header1_cols.extend([
            f'Pre-C Orig ({model_display})',
            f'Pre-C Repro ({model_display})',
            f'Con-C Orig ({model_display})',
            f'Con-C Repro ({model_display})',
        ])
    lines.append('| ' + ' | '.join(header1_cols) + ' |')

    # Separator
    lines.append('| ' + ' | '.join(['---'] * len(header1_cols)) + ' |')

    # Data rows
    for dataset in datasets:
        dataset_display = get_dataset_display_name(dataset)

        for ch_idx, chunker in enumerate(chunkers):
            chunker_display = get_chunker_display_name(chunker)

            # Show dataset name only on the first row of the group
            ds_cell = dataset_display if ch_idx == 0 else ''

            row = [ds_cell, chunker_display]
            for model in models:
                key = (dataset, chunker)
                regular = all_regular_results[model]
                late = all_late_results[model]

                regular_repro = f'{regular[key]:.4f}' if key in regular else '---'
                late_repro = f'{late[key]:.4f}' if key in late else '---'

                # Original columns left blank for user to fill in
                row.append('')           # Pre-C Original
                row.append(regular_repro) # Pre-C Reproduced
                row.append('')           # Con-C Original
                row.append(late_repro)   # Con-C Reproduced

            lines.append('| ' + ' | '.join(row) + ' |')

    md_table = '\n'.join(lines)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(md_table + '\n')
        print(f"Markdown table saved to: {output_path}")

    return md_table


def main():
    """Main function to generate the Markdown results table."""

    BASE_PATH = "/scratch3/wan458/chunking-reproduce/src/chunked_output"

    DATASETS = [
        'scifact',
        'nfcorpus',
        'fiqa',
        'trec-covid',
    ]

    CHUNKERS = [
        'FixedSizeChunker-256',
        'SentenceChunker',
        'SemanticChunker',
    ]

    MODELS = [
        'jina-embeddings-v3',
        'nomic-embed-text-v1',
    ]

    OUTPUT_PATH = 'late_chunking_comparison_v3_nomic.md'

    all_regular = {}
    all_late = {}

    for model in MODELS:
        print(f"Collecting results for {model}...")
        regular, late = collect_results(BASE_PATH, DATASETS, CHUNKERS, model)
        all_regular[model] = regular
        all_late[model] = late
        print(f"  Found {len(regular)} RegularEncoder entries, {len(late)} LateEncoder entries")

    print("\nGenerating Markdown table...")
    md_table = generate_markdown_table(
        all_regular, all_late, MODELS, DATASETS, CHUNKERS, OUTPUT_PATH
    )

    print("\nGenerated Markdown table:")
    print("=" * 80)
    print(md_table)
    print("=" * 80)
    print(f"\nDone! Markdown table saved to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
