#!/usr/bin/env python3
"""
Generate LaTeX table comparing Pre-chunking (RegularEncoder) vs Contextualized Chunking (LateEncoder).
Compares original paper results with reproduced results across BEIR datasets and chunking methods.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_eval_file(eval_path: str) -> Optional[float]:
    """
    Parse an nDCG@10.eval file and extract the average score.
    """
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
        'FixedSizeChunker-256': 'Fixed-Size (256)',
        'SemanticChunker': 'Semantic (Jina-v2)',
    }
    return chunker_mapping.get(chunker_name, chunker_name)


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
    Collect nDCG@10 results for both RegularEncoder and LateEncoder across multiple chunkers.

    Returns:
        Tuple of (regular_results, late_results) dictionaries
        Keys are (dataset, chunker_dir_name) tuples
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
                base_path,
                dataset,
                'results',
                chunker_dir,
                f'RegularEncoder-{model}',
                'nDCG@10.eval'
            )

            score = parse_eval_file(regular_eval_path)
            if score is not None:
                regular_results[(dataset, chunker)] = score
            else:
                print(f"Warning: Could not read RegularEncoder results: {regular_eval_path}")

            # LateEncoder (Contextualized Chunking)
            late_eval_path = os.path.join(
                base_path,
                dataset,
                'results',
                chunker_dir,
                f'LateEncoder-{model}',
                'nDCG@10.eval'
            )

            score = parse_eval_file(late_eval_path)
            if score is not None:
                late_results[(dataset, chunker)] = score
            else:
                print(f"Warning: Could not read LateEncoder results: {late_eval_path}")

    return regular_results, late_results


def generate_latex_table(
    regular_results: Dict[Tuple[str, str], float],
    late_results: Dict[Tuple[str, str], float],
    datasets: List[str],
    chunkers: List[str],
    output_path: Optional[str] = None
) -> str:
    """
    Generate LaTeX table with columns:
    Dataset | Method | Pre-chunking Original | Pre-chunking Reproduced | Contextualized Original | Contextualized Reproduced

    Datasets are grouped with midrules between them.
    """
    latex_lines = []
    latex_lines.append(r'\begin{table}[htbp]')
    latex_lines.append(r'\centering')
    latex_lines.append(r'\caption{Comparison of Pre-chunking (RegularEncoder) vs Contextualized Chunking (LateEncoder) on nDCG@10}')
    latex_lines.append(r'\label{tab:late_chunking_comparison}')
    latex_lines.append(r'\begin{tabular}{llcccc}')
    latex_lines.append(r'\toprule')
    latex_lines.append(r' & & \multicolumn{2}{c}{Pre-chunking} & \multicolumn{2}{c}{Contextualized Chunking} \\')
    latex_lines.append(r'\cmidrule(lr){3-4} \cmidrule(lr){5-6}')
    latex_lines.append(r'Dataset & Method & Original & Reproduced & Original & Reproduced \\')
    latex_lines.append(r'\midrule')

    for ds_idx, dataset in enumerate(datasets):
        dataset_display = get_dataset_display_name(dataset)

        for ch_idx, chunker in enumerate(chunkers):
            chunker_display = get_chunker_display_name(chunker)

            # Use multirow for dataset name on the first chunker row
            if ch_idx == 0:
                ds_cell = rf'\multirow{{{len(chunkers)}}}{{*}}{{{dataset_display}}}'
            else:
                ds_cell = ''

            # Pre-chunking reproduced
            key = (dataset, chunker)
            regular_repr = f'{regular_results[key]:.4f}' if key in regular_results else '---'

            # Contextualized reproduced
            late_repr = f'{late_results[key]:.4f}' if key in late_results else '---'

            # Original columns left blank for user to fill from paper
            latex_lines.append(f'{ds_cell} & {chunker_display} &  & {regular_repr} &  & {late_repr} \\\\')

        # Add midrule between datasets (but not after the last one)
        if ds_idx < len(datasets) - 1:
            latex_lines.append(r'\midrule')

    latex_lines.append(r'\bottomrule')
    latex_lines.append(r'\end{tabular}')
    latex_lines.append(r'\end{table}')

    latex_table = '\n'.join(latex_lines)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(latex_table)
        print(f"LaTeX table saved to: {output_path}")

    return latex_table


def main():
    """Main function to generate the results table."""

    # Configuration
    BASE_PATH = "/scratch3/wan458/chunking-reproduce/src/chunked_output"

    # Define datasets in order
    DATASETS = [
        'scifact',
        'nfcorpus',
        'fiqa',
        'trec-covid',
    ]

    # Define chunking methods to compare
    CHUNKERS = [
        'FixedSizeChunker-256',
        'SentenceChunker',
        'SemanticChunker',
    ]

    # Model (adjust as needed)
    MODEL = 'jina-embeddings-v2-small-en'

    OUTPUT_PATH = 'late_chunking_comparison.tex'

    print(f"Collecting results for {MODEL} with chunkers: {', '.join(CHUNKERS)}...")
    regular_results, late_results = collect_results(BASE_PATH, DATASETS, CHUNKERS, MODEL)

    print(f"Found {len(regular_results)} RegularEncoder entries")
    print(f"Found {len(late_results)} LateEncoder entries")

    print("\nGenerating LaTeX table...")
    latex_table = generate_latex_table(regular_results, late_results, DATASETS, CHUNKERS, OUTPUT_PATH)

    print("\nGenerated LaTeX table:")
    print("=" * 80)
    print(latex_table)
    print("=" * 80)

    print(f"\nDone! LaTeX table saved to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
