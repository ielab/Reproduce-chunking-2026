#!/usr/bin/env python3
"""
Generate LaTeX table comparing chunking strategies across different models and datasets.
Focuses on RegularEncoder with nDCG@10 metric.
"""

import os
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


def parse_eval_file(eval_path: str) -> Optional[float]:
    """
    Parse an nDCG@10.eval file and extract the average score.

    Format expected:
    query_id_1 score_1
    query_id_2 score_2
    ...
    average score_avg

    Returns:
        The average score, or None if file doesn't exist or parsing fails
    """
    try:
        with open(eval_path, 'r') as f:
            lines = f.readlines()
            for line in reversed(lines):  # Start from the end to find 'average' line
                line = line.strip()
                if line.startswith('average'):
                    parts = line.split()
                    if len(parts) == 2:
                        return float(parts[1])
        return None
    except (FileNotFoundError, ValueError, IOError):
        return None


def get_model_display_name(model_name: str) -> str:
    """Convert model names to cleaner display names."""
    model_mapping = {
        'jina-embeddings-v2-small-en': 'Jina-v2',
        'jina-embeddings-v3': 'Jina-v3',
        'nomic-embed-text-v1': 'Nomic',
        'multilingual-e5-large-instruct': 'E5-Large',
    }
    return model_mapping.get(model_name, model_name)


def get_chunker_display_name(chunker_name: str) -> str:
    """Convert chunker names to display names."""
    chunker_mapping = {
        'ParagraphChunker': 'Paragraph',
        'SentenceChunker': 'Sentence',
        'FixedSizeChunker': 'Fixed-Size',
        'SemanticChunker': 'Semantic',
        'LumberChunker': 'Lumber',
        'Proposition': 'Proposition',
    }
    return chunker_mapping.get(chunker_name, chunker_name)


def collect_results(
    base_path: str,
    datasets: List[str],
    models: List[str],
    chunkers: List[str],
    encoder: str = "RegularEncoder"
) -> Dict[Tuple[str, str, str], float]:
    """
    Collect all nDCG@10 or DCG@10 results from the output directory.

    Args:
        base_path: Base path to chunked_output directory
        datasets: List of dataset names
        models: List of model names (cleaned, e.g., 'jina-embeddings-v3')
        chunkers: List of chunker names
        encoder: Encoder name (default: RegularEncoder)

    Returns:
        Dictionary mapping (model, chunker, dataset) -> nDCG@10 or DCG@10 score
    """
    results = {}

    for dataset in datasets:
        for chunker in chunkers:
            for model in models:
                # GutenQA uses DCG@10.eval, others use nDCG@10.eval
                metric_file = 'DCG@10.eval' if dataset == 'GutenQA' else 'nDCG@10.eval'

                # Construct path: dataset/results/chunker/encoder-model/metric_file
                eval_path = os.path.join(
                    base_path,
                    dataset,
                    'results',
                    chunker,
                    f'{encoder}-{model}',
                    metric_file
                )

                score = parse_eval_file(eval_path)
                if score is not None:
                    results[(model, chunker, dataset)] = score
                else:
                    print(f"Warning: Could not read {eval_path}")

    return results


def generate_latex_table(
    results: Dict[Tuple[str, str, str], float],
    datasets: List[str],
    models: List[str],
    chunkers: List[str],
    output_path: Optional[str] = None
) -> str:
    """
    Generate LaTeX table with models as multirow, chunkers as rows, datasets as columns.

    Table structure:
    - First row: Task (GutenQA=document, others=corpus)
    - Second row: Dataset names
    - First column: Model names (multirow spanning all chunkers)
    - Remaining columns: nDCG@10 scores for each dataset
    """

    # Start building the LaTeX table
    num_cols = len(datasets) + 1  # +1 for model/chunker column
    col_spec = 'l' + 'c' * len(datasets)

    latex_lines = []
    latex_lines.append(r'\begin{table}[htbp]')
    latex_lines.append(r'\centering')
    latex_lines.append(r'\caption{Performance comparison across different models, chunking strategies, and datasets using DCG@10 for GutenQA and nDCG@10 for other datasets (RegularEncoder)}')
    latex_lines.append(r'\label{tab:results_regular_encoder}')
    latex_lines.append(r'\begin{tabular}{' + col_spec + r'}')
    latex_lines.append(r'\toprule')

    # First row: Task row
    task_row = ['Model / Chunker & ']
    for dataset in datasets:
        if dataset == 'GutenQA':
            task_row.append(r'\textit{document}')
        else:
            task_row.append(r'\textit{corpus}')
    latex_lines.append(' & '.join(task_row) + r' \\')

    # Second row: Dataset names
    dataset_row = [' & ']
    for dataset in datasets:
        dataset_row.append(r'\textbf{' + dataset + r'}')
    latex_lines.append(' & '.join(dataset_row) + r' \\')
    latex_lines.append(r'\midrule')

    # Data rows: Group by model, then by chunker
    for model_idx, model in enumerate(models):
        model_display = get_model_display_name(model)

        for chunker_idx, chunker in enumerate(chunkers):
            chunker_display = get_chunker_display_name(chunker)

            # For the first chunker of each model, add multirow for model name
            if chunker_idx == 0:
                row = [rf'\multirow{{{len(chunkers)}}}{{*}}{{\textbf{{{model_display}}}}} & {chunker_display}']
            else:
                row = [f'& {chunker_display}']

            # Add scores for each dataset
            for dataset in datasets:
                key = (model, chunker, dataset)
                if key in results:
                    score = results[key]
                    row.append(f'{score:.4f}')
                else:
                    row.append('---')

            latex_lines.append(' & '.join(row) + r' \\')

        # Add midrule between models (except after the last model)
        if model_idx < len(models) - 1:
            latex_lines.append(r'\midrule')

    latex_lines.append(r'\bottomrule')
    latex_lines.append(r'\end{tabular}')
    latex_lines.append(r'\end{table}')

    latex_table = '\n'.join(latex_lines)

    # Save to file if output path is provided
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
        'GutenQA',      # document retrieval
        'fiqa',         # corpus retrieval
        'nfcorpus',
        'scifact',
        'trec-covid',
        'arguana',
        'scidocs',
    ]

    # Define models (cleaned names as they appear in directory structure)
    MODELS = [
        'jina-embeddings-v2-small-en',
        'jina-embeddings-v3',
        'nomic-embed-text-v1',
        'multilingual-e5-large-instruct',
    ]

    # Define chunkers in order
    CHUNKERS = [
        'ParagraphChunker',
        'SentenceChunker',
        'FixedSizeChunker',
        'SemanticChunker',
        'LumberChunker',
        'Proposition',
    ]

    ENCODER = 'RegularEncoder'

    # Output path (relative to current working directory)
    OUTPUT_PATH = 'results_table_regular_encoder.tex'

    print("Collecting results...")
    results = collect_results(BASE_PATH, DATASETS, MODELS, CHUNKERS, ENCODER)

    print(f"Found {len(results)} result entries")

    print("Generating LaTeX table...")
    latex_table = generate_latex_table(results, DATASETS, MODELS, CHUNKERS, OUTPUT_PATH)

    print("\nGenerated LaTeX table:")
    print("=" * 80)
    print(latex_table)
    print("=" * 80)

    print(f"\nDone! LaTeX table saved to {OUTPUT_PATH}")
    print("\nNote: Make sure to include \\usepackage{booktabs} and \\usepackage{multirow} in your LaTeX preamble.")


if __name__ == '__main__':
    main()
