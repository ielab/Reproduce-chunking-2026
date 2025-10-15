#!/usr/bin/env python3
"""
Generate LaTeX table comparing chunking strategies across different models and datasets.
Focuses on LateEncoder with nDCG@10 metric.
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
    encoder: str = "LateEncoder"
) -> Dict[Tuple[str, str, str], float]:
    """
    Collect all nDCG@10 or DCG@10 results from the output directory.

    Args:
        base_path: Base path to chunked_output directory
        datasets: List of dataset names
        models: List of model names (cleaned, e.g., 'jina-embeddings-v3')
        chunkers: List of chunker names
        encoder: Encoder name (default: LateEncoder)

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

    # Find maximum score for each dataset PER MODEL (to bold them)
    max_scores = {}  # key: (model, dataset), value: max_score
    for model in models:
        for dataset in datasets:
            dataset_scores = []
            for chunker in chunkers:
                key = (model, chunker, dataset)
                if key in results:
                    dataset_scores.append(results[key])
            if dataset_scores:
                max_scores[(model, dataset)] = max(dataset_scores)

    # Find maximum average score PER MODEL
    max_avg_per_model = {}  # key: model, value: max_avg
    for model in models:
        model_averages = []
        for chunker in chunkers:
            corpus_scores = []
            for dataset in datasets:
                if dataset != 'GutenQA':
                    key = (model, chunker, dataset)
                    if key in results:
                        corpus_scores.append(results[key])
            if corpus_scores:
                model_averages.append(sum(corpus_scores) / len(corpus_scores))
        if model_averages:
            max_avg_per_model[model] = max(model_averages)

    # Start building the LaTeX table
    num_cols = len(datasets) + 2  # +1 for model/chunker column, +1 for average
    col_spec = 'll' + 'c' * (len(datasets) + 1)  # +1 for average column

    latex_lines = []
    latex_lines.append(r'\begin{table}[htbp]')
    latex_lines.append(r'\centering')
    latex_lines.append(r'\caption{Performance comparison across different models, chunking strategies, and datasets using DCG@10 for GutenQA and nDCG@10 for other datasets (LateEncoder)}')
    latex_lines.append(r'\label{tab:results_late_encoder}')
    latex_lines.append(r'\begin{tabular}{' + col_spec + r'}')
    latex_lines.append(r'\toprule')

    # First row: Task row with multicolumn
    # Count document and corpus datasets
    doc_datasets = [d for d in datasets if d == 'GutenQA']
    corpus_datasets = [d for d in datasets if d != 'GutenQA']

    task_row = [' & ']
    if doc_datasets:
        task_row.append(r'\multicolumn{1}{c}{\textbf{\textit{document}}}')
    if corpus_datasets:
        # +1 for the average column
        task_row.append(r'\multicolumn{' + str(len(corpus_datasets) + 1) + r'}{c}{\textbf{\textit{corpus}}}')
    latex_lines.append(' & '.join(task_row) + r' \\')

    # Add cmidrules under task row
    # Calculate column positions (0-indexed, but cmidrule uses 1-indexed)
    # Columns: [empty, empty, GutenQA, corpus datasets..., Avg]
    if doc_datasets and corpus_datasets:
        # Document: column 3, Corpus: columns 4 to (4 + len(corpus_datasets))
        latex_lines.append(r'\cmidrule(lr){3-3} \cmidrule(lr){4-' + str(3 + len(corpus_datasets) + 1) + r'}')
    elif doc_datasets:
        latex_lines.append(r'\cmidrule(lr){3-3}')
    elif corpus_datasets:
        latex_lines.append(r'\cmidrule(lr){3-' + str(2 + len(corpus_datasets) + 1) + r'}')

    # Second row: Dataset names + Average for corpus
    dataset_row = [' & ']
    for dataset in datasets:
        dataset_row.append(dataset)
    # Add Average column for corpus
    dataset_row.append('Avg')
    latex_lines.append(' & '.join(dataset_row) + r' \\')
    latex_lines.append(r'\midrule')

    # Data rows: Group by model, then by chunker
    for model_idx, model in enumerate(models):
        model_display = get_model_display_name(model)

        # Track per-dataset scores across chunkers for later averaging
        dataset_scores_map = {dataset: [] for dataset in datasets}

        for chunker_idx, chunker in enumerate(chunkers):
            chunker_display = get_chunker_display_name(chunker)

            # For the first chunker of each model, add multirow for model name (rotated)
            if chunker_idx == 0:
                row = [rf'\multirow{{{len(chunkers)}}}{{*}}{{\rotatebox{{90}}{{\textbf{{{model_display}}}}}}} & {chunker_display}']
            else:
                row = [f'& {chunker_display}']

            # Add scores for each dataset and calculate corpus average
            corpus_scores = []
            for dataset in datasets:
                key = (model, chunker, dataset)
                if key in results:
                    score = results[key]
                    # Bold if this is the maximum score for this dataset WITHIN THIS MODEL
                    max_key = (model, dataset)
                    is_max = (max_key in max_scores and abs(score - max_scores[max_key]) < 1e-6)
                    if is_max:
                        row.append(f'\\textbf{{{score:.4f}}}')
                    else:
                        row.append(f'{score:.4f}')
                    # Collect corpus scores for average (exclude GutenQA)
                    if dataset != 'GutenQA':
                        corpus_scores.append(score)
                    dataset_scores_map[dataset].append(score)
                else:
                    row.append('---')

            # Add average column (only for corpus datasets)
            if corpus_scores:
                avg_score = sum(corpus_scores) / len(corpus_scores)
                # Bold if this is the maximum average WITHIN THIS MODEL
                is_max_avg = (model in max_avg_per_model and abs(avg_score - max_avg_per_model[model]) < 1e-6)
                if is_max_avg:
                    row.append(f'\\textbf{{{avg_score:.4f}}}')
                else:
                    row.append(f'{avg_score:.4f}')
            else:
                row.append('---')

            latex_lines.append(' & '.join(row) + r' \\')

        # Add per-model average row across chunkers
        avg_row = ['& \\textbf{Avg}']
        corpus_avg_values = []
        for dataset in datasets:
            dataset_scores = dataset_scores_map[dataset]
            if dataset_scores:
                dataset_avg = sum(dataset_scores) / len(dataset_scores)
                avg_row.append(f'{dataset_avg:.4f}')
                if dataset != 'GutenQA':
                    corpus_avg_values.append(dataset_avg)
            else:
                avg_row.append('---')

        if corpus_avg_values:
            corpus_avg = sum(corpus_avg_values) / len(corpus_avg_values)
            avg_row.append(f'{corpus_avg:.4f}')
        else:
            avg_row.append('---')

        latex_lines.append(' & '.join(avg_row) + r' \\')

        # Separator between model blocks (after average row)
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

    ENCODER = 'LateEncoder'

    # Output path (relative to current working directory)
    OUTPUT_PATH = 'results_table_late_encoder.tex'

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
    print("\nNote: Make sure to include the following packages in your LaTeX preamble:")
    print("  \\usepackage{booktabs}")
    print("  \\usepackage{multirow}")
    print("  \\usepackage{graphicx}  % for \\rotatebox")


if __name__ == '__main__':
    main()
