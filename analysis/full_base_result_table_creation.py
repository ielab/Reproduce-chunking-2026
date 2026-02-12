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
from scipy import stats
import numpy as np


def parse_eval_file(eval_path: str) -> Tuple[Optional[float], Optional[Dict[str, float]]]:
    """
    Parse an nDCG@10.eval file and extract the average score and per-query scores.

    Format expected:
    query_id_1 score_1
    query_id_2 score_2
    ...
    average score_avg

    Returns:
        Tuple of (average score, dict of query_id -> score), or (None, None) if parsing fails
    """
    try:
        with open(eval_path, 'r') as f:
            lines = f.readlines()
            per_query_scores = {}
            average_score = None
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) == 2:
                    query_id, score = parts
                    if query_id == 'average':
                        average_score = float(score)
                    else:
                        per_query_scores[query_id] = float(score)
            return average_score, per_query_scores
    except (FileNotFoundError, ValueError, IOError):
        return None, None


def paired_ttest(scores1: Dict[str, float], scores2: Dict[str, float], alpha: float = 0.05) -> Tuple[bool, float]:
    """
    Perform paired t-test between two sets of per-query scores.

    Args:
        scores1: Dict mapping query_id -> score for method 1
        scores2: Dict mapping query_id -> score for method 2
        alpha: Significance level (default 0.05)

    Returns:
        Tuple of (is_significant, p_value)
    """
    # Get common query ids
    common_ids = set(scores1.keys()) & set(scores2.keys())
    if len(common_ids) < 2:
        return False, 1.0

    # Extract paired scores
    vals1 = [scores1[qid] for qid in sorted(common_ids)]
    vals2 = [scores2[qid] for qid in sorted(common_ids)]

    # Perform paired t-test
    t_stat, p_value = stats.ttest_rel(vals1, vals2)

    return p_value < alpha, p_value


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
        #'FixedSizeChunker': 'Fixed-Size',
        'FixedSizeChunker-256': 'Fixed-Size (256)',
        'SemanticChunker': 'Semantic (Jina-v2)',
        #'LumberChunker': 'Lumber',
        #'LumberChunker-GPT': 'Lumber (GPT)',
        'LumberChunker-Gemini': 'Lumber (Gemini)',
        'Proposition-Gemini': 'Proposition (Gemini)',
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
    models: List[str],
    chunkers: List[str],
    encoder: str = "RegularEncoder"
) -> Tuple[Dict[Tuple[str, str, str], float], Dict[Tuple[str, str, str], Dict[str, float]]]:
    """
    Collect all nDCG@10 or DCG@10 results from the output directory.

    Args:
        base_path: Base path to chunked_output directory
        datasets: List of dataset names
        models: List of model names (cleaned, e.g., 'jina-embeddings-v3')
        chunkers: List of chunker names
        encoder: Encoder name (default: RegularEncoder)

    Returns:
        Tuple of:
        - Dictionary mapping (model, chunker, dataset) -> average score
        - Dictionary mapping (model, chunker, dataset) -> per-query scores dict
    """
    results = {}
    per_query_results = {}

    for dataset in datasets:
        for chunker in chunkers:
            for model in models:
                # Resolve chunker directory in case it has a suffix (e.g., FixedSizeChunker-256)
                chunker_dir_name = resolve_chunker_dir(base_path, dataset, chunker)
                if chunker_dir_name is None:
                    print(f"Warning: No results directory found for {dataset}/{chunker}")
                    continue

                # GutenQA uses DCG@10.eval, others use nDCG@10.eval
                metric_file = 'DCG@10.eval' if dataset == 'GutenQA' else 'nDCG@10.eval'

                # Construct path: dataset/results/chunker/encoder-model/metric_file
                eval_path = os.path.join(
                    base_path,
                    dataset,
                    'results',
                    chunker_dir_name,
                    f'{encoder}-{model}',
                    metric_file
                )

                score, query_scores = parse_eval_file(eval_path)
                if score is not None:
                    results[(model, chunker_dir_name, dataset)] = score
                    per_query_results[(model, chunker_dir_name, dataset)] = query_scores
                else:
                    print(f"Warning: Could not read {eval_path}")

    return results, per_query_results


def generate_latex_table(
    results: Dict[Tuple[str, str, str], float],
    per_query_results: Dict[Tuple[str, str, str], Dict[str, float]],
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

    Significance markers:
    - 'a': significantly different from the best chunker (paired t-test, p < 0.05)
    - 'b': significantly different from the worst chunker (paired t-test, p < 0.05)
    """

    # Find maximum and minimum score chunkers for each (model, dataset)
    max_scores = {}  # key: (model, dataset), value: max_score
    min_scores = {}  # key: (model, dataset), value: min_score
    best_chunker = {}  # key: (model, dataset), value: chunker with best score
    worst_chunker = {}  # key: (model, dataset), value: chunker with worst score

    for model in models:
        for dataset in datasets:
            dataset_scores = []
            for chunker in chunkers:
                key = (model, chunker, dataset)
                if key in results:
                    dataset_scores.append((results[key], chunker))
            if dataset_scores:
                dataset_scores.sort(key=lambda x: x[0], reverse=True)
                max_scores[(model, dataset)] = dataset_scores[0][0]
                best_chunker[(model, dataset)] = dataset_scores[0][1]
                min_scores[(model, dataset)] = dataset_scores[-1][0]
                worst_chunker[(model, dataset)] = dataset_scores[-1][1]

    # Compute significance markers for each (model, chunker, dataset)
    significance_markers = {}  # key: (model, chunker, dataset), value: string like 'a', 'b', 'ab', or ''
    for model in models:
        for dataset in datasets:
            for chunker in chunkers:
                key = (model, chunker, dataset)
                if key not in per_query_results:
                    continue

                markers = []
                best_key = (model, best_chunker.get((model, dataset)), dataset)
                worst_key = (model, worst_chunker.get((model, dataset)), dataset)

                # Compare with best (if not the best itself)
                if best_key in per_query_results and key != best_key:
                    is_sig, _ = paired_ttest(per_query_results[key], per_query_results[best_key])
                    if is_sig:
                        markers.append('a')

                # Compare with worst (if not the worst itself)
                if worst_key in per_query_results and key != worst_key:
                    is_sig, _ = paired_ttest(per_query_results[key], per_query_results[worst_key])
                    if is_sig:
                        markers.append('b')

                significance_markers[key] = ''.join(markers)

    # Find maximum and minimum average score PER MODEL
    max_avg_per_model = {}  # key: model, value: max_avg
    min_avg_per_model = {}  # key: model, value: min_avg
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
            min_avg_per_model[model] = min(model_averages)

    # Start building the LaTeX table
    num_cols = len(datasets) + 2  # +1 for model/chunker column, +1 for average
    col_spec = 'll' + 'c' * (len(datasets) + 1)  # +1 for average column

    latex_lines = []
    latex_lines.append(r'\begin{table*}[htbp]')
    latex_lines.append(r'\centering')
    latex_lines.append(r'\caption{Performance comparison across different models, chunking strategies, and datasets using DCG@10 for GutenQA and nDCG@10 for other datasets (Pre-chunking, Pre-C). \textbf{Bold} indicates the best result per model-dataset; \underline{underline} indicates the worst. Superscripts: $^a$ = significantly different from best ($p < 0.05$, paired t-test); $^b$ = significantly different from worst.}')
    latex_lines.append(r'\label{tab:results_regular_encoder}')
    latex_lines.append(r'\begin{tabular}{' + col_spec + r'}')
    latex_lines.append(r'\toprule')

    # First row: Task row with multicolumn
    # Count document and corpus datasets
    doc_datasets = [d for d in datasets if d == 'GutenQA']
    corpus_datasets = [d for d in datasets if d != 'GutenQA']

    task_row = [' & ']
    if doc_datasets:
        task_row.append(r'\multicolumn{1}{c}{\textbf{\textit{Single-document}}}')
    if corpus_datasets:
        # +1 for the average column
        task_row.append(r'\multicolumn{' + str(len(corpus_datasets) + 1) + r'}{c}{\textbf{\textit{Multi-documents}}}')
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
                    is_min = (max_key in min_scores and abs(score - min_scores[max_key]) < 1e-6)

                    # Get significance markers
                    sig_marker = significance_markers.get(key, '')
                    sig_superscript = f'$^{{{sig_marker}}}$' if sig_marker else ''

                    if is_max:
                        row.append(f'\\textbf{{{score:.4f}}}{sig_superscript}')
                    elif is_min:
                        row.append(f'\\underline{{{score:.4f}}}{sig_superscript}')
                    else:
                        row.append(f'{score:.4f}{sig_superscript}')
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
                is_min_avg = (model in min_avg_per_model and abs(avg_score - min_avg_per_model[model]) < 1e-6)
                if is_max_avg:
                    row.append(f'\\textbf{{{avg_score:.4f}}}')
                elif is_min_avg:
                    row.append(f'\\underline{{{avg_score:.4f}}}')
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
    latex_lines.append(r'\end{table*}')

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
        'FixedSizeChunker-256',
        'SemanticChunker',
        'LumberChunker-Gemini',
        'Proposition-Gemini',
    ]

    ENCODER = 'RegularEncoder'

    # Output path (relative to current working directory)
    OUTPUT_PATH = 'results_table_regular_encoder.tex'

    print("Collecting results...")
    results, per_query_results = collect_results(BASE_PATH, DATASETS, MODELS, CHUNKERS, ENCODER)

    print(f"Found {len(results)} result entries")

    print("Generating LaTeX table...")
    latex_table = generate_latex_table(results, per_query_results, DATASETS, MODELS, CHUNKERS, OUTPUT_PATH)

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
