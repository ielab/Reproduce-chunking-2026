#!/usr/bin/env python3
"""
Generate LaTeX table comparing Pre-chunking (RegularEncoder) vs Contextualized-chunking (LateEncoder) performance.
Shows the percentage difference (LateEncoder - RegularEncoder) for each combination.
Includes paired t-test significance markers comparing Con-C vs Pre-C for the same chunker.
"""

import os
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from scipy import stats


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
        'FixedSizeChunker': 'Fixed-size',
        'FixedSizeChunker-256': 'Fixed-size',
        'SemanticChunker': 'Semantic',
        'LumberChunker': 'LumberChunker',
        'LumberChunker-GPT': 'LumberChunker',
        'LumberChunker-Gemini': 'LumberChunker',
        'Proposition-Gemini': 'Proposition',
    }
    return chunker_mapping.get(chunker_name, chunker_name)


def resolve_chunker_dir(base_path: str, dataset: str, chunker: str) -> Optional[str]:
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
    encoder: str
) -> Tuple[Dict[Tuple[str, str, str], float], Dict[Tuple[str, str, str], Dict[str, float]]]:
    """
    Collect all nDCG@10 or DCG@10 results from the output directory.

    Args:
        base_path: Base path to chunked_output directory
        datasets: List of dataset names
        models: List of model names (cleaned, e.g., 'jina-embeddings-v3')
        chunkers: List of chunker names
        encoder: Encoder name (RegularEncoder or LateEncoder)

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
                chunker_dir_name = resolve_chunker_dir(base_path, dataset, chunker)
                if chunker_dir_name is None:
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

    return results, per_query_results


def generate_comparison_table(
    regular_results: Dict[Tuple[str, str, str], float],
    regular_per_query: Dict[Tuple[str, str, str], Dict[str, float]],
    late_results: Dict[Tuple[str, str, str], float],
    late_per_query: Dict[Tuple[str, str, str], Dict[str, float]],
    datasets: List[str],
    models: List[str],
    chunkers: List[str],
    output_path: Optional[str] = None
) -> str:
    """
    Generate LaTeX table comparing LateEncoder (Con-C) vs RegularEncoder (Pre-C).

    Table shows percentage change relative to RegularEncoder.
    Significance marker (*) added when paired t-test p < 0.05.
    """

    # Start building the LaTeX table
    num_cols = len(datasets) + 2  # +1 for model/chunker column, +1 for average column
    col_spec = 'll' + 'c' * (len(datasets) + 1)

    latex_lines = []
    latex_lines.append(r'\begin{table*}[htbp]')
    latex_lines.append(r'\centering')
    # Define custom light green and light red colors for cell backgrounds
    latex_lines.append(r'\definecolor{lightgreen}{RGB}{220,255,220}')
    latex_lines.append(r'\definecolor{lightred}{RGB}{255,220,220}')
    latex_lines.append(r'\caption{Contextualized-chunking (Con-C) performance relative to Pre-chunking (Pre-C) baseline. Values show percentage change. Green cells indicate improvement; red cells indicate degradation. $^*$ indicates statistically significant difference ($p < 0.05$, paired t-test).}')
    latex_lines.append(r'\label{tab:encoder_comparison}')
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
        task_row.append(r'\multicolumn{' + str(len(corpus_datasets) + 1) + r'}{c}{\textbf{\textit{Multi-documents}}}')
    latex_lines.append(' & '.join(task_row) + r' \\')

    # Add cmidrules under task row
    # Calculate column positions (0-indexed, but cmidrule uses 1-indexed)
    # Columns: [empty, empty, GutenQA, corpus datasets...]
    if doc_datasets and corpus_datasets:
        # Document: column 3, Corpus: columns 4 to (3 + len(corpus_datasets))
        latex_lines.append(r'\cmidrule(lr){3-3} \cmidrule(lr){4-' + str(3 + len(corpus_datasets) + 1) + r'}')
    elif doc_datasets:
        latex_lines.append(r'\cmidrule(lr){3-3}')
    elif corpus_datasets:
        latex_lines.append(r'\cmidrule(lr){3-' + str(2 + len(corpus_datasets) + 1) + r'}')

    # Second row: Dataset names
    dataset_row = [' & ']
    for dataset in datasets:
        dataset_row.append(dataset)
    dataset_row.append('Avg')
    latex_lines.append(' & '.join(dataset_row) + r' \\')
    latex_lines.append(r'\midrule')

    # Data rows: Group by model, then by chunker
    for model_idx, model in enumerate(models):
        model_display = get_model_display_name(model)

        # Track per-dataset percentage changes across chunkers for averaging
        dataset_pct_map = {dataset: [] for dataset in datasets}

        for chunker_idx, chunker in enumerate(chunkers):
            chunker_display = get_chunker_display_name(chunker)

            # For the first chunker of each model, add multirow for model name
            if chunker_idx == 0:
                row = [rf'\multirow{{{len(chunkers)}}}{{*}}{{\textbf{{{model_display}}}}} & {chunker_display}']
            else:
                row = [f'& {chunker_display}']

            # Add scores for each dataset
            corpus_pct_changes = []
            for dataset in datasets:
                key = (model, chunker, dataset)
                regular_score = regular_results.get(key)
                late_score = late_results.get(key)

                pct_change = None
                if regular_score is not None and late_score is not None:
                    if regular_score == 0:
                        # Avoid division by zero
                        cell_content = '---'
                    else:
                        # Calculate percentage change: ((late - regular) / regular) * 100
                        pct_change = ((late_score - regular_score) / regular_score) * 100

                        # Perform paired t-test between Con-C and Pre-C for this chunker
                        is_significant = False
                        if key in regular_per_query and key in late_per_query:
                            is_significant, _ = paired_ttest(late_per_query[key], regular_per_query[key])

                        sig_marker = '$^*$' if is_significant else ''

                        if pct_change > 0:
                            # Positive change - light green background
                            cell_content = f'\\cellcolor{{lightgreen}}{pct_change:+.2f}{sig_marker}'
                        elif pct_change < 0:
                            # Negative change - light red background
                            cell_content = f'\\cellcolor{{lightred}}{pct_change:.2f}{sig_marker}'
                        else:
                            # Zero change - no color
                            cell_content = f'{pct_change:.2f}{sig_marker}'
                else:
                    # Missing data
                    cell_content = '---'

                row.append(cell_content)
                if pct_change is not None and dataset != 'GutenQA':
                    corpus_pct_changes.append(pct_change)
                if pct_change is not None:
                    dataset_pct_map[dataset].append(pct_change)

            # Add average percentage change over corpus datasets
            if corpus_pct_changes:
                avg_pct_change = sum(corpus_pct_changes) / len(corpus_pct_changes)

                if avg_pct_change > 0:
                    avg_cell = f'\\cellcolor{{lightgreen}}{avg_pct_change:+.2f}'
                elif avg_pct_change < 0:
                    avg_cell = f'\\cellcolor{{lightred}}{avg_pct_change:.2f}'
                else:
                    avg_cell = f'{avg_pct_change:.2f}'
            else:
                avg_cell = '---'

            row.append(avg_cell)

            latex_lines.append(' & '.join(row) + r' \\')

        # Add per-model average row
        avg_row = ['& \\textbf{Avg}']
        corpus_avg_changes = []
        for dataset in datasets:
            dataset_pct_changes = dataset_pct_map[dataset]
            if dataset_pct_changes:
                dataset_avg_change = sum(dataset_pct_changes) / len(dataset_pct_changes)
                if dataset_avg_change > 0:
                    avg_row.append(f'\\cellcolor{{lightgreen}}{dataset_avg_change:+.2f}')
                elif dataset_avg_change < 0:
                    avg_row.append(f'\\cellcolor{{lightred}}{dataset_avg_change:.2f}')
                else:
                    avg_row.append(f'{dataset_avg_change:.2f}')
                if dataset != 'GutenQA':
                    corpus_avg_changes.append(dataset_avg_change)
            else:
                avg_row.append('---')

        if corpus_avg_changes:
            overall_avg_change = sum(corpus_avg_changes) / len(corpus_avg_changes)
            if overall_avg_change > 0:
                avg_cell = f'\\cellcolor{{lightgreen}}{overall_avg_change:+.2f}'
            elif overall_avg_change < 0:
                avg_cell = f'\\cellcolor{{lightred}}{overall_avg_change:.2f}'
            else:
                avg_cell = f'{overall_avg_change:.2f}'
        else:
            avg_cell = '---'

        avg_row.append(avg_cell)
        latex_lines.append(' & '.join(avg_row) + r' \\')

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
    """Main function to generate the comparison table."""

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

    # Output path (relative to current working directory)
    OUTPUT_PATH = 'results_table_encoder_comparison.tex'

    print("Collecting RegularEncoder (Pre-C) results...")
    regular_results, regular_per_query = collect_results(BASE_PATH, DATASETS, MODELS, CHUNKERS, 'RegularEncoder')
    print(f"Found {len(regular_results)} RegularEncoder result entries")

    print("Collecting LateEncoder (Con-C) results...")
    late_results, late_per_query = collect_results(BASE_PATH, DATASETS, MODELS, CHUNKERS, 'LateEncoder')
    print(f"Found {len(late_results)} LateEncoder result entries")

    print("Generating comparison table...")

    # Debug: Show some sample comparisons
    print("\nSample comparisons:")
    for i, (model, chunker, dataset) in enumerate(list(regular_results.keys())[:5]):
        reg_score = regular_results.get((model, chunker, dataset))
        late_score = late_results.get((model, chunker, dataset))
        if reg_score and late_score:
            pct = ((late_score - reg_score) / reg_score) * 100
            print(f"  {model} / {chunker} / {dataset}:")
            print(f"    Pre-C: {reg_score:.4f}, Con-C: {late_score:.4f}, Change: {pct:+.2f}%")

    latex_table = generate_comparison_table(
        regular_results, regular_per_query, late_results, late_per_query,
        DATASETS, MODELS, CHUNKERS, OUTPUT_PATH
    )

    print("\nGenerated LaTeX table:")
    print("=" * 80)
    print(latex_table)
    print("=" * 80)

    print(f"\nDone! LaTeX table saved to {OUTPUT_PATH}")
    print("\nNote: Make sure to include the following packages in your LaTeX preamble:")
    print("  \\usepackage{booktabs}")
    print("  \\usepackage{multirow}")
    print("  \\usepackage[table]{xcolor}  % [table] option required for \\cellcolor")


if __name__ == '__main__':
    main()
