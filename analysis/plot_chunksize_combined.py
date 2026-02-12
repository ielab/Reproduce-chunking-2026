#!/usr/bin/env python3
"""
Combined plot: chunk size vs performance.
Layout: 3 rows x 4 columns
  Columns: [GutenQA Jina-v3, GutenQA E5-Large, BEIR Jina-v3, BEIR E5-Large]
  Rows:    [Absolute Pre-C, Absolute Con-C, % Improvement]
Legend with colored dots for chunking methods shown once.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from chunk_size_cache import get_chunk_sizes


def parse_eval_file(eval_path: str) -> Optional[float]:
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


def resolve_chunker_dir(base_path: str, dataset: str, chunker: str) -> Optional[str]:
    results_root = Path(base_path) / dataset / 'results'
    direct = results_root / chunker
    if direct.exists():
        return direct.name
    candidates = sorted(results_root.glob(f"{chunker}-*"))
    if candidates:
        return candidates[0].name
    return None


def collect_gutenqa_scores(base_path, models, chunkers, encoder):
    results = {}
    dataset = 'GutenQA'
    for model in models:
        for chunker in chunkers:
            chunker_dir_name = resolve_chunker_dir(base_path, dataset, chunker)
            if chunker_dir_name is None:
                continue
            eval_path = os.path.join(
                base_path, dataset, 'results', chunker_dir_name,
                f'{encoder}-{model}', 'DCG@10.eval'
            )
            score = parse_eval_file(eval_path)
            if score is not None:
                results[(model, chunker)] = score
    return results


def collect_beir_average(base_path, datasets, models, chunkers, encoder):
    results = {}
    for model in models:
        for chunker in chunkers:
            scores = []
            for dataset in datasets:
                if dataset == 'GutenQA':
                    continue
                chunker_dir_name = resolve_chunker_dir(base_path, dataset, chunker)
                if chunker_dir_name is None:
                    continue
                eval_path = os.path.join(
                    base_path, dataset, 'results', chunker_dir_name,
                    f'{encoder}-{model}', 'nDCG@10.eval'
                )
                score = parse_eval_file(eval_path)
                if score is not None:
                    scores.append(score)
            if scores:
                results[(model, chunker)] = sum(scores) / len(scores)
    return results


def get_chunker_display_name(chunker_name):
    return {
        'ParagraphChunker': 'Paragraph',
        'SentenceChunker': 'Sentence',
        'FixedSizeChunker-256': 'Fixed-256',
        'SemanticChunker': 'Semantic',
        'LumberChunker-Gemini': 'Lumber',
        'Proposition-Gemini': 'Proposition',
    }.get(chunker_name, chunker_name)


def get_model_display_name(model_name):
    return {
        'jina-embeddings-v3': 'Jina-v3',
        'multilingual-e5-large-instruct': 'E5-Large',
    }.get(model_name, model_name)


def main():
    BASE_PATH = "/scratch3/wan458/chunking-reproduce/src/chunked_output"

    DATASETS = [
        'GutenQA', 'fiqa', 'nfcorpus', 'scifact', 'trec-covid', 'arguana', 'scidocs',
    ]

    MODELS = [
        'jina-embeddings-v3',
        'multilingual-e5-large-instruct',
    ]

    CHUNKERS = [
        'ParagraphChunker',
        'SentenceChunker',
        'FixedSizeChunker-256',
        'SemanticChunker',
        'LumberChunker-Gemini',
        'Proposition-Gemini',
    ]

    FALLBACK_CHUNK_SIZES = {
        'ParagraphChunker': 800,
        'SentenceChunker': 150,
        'FixedSizeChunker-256': 256,
        'SemanticChunker': 400,
        'LumberChunker-Gemini': 600,
        'Proposition-Gemini': 100,
    }

    # Collect all scores
    print("Collecting scores...")
    gutenqa_prec = collect_gutenqa_scores(BASE_PATH, MODELS, CHUNKERS, 'RegularEncoder')
    gutenqa_conc = collect_gutenqa_scores(BASE_PATH, MODELS, CHUNKERS, 'LateEncoder')
    beir_prec = collect_beir_average(BASE_PATH, DATASETS, MODELS, CHUNKERS, 'RegularEncoder')
    beir_conc = collect_beir_average(BASE_PATH, DATASETS, MODELS, CHUNKERS, 'LateEncoder')

    chunk_sizes = get_chunk_sizes(BASE_PATH, DATASETS, CHUNKERS, FALLBACK_CHUNK_SIZES)

    # Calculate improvements
    def calc_improvement(prec_scores, conc_scores):
        improvements = {}
        for key in prec_scores:
            prec = prec_scores.get(key)
            conc = conc_scores.get(key)
            if prec is not None and conc is not None and prec > 0:
                improvements[key] = ((conc - prec) / prec) * 100
        return improvements

    gutenqa_improv = calc_improvement(gutenqa_prec, gutenqa_conc)
    beir_improv = calc_improvement(beir_prec, beir_conc)

    # Color map for chunkers
    colors = plt.cm.Set2(np.linspace(0, 1, len(CHUNKERS)))
    chunker_colors = {chunker: colors[i] for i, chunker in enumerate(CHUNKERS)}

    # --- Build the grid ---
    # Columns: GutenQA-Jina-v3, GutenQA-E5-Large, BEIR-Jina-v3, BEIR-E5-Large
    # Rows:    Absolute Pre-C, Absolute Con-C, % Improvement
    col_configs = [
        ('GutenQA', MODELS[0]),
        ('GutenQA', MODELS[1]),
        ('BEIR', MODELS[0]),
        ('BEIR', MODELS[1]),
    ]

    row_configs = [
        ('Pre-C', gutenqa_prec, beir_prec),
        ('Con-C', gutenqa_conc, beir_conc),
        ('Improvement', gutenqa_improv, beir_improv),
    ]

    # Y-axis limits per row
    y_limits = {
        0: {'GutenQA': (0, 1.0), 'BEIR': (0.25, 0.55)},    # Pre-C
        1: {'GutenQA': (0, 1.0), 'BEIR': (0.25, 0.55)},    # Con-C
        2: {'GutenQA': (-70, 20), 'BEIR': (-7, 30)},        # Improvement
    }

    # Y-axis labels
    y_labels = {
        0: {'GutenQA': 'DCG@10 (Pre-C)', 'BEIR': 'nDCG@10 (Pre-C)'},
        1: {'GutenQA': 'DCG@10 (Con-C)', 'BEIR': 'nDCG@10 (Con-C)'},
        2: {'GutenQA': '% Improvement', 'BEIR': '% Improvement'},
    }

    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(25, 10))

    # Use GridSpec with 5 columns: [GutenQA1, GutenQA2, spacer, BEIR1, BEIR2]
    gs = gridspec.GridSpec(3, 5, figure=fig,
                           width_ratios=[1, 1, 0.15, 1, 1],
                           wspace=0.05, hspace=0.10,
                           left=0.06, right=0.98, top=0.92, bottom=0.14)

    # Map logical col_idx (0-3) to gridspec columns (0,1,3,4) skipping spacer col 2
    gs_col_map = [0, 1, 3, 4]

    axes = {}
    for row_idx in range(3):
        for col_idx in range(4):
            axes[(row_idx, col_idx)] = fig.add_subplot(gs[row_idx, gs_col_map[col_idx]])

    for row_idx, (row_label, gutenqa_scores, beir_scores) in enumerate(row_configs):
        for col_idx, (task_name, model) in enumerate(col_configs):
            ax = axes[(row_idx, col_idx)]

            scores = gutenqa_scores if task_name == 'GutenQA' else beir_scores

            x_vals, y_vals, point_colors = [], [], []
            for chunker in CHUNKERS:
                key = (model, chunker)
                if key in scores:
                    x_vals.append(chunk_sizes[chunker])
                    y_vals.append(scores[key])
                    point_colors.append(chunker_colors[chunker])

            ax.scatter(x_vals, y_vals, c=point_colors, s=200, edgecolors='black', linewidths=2, zorder=3)

            # Trend line + correlation
            if len(x_vals) >= 2:
                z = np.polyfit(x_vals, y_vals, 1)
                p = np.poly1d(z)
                x_line = np.linspace(min(x_vals) - 50, max(x_vals) + 50, 100)
                ax.plot(x_line, p(x_line), 'r--', alpha=0.5, linewidth=2)

                corr, p_value = pearsonr(x_vals, y_vals)
                sig_marker = '*' if p_value < 0.05 else ''
                ax.text(0.95, 0.95, f'r = {corr:.2f}{sig_marker}', transform=ax.transAxes, fontsize=18,
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            # Horizontal line at y=0 for improvement row
            if row_idx == 2:
                ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)

            # Y-axis limits
            ylim = y_limits[row_idx][task_name]
            ax.set_ylim(ylim)

            # Share y-axis within each task group
            if col_idx == 1:
                ax.sharey(axes[(row_idx, 0)])
            elif col_idx == 3:
                ax.sharey(axes[(row_idx, 2)])

            # Y-axis label only on leftmost column of each task group
            if col_idx == 0:
                ax.set_ylabel(y_labels[row_idx]['GutenQA'], fontsize=18)
            elif col_idx == 2:
                ax.set_ylabel(y_labels[row_idx]['BEIR'], fontsize=18)

            # Hide y tick labels on cols 1 and 3
            if col_idx in (1, 3):
                ax.tick_params(axis='y', labelleft=False)

            # Column titles on the top row only
            if row_idx == 0:
                ax.set_title(f'{get_model_display_name(model)}', fontsize=20, fontweight='bold')

            ax.tick_params(axis='both', labelsize=16)

            # Hide x tick labels on rows 0 and 1
            if row_idx < 2:
                ax.tick_params(axis='x', labelbottom=False)

            # X-axis label only on bottom-right corner
            if row_idx == 2 and col_idx == 3:
                ax.set_xlabel('Avg Chunk Size (chars)', fontsize=18)

    # Add task group labels at the top
    # Center over cols 0-1 and cols 3-4 of the gridspec
    fig.text(0.265, 0.96, 'In-document (GutenQA)', ha='center', fontsize=22, fontweight='bold')
    fig.text(0.755, 0.96, 'In-corpus (BEIR)', ha='center', fontsize=22, fontweight='bold')

    # Legend with colored dots for chunking methods
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=chunker_colors[c],
                   markeredgecolor='black', markeredgewidth=1.5, markersize=14,
                   label=get_chunker_display_name(c))
        for c in CHUNKERS
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=len(CHUNKERS),
               fontsize=18, frameon=True, fancybox=True, shadow=False,
               borderpad=0.8, handletextpad=0.5, columnspacing=1.5,
               bbox_to_anchor=(0.52, 0.005))

    os.makedirs('figures', exist_ok=True)
    output_path = 'figures/rq4_chunksize_combined.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.savefig('figures/rq4_chunksize_combined.png', dpi=150, bbox_inches='tight')
    print(f"Plot also saved as PNG")
    plt.show()


if __name__ == '__main__':
    main()
