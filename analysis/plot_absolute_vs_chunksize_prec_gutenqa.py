#!/usr/bin/env python3
"""
Plot showing absolute Pre-C (RegularEncoder) performance vs average chunk size for GutenQA.
One subplot per model.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from adjustText import adjust_text

from chunk_size_cache import get_chunk_sizes


def parse_eval_file(eval_path: str) -> Optional[float]:
    """Parse an eval file and extract the average score."""
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
    """Resolve chunker directory name."""
    results_root = Path(base_path) / dataset / 'results'
    direct = results_root / chunker
    if direct.exists():
        return direct.name
    candidates = sorted(results_root.glob(f"{chunker}-*"))
    if candidates:
        return candidates[0].name
    return None


def collect_gutenqa_scores(
    base_path: str,
    models: List[str],
    chunkers: List[str],
    encoder: str
) -> Dict[Tuple[str, str], float]:
    """Collect GutenQA DCG@10 scores for each (model, chunker)."""
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


def get_chunker_display_name(chunker_name: str) -> str:
    chunker_mapping = {
        'ParagraphChunker': 'Paragraph',
        'SentenceChunker': 'Sentence',
        'FixedSizeChunker-256': 'Fixed-256',
        'SemanticChunker': 'Semantic',
        'LumberChunker-Gemini': 'Lumber',
        'Proposition-Gemini': 'Proposition',
    }
    return chunker_mapping.get(chunker_name, chunker_name)


def get_model_display_name(model_name: str) -> str:
    model_mapping = {
        'jina-embeddings-v2-small-en': 'Jina-v2',
        'jina-embeddings-v3': 'Jina-v3',
        'nomic-embed-text-v1': 'Nomic',
        'multilingual-e5-large-instruct': 'E5-Large',
    }
    return model_mapping.get(model_name, model_name)


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

    print("Collecting Pre-C (RegularEncoder) GutenQA results...")
    scores = collect_gutenqa_scores(BASE_PATH, MODELS, CHUNKERS, 'RegularEncoder')
    print(f"Found {len(scores)} entries")

    # Get chunk sizes (using cache)
    chunk_sizes = get_chunk_sizes(BASE_PATH, DATASETS, CHUNKERS, FALLBACK_CHUNK_SIZES)

    # Create the plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    colors = plt.cm.Set2(np.linspace(0, 1, len(CHUNKERS)))
    chunker_colors = {chunker: colors[i] for i, chunker in enumerate(CHUNKERS)}

    # Fixed y-axis limits
    y_min, y_max = 0, 1

    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        x_vals, y_vals, labels, point_colors = [], [], [], []

        for chunker in CHUNKERS:
            key = (model, chunker)
            if key in scores:
                x_vals.append(chunk_sizes[chunker])
                y_vals.append(scores[key])
                labels.append(get_chunker_display_name(chunker))
                point_colors.append(chunker_colors[chunker])

        scatter = ax.scatter(x_vals, y_vals, c=point_colors, s=200, edgecolors='black', linewidths=2, zorder=3)

        texts = []
        for x, y, label in zip(x_vals, y_vals, labels):
            texts.append(ax.text(x, y, label, fontsize=14, fontweight='medium'))
        adjust_text(texts, ax=ax, force_text=(2, 2), force_points=(2, 2),
                    arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

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

        if idx == 0:
            ax.set_ylabel('DCG@10 (Pre-C)', fontsize=18)

        ax.set_title(f'{get_model_display_name(model)}', fontsize=18, fontweight='bold')
        ax.tick_params(axis='both', labelsize=14)
        ax.set_ylim(y_min, y_max)

    plt.subplots_adjust(wspace=0.05, left=0.06, right=0.98, top=0.92, bottom=0.15)
    fig.text(0.52, 0.03, 'Average Chunk Size (characters)', ha='center', fontsize=18)

    os.makedirs('figures', exist_ok=True)
    output_path = 'figures/rq4_chunksize_vs_absolute_prec_gutenqa.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.savefig('figures/rq4_chunksize_vs_absolute_prec_gutenqa.png', dpi=150, bbox_inches='tight')
    print(f"Plot also saved as PNG")
    plt.show()


if __name__ == '__main__':
    main()
