#!/usr/bin/env python3
"""
RQ4 plot: Number of chunks per document vs retrieval performance.

Generates one figure per model (4 models total).

Both panels use avg chunk size in tokens (model-specific tokenizer) on the x-axis.

For GutenQA (in-document): per-book scatter plot.
  - x-axis: avg chunk size in tokens for that book
  - y-axis: DCG@10 averaged over 30 queries per book

For BEIR (in-corpus): per-document scatter, pooled across datasets.
  - x-axis: avg chunk size in tokens for that document
  - y-axis: avg nDCG@10 across queries where this document is relevant

Layout per figure: 3 rows x 4 columns
  Columns: [GutenQA Structure-based, GutenQA Semantic/LLM-guided, BEIR Structure-based, BEIR Semantic/LLM-guided]
  Rows:    [Pre-C, Con-C, % Improvement]
"""

import os
import json
import hashlib
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.stats import pearsonr
from transformers import AutoTokenizer

# ── Configuration ────────────────────────────────────────────────────────────

_REMOTE_PATH = "/scratch3/wan458/chunking-reproduce/src/chunked_output"
_LOCAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'chunked_output')
BASE_PATH = _REMOTE_PATH if os.path.isdir(_REMOTE_PATH) else _LOCAL_PATH

MODELS = [
    'jina-embeddings-v2-small-en',
    'jina-embeddings-v3',
    'nomic-embed-text-v1',
    'multilingual-e5-large-instruct',
]

# Chunker groups
HEURISTIC_CHUNKERS = [
    'ParagraphChunker',
    'SentenceChunker',
    'FixedSizeChunker-256',
]

SEMANTIC_CHUNKERS = [
    'SemanticChunker',
    'LumberChunker-Gemini',
    'Proposition-Gemini',
]

ALL_CHUNKERS = HEURISTIC_CHUNKERS + SEMANTIC_CHUNKERS

CHUNKER_DISPLAY = {
    'ParagraphChunker': 'Paragraph',
    'SentenceChunker': 'Sentence',
    'FixedSizeChunker-256': 'Fixed-Size',
    'SemanticChunker': 'Semantic',
    'LumberChunker-Gemini': 'LumberChunker',
    'Proposition-Gemini': 'Proposition',
}

MODEL_DISPLAY = {
    'jina-embeddings-v2-small-en': 'Jina-v2',
    'jina-embeddings-v3': 'Jina-v3',
    'nomic-embed-text-v1': 'Nomic',
    'multilingual-e5-large-instruct': 'E5-Large',
}

# HuggingFace model IDs for tokenizer loading
MODEL_HF_ID = {
    'jina-embeddings-v2-small-en': 'jinaai/jina-embeddings-v2-small-en',
    'jina-embeddings-v3': 'jinaai/jina-embeddings-v3',
    'nomic-embed-text-v1': 'nomic-ai/nomic-embed-text-v1',
    'multilingual-e5-large-instruct': 'intfloat/multilingual-e5-large-instruct',
}

# BEIR datasets to include (comment out any you want to exclude)
BEIR_DATASETS = [
    'fiqa',
    'nfcorpus',
    'scifact',
    'trec-covid',
    'arguana',
    'scidocs',
]
BEIR_LABEL = 'BEIR'  # Label shown in the figure title


# ── Data loading ─────────────────────────────────────────────────────────────

def count_chunks_per_doc(dataset: str, chunker: str) -> Dict[str, int]:
    """Count number of chunks per document for a given chunker."""
    chunks_path = Path(BASE_PATH) / dataset / 'chunks' / chunker / 'chunks.jsonl'
    if not chunks_path.exists():
        return {}
    counts = Counter()
    with open(chunks_path) as f:
        for line in f:
            try:
                d = json.loads(line.strip())
                counts[d['doc_id']] += 1
            except (json.JSONDecodeError, KeyError):
                pass
    return dict(counts)


CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.chunk_token_cache')
BATCH_SIZE = 4096  # texts per tokenizer batch

# Global tokenizer cache to avoid reloading
_tokenizers: Dict[str, object] = {}


def _get_tokenizer(model: str):
    """Load and cache a fast (Rust-backed) tokenizer."""
    if model not in _tokenizers:
        hf_id = MODEL_HF_ID.get(model, model)
        print(f"  Loading tokenizer for {hf_id}...")
        _tokenizers[model] = AutoTokenizer.from_pretrained(
            hf_id, trust_remote_code=True, use_fast=True)
    return _tokenizers[model]


def avg_chunk_tokens_per_doc(dataset: str, chunker: str, model: str) -> Dict[str, float]:
    """
    Compute average chunk size (in tokens) per document for a given chunker and model.
    Uses fast tokenizer with batched encoding. Results are cached to disk.
    """
    # Check cache first
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_key = f"{dataset}_{chunker}_{model.replace('/', '_')}"
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    chunks_path = Path(BASE_PATH) / dataset / 'chunks' / chunker / 'chunks.jsonl'
    if not chunks_path.exists():
        return {}

    tokenizer = _get_tokenizer(model)

    # Read all chunks, keep (doc_id, text) pairs
    doc_ids = []
    texts = []
    with open(chunks_path) as f:
        for line in f:
            try:
                d = json.loads(line.strip())
                doc_ids.append(d['doc_id'])
                texts.append(d.get('text', ''))
            except (json.JSONDecodeError, KeyError):
                pass

    if not texts:
        return {}

    total = len(texts)
    print(f"  Tokenizing {total} chunks for {dataset}/{chunker} with {model}...")

    # Batch tokenize all chunks at once for maximum throughput
    token_counts = []
    for i in range(0, total, BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        encoded = tokenizer(batch, add_special_tokens=False,
                            return_attention_mask=False,
                            return_token_type_ids=False)
        token_counts.extend(len(ids) for ids in encoded['input_ids'])
        if (i + BATCH_SIZE) % (BATCH_SIZE * 10) == 0:
            print(f"    {min(i + BATCH_SIZE, total)}/{total} done")

    # Aggregate per document
    doc_total_tokens = defaultdict(int)
    doc_chunk_count = defaultdict(int)
    for doc_id, tc in zip(doc_ids, token_counts):
        doc_total_tokens[doc_id] += tc
        doc_chunk_count[doc_id] += 1

    result = {doc_id: doc_total_tokens[doc_id] / doc_chunk_count[doc_id]
              for doc_id in doc_total_tokens}

    # Save cache
    with open(cache_path, 'w') as f:
        json.dump(result, f)
    print(f"  Cached to {cache_path}")

    return result


def resolve_chunker_dir(dataset: str, chunker: str) -> Optional[str]:
    """Resolve chunker directory name (handles suffixed names)."""
    results_root = Path(BASE_PATH) / dataset / 'results'
    if (results_root / chunker).exists():
        return chunker
    candidates = sorted(results_root.glob(f"{chunker}-*"))
    return candidates[0].name if candidates else None


def load_per_query_scores(dataset: str, chunker: str, encoder: str,
                          model: str, metric: str) -> Dict[str, float]:
    """Load per-query scores from an eval file."""
    chunker_dir = resolve_chunker_dir(dataset, chunker)
    if chunker_dir is None:
        return {}
    eval_path = (Path(BASE_PATH) / dataset / 'results' / chunker_dir
                 / f'{encoder}-{model}' / f'{metric}.eval')
    if not eval_path.exists():
        return {}
    scores = {}
    with open(eval_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2 and parts[0] != 'average':
                try:
                    scores[parts[0]] = float(parts[1])
                except ValueError:
                    pass
    return scores


def load_qrels(dataset: str) -> Dict[str, List[str]]:
    """Load qrels: query_id -> list of relevant doc_ids."""
    # Try repo-relative path first, then alongside BASE_PATH
    candidates = [
        Path(os.path.dirname(os.path.abspath(__file__))) / '..' / 'src' / 'data' / dataset / 'qrels' / 'test.tsv',
        Path(BASE_PATH).parent / 'data' / dataset / 'qrels' / 'test.tsv',
    ]
    qrels_path = None
    for c in candidates:
        if c.exists():
            qrels_path = c
            break
    if qrels_path is None:
        return {}
    qrels = defaultdict(list)
    with open(qrels_path) as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                qid, did = parts[0], parts[1]
                try:
                    score = int(parts[2])
                except ValueError:
                    continue
                if score > 0:
                    qrels[qid].append(did)
    return dict(qrels)


# ── GutenQA: per-book aggregation ────────────────────────────────────────────

def get_gutenqa_book_data(chunker: str, encoder: str, model: str):
    """
    Returns dict: book_id -> (avg_chunk_tokens, avg_score).
    Each book has 30 queries; we average them.
    x-axis: average chunk size in tokens (using the model's tokenizer).
    """
    chunk_sizes = avg_chunk_tokens_per_doc('GutenQA', chunker, model)
    if not chunk_sizes:
        return {}

    scores = load_per_query_scores('GutenQA', chunker, encoder, model, 'DCG@10')
    if not scores:
        return {}

    # Group query scores by book
    book_scores = defaultdict(list)
    for qid, score in scores.items():
        book = '-'.join(qid.split('-')[:2])  # Book-X-Query-Y -> Book-X
        book_scores[book].append(score)

    result = {}
    for book, qscores in book_scores.items():
        if book in chunk_sizes:
            result[book] = (chunk_sizes[book], np.mean(qscores))
    return result


# ── BEIR: per-document analysis ───────────────────────────────────────────────

def get_beir_query_data(dataset: str, chunker: str, encoder: str, model: str):
    """
    Returns dict: query_id -> (avg_chunk_tokens_of_relevant_docs, nDCG@10).
    Per-query scatter: each point is a query.
      x = avg chunk size (tokens) of the query's relevant documents.
      y = nDCG@10 for that query.
    """
    chunk_sizes = avg_chunk_tokens_per_doc(dataset, chunker, model)
    if not chunk_sizes:
        return {}

    scores = load_per_query_scores(dataset, chunker, encoder, model, 'nDCG@10')
    if not scores:
        return {}

    qrels = load_qrels(dataset)
    if not qrels:
        return {}

    result = {}
    for qid, rel_docs in qrels.items():
        if qid not in scores:
            continue
        # Avg chunk size of this query's relevant documents
        doc_sizes = [chunk_sizes[did] for did in rel_docs if did in chunk_sizes]
        if doc_sizes:
            result[qid] = (np.mean(doc_sizes), scores[qid])
    return result


# ── Plotting helpers ─────────────────────────────────────────────────────────

def remove_outliers(x, y, factor=1.5):
    """Remove outliers using IQR on both x and y."""
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    mask = np.ones(len(x), dtype=bool)
    for arr in [x, y]:
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        mask &= (arr >= q1 - factor * iqr) & (arr <= q3 + factor * iqr)
    return x[mask].tolist(), y[mask].tolist()


def add_trend_and_corr(ax, x_arr, y_arr, color='black', linewidth=2, alpha=0.6):
    """Add log-linear trend line. Returns (corr, pval, sig_marker) or None."""
    if len(x_arr) < 5:
        return None
    x_arr, y_arr = np.array(x_arr, dtype=float), np.array(y_arr, dtype=float)
    # Filter out zero/negative x values before log transform
    mask = x_arr > 0
    x_arr, y_arr = x_arr[mask], y_arr[mask]
    if len(x_arr) < 5:
        return None
    log_x = np.log10(x_arr)

    # Skip if x is constant (no variance to fit/correlate)
    if np.ptp(log_x) == 0:
        return None

    z = np.polyfit(log_x, y_arr, 1)
    p = np.poly1d(z)
    x_line = np.linspace(log_x.min(), log_x.max(), 100)
    ax.plot(10**x_line, p(x_line), '--', color=color, alpha=alpha,
            linewidth=linewidth, zorder=4)

    corr, pval = pearsonr(log_x, y_arr)
    if np.isnan(corr):
        return None
    sig = '*' if pval < 0.05 else ''
    return corr, pval, sig


def annotate_corr_list(ax, corr_entries):
    """
    Display stacked per-method r values in a box in top-right corner.
    corr_entries: list of (display_name, corr, sig, color)
    """
    lines = [f'{name}: r={corr:.2f}{sig}' for name, corr, sig, _ in corr_entries]
    text = '\n'.join(lines)
    ax.text(0.97, 0.97, text,
            transform=ax.transAxes, fontsize=12,
            va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


# ── Main ─────────────────────────────────────────────────────────────────────

def generate_figure_for_model(model: str, chunker_colors: dict):
    """Generate one 3x4 figure for a single model."""
    model_display = MODEL_DISPLAY.get(model, model)
    model_slug = model  # used in filenames

    # ── Collect all data ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Loading data for {model_display} ({model})...")
    print(f"{'='*60}")
    gutenqa_data = {}  # (encoder, chunker) -> {book_id: (avg_tokens, score)}
    beir_data = {}     # (encoder, chunker) -> {uid: (avg_tokens, score)}

    for encoder in ['RegularEncoder', 'LateEncoder']:
        for chunker in ALL_CHUNKERS:
            gd = get_gutenqa_book_data(chunker, encoder, model)
            if gd:
                gutenqa_data[(encoder, chunker)] = gd
                print(f"  GutenQA {encoder} {CHUNKER_DISPLAY[chunker]}: {len(gd)} books")

            # Pool across all BEIR datasets (per-query)
            pooled = {}
            for beir_ds in BEIR_DATASETS:
                bd = get_beir_query_data(beir_ds, chunker, encoder, model)
                if bd:
                    for qid, val in bd.items():
                        pooled[f"{beir_ds}:{qid}"] = val
                    print(f"  {beir_ds} {encoder} {CHUNKER_DISPLAY[chunker]}: {len(bd)} queries")
            if pooled:
                beir_data[(encoder, chunker)] = pooled

    # ── Figure: 3 rows × 4 cols ──────────────────────────────────────────
    col_configs = [
        ('GutenQA', 'Structure-based', HEURISTIC_CHUNKERS),
        ('GutenQA', 'Semantic and LLM-guided', SEMANTIC_CHUNKERS),
        (BEIR_LABEL, 'Structure-based', HEURISTIC_CHUNKERS),
        (BEIR_LABEL, 'Semantic and LLM-guided', SEMANTIC_CHUNKERS),
    ]

    row_configs = [
        ('RegularEncoder', 'Pre-C'),
        ('LateEncoder', 'Con-C'),
    ]

    # Y-axis limits per row
    y_limits = {
        0: {'GutenQA': (0, 1), 'BEIR': (0, 1)},
        1: {'GutenQA': (0, 1), 'BEIR': (0, 1)},
        2: {'GutenQA': (-100, 120), 'BEIR': (-100, 120)},
    }

    fig = plt.figure(figsize=(25, 10))
    gs = gridspec.GridSpec(3, 5, figure=fig,
                           width_ratios=[1, 1, 0.15, 1, 1],
                           wspace=0.05, hspace=0.12,
                           left=0.06, right=0.98, top=0.90, bottom=0.14)
    gs_col_map = [0, 1, 3, 4]

    axes = {}
    for ri in range(3):
        for ci in range(4):
            axes[(ri, ci)] = fig.add_subplot(gs[ri, gs_col_map[ci]])

    # ── Rows 0-1: Pre-C and Con-C ───────────────────────────────────────
    for ri, (encoder, row_label) in enumerate(row_configs):
        for ci, (task, group_label, chunker_list) in enumerate(col_configs):
            ax = axes[(ri, ci)]
            is_gutenqa = (task == 'GutenQA')
            data_store = gutenqa_data if is_gutenqa else beir_data

            corr_entries = []
            for chunker in chunker_list:
                key = (encoder, chunker)
                if key not in data_store:
                    continue
                entries = data_store[key]
                x = [v[0] for v in entries.values()]
                y = [v[1] for v in entries.values()]
                x, y = remove_outliers(x, y)
                ax.scatter(x, y, c=[chunker_colors[chunker]], s=50 if is_gutenqa else 15,
                           alpha=0.75 if is_gutenqa else 0.6, edgecolors='none', zorder=3)
                result = add_trend_and_corr(ax, x, y, color='black',
                                            linewidth=2, alpha=0.8)
                if result:
                    corr, _, sig = result
                    corr_entries.append((CHUNKER_DISPLAY[chunker], corr, sig,
                                        chunker_colors[chunker]))

            if corr_entries:
                annotate_corr_list(ax, corr_entries)

            ax.set_xscale('log')

            ylim_key = 'GutenQA' if is_gutenqa else 'BEIR'
            ax.set_ylim(y_limits[ri][ylim_key])

            if ci == 0:
                ax.set_ylabel(f'DCG@10 ({row_label})', fontsize=18)
            elif ci == 2:
                ax.set_ylabel(f'nDCG@10 ({row_label})', fontsize=18)

            if ci == 1:
                ax.sharey(axes[(ri, 0)])
                ax.tick_params(axis='y', labelleft=False)
            elif ci == 3:
                ax.sharey(axes[(ri, 2)])
                ax.tick_params(axis='y', labelleft=False)

            if ri == 0:
                ax.set_title(group_label, fontsize=20)
            ax.tick_params(axis='both', labelsize=16)
            if ri < 2:
                ax.tick_params(axis='x', labelbottom=False)

    # ── Row 2: % Improvement ─────────────────────────────────────────────
    for ci, (task, group_label, chunker_list) in enumerate(col_configs):
        ax = axes[(2, ci)]
        is_gutenqa = (task == 'GutenQA')
        data_store = gutenqa_data if is_gutenqa else beir_data

        corr_entries = []
        for chunker in chunker_list:
            key_pre = ('RegularEncoder', chunker)
            key_con = ('LateEncoder', chunker)
            if key_pre not in data_store or key_con not in data_store:
                continue
            pre_entries = data_store[key_pre]
            con_entries = data_store[key_con]

            common_ids = set(pre_entries.keys()) & set(con_entries.keys())
            x_imp, y_imp = [], []
            for doc_id in common_ids:
                nchunks_pre, score_pre = pre_entries[doc_id]
                _, score_con = con_entries[doc_id]
                if score_pre > 0:
                    improvement = ((score_con - score_pre) / score_pre) * 100
                    x_imp.append(nchunks_pre)
                    y_imp.append(improvement)

            x_imp, y_imp = remove_outliers(x_imp, y_imp)
            if x_imp:
                ax.scatter(x_imp, y_imp, c=[chunker_colors[chunker]],
                           s=50 if is_gutenqa else 15,
                           alpha=0.75 if is_gutenqa else 0.6,
                           edgecolors='none', zorder=3)
                result = add_trend_and_corr(ax, x_imp, y_imp, color='black',
                                            linewidth=2, alpha=0.8)
                if result:
                    corr, _, sig = result
                    corr_entries.append((CHUNKER_DISPLAY[chunker], corr, sig,
                                        chunker_colors[chunker]))

        if corr_entries:
            annotate_corr_list(ax, corr_entries)

        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)

        ax.set_xscale('log')

        ylim_key = 'GutenQA' if is_gutenqa else 'BEIR'
        ax.set_ylim(y_limits[2][ylim_key])

        if ci == 0:
            ax.set_ylabel('% Improvement', fontsize=18)
        elif ci == 2:
            ax.set_ylabel('% Improvement', fontsize=18)
        if ci == 1:
            ax.sharey(axes[(2, 0)])
            ax.tick_params(axis='y', labelleft=False)
        elif ci == 3:
            ax.sharey(axes[(2, 2)])
            ax.tick_params(axis='y', labelleft=False)

        ax.tick_params(axis='both', labelsize=16)
        if ci == 0 or ci == 3:
            ax.set_xlabel('Avg Chunk Size (tokens)', fontsize=18)

    # ── Suptitle with model name ─────────────────────────────────────────
    fig.suptitle(f'{model_display}', fontsize=26, fontweight='bold', y=0.99)

    # ── Group labels ─────────────────────────────────────────────────────
    fig.text(0.265, 0.94, 'In-document (GutenQA)', ha='center', fontsize=22, fontweight='bold')
    beir_title = f'In-corpus ({BEIR_LABEL})' if len(BEIR_DATASETS) > 1 else f'In-corpus ({BEIR_DATASETS[0]})'
    fig.text(0.755, 0.94, beir_title, ha='center', fontsize=22, fontweight='bold')

    # ── Legend ────────────────────────────────────────────────────────────
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=chunker_colors[c], markersize=14,
                   markeredgecolor='black', markeredgewidth=1.5,
                   label=CHUNKER_DISPLAY[c])
        for c in ALL_CHUNKERS
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=len(ALL_CHUNKERS),
               fontsize=18, frameon=True, fancybox=True, borderpad=0.8,
               handletextpad=0.5, columnspacing=1.5,
               bbox_to_anchor=(0.52, 0.005))

    os.makedirs('figures', exist_ok=True)
    for ext in ['pdf', 'png']:
        out = f'figures/rq4_chunkcount_vs_performance_{model_slug}.{ext}'
        fig.savefig(out, dpi=200, bbox_inches='tight')
        print(f"Saved: {out}")
    plt.close()


def main():
    colors = plt.cm.Set2(np.linspace(0, 1, len(ALL_CHUNKERS)))
    chunker_colors = {c: colors[i] for i, c in enumerate(ALL_CHUNKERS)}

    for model in MODELS:
        generate_figure_for_model(model, chunker_colors)

    print("\nDone. Generated figures for all models.")


if __name__ == '__main__':
    main()
