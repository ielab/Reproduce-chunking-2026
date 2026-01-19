#!/usr/bin/env python3
"""
Utility script to compute GutenQA DCG metrics directly from a result.trec file.

Example:
    python scripts/check_gutenqa_dcg.py \
        --source-path src/chunked_output \
        --dataset-name GutenQA \
        --chunk-run-id Proposition-Gemini \
        --query-run-id 20250924-133046-GutenQA-2266af06 \
        --trec-file /scratch3/.../result.trec \
        --query-id Book-0-Query-0
"""

import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

# Ensure repo root is on sys.path when running as a standalone script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.io import load_chunks, load_queries, load_trec_file
from src.types import Query
from src.evaluators.qutenqa_evaluator import (
    compute_DCG,
    compute_length_normalized_dcg,
    compute_length_normalized_recall,
    compute_relevance_and_penalties,
    compute_Recall,
)
from src.runner import _is_gutenqa_proposition_run


def _load_chunk_texts(source_path: str, dataset_name: str, chunk_run_id: str) -> Dict[str, str]:
    """
    Load chunk text for the given run. Proposition runs reuse the paragraph chunks.
    """
    dataset_dir = os.path.join(source_path, dataset_name, "chunks")
    if _is_gutenqa_proposition_run(dataset_name, chunk_run_id):
        chunk_dir = "ParagraphChunker"
    else:
        chunk_dir = chunk_run_id

    chunk_path = os.path.join(dataset_dir, chunk_dir, "chunks.jsonl")
    chunks = load_chunks(chunk_path)
    return {chunk.chunk_id: chunk.text for chunk in chunks}


def _load_queries(source_path: str, dataset_name: str, query_run_id: str) -> Dict[str, Query]:
    query_path = os.path.join(source_path, dataset_name, "queries", query_run_id, "queries.jsonl")
    return {query.query_id: query for query in load_queries(query_path)}


def _sorted_ranking_from_trec(trec_path: str) -> Dict[str, List[Tuple[str, float]]]:
    """
    Convert a TREC file into {query_id: [(chunk_id, score), ...]} sorted by score desc.
    """
    raw = load_trec_file(trec_path)
    return {
        qid: sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
        for qid, doc_scores in raw.items()
    }


def _compute_query_metrics(
    chunk_id2text: Dict[str, str],
    query,
    ranking: List[Tuple[str, float]],
    k_values: List[int],
) -> Dict[str, float]:
    re_chunk_list = [[chunk_id2text.get(chunk_id)] for chunk_id, _ in ranking]
    relevance, penalties = compute_relevance_and_penalties(re_chunk_list, query.chunk_must_Contain)

    metrics: Dict[str, float] = {}
    for k in k_values:
        metrics[f"DCG@{k}"] = compute_DCG(relevance[:k])
        metrics[f"Recall@{k}"] = compute_Recall(relevance[:k])
        metrics[f"lnDCG@{k}"] = compute_length_normalized_dcg(relevance[:k], penalties[:k])
        metrics[f"lnRecall@{k}"] = compute_length_normalized_recall(relevance[:k], penalties[:k])

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect GutenQA DCG metrics from a TREC file.")
    parser.add_argument("--source-path", default="src/chunked_output", help="Base directory containing dataset outputs.")
    parser.add_argument("--dataset-name", default="GutenQA")
    parser.add_argument("--chunk-run-id", required=True, help="Chunk run identifier (e.g., ParagraphChunker or Proposition-Gemini).")
    parser.add_argument("--query-run-id", required=True, help="Query run identifier (folder name under dataset/queries).")
    parser.add_argument("--trec-file", required=True, help="Path to the result.trec file to inspect.")
    parser.add_argument("--query-id", help="Restrict output to a single query id.")
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 2, 5, 10, 20], help="Cutoffs at which to compute metrics.")
    return parser.parse_args()


def main():
    args = parse_args()

    chunk_id2text = _load_chunk_texts(args.source_path, args.dataset_name, args.chunk_run_id)
    queries = _load_queries(args.source_path, args.dataset_name, args.query_run_id)
    rankings = _sorted_ranking_from_trec(args.trec_file)

    if args.query_id:
        query_ids = [args.query_id]
    else:
        # Intersect: only queries present in both TREC file and queries list
        query_ids = [qid for qid in rankings if qid in queries]

    if not query_ids:
        raise ValueError("No overlapping queries between the TREC file and the provided query set.")

    aggregate = defaultdict(list)

    processed_queries = 0

    for qid in query_ids:
        query = queries.get(qid)
        if query is None:
            print(f"[WARN] Query {qid} not found in query file. Skipping.")
            continue

        metrics = _compute_query_metrics(chunk_id2text, query, rankings.get(qid, []), args.k_values)
        print(f"\nQuery: {qid}")
        print(f"  must_contain: {query.chunk_must_Contain}")
        for k in args.k_values:
            print(
                f"  DCG@{k}: {metrics[f'DCG@{k}']:.4f} | "
                f"Recall@{k}: {metrics[f'Recall@{k}']:.4f} | "
                f"lnDCG@{k}: {metrics[f'lnDCG@{k}']:.4f} | "
                f"lnRecall@{k}: {metrics[f'lnRecall@{k}']:.4f}"
        )
        for name, value in metrics.items():
            aggregate[name].append(value)
        processed_queries += 1

    if processed_queries == 0:
        raise ValueError("None of the requested queries were found in the query set.")

    print("\nOverall averages across processed queries:")
    for k in args.k_values:
        dcg_avg = sum(aggregate[f"DCG@{k}"]) / len(aggregate[f"DCG@{k}"])
        recall_avg = sum(aggregate[f"Recall@{k}"]) / len(aggregate[f"Recall@{k}"])
        ln_dcg_avg = sum(aggregate[f"lnDCG@{k}"]) / len(aggregate[f"lnDCG@{k}"])
        ln_recall_avg = sum(aggregate[f"lnRecall@{k}"]) / len(aggregate[f"lnRecall@{k}"])
        print(
            f"  DCG@{k}: {dcg_avg:.4f} | Recall@{k}: {recall_avg:.4f} | "
            f"lnDCG@{k}: {ln_dcg_avg:.4f} | lnRecall@{k}: {ln_recall_avg:.4f}"
        )


if __name__ == "__main__":
    main()
