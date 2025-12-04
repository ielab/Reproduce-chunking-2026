from typing import List, Dict, Tuple
from collections import defaultdict
import numpy as np
from tqdm import tqdm

from src.evaluators.base_evaluator import BaseEvaluator
from src.registry import EVALUATOR_REG
from src.types import Query, Chunk, ChunkEmbedding, QueryEmbedding


from src.evaluators.ranker import SimpleRanker


def _token_length(text: str) -> int:
    return max(1, len(text.split()))


def compute_relevance_and_penalties(retrieval_chunk_list: List[List[str]], gold_label: str) -> Tuple[List[int], List[float]]:
    """
    Compute relevance and penalties for retrieved chunks.
    Now handles multiple propositions per chunk (for proposition-based chunking).

    Args:
        retrieval_chunk_list: List of chunks, where each chunk is a list of propositions (or single-item list for regular chunks)
        gold_label: The gold answer string to match

    Returns:
        relevance: Binary relevance scores (1 if gold label found in any proposition, 0 otherwise)
        penalties: Length-normalized penalties
    """
    relevance: List[int] = []
    penalties: List[float] = []

    gold_label_lower = gold_label.lower()
    gold_len = _token_length(gold_label)
    match_found = False

    for chunk_propositions in retrieval_chunk_list:
        if not chunk_propositions or all(p is None for p in chunk_propositions):
            relevance.append(0)
            penalties.append(1.0)
            continue

        # Check if gold label appears in ANY proposition for this chunk
        found_in_propositions = []
        penalties_for_propositions = []

        for proposition in chunk_propositions:
            if proposition is None:
                continue

            prop_len = _token_length(proposition)
            prop_penalty = min(1.0, gold_len / prop_len)

            if gold_label_lower in proposition.lower():
                found_in_propositions.append(True)
                penalties_for_propositions.append(prop_penalty)
            else:
                found_in_propositions.append(False)
                penalties_for_propositions.append(prop_penalty)

        # If gold label found in any proposition, mark as relevant
        if not match_found and any(found_in_propositions):
            relevance.append(1)
            # Use the minimum penalty among propositions that contain the gold label
            matching_penalties = [p for found, p in zip(found_in_propositions, penalties_for_propositions) if found]
            penalties.append(min(matching_penalties) if matching_penalties else 1.0)
            match_found = True
        else:
            relevance.append(0)
            # Use the minimum penalty across all propositions
            penalties.append(min(penalties_for_propositions) if penalties_for_propositions else 1.0)

    return relevance, penalties


def compute_DCG(relevance):

    aux = 0
    for i in range(1, len(relevance)+1):
        aux = aux + (np.power(2, relevance[i - 1]) - 1) / (np.log2(i + 1))
    return aux


def compute_Recall(rel):
    if 1 in rel:
        return 1
    else:
        return 0


def compute_length_normalized_dcg(relevance: List[int], penalties: List[float]) -> float:
    aux = 0.0
    for i, rel in enumerate(relevance, start=1):
        if rel <= 0:
            continue
        gain = (np.power(2, rel) - 1) * penalties[i - 1]
        aux += gain / (np.log2(i + 1))
    return aux


def compute_length_normalized_recall(relevance: List[int], penalties: List[float]) -> float:
    for rel, penalty in zip(relevance, penalties):
        if rel > 0:
            return penalty
    return 0.0


@EVALUATOR_REG.register('GutenQA')
class QutenQAEvaluator(BaseEvaluator):

    def __init__(self,
                 scope: str = 'document',
                 similarity: str = 'cosine',
                 **kwargs):

        # if metrics is None:
        #     metrics = ['nDCG', 'Recall']

        self.k_values = kwargs.get('k_values', [1, 2, 5, 10, 20])

        self.scope = scope
        self.similarity = similarity


    def evaluate(self,
                 queries: List[Query],
                 query_embeddings: List[QueryEmbedding],
                 chunks: List[Chunk],
                 chunk_embeddings: List[ChunkEmbedding], ):


        assert len(queries) == len(query_embeddings)
        # assert len(chunks) == len(chunk_embeddings)

        ranker = SimpleRanker(chunk_embs=chunk_embeddings, similarity=self.similarity)
        ranking_result = ranker.rank(
            query_embs=query_embeddings,
            top_k_max=max(self.k_values),
            scope=self.scope)

        # get query-relevance mapping
        # Build mapping: chunk_id -> original chunk text
        # Note: For proposition-based evaluation, chunks should be the ORIGINAL paragraphs,
        # not the propositions. Propositions are only used for embedding/ranking.
        chunk_id2text = {c.chunk_id: c.text for c in chunks}

        ranked_relevance_dict: Dict[str: List[int]] = {}
        ranked_penalty_dict: Dict[str: List[float]] = {}

        for query in tqdm(queries, desc="Calculating GutenQA Metrics"):

            query_id = query.query_id
            # Get original chunk text for each ranked chunk_id
            # For propositions: chunk_id represents the original paragraph,
            # and we check gold label against that original text
            re_chunk_list = [[chunk_id2text.get(c_id)] for c_id, _ in ranking_result.get(query_id, [])]

            relevance, penalties = compute_relevance_and_penalties(re_chunk_list, query.chunk_must_Contain)

            ranked_relevance_dict[query_id] = relevance
            ranked_penalty_dict[query_id] = penalties

        dcg_dict = defaultdict(list)
        recall_dict = defaultdict(list)
        ln_dcg_dict = defaultdict(list)
        ln_recall_dict = defaultdict(list)

        for top_k in self.k_values:
            for query_id, relevance in ranked_relevance_dict.items():
                penalties = ranked_penalty_dict.get(query_id, [])
                dcg = compute_DCG(relevance[:top_k])
                recall = compute_Recall(relevance[:top_k])
                ln_dcg = compute_length_normalized_dcg(relevance[:top_k], penalties[:top_k])
                ln_recall = compute_length_normalized_recall(relevance[:top_k], penalties[:top_k])
                dcg_dict[top_k].append(dcg)
                recall_dict[top_k].append(recall)
                ln_dcg_dict[top_k].append(ln_dcg)
                ln_recall_dict[top_k].append(ln_recall)

        per_query_eval = defaultdict(dict)
        for top_k in self.k_values:
            for i, (query_id, relevance) in enumerate(ranked_relevance_dict.items()):
                penalties = ranked_penalty_dict.get(query_id, [])
                dcg = compute_DCG(relevance[:top_k])
                recall = compute_Recall(relevance[:top_k])
                ln_dcg = compute_length_normalized_dcg(relevance[:top_k], penalties[:top_k])
                ln_recall = compute_length_normalized_recall(relevance[:top_k], penalties[:top_k])
                per_query_eval[query_id][f'DCG@{top_k}'] = dcg
                per_query_eval[query_id][f'Recall@{top_k}'] = recall
                per_query_eval[query_id][f'lnDCG@{top_k}'] = ln_dcg
                per_query_eval[query_id][f'lnRecall@{top_k}'] = ln_recall

        final_dcg_dict = {f'DCG@{k}': round(np.mean(v), 5) for k, v in dcg_dict.items()}
        final_recall_dict = {f'Recall@{k}': round(np.mean(v), 5) for k, v in recall_dict.items()}
        final_ln_dcg_dict = {f'lnDCG@{k}': round(np.mean(v), 5) for k, v in ln_dcg_dict.items()}
        final_ln_recall_dict = {f'lnRecall@{k}': round(np.mean(v), 5) for k, v in ln_recall_dict.items()}

        # print([f'Top-{k}' for k in self.ks])
        print(final_dcg_dict)
        print(final_recall_dict)
        print(final_ln_dcg_dict)
        print(final_ln_recall_dict)

        return {
            'dcg': final_dcg_dict,
            'recall': final_recall_dict,
            'length_normalized_dcg': final_ln_dcg_dict,
            'length_normalized_recall': final_ln_recall_dict,
            'per_query_eval': per_query_eval,
            'ranking_results': ranking_result
        }



