from typing import List, Dict, Generator
from collections import defaultdict
import numpy as np

from src.evaluators.base_evaluator import BaseEvaluator
from src.registry import EVALUATOR_REG
from src.types import Query, Chunk, ChunkEmbedding, QueryEmbedding


from src.evaluators.ranker import SimpleRanker


def find_index_of_match(retrieval_chunk_list, gold_label):

    relevance = []
    gold_label = gold_label.lower()

    for chunk in retrieval_chunk_list:
        if gold_label in chunk.lower():
            relevance.append(1)
            relevance = relevance + ((len(retrieval_chunk_list) - len(relevance))*([0]))
            break
        else:
            relevance.append(0)
    return relevance


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


@EVALUATOR_REG.register('GutenQA')
class QutenQAEvaluator(BaseEvaluator):

    def __init__(self,
                 scope: str = 'document',
                 similarity: str = 'dot',
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
        assert len(chunks) == len(chunk_embeddings)

        ranker = SimpleRanker(chunk_embs=chunk_embeddings)
        ranking_result = ranker.rank(query_embs=query_embeddings, top_k_max=max(self.k_values), scope=self.scope)

        # get query-relevance mapping
        chunk2id = {c.chunk_id:c.text for c in chunks}
        ranked_relevance_dict: Dict[str: List[int]] = {}

        for query in queries:

            query_id = query.query_id
            re_chunk_list = [chunk2id.get(c_id) for c_id, _ in ranking_result.get(query_id, [])]
            relevance = find_index_of_match(re_chunk_list, query.chunk_must_Contain)

            ranked_relevance_dict[query_id] = relevance

        dcg_dict = defaultdict(list)
        recall_dict = defaultdict(list)

        for top_k in self.k_values:
            for _, relevance in ranked_relevance_dict.items():
                dcg = compute_DCG(relevance[:top_k])
                recall = compute_Recall(relevance[:top_k])
                dcg_dict[top_k].append(dcg)
                recall_dict[top_k].append(recall)

        final_dcg_dict = {f'DCG@{k}': round(np.mean(v), 5) for k, v in dcg_dict.items()}
        final_recall_dict = {f'Recall@{k}': round(np.mean(v), 5) for k, v in recall_dict.items()}

        # print([f'Top-{k}' for k in self.ks])
        print(final_dcg_dict)
        print(final_recall_dict)

        return {'dcg': final_dcg_dict, 'recall': final_recall_dict}




