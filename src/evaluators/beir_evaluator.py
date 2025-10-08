from typing import List, Dict, Generator
from collections import defaultdict

import numpy as np
import pandas as pd
from beir.retrieval.evaluation import EvaluateRetrieval
from tqdm import tqdm

from src.evaluators.base_evaluator import BaseEvaluator
from src.registry import EVALUATOR_REG
from src.types import Query, Chunk, ChunkEmbedding, QueryEmbedding


@EVALUATOR_REG.register("beir")
class BeirEvaluator(BaseEvaluator):
    """
    Similarity function: dot
    score: max score in query and document
    """

    def __init__(self, **kwargs):

        self.k_values = kwargs.get('k_values', [1, 2, 5, 10, 20])


        # aggregation function
        aggregation = kwargs.get('aggregation', 'max')
        agg_functions = {
            "max": lambda df: df.groupby("doc_id")["score"].max(),
            "mean": lambda df: df.groupby("doc_id")["score"].mean(),
        }

        self.aggregation_function = agg_functions[aggregation]

        similarity = kwargs.get('similarity', 'cosine')

        self.similarity_function = {
            'cosine': self.cosine_similarity,
            'dot': self.dot_product
        }[similarity]



    def dot_product(self, query_embeddings, chunk_embeddings):
        return np.dot(query_embeddings, chunk_embeddings.T)

    def cosine_similarity(self, query_embeddings, chunk_embeddings):
        # Normalize query and chunk embeddings
        query_norm = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        chunk_norm = chunk_embeddings / np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)

        # Compute cosine similarity as dot product of normalized vectors
        return np.dot(query_norm, chunk_norm.T)

    def rank(self,
             chunk_embeddings: List[ChunkEmbedding],
             query_embeddings: List[QueryEmbedding]
             ) -> Dict[str, Dict[str, float]]:
            # {query_id: {doc_id: score}}

        doc_id_list = [c.doc_id for c in chunk_embeddings]
        # chunk_id_list = [c.chunk_id for c in chunk_embeddings]
        chunk_emb_list = [c.vector for c in chunk_embeddings]

        query_id_list = [q.query_id for q in query_embeddings]
        query_emb_list = [q.vector for q in query_embeddings]


        C_full = np.asarray(chunk_emb_list)
        Q_full = np.asarray(query_emb_list)

        # similarity_matrix = np.dot(Q_full, C_full.T)
        similarity_matrix = self.similarity_function(Q_full, C_full)

        results: Dict[str, Dict[str, float]] = {}

        for i, query_id in tqdm(enumerate(query_id_list), total=len(query_id_list), desc="Ranking"):
            scores = similarity_matrix[i, :]
            df = pd.DataFrame({
                'doc_id': doc_id_list,
                'score': scores
            })

            doc_scores = self.aggregation_function(df)

            results[query_id] = dict(doc_scores.sort_values(ascending=False))

        return results


    def evaluate(self,
                 queries: List[Query],
                 query_embeddings: List[QueryEmbedding],
                 chunks: List[Chunk],
                 chunk_embeddings: List[ChunkEmbedding], ):


        ranking_results = self.rank(chunk_embeddings, query_embeddings)

        qrels = {q.query_id:q.qrels for q in queries}

        retriever = EvaluateRetrieval()
        # Evaluate all queries at once for the aggregated score
        ndcg, _map, recall, precision = retriever.evaluate(qrels, ranking_results, self.k_values)

        # Calculate per-query scores by iterating
        per_query_eval = {}
        for q_id in tqdm(qrels.keys(), desc="Calculating Per-Query Metrics"):
            qrels_single = {q_id: qrels[q_id]}
            results_single = {q_id: ranking_results.get(q_id, {})}

            # Evaluate each query individually
            ndcg_q, _, recall_q, _ = retriever.evaluate(qrels_single, results_single, self.k_values)
            
            query_scores = {}
            # Check if the results are valid before processing
            if ndcg_q and recall_q:
                for k in self.k_values:
                    query_scores[f"NDCG@{k}"] = ndcg_q.get(f"NDCG@{k}", 0.0)
                    query_scores[f"Recall@{k}"] = recall_q.get(f"Recall@{k}", 0.0)
            
            per_query_eval[q_id] = query_scores

        print(ndcg)
        print(recall)

        return {'ndcg': ndcg, 'recall': recall, 'per_query_eval': per_query_eval, 'ranking_results': ranking_results}