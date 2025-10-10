from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm

from src.types import Query, Chunk, ChunkEmbedding, QueryEmbedding

from src.utils.docid_utils import get_book_range


def doc_sim(q_vec: np.array, d_vec: np.array):

        scores = np.dot(q_vec, d_vec.T)

        return scores

def cos_sim(q_vec: np.array, d_vec: np.array):

    query_norm = q_vec / np.linalg.norm(q_vec, axis=1, keepdims=True)
    chunk_norm = d_vec / np.linalg.norm(d_vec, axis=1, keepdims=True)

    return np.dot(query_norm, chunk_norm.T)


def _top_k_rows(scores: np.array, k: int) -> Tuple[np.array, np.array]:

    top_indices = np.argsort(scores, axis=1)[:, -k:][:, ::-1]
    top_scores = np.take_along_axis(scores, top_indices, axis=1)

    return top_indices, top_scores


def keep_highest_scores(pairs: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """
    If there are duplicate chunk_ids in the ranking_result, keep only the one with the highest score.
    :param pairs:
    :return:
    """

    best_scores = {}
    order = []

    for chunk_id, score in pairs:
        if chunk_id not in best_scores:
            best_scores[chunk_id] = score
            order.append(chunk_id)
        else:
            if score > best_scores[chunk_id]:
                best_scores[chunk_id] = score

    return [(cid, best_scores[cid]) for cid in order]



class SimpleRanker:
    def __init__(self,
                 chunk_embs: List[ChunkEmbedding],
                 similarity: str):
        self.chunk_embs = chunk_embs
        # self.chunks = chunks

        # self.chunk_doc_id_list = [c.doc_id for c in self.chunk_embs]
        self.c_chunk_id_list = [c.chunk_id for c in self.chunk_embs]
        self.C_full = [c.vector for c in self.chunk_embs]

        self.similarity_func = {
            'dot': doc_sim,
            'cosine': cos_sim,
        }[similarity]


    @staticmethod
    def _get_book_range(doc_id_list: List[str]) -> Dict[str, Dict[str, int]]:
        """
        this function works getting the boundary in the doc_id_list
        :return:
        """
        return get_book_range(doc_id_list)

    def rank(self,
             query_embs: List[QueryEmbedding],
             top_k_max: int,
             scope: str = 'document') -> Dict[str, List[Tuple[str, float]]]:
        """

        :param query_embs:
        :param top_k_max:
        :param scope:
        :return: {'Book-0-Query-0': [('Book-0-Chunk-31', 0.8853517067908449), ('Book-0-Chunk-54', 0.8508707152876265)]}
        """

        query_ids: List[str] = [q_e.query_id for q_e in query_embs]
        Q_full = np.asarray([q_e.vector for q_e in query_embs], dtype=np.float32)

        out: Dict[str, List[Tuple[str, float]]] = {}

        if scope == 'document':

            range_in_chunk = self._get_book_range(self.c_chunk_id_list)

            for book_id, position_idx in tqdm(self._get_book_range(query_ids).items(), desc="Ranking (Document Scope)"):
                start_idx, end_idx = position_idx['start'], position_idx['end']

                query_ids_sub = query_ids[start_idx:end_idx]
                Q_sub = Q_full[start_idx:end_idx]         # (x, dim)

                start_end_idx_in_chunk = range_in_chunk.get(book_id)
                if start_end_idx_in_chunk is not None:
                    c_chunk_ids_sub = self.c_chunk_id_list[start_end_idx_in_chunk['start']:start_end_idx_in_chunk['end']]
                    C_sub = self.C_full[start_end_idx_in_chunk['start']:start_end_idx_in_chunk['end']]

                    assert len(c_chunk_ids_sub) == len(C_sub)
                    assert len(query_ids_sub) == len(Q_sub)

                    # score_matrix = doc_sim(Q_sub, np.array(C_sub))
                    score_matrix = self.similarity_func(Q_sub, np.array(C_sub))
                    # print(score_matrix)

                    top_indices, top_scores = _top_k_rows(score_matrix, top_k_max*100) # for proposition case, expand ranking number

                    for i, qid in enumerate(query_ids_sub):
                        rows = top_indices[i]
                        scores = top_scores[i]
                        pairs = [(c_chunk_ids_sub[j], float(s)) for j, s in zip(rows, scores)]
                        keep_highest_pairs = keep_highest_scores(pairs)
                        # print(len(pairs), len(keep_highest_pairs))
                        out[qid] = keep_highest_pairs
                        # out[qid] = [(c_chunk_ids_sub[j], float(s)) for j, s in zip(rows, scores)]

        elif scope == 'corpus':

            # This step is to batch the queries, to avoid out of memory
            # batch by book range
            for _, position_idx in self._get_book_range(query_ids).items():
                start_idx, end_idx = position_idx['start'], position_idx['end']

                query_ids_sub = query_ids[start_idx:end_idx + 1]
                Q_sub = Q_full[start_idx:end_idx + 1]  # (x, dim)

                assert len(query_ids_sub) == len(Q_sub)

                score_matrix = doc_sim(Q_sub, np.array(self.C_full))
                top_indices, top_scores = _top_k_rows(score_matrix, top_k_max*100) # for proposition case, expand ranking number

                for i, qid in enumerate(query_ids_sub):
                    rows = top_indices[i]
                    scores = top_scores[i]
                    pairs = [(self.c_chunk_id_list[j], float(s)) for j, s in zip(rows, scores)]
                    keep_highest_pairs = keep_highest_scores(pairs)
                    out[qid] = keep_highest_pairs
                    # out[qid] = [(self.c_chunk_id_list[j], float(s)) for j, s in zip(rows, scores)]


        else:
            raise ValueError(f'Unknown scope {scope}')

        return out


if __name__ == '__main__':
    dim = 4
    num_queries = 5
    num_chunks = 10
    query_vec = np.random.rand(5, dim)
    chunk_matrix = np.random.rand(num_chunks, dim)

    print(query_vec.shape, chunk_matrix.shape)

    s = doc_sim(query_vec, chunk_matrix)
    print(s)
    # top_indices = np.argsort(s, axis=1)[:, -3:][:, ::-1]
    # top_scores = np.take_along_axis(s, top_indices, axis=1)
    #
    # print(top_indices)
    # print(top_scores)



