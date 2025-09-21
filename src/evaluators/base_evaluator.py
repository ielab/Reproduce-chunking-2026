from abc import ABC, abstractmethod
from typing import List, Iterable

from src.types import Query, Chunk, ChunkEmbedding, QueryEmbedding


class BaseEvaluator(ABC):

    @abstractmethod
    def evaluate(self,
                 queries: Iterable[Query],
                 query_embeddings: Iterable[QueryEmbedding],
                 chunks: Iterable[Chunk],
                 chunk_embeddings: Iterable[ChunkEmbedding]):
        pass
        #                  *args,
        #                  **kwargs)


# component:
# 1. similarity function:
#       - dot function
#       - cosine function
#       - sim max (Colbert)


# 2. ranking, when we get the similarity, we should rank the result
#       top-k = [1, 2, 5, 10, 20]
# 2. metrics
#       - DCG
#       - Recall

# 3. retrieval mode
#       - in a document
#       - in all corpus


# ---------------------------------
# What do we have right now?
# For GutenQA
#       - chunk files
#       - query files
#       - query embeddings
#       - chunk embeddings

