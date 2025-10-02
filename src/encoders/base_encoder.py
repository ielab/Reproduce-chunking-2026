from abc import ABC, abstractmethod
from typing import List

import numpy as np

from src.types import Chunk, Query, QueryEmbedding
from src.registry import EMD_BACKBONE_REG
from src.models.embedding.base_embedding import BaseEmbeddingModel
from src.io.sink import JsonlSink


class BaseEncoder(ABC):

    def __init__(self,
                 backbone,
                 embed_sink_path,
                 backbone_kwargs: dict | None = None):

        self.backbone = backbone

        backbone_cls = EMD_BACKBONE_REG.get(backbone)

        self.model: BaseEmbeddingModel = backbone_cls(**(backbone_kwargs or {}))
        self._sink = JsonlSink(embed_sink_path) if embed_sink_path else None

    def encode_queries(self,
                       queries: List[Query],
                       query_sink_path: str=None,
                       batch_size: int=32,
                       **kwargs):

        register_instruction = {
            "JinaaiV3": ("task", "retrieval.query"),
            "Qwen3": ("prompt_name", "query"),
            "Normic": ("instruction", "search_query"),
            "IntFloatE5": ("task", "query")
        }

        output: List[QueryEmbedding] = []

        call_kwargs = {}
        pair = register_instruction.get(self.backbone)
        if pair is not None:
            call_kwargs[pair[0]] = pair[1]

        query_sink = JsonlSink(query_sink_path) if query_sink_path else None

        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]

            vecs = self.model.get_embeddings(
                texts=[c.text for c in batch],
                **call_kwargs
            )

            if isinstance(vecs, np.ndarray):
                vecs = vecs.tolist()

            for query, vec in zip(batch, vecs):

                embedding = QueryEmbedding(
                    query_id=query.query_id,
                    vector=vec,
                )

                output.append(embedding)

        if query_sink:
            query_sink.write_batch(output)

        return output


    @abstractmethod
    def encode_passages(self,
               chunks: List[Chunk],
               batch_size: int,
               **kwargs):
        pass
