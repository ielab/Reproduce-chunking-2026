from typing import List

from tqdm import tqdm
import numpy as np

from src.types import Chunk, Embedding, Query, QueryEmbedding
from src.encoders.base_encoder import BaseEncoder
from src.registry import ENCODER_REG, EMD_BACKBONE_REG
from src.io.sink import JsonlSink, ParquetSink, JsonlZstSink
from src.models.embedding.base_embedding import BaseEmbeddingModel


@ENCODER_REG.register('RegularEncoder')
class RegularEncoder(BaseEncoder):

    def __init__(self,
                 backbone: str,
                 embed_sink_path,
                 backbone_kwargs: dict | None = None):

        self.backbone = backbone

        backbone_cls = EMD_BACKBONE_REG.get(backbone)

        self.model: BaseEmbeddingModel = backbone_cls(**(backbone_kwargs or {}))
        self._sink = JsonlSink(embed_sink_path)
        # self._sink = JsonlZstSink(embed_sink_path)


    def encode(self,
               chunks: List[Chunk],
               batch_size: int=32,
               **kwargs):

        output: List[Embedding] = []

        call_kwargs = {}
        if self.backbone == 'JinaV3':
            call_kwargs['task'] = 'retrieval.passage'


        for i in tqdm(range(0, len(chunks), batch_size)):
            batch = chunks[i:i+batch_size]

            # if self.backbone == EMD_BACKBONE_REG.get(self.backbone)

            vecs = self.model.get_embeddings(
                texts=[c.text for c in batch],
                **call_kwargs
            )

            if isinstance(vecs, np.ndarray):
                vecs = vecs.tolist()

            for chunk, vec in zip(batch, vecs):

                embedding = Embedding(
                    doc_id=chunk.doc_id,
                    chunk_id=chunk.chunk_id,
                    vector=vec,
                )

                if self._sink:
                    self._sink.write_batch([embedding])

                # output.append(embedding)

        self._sink.close()
        # self._sink.write_batch(output)

        return output


    def encode_queries(self,
                       queries: List[Query],
                       query_sink_path: str,
                       batch_size: int=32,
                       **kwargs):

        output: List[QueryEmbedding] = []

        call_kwargs = {}
        if self.backbone == 'JinaV3':
            call_kwargs['task'] = 'retrieval.query'

        elif self.backbone == 'Qwen3':
            call_kwargs['prompt_name'] = 'query'

        query_sink = JsonlSink(query_sink_path)

        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]

            vecs = self.model.get_embeddings(
                texts=[c.text for c in batch],
                prompt_name='query',
                **call_kwargs
            )

            if isinstance(vecs, np.ndarray):
                vecs = vecs.tolist()

            for query, vec in zip(batch, vecs):

                embedding = QueryEmbedding(
                    query_id=query.query_id,
                    vector=vec,
                )

                if query_sink:
                    query_sink.write_batch([embedding])

        return output

