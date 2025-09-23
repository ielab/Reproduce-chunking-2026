from typing import List

from tqdm import tqdm
import numpy as np

from src.types import Chunk, ChunkEmbedding, Query, QueryEmbedding
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
        self._sink = JsonlSink(embed_sink_path) if embed_sink_path else None
        # self._sink = JsonlZstSink(embed_sink_path)


    def encode(self,
               chunks: List[Chunk],
               batch_size: int=32,
               **kwargs):

        output: List[ChunkEmbedding] = []

        call_kwargs = {}
        if self.backbone == 'JinaaiV3':
            call_kwargs['task'] = 'retrieval.passage'

        elif self.backbone == 'Normic':
            call_kwargs['instruction'] = 'search_document: '


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

                embedding = ChunkEmbedding(
                    doc_id=chunk.doc_id,
                    chunk_id=chunk.chunk_id,
                    vector=vec,
                )


                output.append(embedding)

        if self._sink is not None:
            self._sink.write_batch(output)

        return output


    def encode_queries(self,
                       queries: List[Query],
                       query_sink_path: str,
                       batch_size: int=32,
                       **kwargs):

        output: List[QueryEmbedding] = []

        call_kwargs = {}
        if self.backbone == 'JinaaiV3':
            call_kwargs['task'] = 'retrieval.query'

        elif self.backbone == 'Qwen3':
            call_kwargs['prompt_name'] = 'query'

        elif self.backbone == 'Normic':
            call_kwargs['instruction'] = 'search_query: '

        query_sink = JsonlSink(query_sink_path)

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


if __name__ == '__main__':

    backbone = 'Normic'
    model_name = 'nomic-ai/nomic-embed-text-v1'

    text_list = [
    'How is the weather today?',
    'What is the current weather like today?'
]
    chunk_list = []

    for idx, text in enumerate(text_list):

        c = Chunk(
            doc_id=str(idx),
            chunk_id=str(idx),
            text=text,
        )

        chunk_list.append(c)

    encoder = RegularEncoder(backbone,
                             embed_sink_path=None,
                             backbone_kwargs={'model_name': model_name})

    test_output = encoder.encode(chunk_list)
    # print(test_output)

    vec1 = test_output[0].vector
    vec2 = test_output[1].vector
    from sentence_transformers.util import cos_sim
    print(cos_sim(vec1, vec2))
