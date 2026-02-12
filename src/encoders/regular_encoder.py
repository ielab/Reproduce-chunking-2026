from typing import List

from tqdm import tqdm
import numpy as np

from src.types import Chunk, ChunkEmbedding, Query
from src.encoders.base_encoder import BaseEncoder
from src.registry import ENCODER_REG, EMD_BACKBONE_REG
from src.io.sink import PickleSink
from src.models.embedding.base_embedding import BaseEmbeddingModel


@ENCODER_REG.register('RegularEncoder')
class RegularEncoder(BaseEncoder):

    def __init__(self,
                 backbone: str,
                 embed_sink_path,
                 backbone_kwargs: dict | None = None):

        super().__init__(backbone, embed_sink_path, backbone_kwargs)
        self.backbone = backbone

        backbone_cls = EMD_BACKBONE_REG.get(backbone)

        # self.model: BaseEmbeddingModel = backbone_cls(**(backbone_kwargs or {}))
        # self._sink = PickleSink(embed_sink_path) if embed_sink_path else None


    def encode_passages(self,
               chunks: List[Chunk],
               batch_size: int=32,
               incremental: bool = True,
               flush_every_n_batches: int = 100,
               **kwargs):

        call_kwargs = {}
        if self.backbone == 'JinaaiV3':
            call_kwargs['task'] = 'retrieval.passage'

        elif self.backbone == 'Normic':
            call_kwargs['instruction'] = 'search_document'

        # Use incremental mode to avoid memory accumulation
        if incremental and self._sink is not None:
            # Re-initialize sink with incremental=True
            from src.io.sink import PickleSink
            self._sink = PickleSink(self._sink.path, incremental=True)

            accumulated = []
            batch_count = 0

            for i in tqdm(range(0, len(chunks), batch_size)):
                batch = chunks[i:i+batch_size]

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
                    accumulated.append(embedding)

                batch_count += 1

                # Flush to temp file every N batches
                if batch_count >= flush_every_n_batches:
                    self._sink.write_batch(accumulated)
                    accumulated = []
                    batch_count = 0

            # Flush any remaining embeddings
            if accumulated:
                self._sink.write_batch(accumulated)

            # Merge all temp files into final output
            self._sink.finalize()
            return []

        else:
            # Original behavior for backward compatibility
            output: List[ChunkEmbedding] = []

            for i in tqdm(range(0, len(chunks), batch_size)):
                batch = chunks[i:i+batch_size]

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


if __name__ == '__main__':

    # backbone = 'IntFloatE5'
    # model_name = 'intfloat/multilingual-e5-large-instruct'

    backbone = 'JinaaiV2'
    model_name = 'jinaai/jina-embeddings-v2-small-en'

    # backbone = "Normic"
    # model_name = "nomic-ai/nomic-embed-text-v1"

    # backbone = "JinaaiV3"
    # model_name = "jinaai/jina-embeddings-v3"

    print(backbone, model_name)
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

    query_list =[]
    for idx, text in enumerate(text_list):
        c = Query(
            query_id=str(1),
            text=text,
        )

        query_list.append(c)

    encoder = RegularEncoder(backbone,
                             embed_sink_path=None,
                             backbone_kwargs={'model_name': model_name})

    test_output = encoder.encode_passages(chunk_list)
    test_output = encoder.encode_queries(query_list, query_sink_path=None)
    # print(test_output)

    vec1 = test_output[0].vector
    vec2 = test_output[1].vector
    from sentence_transformers.util import cos_sim
    print(cos_sim(vec1, vec2))
