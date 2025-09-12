from typing import List, Dict, Tuple

from tqdm import tqdm
import numpy as np

from transformers import AutoTokenizer

from src.types import Chunk, Embedding, Query, QueryEmbedding
from src.encoders.base_encoder import BaseEncoder
from src.registry import ENCODER_REG, EMD_BACKBONE_REG
from src.io.sink import JsonlSink
from src.utils.docid_utils import get_book_range
from src.models.embedding.base_embedding import BaseEmbeddingModel


@ENCODER_REG.register('LateEncoder')
class LateEncoder(BaseEncoder):

    def __init__(self,
                 backend: str,
                 embed_sink_path,
                 backend_kwargs: dict | None = None):

        backend_cls = EMD_BACKBONE_REG.get(backend)

        self.model: BaseEmbeddingModel = backend_cls(**(backend_kwargs or {}))
        self._sink = JsonlSink(embed_sink_path)

        self.tokenizer = AutoTokenizer.from_pretrained(backend_kwargs.get('model_name'),
                                                       trust_remote_code=True)

        self.chunk_joint_symbol = backend_kwargs.get('chunk_joint_symbol', '\n')

    @staticmethod
    def _get_book_range(doc_id_list: List[str]) -> Dict[str, Dict[str, int]]:
        """
        this function works getting the boundary in the doc_id_list
        :return:
        """
        return get_book_range(doc_id_list)


    def _get_chunk_span(self, text_list: List[str]) -> List[Tuple[int, int]]:

        chunk_spans: List[Tuple[int, int]] = []

        input_ids = self.tokenizer.batch_encode_plus(text_list)['input_ids']

        for token_ids in input_ids:
            if chunk_spans:
                _, start = chunk_spans[-1]
                end = start + len(token_ids) - 1
            else:
                start, end = 0, len(token_ids) - 1

            chunk_spans.append((start, end))

        return chunk_spans

    def merge_chunks(self, text_list: List[str]) -> str:

        return self.chunk_joint_symbol.join(text_list)


    def encode(self,
               chunks: List[Chunk],
               batch_size: int,
               **kwargs):



        doc_id_list = [c.doc_id for c in chunks][:10]
        doc_range = self._get_book_range(doc_id_list)
        print(doc_range)

        for doc_id, position_idx in doc_range.items():

            doc_start_idx, doc_end_idx = position_idx['start'], position_idx['end']

            chunks_sub = chunks[doc_start_idx:doc_end_idx]
            doc_text = [c.text for c in chunks_sub]

            chunk_spans = self._get_chunk_span(doc_text)

            assert len(chunks_sub) == len(doc_id_list)

            texts = self.merge_chunks(doc_text)

            # outputs = self.model.get_all_token_embeddings(texts=[texts])

            vectors = self.model.get_all_token_embeddings(texts=[texts]).last_hidden_state # 1, N, D

            vectors = vectors[0].detach().numpy()


            for span, chunk in zip(chunk_spans, chunks_sub):
                start, end = span

                chunk_vector = np.mean(vectors[start:end], axis=0)

                if isinstance(chunk_vector, np.ndarray):
                    chunk_vector = chunk_vector.tolist()

                embedding = Embedding(
                    doc_id=chunk.doc_id,
                    chunk_id=chunk.chunk_id,
                    vector=chunk_vector,
                )

                self._sink.write_batch([embedding])



    def encode_queries(self,
                       queries: List[Query],
                       query_sink_path: str,
                       batch_size: int,
                       **kwargs):
        pass


# function:
#   - inputs: text, and start and end position of each token
#       - overlapping ?, tokenizer ?
