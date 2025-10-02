from typing import List, Dict, Tuple

from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer

from src.types import Chunk, ChunkEmbedding, Query
from src.encoders.base_encoder import BaseEncoder
from src.registry import ENCODER_REG, EMD_BACKBONE_REG
from src.io.sink import JsonlSink
from src.utils.docid_utils import get_book_range
from src.models.embedding.base_embedding import BaseEmbeddingModel


@ENCODER_REG.register('LateEncoder')
class LateEncoder(BaseEncoder):

    def __init__(self,
                 backbone: str,
                 embed_sink_path,
                 backbone_kwargs: dict | None = None):

        super().__init__(backbone, embed_sink_path, backbone_kwargs)

        backbone_cls = EMD_BACKBONE_REG.get(backbone)
        self.backbone = backbone

        self.model: BaseEmbeddingModel = backbone_cls(**(backbone_kwargs or {}))
        self._sink = JsonlSink(embed_sink_path) if embed_sink_path else None

        self.tokenizer = AutoTokenizer.from_pretrained(backbone_kwargs.get('model_name'),
                                                       trust_remote_code=True)

        self.chunk_joint_symbol = backbone_kwargs.get('chunk_joint_symbol', '\n')

    @staticmethod
    def _get_book_range(doc_id_list: List[str]) -> Dict[str, Dict[str, int]]:
        """
        this function works getting the boundary in the doc_id_list
        """
        return get_book_range(doc_id_list)

    @staticmethod
    def _get_document_range_by_doc_id(doc_id_list: List[str]) -> Dict[str, Dict[str, int]]:
        """
        For this method, we must ensure that chunks split from one document have the same doc id.
        """

        document_range = {}
        for idx, doc_id in enumerate(doc_id_list):
            if doc_id not in document_range:
                document_range[doc_id] = {'start': idx, 'end': idx+1}
            else:
                document_range[doc_id]['end'] = idx + 1

        return document_range


    def _get_chunk_span(self, text_list: List[str], prefix='') -> List[Tuple[int, int]]:
        """
        In the paper Late Chunking:
            - All prepended tokens, such as [CLS] and the Instruction string, are considered the part of first chunk.
            - All appended tokens, such as [SEP], are considered part of last chunk.
        """

        chunk_spans: List[Tuple[int, int]] = []

        # add predix to first chunk
        new_text_list = [prefix + text_list[0]] + text_list[1:]

        input_ids = self.tokenizer.batch_encode_plus(new_text_list)['input_ids']

        for idx, token_ids in enumerate(input_ids):
            if idx == 0:
                start, end = 0, len(token_ids) - 1 # del [SEP]
            elif idx == len(input_ids) - 1:
                _, start = chunk_spans[-1]
                end = start + len(token_ids) - 1    # del [CLS]
            else:
                _, start = chunk_spans[-1]
                end = start + len(token_ids) - 2  # del [CLS], [SEP]

            chunk_spans.append((start, end))

        return chunk_spans

    def merge_chunks(self, text_list: List[str]) -> str:
        """
        merge chunks by doc_ids
        """

        return self.chunk_joint_symbol.join(text_list)


    def encode_passages(self,
               chunks: List[Chunk],
               batch_size: int=32,
               **kwargs):

        output: List[ChunkEmbedding] = []

        prefix = ""

        call_kwargs = {}
        if self.backbone == 'JinaaiV3':
            prefix = 'Represent the document for retrieval: '
            call_kwargs['task'] = 'retrieval.passage'

        elif self.backbone == 'Normic':
            prefix = 'search_document: '
            call_kwargs['instruction'] = 'search_document'


        doc_id_list = [c.doc_id for c in chunks]
        # doc_range = self._get_book_range(doc_id_list)
        doc_range = self._get_document_range_by_doc_id(doc_id_list)


        for doc_id, position_idx in tqdm(doc_range.items()):

            doc_start_idx, doc_end_idx = position_idx['start'], position_idx['end']

            chunks_sub = chunks[doc_start_idx:doc_end_idx]
            doc_text = [c.text for c in chunks_sub]

            chunk_spans = self._get_chunk_span(doc_text, prefix=prefix)

            assert len(chunks_sub) == len(chunks_sub)

            texts = self.merge_chunks(doc_text)


            vectors = self.model.get_all_token_embeddings(texts=[texts], **call_kwargs).last_hidden_state # 1, N, D
            vectors = vectors[0].detach().cpu().numpy()

            assert(len(vectors) == chunk_spans[-1][-1])


            for span, chunk in zip(chunk_spans, chunks_sub):
                start, end = span


                chunk_vector = np.mean(vectors[start:end], axis=0)

                if isinstance(chunk_vector, np.ndarray):
                    chunk_vector = chunk_vector.tolist()

                embedding = ChunkEmbedding(
                    doc_id=chunk.doc_id,
                    chunk_id=chunk.chunk_id,
                    vector=chunk_vector,
                )

                output.append(embedding)

                # self._sink.write_batch([embedding])

        if self._sink is not None:
            self._sink.write_batch(output)


        return output


if __name__ == '__main__':

    backbone = 'IntFloatE5'
    model_name = 'intfloat/multilingual-e5-large-instruct'

    # backbone = 'JinaaiV2'
    # model_name = 'jinaai/jina-embeddings-v2-small-en'

    # backbone = "Normic"
    # model_name = "nomic-ai/nomic-embed-text-v1"

    # backbone = "JinaaiV3"
    # model_name = "jinaai/jina-embeddings-v3"

    text_list = [
        'How is the weather today?',
        'What is the current weather like today?',
        'Who are you?',
    ]
    chunk_list = []

    for idx, text in enumerate(text_list):
        c = Chunk(
            doc_id=str(1),
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

    encoder = LateEncoder(backbone,
                             embed_sink_path=None,
                             backbone_kwargs={'model_name': model_name})

    test_output = encoder.encode_passages(chunk_list)
    test_output = encoder.encode_queries(query_list, query_sink_path=None)
    # print(test_output)

    vec1 = test_output[0].vector
    vec2 = test_output[1].vector
    from sentence_transformers.util import cos_sim

    print(backbone, model_name)

    print(cos_sim(vec1, vec2))
