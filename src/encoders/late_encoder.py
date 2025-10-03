from typing import List, Dict, Tuple

from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer
import torch

from src.types import Chunk, ChunkEmbedding, Query
from src.encoders.base_encoder import BaseEncoder
from src.registry import ENCODER_REG, EMD_BACKBONE_REG
from src.io.sink import PickleSink
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
        self._sink = PickleSink(embed_sink_path) if embed_sink_path else None

        self.tokenizer = AutoTokenizer.from_pretrained(backbone_kwargs.get('model_name'),
                                                       trust_remote_code=True)

        self.chunk_joint_symbol = backbone_kwargs.get('chunk_joint_symbol', '\n')

        # set tokenizer, model_max_length
        # in this case, the model_max_length is 2147483648, so set it up to 8192.
        if backbone_kwargs.get('model_name') == "jinaai/jina-embeddings-v2-small-en":
            self.tokenizer.model_max_length = 8192

        self.model_max_length = self.tokenizer.model_max_length
        self.long_late_chunking_overlap_size = 256

        if self.model_max_length < self.long_late_chunking_overlap_size:
            raise ValueError('long_late_chunking_overlap_size must be larger than model_max_length')

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

        input_ids = self.tokenizer.batch_encode_plus(new_text_list, add_special_tokens=False)['input_ids']
        start = 0
        for idx, token_ids in enumerate(input_ids):

            end = start + len(token_ids)
            if idx == 0:
                end += 1  # for [ClS], add 1

            chunk_spans.append((start, end))
            start = end

        # add 1 for [SEP]
        chunk_spans[-1] = (chunk_spans[-1][0], chunk_spans[-1][1]+1)

        return chunk_spans


    def merge_text(self, text_list: List[str]) -> str:
        """
        merge chunks by doc_ids
        """

        return self.chunk_joint_symbol.join(text_list)


    def _split_doc_chunks(self, doc_chunks: List[Chunk], prefix: str) -> List[List[Chunk]]:
        """
        This function helps implement long-late-chunking.
        Each model has model_max_length, so the long document should be split into segments, the input the of model
         would not be truncation.
        """

        final_list = []

        prefix_length = len(self.tokenizer.encode(prefix, add_special_tokens=False))

        current_sub_list = []
        current_length = prefix_length + 2  # prefix string, [CLS] and [SEP]

        for chunk in doc_chunks:

            token_ids = self.tokenizer.encode(chunk.text, add_special_tokens=False)
            token_len = len(token_ids)

            if token_len > self.tokenizer.model_max_length:
                if current_sub_list:
                    final_list.append(current_sub_list)
                final_list.append([chunk])

                current_sub_list = []
                current_length = prefix_length

            elif token_len + current_length > self.tokenizer.model_max_length:
                if current_sub_list:
                    final_list.append(current_sub_list)

                current_sub_list = [chunk]
                current_length = prefix_length + token_len

            else:

                current_sub_list.append(chunk)
                current_length += token_len

        if current_sub_list:
            final_list.append(current_sub_list)

        return final_list


    def late_encode(self,
                    texts: str,
                    call_kwargs:dict):


        vectors = self.model.get_all_token_embeddings(texts=[texts], **call_kwargs).last_hidden_state  # 1, N, D
        vectors = vectors[0].detach().cpu().numpy()

        return vectors


    def long_late_encode(self,
                    texts: str,
                    prefix: str):

        input_text = prefix + texts
        # print(input_text)

        model_inputs = self.tokenizer(
            input_text,
            padding=True,
            return_tensors="pt",
            truncation=False,
        )

        len_tokens = len(model_inputs['input_ids'][0])

        # print(f'len_tokens: {len_tokens}, model_max_length: {self.model_max_length}' )

        indices = []

        for i in range(0, len_tokens, self.model_max_length - self.long_late_chunking_overlap_size):

            start = i
            end = min(i + self.model_max_length, len_tokens)
            indices.append((start, end))

        sub_output = []

        for start, end in indices:

            batch_inputs = {k: v[:, start:end] for k,v in model_inputs.items()}

            last_state = self.model.get_embeddings_for_inputs(inputs=batch_inputs).last_hidden_state

            if start > 0:
                sub_output.append(last_state[:, self.long_late_chunking_overlap_size:])

            else:
                sub_output.append(last_state)

        vectors = torch.cat(sub_output, dim=1)  # [1, N, D]
        vectors = vectors[0].detach().cpu().numpy()  # [N, D]

        return vectors

    def whether_exceed_model_max_length(self, text: str):

        token_ids = self.tokenizer(text, add_special_tokens=True)['input_ids']

        if len(token_ids) > self.tokenizer.model_max_length:
            return True
        else:
            return False


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
        doc_range = self._get_document_range_by_doc_id(doc_id_list)

        for doc_id, position_idx in tqdm(doc_range.items()):

            doc_start_idx, doc_end_idx = position_idx['start'], position_idx['end']

            doc_chunks = chunks[doc_start_idx:doc_end_idx]
            doc_text = [c.text for c in doc_chunks]
            doc_chunk_spans = self._get_chunk_span(doc_text, prefix=prefix)
            texts = self.merge_text(doc_text)   # merge doc text

            assert len(doc_chunks) == len(doc_chunk_spans)

            if self.whether_exceed_model_max_length(texts) is True:

                vectors = self.long_late_encode(texts, prefix=prefix)
            else:

                vectors = self.late_encode(texts, call_kwargs=call_kwargs)

            assert (len(vectors) == doc_chunk_spans[-1][-1])

            for span, chunk in zip(doc_chunk_spans, doc_chunks):
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


        if self._sink is not None:
            self._sink.write_batch(output)


        return output


if __name__ == '__main__':

    # backbone = 'IntFloatE5'
    # model_name = 'intfloat/multilingual-e5-large-instruct'

    # backbone = 'JinaaiV2'
    # model_name = 'jinaai/jina-embeddings-v2-small-en'

    backbone = "Normic"
    model_name = "nomic-ai/nomic-embed-text-v1"

    # backbone = "JinaaiV3"
    # model_name = "jinaai/jina-embeddings-v3"

    passage_list = [
        'How is the weather today?',
        'What is the current weather like today?',
        'Who are you?',
    ]
    chunk_list = []

    for idx, p in enumerate(passage_list):
        c = Chunk(
            doc_id=str(1),
            chunk_id=str(idx),
            text=p,
        )

        chunk_list.append(c)

    query_list =[]
    for idx, p in enumerate(passage_list):
        c = Query(
            query_id=str(1),
            text=p,
        )

        query_list.append(c)

    encoder = LateEncoder(backbone,
                             embed_sink_path=None,
                             backbone_kwargs={'model_name': model_name})

    test_output = encoder.encode_passages(chunk_list)
    # test_output = encoder.encode_queries(query_list, query_sink_path=None)
    # print(test_output)

    vec1 = test_output[0].vector
    vec2 = test_output[1].vector
    from sentence_transformers.util import cos_sim

    print(backbone, model_name)

    print(cos_sim(vec1, vec2))
