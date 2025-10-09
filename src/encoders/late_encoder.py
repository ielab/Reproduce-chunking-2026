from typing import List, Dict, Tuple

from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer
import torch
from torch.nn.utils.rnn import pad_sequence

from src.types import Chunk, ChunkEmbedding, Query
from src.encoders.base_encoder import BaseEncoder
from src.registry import ENCODER_REG, EMD_BACKBONE_REG
from src.io.sink import PickleSink
from src.models.embedding.base_embedding import BaseEmbeddingModel

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@ENCODER_REG.register('LateEncoder')
class LateEncoder(BaseEncoder):

    def __init__(self,
                 backbone: str,
                 embed_sink_path,
                 backbone_kwargs: dict | None = None):

        super().__init__(backbone, embed_sink_path, backbone_kwargs)

        #backbone_cls = EMD_BACKBONE_REG.get(backbone)
        self.backbone = backbone

        #self.model: BaseEmbeddingModel = backbone_cls(**(backbone_kwargs or {}))
        #self._sink = PickleSink(embed_sink_path) if embed_sink_path else None

        self.tokenizer = AutoTokenizer.from_pretrained(backbone_kwargs.get('model_name'),
                                                       trust_remote_code=True)

        self.chunk_joint_symbol = backbone_kwargs.get('chunk_joint_symbol', '\n')

        # set tokenizer, model_max_length
        # in this case, the model_max_length is 2147483648, so set it up to 8192.
        if backbone_kwargs.get('model_name') == "jinaai/jina-embeddings-v2-small-en":
            self.tokenizer.model_max_length = 8192
        elif backbone_kwargs.get('model_name') == "nomic-ai/nomic-embed-text-v1":
            self.tokenizer.model_max_length = 8192

        self.model_max_length = self.tokenizer.model_max_length
        self.long_late_chunking_overlap_size = 256

        if self.model_max_length < self.long_late_chunking_overlap_size:
            raise ValueError('long_late_chunking_overlap_size must be larger than model_max_length')

    @staticmethod
    def _get_document_range_by_doc_id(doc_id_list: List[str]) -> Dict[str, Dict[str, int]]:
        """
        For this method, we must ensure that chunks split from one document have the same doc id.
        :return {doc_id: {'start': int, 'end': int}, ...}
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

    def late_encode(self,
                    texts: str,
                    call_kwargs:dict):


        vectors = self.model.get_all_token_embeddings(texts=[texts], **call_kwargs).last_hidden_state  # 1, N, D
        vectors = vectors[0].detach().cpu().numpy()

        return vectors

    @staticmethod
    def _slice_one(model_inputs: Dict[str, List], window: Tuple[int, int, int, int]):
        i, start, end, _ = window
        # Slice each field along sequence dim for the chosen example
        sliced = {k: v[i][start:end] for k, v in model_inputs.items()}
        return sliced

    @staticmethod
    def _pad_batch(sliced_list: List[Dict], pad_token_id: int = 0):

        keys = sliced_list[0].keys()

        # Prepare sequences per key (list of 1D tensors)
        seqs = {k: [d[k] for d in sliced_list] for k in keys}

        input_ids_padded = pad_sequence(seqs["input_ids"], batch_first=True, padding_value=pad_token_id)

        batch = {"input_ids": input_ids_padded}

        # token_type_ids (if provided)
        if "token_type_ids" in keys:
            tti_padded = pad_sequence(
                seqs["token_type_ids"], batch_first=True, padding_value=0
            )
            batch["token_type_ids"] = tti_padded

        # attention_mask (prefer sliced original if present; otherwise derive)
        if "attention_mask" in keys:
            am_padded = pad_sequence(
                seqs["attention_mask"], batch_first=True, padding_value=0
            )
            batch["attention_mask"] = am_padded
        else:
            batch["attention_mask"] = (input_ids_padded != pad_token_id).long()

        # Keep original lengths so we can unpad results later
        lengths = torch.tensor([d["input_ids"].shape[0] for d in sliced_list], device=input_ids_padded.device)
        return batch, lengths

    @staticmethod
    def add_special_tokens(sliced_list: List[Dict],
                       prefix_token_ids: torch.Tensor,
                       cls_token_id: torch.Tensor,
                       sep_token_id: torch.Tensor):

        new_sliced_list = []

        if prefix_token_ids.numel() > 0:
            prefix_special_token_ids = torch.cat([cls_token_id, prefix_token_ids])
        else:
            prefix_special_token_ids = cls_token_id

        for data in sliced_list:

            new_data = {'input_ids': torch.cat([
                prefix_special_token_ids,
                data['input_ids'],
                sep_token_id
            ])}

            if 'token_type_ids' in data:
                new_data['token_type_ids'] = torch.cat([
                    torch.tensor([0] * len(prefix_special_token_ids)),
                    data['token_type_ids'],
                    torch.tensor([0])
                ])

            if 'attention_mask' in data:
                new_data['attention_mask'] = torch.cat([
                    torch.tensor([1] * len(prefix_special_token_ids)),
                    data['attention_mask'],
                    torch.tensor([1])
                ])

            new_sliced_list.append(new_data)

        return new_sliced_list

    def long_late_encode_with_prefix(self, texts: List[str], prefix: str, batch_size: int):
        """
        the logic:
        1. tokenize the texts
        2. record cls, sep, prefix
        3. cut the text into sub_texts with max_model_length - overlap_size - prefix_length - cls - sep
        4. combine each text into new sub_text, new_sub_text = cls + prefix + overlap_text + sub_text + sep
        5. add padding, generate the input of the model
        6. feed the input to the model, get embeddings
        7. get the original text embedding,
            - if start = 0 and end = last -> text_embedding = embedding
            - if start = 0 and end != last -> text_embedding[:-1,:]
            - if start != 0 and end = last -> text_embedding[cls+prefix_length + overlap_size:,:]
            if start != 0 and end != last -> text_embedding[cls+prefix_length + overlap_size:-1,:]
        """

        model_inputs_no_padding = self.tokenizer(
            texts,
            padding=False,
            truncation=False,
            add_special_tokens=False,
        )

        model_inputs_no_padding = {k: [torch.tensor(x) for x in v] for k,v in model_inputs_no_padding.items()}

        len_tokens = [len(input_id) for input_id in model_inputs_no_padding['input_ids']]

        prefix_token_ids = self.tokenizer(prefix, add_special_tokens=False, return_tensors='pt')['input_ids'][0]
        prefix_token_len = len(prefix_token_ids)

        special_tokens_len = prefix_token_len + 2

        indices = []

        for text_id, length in enumerate(len_tokens):

            for i in range(0, length, self.model_max_length - self.long_late_chunking_overlap_size - special_tokens_len):
                start = i
                end = min(i + self.model_max_length - special_tokens_len, length)
                indices.append((text_id, start, end, length))
                if end == length:
                    break

        text_id2embedding = {}

        cls_token_tensor = torch.tensor([self.tokenizer.cls_token_id])
        sep_token_tensor = torch.tensor([self.tokenizer.sep_token_id])

        for i in tqdm(range(0, len(indices), batch_size)):

            sub_indices = indices[i:i+batch_size]

            sliced_list = [self._slice_one(model_inputs_no_padding, index_tuple) for index_tuple in sub_indices]

            add_special_tokens_sliced_list = self.add_special_tokens(sliced_list,
                                                                prefix_token_ids=prefix_token_ids,
                                                                cls_token_id=cls_token_tensor,
                                                                sep_token_id=sep_token_tensor)


            batch_inputs, lengths = self._pad_batch(add_special_tokens_sliced_list, pad_token_id=self.tokenizer.pad_token_id)

            last_hidden_state = self.model.get_embeddings_for_inputs(inputs=batch_inputs).last_hidden_state

            assert len(last_hidden_state) == len(sub_indices)

            for b_idx, L in enumerate(lengths):

                text_id, start, end, text_L = sub_indices[b_idx]

                embedding = last_hidden_state[b_idx, :L]

                if start == 0 and end == text_L:
                    pass
                elif start == 0 and end != text_L:
                    embedding = embedding[:-1, ]
                elif start != 0 and end == text_L:
                    embedding = embedding[self.long_late_chunking_overlap_size + prefix_token_len + 1:, ]
                else:
                    embedding = embedding[self.long_late_chunking_overlap_size + prefix_token_len + 1:-1, ]

                if embedding.numel() != 0:
                    text_id2embedding.setdefault(text_id, []).append(embedding)

        vectors = []

        for text_id, embeddings in text_id2embedding.items():

            V = torch.cat(embeddings, dim=0)
            V = V.detach().cpu().numpy()

            vectors.append(V)

        return vectors


    def encode_passages(self,
               chunks: List[Chunk],
               batch_size: int=32,
               **kwargs):

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

        doc_chunks_list: List[List[Chunk]] = []
        doc_chunk_spans_list: List[List[Tuple]] = []

        text_list: List[str] = []

        for doc_id, position_idx in doc_range.items():

            doc_start_idx, doc_end_idx = position_idx['start'], position_idx['end']

            doc_chunks = chunks[doc_start_idx:doc_end_idx]
            doc_text = [c.text for c in doc_chunks]
            doc_chunk_spans = self._get_chunk_span(doc_text, prefix=prefix)
            text = self.merge_text(doc_text)   # merge doc text

            # here we need the pair (doc_chunks, doc_chunk_spans, texts)
            # text is used for generating vector
            # doc_chunks and doc_chunk_spans are used for create ChunkEmbedding
            assert len(doc_chunks) == len(doc_chunk_spans)

            doc_chunks_list.append(doc_chunks)
            doc_chunk_spans_list.append(doc_chunk_spans)
            text_list.append(text)

        vectors: List[np.ndarray] = self.long_late_encode_with_prefix(texts=text_list, prefix=prefix, batch_size=batch_size)

        output: List[ChunkEmbedding] = []

        # Get the embedding dimension from the model config
        embed_dim = self.model.get_embed_dim()

        for doc_chunks, doc_chunk_spans, V in zip(doc_chunks_list, doc_chunk_spans_list, vectors):

            # Handle cases where a document produced no embeddings
            if V.shape[0] == 0:
                print(f"Warning: Document {doc_chunks[0].doc_id} produced an empty embedding vector. Assigning zero-vectors to its chunks.")
                for chunk in doc_chunks:
                    embedding = ChunkEmbedding(
                        doc_id=chunk.doc_id,
                        chunk_id=chunk.chunk_id,
                        vector=[0.0] * embed_dim,
                    )
                    output.append(embedding)
                continue

            assert (len(V) == doc_chunk_spans[-1][-1])

            for span, chunk in zip(doc_chunk_spans, doc_chunks):
                start, end = span

                # Handle cases where a chunk span is empty
                if start >= end:
                    print(f"Warning: Chunk {chunk.chunk_id} in doc {chunk.doc_id} has an empty span. Assigning a zero-vector.")
                    chunk_vector = [0.0] * embed_dim
                else:
                    chunk_vector = np.mean(V[start:end], axis=0)

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
    #
    # # backbone = "JinaaiV3"
    # # model_name = "jinaai/jina-embeddings-v3"
    #
    passage_list = ['What is the current weather like today?',
         'What is your favourite food? I have three options: meat, fries and fish, What is your favourite food? I have three options: meat, fries and fish',]


    # passage_list = ['How are you?']
    # # passage_list = ['What is your favourite food? I have three options: meat, fries and fish']
    chunk_list = []

    for idx, p in enumerate(passage_list):
        c = Chunk(
            doc_id=str(idx),
            chunk_id=str(idx),
            text=p,
        )

        chunk_list.append(c)
    #
    # query_list =[]
    # for idx, p in enumerate(passage_list):
    #     c = Query(
    #         query_id=str(1),
    #         text=p,
    #     )
    #
    #     query_list.append(c)
    #
    encoder = LateEncoder(backbone,
                             embed_sink_path=None,
                             backbone_kwargs={'model_name': model_name})
    #
    print("batch: ==============")
    batch_output = encoder.encode_passages(chunk_list)
    #
    vec1 = batch_output[0].vector
    vec2 = batch_output[1].vector
    from sentence_transformers.util import cos_sim
    print(cos_sim(vec1, vec2))


