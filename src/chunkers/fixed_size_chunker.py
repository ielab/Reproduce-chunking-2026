from typing import List, Iterable
from itertools import count

from src.chunkers.base_chunker import BaseChunker
from src.types import Document, Chunk
from src.registry import CHUNKER_REG
from src.io.sink import JsonlSink
from transformers import AutoTokenizer
import tiktoken


@CHUNKER_REG.register("FixedSizeChunker")
class FixedSizeChunker(BaseChunker):

    def __init__(self,
                 chunk_sink_path: str|None=None,
                 fixed_size:int=256,
                 **kwargs
                 ):

        tokenizer_name = kwargs.get("tokenizer_name") or 'Qwen/Qwen3-Embedding-0.6B'

        # if tokenizer_name is not None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        # else:
        #     self.tokenizer = tiktoken.encoding_for_model("text-embedding-ada-002")


        self.chunk_sink_path = chunk_sink_path
        self.fixed_size = fixed_size

        self._sink = JsonlSink(chunk_sink_path) if chunk_sink_path else None
        self._sample = kwargs.get("sample")


    def chunk(self, raw_docs: Iterable[Document]):


        chunks = []


        if self._sample is not None:
            raw_docs = raw_docs[:self._sample]

        for document in raw_docs:

            chunk_counter = count()
            tokens = self.tokenizer.encode_plus(
                document.text,
                return_offsets_mapping=True,
                add_special_tokens=False,
            )

            token_offsets = tokens.offset_mapping

            last_offset = (0, 0)
            for i in range(self.fixed_size, len(token_offsets), self.fixed_size):

                start = last_offset[-1]
                end = token_offsets[i][-1]

                last_offset = token_offsets[i]
                chunk = Chunk(
                    doc_id=document.doc_id,
                    chunk_id=f'{document.doc_id}-Chunk-{next(chunk_counter)}',
                    text=document.text[start:end],
                )

                chunks.append(chunk)

                if self._sink is not None:
                    self._sink.write_batch([chunk])

            # add the remaining part of the text.
            if last_offset[1] != len(document.text):

                chunk = Chunk(
                    doc_id=document.doc_id,
                    chunk_id=f'{document.doc_id}-Chunk-{next(chunk_counter)}',
                    text=document.text[last_offset[1]:],
                )

                if self._sink is not None:
                    self._sink.write_batch([chunk])

        return chunks
