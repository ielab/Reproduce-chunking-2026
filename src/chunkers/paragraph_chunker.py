from typing import List, Iterable
from itertools import count

from src.chunkers.base_chunker import BaseChunker
from src.types import Document, Chunk
from src.registry import CHUNKER_REG
from src.io.sink import JsonlSink


@CHUNKER_REG.register("ParagraphChunker")
class ParagraphChunker(BaseChunker):

    def __init__(self,
                 chunk_sink_path: str|None = None,
                 **kwargs):

        self._sink = JsonlSink(chunk_sink_path) if chunk_sink_path else None
        self._sample = kwargs.get("sample")

    def chunk(self, raw_docs: Iterable[Document]) -> List[Chunk]:

        chunks: List[Chunk] = []

        if self._sample is not None:
            raw_docs = raw_docs[:self._sample]

        for document in raw_docs:

            paragraph_list = [x.strip() for x in document.text.split('\n') if x.strip() != '']

            chunk_counter = count()

            for paragraph in paragraph_list:
                chunk = Chunk(doc_id=document.doc_id,
                              chunk_id=f'{document.doc_id}-Chunk-{next(chunk_counter)}',
                              text=paragraph)

                chunks.append(chunk)


                if self._sink:
                    self._sink.write_batch([chunk])

        return chunks
