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

    def chunk(self, raw_docs: Iterable[Document]) -> List[Chunk]:

        chunks: List[Chunk] = []

        for document in raw_docs:

            doc_id = document.doc_id.split('-Paragraph-')[0]

            paragraph_list = [x.strip() for x in document.text.split('\n') if x.strip() != '']

            chunk_counter = count()

            for paragraph in paragraph_list:
                chunk = Chunk(doc_id=document.doc_id,
                              # chunk_id=f'Chunk-{next(chunk_counter)}',
                              chunk_id=f'{doc_id}-Chunk-{next(chunk_counter)}',
                              text=paragraph)

                chunks.append(chunk)


                if self._sink:
                    self._sink.write_batch([chunk])

        return chunks
