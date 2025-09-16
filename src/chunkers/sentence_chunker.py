from typing import List, Iterable
from itertools import count

import re

from src.chunkers.base_chunker import BaseChunker
from src.types import Document, Chunk
from src.registry import CHUNKER_REG
from src.io.sink import JsonlSink


@CHUNKER_REG.register('SentenceChunker')
class SentenceChunker(BaseChunker):

    def __init__(self,
                 n_sentences: int = 5,
                 chunk_sink_path: str = None,
                 **kwargs):

        self.n_sentences = n_sentences
        self._sample = kwargs.get('sample')

        self._sink = JsonlSink(chunk_sink_path) if chunk_sink_path else None


        self.pattern = re.compile(r"(?<!\w\.\w.)(?<![A-Z]\.)(?<![A-Z][a-z]\.)(?<! [a-z]\.)(?<![A-Z][a-z][a-z]\.)("
                             r"?<=\.|\?|!)\"*\s*\s*(?:\W*)([A-Z])")


    def _segment_sentence(self, text: str) -> List[str]:

        find_list = self.pattern.split(text)

        sentences = find_list[:1]
        for ids in range(1, len(find_list), 2):
            sentences.append(find_list[ids] + find_list[ids+1])

        sentences = [s for s in sentences if s.strip()]

        return sentences



    def chunk(self, raw_docs: Iterable[Document]):

        chunks = []


        if self._sample is not None:
            raw_docs = raw_docs[:self._sample]


        for document in raw_docs:

            sentences = self._segment_sentence(document.text)
            chunk_counter = count()
            for i in range(0, len(sentences), self.n_sentences):

                chunk = Chunk(
                    doc_id=document.doc_id,
                    chunk_id=f'{document.doc_id}-Chunk-{next(chunk_counter)}',
                    text=' '.join(sentences[i:i+self.n_sentences])
                )

                if self._sink is not None:

                    self._sink.write_batch([chunk])

        return chunks