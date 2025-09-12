from typing import List, Iterable
from itertools import count

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import Document as SchemaDocument
from tqdm import tqdm


from src.chunkers.base_chunker import BaseChunker
from src.types import Document, Chunk
from src.registry import CHUNKER_REG
from src.io.sink import JsonlSink
from transformers import AutoTokenizer


@CHUNKER_REG.register("SemanticChunker")
class SemanticChunker(BaseChunker):

    def __init__(self,
                 embedding_model_name: str='BAAI/bge-small-en',
                 chunk_sink_path: str|None=None,
                 sample: int = None,
                 **kwargs):

        self.embedding_model_name = embedding_model_name
        self.embedding_model = None
        self.splitter = None

        if self.embedding_model_name is None:
            self.embedding_model_name = 'BAAI/bge-small-en'

        self._setup_semantic_chunking()

        self._sink = JsonlSink(chunk_sink_path)
        self._sample = sample


    def _setup_semantic_chunking(self):

        self.embed_model = HuggingFaceEmbedding(
            model_name=self.embedding_model_name,
            trust_remote_code=True,
            embed_batch_size=1
        )

        self.splitter = SemanticSplitterNodeParser(
            embed_model=self.embed_model,
            # show_progress=False
        )


    def chunk(self, raw_docs: Iterable[Document]):

        chunks = []

        chunk_counter = count()

        if self._sample is not None:
            raw_docs = raw_docs[:self._sample]


        for document in tqdm(raw_docs):

            nodes = [
                (node.start_char_idx, node.end_char_idx)
                for node in self.splitter.get_nodes_from_documents(
                    [SchemaDocument(text=document.text)], show_progress=False
                )
            ]

            for node in nodes:
                chunk = Chunk(
                    doc_id=document.doc_id,
                    chunk_id=f'{document.doc_id}-Chunk-{next(chunk_counter)}',
                    text=document.text[node[0]:node[1]]
                )

                chunks.append(chunk)

                if self._sink is not None:
                    self._sink.write_batch([chunk])

