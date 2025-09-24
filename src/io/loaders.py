from typing import List, Generator

from tqdm import tqdm
import numpy as np

from src.types import Chunk, Query, ChunkEmbedding, QueryEmbedding
from src.io.jsonl_io import read_jsonl, read_jsonl_gz


def load_chunks(path: str) -> List[Chunk]:

    out: List[Chunk] = []

    for r in read_jsonl(path):
        chunk = Chunk(
            doc_id=r['doc_id'],
            chunk_id=r['chunk_id'],
            text=r['text']
        )
        out.append(chunk)
    return out


def load_queries(path: str) -> List[Query]:

    out: List[Query] = []
    for r in read_jsonl(path):
        out.append(Query(**r))
    return out


def load_embeddings(path: str) -> Generator[ChunkEmbedding, None, None]:

    # out: List[Embedding] = []

    if path.endswith('.gz'):
        reader = read_jsonl_gz
    else:
        reader = read_jsonl

    for r in reader(path):

        # yield Embedding(**r)
        yield ChunkEmbedding(
            doc_id=r['doc_id'],
            chunk_id=r['chunk_id'],
            vector=np.array(r['vector'])
        )

def load_queries_embeddings(path: str) -> List[QueryEmbedding]:

    out: List[QueryEmbedding] = []
    for r in read_jsonl(path):
        out.append(QueryEmbedding(**r))
    return out