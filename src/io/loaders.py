from typing import List, Generator

from tqdm import tqdm
import numpy as np
import pickle

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


def load_pkl_embeddings(path: str) -> List[QueryEmbedding|ChunkEmbedding]:
    import os
    import glob

    # Check if path is a directory of sharded batch files
    if os.path.isdir(path):
        batch_files = sorted(glob.glob(os.path.join(path, "batch_*.pkl")))
        if not batch_files:
            raise ValueError(f"No batch_*.pkl files found in {path}")

        all_embeddings = []
        for batch_file in tqdm(batch_files, desc="Loading sharded embeddings"):
            with open(batch_file, 'rb') as f:
                batch = pickle.load(f)
                all_embeddings.extend(batch)
        return all_embeddings

    # Original behavior: single pickle file (preferred if it exists)
    if os.path.isfile(path):
        with open(path, 'rb') as f:
            loaded = pickle.load(f)
        return loaded

    # Fallback: check for sibling shards directory if .pkl doesn't exist
    if path.endswith('.pkl'):
        shards_dir = path.replace('.pkl', '_shards')
        if os.path.isdir(shards_dir):
            return load_pkl_embeddings(shards_dir)

    raise FileNotFoundError(f"No embeddings found at {path}")