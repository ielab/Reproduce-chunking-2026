import os
import json
from typing import List
from dataclasses import asdict
import pickle

import pyarrow as pa
import pyarrow.parquet as pq
import zstandard as zstd
import gzip

from src.types import Chunk, ChunkEmbedding, Query, QueryEmbedding


class JsonlSink:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        if path.endswith(".jsonl"):
            self.f = open(path, "a", encoding="utf-8")
        elif path.endswith(".jsonl.gz"):
            self.f = gzip.open(path, "at", encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file type: {path}")

    def write_batch(self, objs: List[Chunk | ChunkEmbedding | Query | QueryEmbedding]):

        for o in objs:
            rec = asdict(o)
            self.f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        self.f.flush()
        os.fsync(self.f.fileno())

    def close(self):
        if hasattr(self, 'f') and self.f:
            try:
                self.f.flush()
                if hasattr(self.f, 'fileno'):
                    os.fsync(self.f.fileno())
                self.f.close()
            except Exception as e:
                print(f"Error closing sink: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class PickleSink:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        if path.endswith(".pkl"):
            self.path = path
        else:
            raise ValueError(f"Unsupported file type: {path}")

    def write_batch(self, objs: List[ChunkEmbedding|QueryEmbedding]):

        with open(self.path, "wb") as f:
            pickle.dump(objs, f, protocol=pickle.HIGHEST_PROTOCOL)



class ParquetSink:

    def __init__(self,
                 path: str):

        self.path = path

        self._schema = pa.schema([
            pa.field('doc_id', pa.string()),
            pa.field('chunk_id', pa.string()),
            pa.field('vector', pa.list_(pa.float32()))
        ])

        self.writer = pq.ParquetWriter(self.path, schema=self._schema, compression="zstd")


    def _batch_to_table(self, embs: List[ChunkEmbedding]):

        doc_ids = [e.doc_id for e in embs]
        chunk_ids = [e.chunk_id for e in embs]
        vectors = [e.vector for e in embs]

        arrays = {
            'doc_id': pa.array(doc_ids, type=pa.string()),
            'chunk_id': pa.array(chunk_ids, type=pa.string()),
            'vector': pa.array(vectors, type=pa.list_(pa.float32()))
        }

        return pa.table(arrays, schema=self._schema)


    def write_batch(self, embs: List[ChunkEmbedding]):

        table = self._batch_to_table(embs)

        self.writer.write_table(table)


    def close(self):

        if self.writer is not None:
            self.writer.close()
            self.writer = None


class JsonlZstSink:
    def __init__(self, path: str, level=10):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        self._fh = open(path, "ab")
        self._cctx = zstd.ZstdCompressor(level=level)
        self._writer = self._cctx.stream_writer(self._fh)

    def write_batch(self, objs: List[ChunkEmbedding]):

        for o in objs:
            rec = asdict(o)
            line = json.dumps(rec, ensure_ascii=False) + "\n"
            self._writer.write(line.encode("utf-8"))

        self._writer.flush()

    def close(self):
        try:
            self._writer.close()
            self._fh.close()
        except Exception:
            pass


def write_trec_file(path: str, results: dict, run_name: str, top_k: int = 1000):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for query_id, doc_scores in results.items():
            # doc_scores can be a dict {doc_id: score} or a list of tuples [(doc_id, score)]
            if isinstance(doc_scores, dict):
                # Sort items by score descending
                scores_list = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
            else:  # it's a list of tuples
                scores_list = doc_scores

            for rank, (doc_id, score) in enumerate(scores_list[:top_k]):
                f.write(f"{query_id} Q0 {doc_id} {rank + 1} {score} {run_name}\n")


def load_trec_file(path: str) -> dict:
    """
    Load a TREC format file and return ranking results.

    TREC format: query_id Q0 doc_id rank score run_name
    Returns: {query_id: {doc_id: score}} or {query_id: [(doc_id, score)]} depending on evaluator needs
    """
    results = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            query_id, _, doc_id, rank, score, run_name = parts[:6]

            if query_id not in results:
                results[query_id] = {}

            results[query_id][doc_id] = float(score)

    return results


