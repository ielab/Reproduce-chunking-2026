import os
import json
from typing import List
from dataclasses import asdict

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

    def close(self):
        try:
            self.f.close()
        except:
            pass



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

