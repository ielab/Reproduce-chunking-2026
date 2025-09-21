from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import numpy as np

@dataclass
class Document:
    doc_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    # start: int
    # end: int
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ChunkEmbedding:
    doc_id: str
    chunk_id: str
    vector: List[float] | np.ndarray
    # metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryEmbedding:
    query_id: str
    vector: List[float] | np.ndarray



@dataclass
class Query:
    query_id: str
    text: str
    qrels: Dict[str, int] = field(default_factory=Dict)
    chunk_must_Contain: str =field(default_factory=str)
    metadata: Dict[str, Any] = field(default_factory=dict)




