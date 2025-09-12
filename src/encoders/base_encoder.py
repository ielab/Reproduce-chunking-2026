from abc import ABC, abstractmethod
from typing import List

from src.types import Chunk, Query


class BaseEncoder(ABC):

    @abstractmethod
    def encode(self,
               chunks: List[Chunk],
               batch_size: int,
               **kwargs):
        pass


    @abstractmethod
    def encode_queries(self,
                       queries: List[Query],
                       query_sink_path: str,
                       batch_size: int,
                       **kwargs):
        pass
