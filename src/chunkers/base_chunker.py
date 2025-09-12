from abc import ABC, abstractmethod
from typing import List, Iterable

from src.types import Document, Chunk



class BaseChunker(ABC):

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def chunk(self, raw_docs: Iterable[Document]):
        pass

