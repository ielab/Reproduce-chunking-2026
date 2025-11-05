from abc import ABC, abstractmethod
from typing import List, Iterable, Union

from src.types import Document, Chunk



class BaseChunker(ABC):

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def chunk(self, raw_docs: Union[Iterable[Document], Iterable[Chunk]]):
        pass

