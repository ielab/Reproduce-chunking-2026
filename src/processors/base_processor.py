from abc import ABC, abstractmethod
from typing import List
from src.types import Document, Query


class BaseProcessor(ABC):

    @abstractmethod
    def load_corpus(self) ->List[Document]:
        pass


    @abstractmethod
    def load_query(self, sink_path: str|None=None) ->List[Query]:
        pass

    # @staticmethod
    # def save_obj():


# processor name
# dataset name