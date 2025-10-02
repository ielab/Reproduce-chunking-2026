from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddingModel(ABC):

    @property
    @abstractmethod
    def model_id(self) -> str:
        pass

    @abstractmethod
    def get_embeddings(self, texts: List[str], **kwargs):  # [B, Dim]
        pass

    @abstractmethod
    def get_all_token_embeddings(self, texts: List[str], **kwargs):
        pass

