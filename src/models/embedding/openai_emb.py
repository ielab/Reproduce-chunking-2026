from typing import List

from openai import OpenAI

from src.models.embedding.base_embedding import BaseEmbeddingModel
from src.registry import EMD_BACKBONE_REG



@EMD_BACKBONE_REG.register("openai")
class OpenAIEmbeddingModel(BaseEmbeddingModel):

    def __init__(self, api_key: str, model_name: str):

        self.api_key = api_key
        self.model_name = model_name

        self.client = OpenAI(api_key=self.api_key)

    @property
    def model_id(self) -> str:
        return f"openai:{self.model_name}"

    def get_embeddings(self, texts: List[str], **kwargs) -> List[List[float]]:

        response = self.client.embeddings.create(
            input=texts,
            model=self.model_name,
        )

        embeddings = [per.embedding for per in response.data]

        return embeddings

    def get_all_token_embeddings(self, texts: List[str], **kwargs) -> List[float]:
        pass