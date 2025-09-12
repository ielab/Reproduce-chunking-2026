from typing import List

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

from src.models.embedding.base_embedding import BaseEmbeddingModel
from src.registry import EMD_BACKBONE_REG


@EMD_BACKBONE_REG.register('JinaV3')
class JinaV3EmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name: str):

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = SentenceTransformer(model_name, trust_remote_code=True)

    @property
    def model_id(self) -> str:
        return f"JinaV3: {self.model_name}"

    def get_embeddings(self, texts: List[str], **kwargs):

        task = kwargs.get('task', None)

        if task is None:

            embeddings = self.model.encode(texts)

        else:
            embeddings = self.model.encode(texts, task=task)

        return embeddings

    def get_all_token_embeddings(self, texts: List[str], **kwargs):

        inputs = self.tokenizer(
            texts,
            padding=True,
            return_tensors='pt',
            truncation=True
        )

        outputs = self.model(**inputs).last_hidden_state

        return outputs
