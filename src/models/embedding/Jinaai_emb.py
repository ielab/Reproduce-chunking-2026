from typing import List

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

from src.models.embedding.base_embedding import BaseEmbeddingModel
from src.registry import EMD_BACKBONE_REG


@EMD_BACKBONE_REG.register('Jinaai')
class JinaV3EmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name: str):

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.emb_model = SentenceTransformer(model_name, trust_remote_code=True)

        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    @property
    def model_id(self) -> str:
        return f"Jinaai: {self.model_name}"

    def get_embeddings(self, texts: List[str], **kwargs):

        task = kwargs.get('task', None)

        if task is None:

            embeddings = self.emb_model.encode(texts)

        else:
            embeddings = self.emb_model.encode(texts, task=task)

        return embeddings

    def get_all_token_embeddings(self, texts: List[str], **kwargs):

        inputs = self.tokenizer(
            texts,
            padding=True,
            return_tensors='pt',
            truncation=True
        )

        outputs = self.model(**inputs)

        return outputs
