from typing import List

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

from src.models.embedding.base_embedding import BaseEmbeddingModel
from src.registry import EMD_BACKBONE_REG


@EMD_BACKBONE_REG.register('JinaaiV2')
class JinaaiEmbeddingModelV2(BaseEmbeddingModel):
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
            try:
                embeddings = self.emb_model.encode(texts, batch_size=kwargs.get('batch_size', 32))
            except RuntimeError as e:
                print(len(texts))
                print(len(texts[0]))
                raise e

        else:
            embeddings = self.emb_model.encode(texts, task=task, batch_size=kwargs.get('batch_size', 32))

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


@EMD_BACKBONE_REG.register('JinaaiV3')
class JinaaiEmbeddingModelV3(BaseEmbeddingModel):
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

            embeddings = self.emb_model.encode(texts, batch_size=kwargs.get('batch_size', 32))

        else:
            embeddings = self.emb_model.encode(texts, task=task, batch_size=kwargs.get('batch_size', 32))

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



if __name__ == '__main__':

    import numpy as np
    from sentence_transformers.util import cos_sim

    name = 'jinaai/jina-embeddings-v2-small-en'
    emb_model = JinaaiEmbeddingModelV2(name)

    e = emb_model.get_embeddings([
    'How is the weather tomorrow?',
    'How is the weather today?'
])

    print(cos_sim(e[0], e[1]))

