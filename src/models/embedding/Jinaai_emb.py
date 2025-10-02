from typing import List

import torch
from transformers import AutoTokenizer, AutoModel

from src.models.embedding.base_embedding import BaseEmbeddingModel
from src.registry import EMD_BACKBONE_REG


@EMD_BACKBONE_REG.register('JinaaiV2')
class JinaaiEmbeddingModelV2(BaseEmbeddingModel):
    def __init__(self, model_name: str):

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        model_kwargs = {"trust_remote_code": True}
        if torch.cuda.is_available():
            model_kwargs["attn_implementation"] = "flash_attention_2"
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["device_map"] = "auto"

        self.model = AutoModel.from_pretrained(model_name, **model_kwargs)

    @property
    def model_id(self) -> str:
        return f"Jinaai: {self.model_name}"

    def get_embeddings(self, texts: List[str], **kwargs):

        with torch.no_grad():
            outputs = self.model.encode(texts)

        return outputs

    def get_all_token_embeddings(self, texts: List[str], **kwargs):

        inputs = self.tokenizer(
            texts,
            padding=True,
            return_tensors='pt',
            truncation=True
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs


@EMD_BACKBONE_REG.register('JinaaiV3')
class JinaaiEmbeddingModelV3(BaseEmbeddingModel):
    def __init__(self, model_name: str):

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        model_kwargs = {"trust_remote_code": True}
        if torch.cuda.is_available():
            model_kwargs["attn_implementation"] = "flash_attention_2"
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["device_map"] = "auto"

        self.model = AutoModel.from_pretrained(model_name, **model_kwargs)

        self.prompts = {
        "retrieval.query": "Represent the query for retrieving evidence documents: ",
        "retrieval.passage": "Represent the document for retrieval: ",
        "separation": "",
        "classification": "",
        "text-matching": ""
    }

    @property
    def model_id(self) -> str:
        return f"Jinaai: {self.model_name}"

    def get_embeddings(self, texts: List[str], **kwargs):

        task = kwargs.get("task")

        with torch.no_grad():

            if task is not None:
                embeddings = self.model.encode(texts, task=task)
            else:
                embeddings = self.model.encode(texts)

        return embeddings


    def get_all_token_embeddings(self, texts: List[str], **kwargs):

        task = kwargs.get("task")
        if task is not None:
            prefix = self.prompts[task]
            texts = [prefix + text for text in texts]

        inputs = self.tokenizer(
            texts,
            padding=True,
            return_tensors='pt',
            truncation=True
        ).to(self.model.device)

        with torch.no_grad():
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

