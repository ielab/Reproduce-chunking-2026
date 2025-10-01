from typing import List

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

from src.models.embedding.base_embedding import BaseEmbeddingModel
from src.registry import EMD_BACKBONE_REG


import torch.nn.functional as F
from torch import Tensor


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

    @staticmethod
    def mean_pooling(model_output, attention_mask: Tensor) -> Tensor:
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embeddings(self, texts: List[str], **kwargs):
        batch_dict = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**batch_dict)

        embeddings = self.mean_pooling(outputs, batch_dict['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()

    def get_all_token_embeddings(self, texts: List[str], **kwargs):

        inputs = self.tokenizer(
            texts,
            padding=True,
            return_tensors='pt',
            truncation=True
        ).to(self.model.device)

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

    @property
    def model_id(self) -> str:
        return f"Jinaai: {self.model_name}"

    @staticmethod
    def mean_pooling(model_output, attention_mask: Tensor) -> Tensor:
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embeddings(self, texts: List[str], **kwargs):
        batch_dict = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**batch_dict)

        embeddings = self.mean_pooling(outputs, batch_dict['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()

    def get_all_token_embeddings(self, texts: List[str], **kwargs):

        inputs = self.tokenizer(
            texts,
            padding=True,
            return_tensors='pt',
            truncation=True
        ).to(self.model.device)

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

