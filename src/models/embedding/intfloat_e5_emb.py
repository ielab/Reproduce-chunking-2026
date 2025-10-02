from typing import List

import torch
from transformers import AutoTokenizer, AutoModel
from torch import Tensor
from torch.nn import functional as F

from src.models.embedding.base_embedding import BaseEmbeddingModel
from src.registry import EMD_BACKBONE_REG


@EMD_BACKBONE_REG.register("IntFloatE5")
class IntFloatE5EmbeddingModel(BaseEmbeddingModel):

    def __init__(self, model_name: str = None, **kwargs):

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        model_kwargs = {"trust_remote_code": True}
        if torch.cuda.is_available():
            # model_kwargs["attn_implementation"] = "flash_attention_2"
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["device_map"] = "auto"

        self.model = AutoModel.from_pretrained(model_name, **model_kwargs)


    @property
    def model_id(self) -> str:

        return f"IntFloat_E5: {self.model_name}"

    @staticmethod
    def average_pool(last_hidden_states: Tensor,
                     attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    @staticmethod
    def get_detailed_instruct(query: str) -> str:

        task_description = "Given a web search query, retrieve relevant passages that answer the query"

        return f'Instruct: {task_description}\nQuery: {query}'

    def get_embeddings(self, texts: List[str], **kwargs):

        task = kwargs.get('task')
        if task == 'query':
            texts = [self.get_detailed_instruct(text) for text in texts]

        inputs = self.tokenizer(
            texts,
            padding=True,
            return_tensors='pt',
            truncation=True
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = self.average_pool(outputs.last_hidden_state, inputs['attention_mask'])

        return embeddings.detach().cpu().numpy()


    def get_all_token_embeddings(self, texts: List[str], **kwargs):

        task = kwargs.get('task')
        if task == 'query':
            texts = [self.get_detailed_instruct(text) for text in texts]

        inputs = self.tokenizer(
            texts,
            padding=True,
            return_tensors='pt',
            truncation=True
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)


        return outputs