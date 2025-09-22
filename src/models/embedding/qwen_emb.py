from typing import List

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from torch import Tensor

from src.models.embedding.base_embedding import BaseEmbeddingModel
from src.registry import EMD_BACKBONE_REG


@EMD_BACKBONE_REG.register('Qwen3')
class Qwen3EmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name: str):

        self.model_name = model_name
        self.emb_model = SentenceTransformer(model_name)

        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        self.task = 'Given a web search query, retrieve relevant passages that answer the query'

    @property
    def model_id(self) -> str:
        return f"Qwen3:{self.model_name}"

    @staticmethod
    def get_detailed_instruct(task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery:{query}'

    @staticmethod
    def last_token_pool(last_hidden_states: Tensor,
                        attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def get_embeddings2(self, texts: List[str], **kwargs):

        prompt_name = kwargs.get("prompt_name", None)

        if prompt_name is not None:
            texts = [
                self.get_detailed_instruct(self.task, text) for text in texts
            ]

        batch_dict = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        outputs = self.model(**batch_dict)

        embeddings = self.last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    def get_embeddings(self, texts: List[str], **kwargs):

        prompt_name = kwargs.get('prompt_name', None)

        if prompt_name is None:

            embeddings = self.emb_model.encode(texts)

        else:
            embeddings = self.emb_model.encode(texts, prompt_name=prompt_name)

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