from typing import List
import torch
from sentence_transformers import SentenceTransformer
from src.models.embedding.base_embedding import BaseEmbeddingModel
from src.registry import EMD_BACKBONE_REG


@EMD_BACKBONE_REG.register('Qwen3')
class Qwen3EmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name: str):
        self.model_name = model_name
        model_kwargs = {}
        tokenizer_kwargs = {}
        if torch.cuda.is_available():
            model_kwargs = {"attn_implementation": "flash_attention_2", "device_map": "auto", "torch_dtype": torch.float16}
            tokenizer_kwargs = {"padding_side": "left"}

        self.model = SentenceTransformer(
            model_name,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            trust_remote_code=True
        )

    @property
    def model_id(self) -> str:
        return f"Qwen3:{self.model_name}"

    def get_embeddings(self, texts: List[str], **kwargs):
        prompt_name = kwargs.get('prompt_name', None)
        with torch.no_grad():
            embeddings = self.model.encode(texts, prompt_name=prompt_name)
        return embeddings

    def get_all_token_embeddings(self, texts: List[str], **kwargs):
        raise NotImplementedError(
            "This model is designed for single-vector sentence embeddings, not for retrieving all token embeddings."
        )
