from abc import ABC, abstractmethod
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModel
from transformers.tokenization_utils_base import BatchEncoding


class BaseEmbeddingModel(ABC):

    def __init__(self, model_name: str):

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.model_name == "jinaai/jina-embeddings-v2-small-en":
            self.tokenizer.model_max_length = 8192

        model_kwargs = {"trust_remote_code": True}

        if torch.cuda.is_available():
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["device_map"] = "auto"

            # Try flash attention first, fall back gracefully if not available
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                self.model = AutoModel.from_pretrained(model_name, **model_kwargs).cuda()
                print(f"Loaded model {self.model_name} on {self.model.device} with flash_attention_2")
            except Exception as e:
                print(f"Flash attention not available, loading without it: {e}")
                model_kwargs.pop("attn_implementation")  # Remove flash attention
                self.model = AutoModel.from_pretrained(model_name, **model_kwargs).cuda()
                print(f"Loaded model {self.model_name} on {self.model.device}")
        else:
            self.model = AutoModel.from_pretrained(model_name, **model_kwargs)
            print(f"Loaded model {self.model_name} on CPU")
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

    def get_embed_dim(self) -> int:
        """Returns the embedding dimension of the model."""
        return self.model.config.hidden_size

    def get_embeddings_for_inputs(self, inputs: Dict, **kwargs):

        with torch.no_grad():

            inputs = BatchEncoding(inputs).to(self.model.device)
            outputs = self.model(**inputs)

        return outputs

