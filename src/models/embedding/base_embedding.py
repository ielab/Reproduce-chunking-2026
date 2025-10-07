from abc import ABC, abstractmethod
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModel
from transformers.tokenization_utils_base import BatchEncoding


class BaseEmbeddingModel(ABC):

    def __init__(self, model_name: str):

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_kwargs = {"trust_remote_code": True}

        if torch.cuda.is_available():
            model_kwargs["attn_implementation"] = "flash_attention_2"
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["device_map"] = "auto"
        try:
            self.model = AutoModel.from_pretrained(model_name, **model_kwargs).cuda()
        except Exception as e:
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto").cuda()

        print(f"Loaded model {self.model_name} on {self.model.device}")
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

    def get_embeddings_for_inputs(self, inputs: Dict, **kwargs):

        with torch.no_grad():

            inputs = BatchEncoding(inputs).to(self.model.device)
            outputs = self.model(**inputs)

        return outputs

