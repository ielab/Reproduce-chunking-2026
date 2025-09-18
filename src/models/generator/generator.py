from abc import ABC, abstractmethod
from typing import List, Union, Optional

from openai import OpenAI


class BaseGenerator(ABC):

    @abstractmethod
    def generate(self,
                 input_text: Union[str, List[str]],
                 temperature: float = 1.0):
        pass


class OpenAIGenerator(BaseGenerator):

    def __init__(self,
                 model_name,
                 api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name


    def generate(self,
                 input_text: Union[str, List[str]],
                 temperature: float = 1.0):


        response = self.client.responses.create(
            model=self.model_name,
            input=input_text,
            temperature=temperature
        )

        return response.output_text






