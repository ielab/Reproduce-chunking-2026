from typing import List, Iterable
from itertools import count

from src.chunkers.base_chunker import BaseChunker
from src.types import Document, Chunk
from src.registry import CHUNKER_REG
from src.io.sink import JsonlSink



class PropositionChunker(BaseChunker):
    """
    For Proposition Chunking, we do not implement it in this class. since we request openai api, to improve the
    efficiency and reduce the cost, we use the batch api to chunk, updating the file rather running online.

    The entire process of proposition chunking is quite simple, just providing the prompt and the LLM will output the result.
    Here is the instruction, more details about how to request batch api, please access Openai webpage:

    System prompt: Decompose the "Content" into clear statements, ensuring they are interpretable out of context.
    1. Split compound sentence into simple sentences. Maintain the original phrasing from the input whenever possible.
    2. For any named entity that is accompanied by additional descriptive information, separate this information into its own distinct statement.
    3. Decontextualize the statement by adding necessary modifier to nouns or entire sentences and replacing pronouns (e.g., "it", "he", "she", "they", "this", "that") with the full name of the entities they refer to.

    user prompt: text

    Note: To ensure the output is Json format, we need to use 'Structure Output'.
                    "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "string_list_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "propositions": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["propositions"]
                    }
                }
            }


    Output: {
        'propositions': [proposition1, proposition2, ...]
    }

    """

    def __init__(self, **kwargs):
        pass

    def chunk(self, row_docs: Iterable[Document]) -> List[Chunk]:

        pass