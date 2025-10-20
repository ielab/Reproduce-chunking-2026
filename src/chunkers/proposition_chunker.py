from typing import List, Dict
from itertools import count
from dataclasses import dataclass, field
import re
from tqdm import tqdm
import ast

from src.chunkers.base_chunker import BaseChunker
from src.types import Document, Chunk
from src.registry import CHUNKER_REG, GENERATOR_REG
from src.io.sink import JsonlSink
from src.models.generator.base_generator import BaseGenerator


system_prompt = """Decompose the "Content" into clear statements, ensuring they are interpretable out of context.
    1. Split compound sentence into simple sentences. Maintain the original phrasing from the input whenever possible.
    2. For any named entity that is accompanied by additional descriptive information, separate this information into its own distinct statement.  
    3. Decontextualize the statement by adding necessary modifier to nouns or entire sentences and replacing pronouns (e.g., "it", "he", "she", "they", "this", "that") with the full name of the entities they refer to."""


@dataclass
class DocumentGenTracker:
    doc_id: str
    paragraphs: List[str]
    propositions: List[str]


@CHUNKER_REG.register("Proposition")
class PropositionChunker(BaseChunker):

    def __init__(self,
                 gen_backbone: str,
                 batch_size: int = 20000,
                 chunk_sink_path: str | None = None,
                 **kwargs):

        # gen model
        gen_cls = GENERATOR_REG.get(gen_backbone)
        generative_model_name = kwargs.get("generative_model_name")

        if generative_model_name is None:
            raise KeyError("Generative model name not found")

        gen_backbone_kwargs = {"model": generative_model_name}
        self.generation_model: BaseGenerator = gen_cls(**gen_backbone_kwargs or {})

        self.batch_size = batch_size
        self.system_instruction = system_prompt

        self._sink = JsonlSink(chunk_sink_path) if chunk_sink_path else None
        self._sample = kwargs.get("sample")

    @staticmethod
    def _segment_paragraph(text: str) -> List[str]:

        pattern = re.compile(r'\n')

        return [x.strip() for x in pattern.split(text) if x.strip()]


    def request_generative_llm(self, traker_dict: Dict[str, DocumentGenTracker]):

        prompts = []
        doc_id_list = []

        for doc_id, tracker in traker_dict.items():

            for paragraph in tracker.paragraphs:
                prompts.append(paragraph)
                doc_id_list.append(doc_id)

        in_batch = False if len(prompts) <= 10 else True

        llm_output: Dict = self.generation_model.generate(
            prompts=prompts,
            system_instruction=self.system_instruction,
            temperature=0,
            display_name="Generate propositions",
            in_batch=in_batch,
            structured_output='array'

        )

        responses = llm_output['responses']

        assert len(responses) == len(doc_id_list)

        for doc_id, response in zip(doc_id_list, responses):

            if response:
                try:
                    if isinstance(response, str):
                        propositions = ast.literal_eval(response)
                    else:
                        propositions = response
                    traker_dict[doc_id].propositions.extend(propositions)
                except Exception as e:
                    print(f"LLM response Error: {e}")


    def proposition_pipeline(self, batch_documents: List[Document]) -> List[Chunk]:

        tracker_dict: Dict[str, DocumentGenTracker] = {}
        chunks: List[Chunk] = []

        # create tracker objs
        for doc in batch_documents:

            paragraphs = self._segment_paragraph(doc.text)

            tracker_dict[doc.doc_id] = DocumentGenTracker(
                doc_id=doc.doc_id,
                paragraphs=paragraphs,
                propositions=[]
            )

        self.request_generative_llm(tracker_dict)

        for doc_id, tracker in tracker_dict.items():

            chunk_counter = count()
            for proposition in tracker.propositions:

                chunk = Chunk(
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}-Chunk-{next(chunk_counter)}",
                    text=proposition
                )

                chunks.append(chunk)

        return chunks



    def chunk(self, raw_docs: List[Document]) -> List[Chunk]:

        chunks: List[Chunk] = []

        if self._sample is not None:
            raw_docs = raw_docs[:self._sample]

        for i in tqdm(range(0, len(raw_docs), self.batch_size)):

            batch_documents = raw_docs[i:i + self.batch_size]

            batch_chunks = self.proposition_pipeline(batch_documents)

            if self._sink:
                self._sink.write_batch(batch_chunks)

            chunks.extend(batch_chunks)

        return chunks