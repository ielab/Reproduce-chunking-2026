from typing import List, Dict
from itertools import count
from dataclasses import dataclass
import re
from tqdm import tqdm
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import pandas as pd
import json

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
    """
    Thread-based version of PropositionChunker that avoids multiprocessing pickling issues.

    Uses ThreadPoolExecutor instead of ProcessPoolExecutor:
    - Pros: No pickling issues, simpler error handling, shared memory
    - Cons: Limited by Python's GIL for CPU-bound tasks
    - Best for: I/O-bound tasks like API calls (which this is!)

    Since most time is spent waiting for API responses, threads work well here.
    """

    def __init__(self,
                 gen_backbone: str,
                 batch_size: int = 20000,
                 chunk_sink_path: str | None = None,
                 llm_workers: int = 4,
                 **kwargs):

        self.gen_backbone = gen_backbone
        self.generative_model_name = kwargs.get("generative_model_name")

        if self.generative_model_name is None:
            raise KeyError("Generative model name not found")

        self.batch_size = batch_size
        self.system_instruction = system_prompt
        self.num_workers = llm_workers

        self._sink = JsonlSink(chunk_sink_path) if chunk_sink_path else None
        self._sample = kwargs.get("sample")

        self._failed_batches_data = []
        self.failed_batches_log_path = chunk_sink_path.replace("chunks.jsonl", "_failed_batches.json")

        # Thread-safe lock for sink writes
        self._write_lock = threading.Lock()

    @staticmethod
    def _segment_paragraph(text: str) -> List[str]:
        pattern = re.compile(r'\n')
        return [x.strip() for x in pattern.split(text) if x.strip()]

    def process_batch(self, batch_docs: List[Document]) -> List[Chunk]:
        """
        Process a batch of documents.
        This method runs in a thread, so no pickling needed!
        """
        try:
            # Create generator instance for this thread
            gen_cls = GENERATOR_REG.get(self.gen_backbone)
            generation_model: BaseGenerator = gen_cls(model=self.generative_model_name)

            # Create tracker dict
            tracker_dict: Dict[str, DocumentGenTracker] = {}

            for doc in batch_docs:
                paragraphs = self._segment_paragraph(doc.text)
                tracker_dict[doc.doc_id] = DocumentGenTracker(
                    doc_id=doc.doc_id,
                    paragraphs=paragraphs,
                    propositions=[]
                )

            # Request LLM
            prompts = []
            doc_id_list = []

            for doc_id, tracker in tracker_dict.items():
                for paragraph in tracker.paragraphs:
                    prompts.append(paragraph)
                    doc_id_list.append(doc_id)

            in_batch = False if len(prompts) <= 10 else True

            llm_output: Dict = generation_model.generate(
                prompts=prompts,
                system_instruction=self.system_instruction,
                temperature=0,
                top_k=1,
                display_name=f"Generate propositions",
                in_batch=in_batch,
                structured_output='array'
            )

            responses = llm_output['responses']

            # Process responses
            for doc_id, response in zip(doc_id_list, responses):
                if response:
                    try:
                        if isinstance(response, str):
                            propositions = ast.literal_eval(response)
                        else:
                            propositions = response
                        tracker_dict[doc_id].propositions.extend(propositions)
                    except Exception as e:
                        print(f"LLM response parsing error for {doc_id}: {e}")

            # Create chunks
            chunks: List[Chunk] = []
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

        except Exception as e:
            import traceback
            print(f"Error processing batch: {str(e)}\n{traceback.format_exc()}")
            return []

    def _save_failed_batches(self, failed_batch_indices: List[int]):
        """Save failed batch information to a JSON file for later retry."""

        failure_record = {
            'timestamp': str(pd.Timestamp.now()),
            'batch_size': self.batch_size,
            'failed_batch_indices': failed_batch_indices,
            'detailed_failures': self._failed_batches_data,
            'total_failed_docs': sum(d['doc_count'] for d in self._failed_batches_data)
        }

        with open(self.failed_batches_log_path, 'w') as f:
            json.dump(failure_record, f, indent=2)

        print(f"Failed batch information saved to {self.failed_batches_log_path}")

    def chunk(self, raw_docs: List[Document]) -> List[Chunk]:
        """
        Main chunking method using ThreadPoolExecutor for parallel processing.
        """
        chunks: List[Chunk] = []
        failed_batches = []

        if self._sample is not None:
            raw_docs = raw_docs[:self._sample]

        # Split documents into batches
        batches = [raw_docs[i:i + self.batch_size]
                   for i in range(0, len(raw_docs), self.batch_size)]

        retry_batch_indices = []
        # If retrying, only process specified batches
        if retry_batch_indices:
            print(f"Retrying {len(retry_batch_indices)} failed batches: {retry_batch_indices}")
            batches_to_process = [(idx, batches[idx]) for idx in retry_batch_indices if idx < len(batches)]
        else:
            batches_to_process = list(enumerate(batches))

        print(f"Processing {len(raw_docs)} documents in {len(batches_to_process)} batches using {self.num_workers} threads")



        try:
            # Process batches with threads
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                # Submit all batches
                future_to_batch = {
                    executor.submit(self.process_batch, batch): idx
                    for idx, batch in batches_to_process
                }

                # Collect results as they complete
                with tqdm(total=len(batches), desc="Processing batches") as pbar:
                    for future in as_completed(future_to_batch):
                        batch_idx = future_to_batch[future]
                        try:
                            batch_chunks = future.result()

                            if self._sink and batch_chunks:
                                with self._write_lock:
                                    self._sink.write_batch(batch_chunks)
                                    print(f"Batch {batch_idx}: Saved {len(batch_chunks)} chunks")

                            chunks.extend(batch_chunks)
                            pbar.update(1)

                        except Exception as e:
                            print(f"Error retrieving results for batch {batch_idx}: {e}")
                            failed_batches.append(batch_idx)

                            batch_docs = batches[batch_idx] if batch_idx < len(batches) else []
                            self._failed_batches_data.append({
                                'batch_idx': batch_idx,
                                'error': str(e),
                                'doc_ids': [doc.doc_id for doc in batch_docs],
                                'doc_count': len(batch_docs)
                            })

                            pbar.update(1)
            if failed_batches:
                print(f"WARNING: {len(failed_batches)} batches failed: {failed_batches}")

                if self.failed_batches_log_path:
                    self._save_failed_batches(failed_batches)

            return chunks
        finally:
            # Always close the sink
            if self._sink:
                self._sink.close()


# Example usage
if __name__ == "__main__":
    from src.types import Document

    # Test documents
    test_docs = [
        Document(
            doc_id=f"doc_{i}",
            text="This is a test sentence. It has multiple parts. Each part should be separated."
        )
        for i in range(100)
    ]

    docs2 = [
        Document(
            doc_id=f"doc_{i}",
            text="please call the model, then you will get the llm result"
        )
    for i in range(100, 200)
    ]

    test_docs = test_docs + docs2

    # Create chunker
    chunker = PropositionChunker(
        gen_backbone="gemini",
        batch_size=100,
        llm_workers=8,
        generative_model_name="gemini-2.5-flash-lite",
        chunk_sink_path = "llm_multi_processing.jsonl"
    )

    # Process
    chunks = chunker.chunk(test_docs)
    print(f"Created {len(chunks)} chunks from {len(test_docs)} documents")