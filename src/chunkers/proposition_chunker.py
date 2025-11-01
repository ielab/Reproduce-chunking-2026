from typing import List, Dict, Set
from itertools import count
from dataclasses import dataclass
import re
from tqdm import tqdm
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import pandas as pd
import json
import gzip
import os

from src.chunkers.base_chunker import BaseChunker
from src.types import Document, Chunk
from src.registry import CHUNKER_REG, GENERATOR_REG
from src.io.sink import JsonlSink
from src.models.generator.base_generator import BaseGenerator


task_instruction = """Decompose the "Content" into clear and simple propositions, ensuring they are interpretable out of context.
1. Split compound sentence into simple sentences. Maintain the original phrasing from the input whenever possible.
2. For any named entity that is accompanied by additional descriptive information, separate this information into its own distinct proposition.
3. Decontextualize the proposition by adding necessary modifier to nouns or entire sentences and replacing pronouns (e.g., "it", "he", "she", "they", "this", "that") with the full name of the entities they refer to.

Input: Title: Eostre. Section: Theories and interpretations, Connection to Easter Hares. Content: The earliest evidence for the Easter Hare (Osterhase) was recorded in south-west Germany in 1678 by the professor of medicine Georg Franck von Franckenau, but it remained unknown in other parts of Germany until the 18th century. Scholar Richard Sermon writes that "hares were frequently seen in gardens in spring, and thus may have served as a convenient explanation for the origin of the colored eggs hidden there for children. Alternatively, there is a European tradition that hares laid eggs, since a hare’s scratch or form and a lapwing’s nest look very similar, and both occur on grassland and are first seen in the spring. In the nineteenth century the influence of Easter cards, toys, and books was to make the Easter Hare/Rabbit popular throughout Europe. German immigrants then exported the custom to Britain and America where it evolved into the Easter Bunny."
Output: [ "The earliest evidence for the Easter Hare was recorded in south-west Germany in 1678 by Georg Franck von Franckenau.", "Georg Franck von Franckenau was a professor of medicine.", "The evidence for the Easter Hare remained unknown in other parts of Germany until the 18th century.", "Richard Sermon was a scholar.", "Richard Sermon writes a hypothesis about the possible explanation for the connection between hares and the tradition during Easter", "Hares were frequently seen in gardens in spring.", "Hares may have served as a convenient explanation for the origin of the colored eggs hidden in gardens for children.", "There is a European tradition that hares laid eggs.", "A hare’s scratch or form and a lapwing’s nest look very similar.", "Both hares and lapwing’s nests occur on grassland and are first seen in the spring.", "In the nineteenth century the influence of Easter cards, toys, and books was to make the Easter Hare/Rabbit popular throughout Europe.", "German immigrants exported the custom of the Easter Hare/Rabbit to Britain and America.", "The custom of the Easter Hare/Rabbit evolved into the Easter Bunny in Britain and America." ]"""


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
                 **kwargs):

        self.gen_backbone = gen_backbone
        self.generative_model_name = kwargs.get("generative_model_name")

        if self.generative_model_name is None:
            raise KeyError("Generative model name not found")

        self.batch_size = batch_size
        self.task_instruction = task_instruction

        self._sink = JsonlSink(chunk_sink_path) if chunk_sink_path else None
        self._sample = kwargs.get("sample")

        # Thread-safe lock for sink writes and doc_id tracking
        self._write_lock = threading.Lock()

        def _safe_int(value, default):
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        # Resume functionality
        def _safe_bool(value, default):
            if value is None:
                return default
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                lower = value.strip().lower()
                if lower in {"true", "1", "yes", "y"}:
                    return True
                if lower in {"false", "0", "no", "n"}:
                    return False
            return default

        self.num_parallel_batches = _safe_int(kwargs.get("num_parallel_batches"), 4)

        workers_candidate = (
                kwargs.get("llm_workers")
                or kwargs.get("num_workers")
                or kwargs.get("llm_max_workers")
        )
        self.llm_max_workers = max(_safe_int(workers_candidate, 1), 1)

        self.resume = _safe_bool(kwargs.get("resume"), False)
        self.processed_doc_ids: Set[str] = set()

        if self.resume and chunk_sink_path:
            if os.path.exists(chunk_sink_path):
                self.processed_doc_ids = self._load_processed_doc_ids(chunk_sink_path)
                print(f"[PropositionChunker] Loaded {len(self.processed_doc_ids)} completed documents for resume.")
            else:
                print(
                    f"[PropositionChunker] Resume requested but no existing chunk file at {chunk_sink_path}. Starting fresh.")
                self.resume = False

    @staticmethod
    def _segment_sentence(text: str) -> List[str]:
        """Segment text into sentences using regex pattern."""

        sentence_pattern = re.compile(
            r"(?<!\w\.\w.)(?<![A-Z]\.)(?<![A-Z][a-z]\.)(?<! [a-z]\.)(?<![A-Z][a-z][a-z]\.)("
            r"?<=\.|\?|!)\"*\s*\s*(?:\W*)([A-Z])"
        )

        find_list = sentence_pattern.split(text)

        sentences = find_list[:1]
        for ids in range(1, len(find_list), 2):
            sentences.append(find_list[ids] + find_list[ids + 1])

        sentences = [s for s in sentences if s.strip()]

        return sentences

    def _load_processed_doc_ids(self, path: str) -> Set[str]:
        """Load processed document IDs from existing chunk file."""
        doc_ids: Set[str] = set()

        if not os.path.exists(path):
            return doc_ids

        opener = gzip.open if path.endswith(".gz") else open

        try:
            with opener(path, "rt", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    doc_id = record.get("doc_id")
                    if doc_id is not None:
                        doc_ids.add(doc_id)
        except FileNotFoundError:
            return set()
        except Exception as exc:
            print(f"[PropositionChunker] Failed to read existing chunks for resume: {exc}")

        return doc_ids

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
                if not paragraphs:
                    print(f"⚠️ Document {doc.doc_id} has no paragraphs (empty/whitespace)")
                tracker_dict[doc.doc_id] = DocumentGenTracker(
                    doc_id=doc.doc_id,
                    paragraphs=paragraphs,
                    propositions=[]
                )

            # Request LLM
            prompts = []
            paragraph_list = []
            doc_id_list = []
            # pack prompt
            for doc_id, tracker in tracker_dict.items():
                for paragraph in tracker.paragraphs:
                    new_prompt = self.task_instruction + "\n\n" + f"Input: {paragraph}" + "\nOutput:"
                    prompts.append(new_prompt)
                    doc_id_list.append(doc_id)
                    paragraph_list.append(paragraph)


            in_batch = False if len(prompts) <= 10 else True

            llm_output: Dict = generation_model.generate(
                prompts=prompts,
                temperature=0,
                top_k=1,
                display_name=f"Generate propositions",
                in_batch=in_batch,
                structured_output='array',
                max_workers=self.llm_max_workers
            )

            responses = llm_output['responses']

            # Process responses
            for doc_id, response, paragraph in zip(doc_id_list, responses, paragraph_list):
                if response:
                    try:
                        if isinstance(response, str):
                            propositions = ast.literal_eval(response)
                        else:
                            propositions = response
                        tracker_dict[doc_id].propositions.extend(propositions)
                    except Exception as e:
                        # print(f"LLM response parsing error for {doc_id}: {e}")
                        # print(f"Falling back to sentence segmentation for this paragraph")
                        sentences = self._segment_sentence(paragraph)
                        tracker_dict[doc_id].propositions.extend(sentences)
                else:
                    sentences = self._segment_sentence(paragraph)
                    tracker_dict[doc_id].propositions.extend(sentences)



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

    def chunk(self, raw_docs: List[Document]) -> List[Chunk]:
        """
        Main chunking method using ThreadPoolExecutor for parallel processing.
        """
        chunks: List[Chunk] = []
        failed_batches = []

        if self._sample is not None:
            raw_docs = raw_docs[:self._sample]

        # Resume functionality: filter out already-processed documents
        if self.resume and self.processed_doc_ids:
            original_count = len(raw_docs)
            raw_docs = [doc for doc in raw_docs if doc.doc_id not in self.processed_doc_ids]
            skipped = original_count - len(raw_docs)
            if skipped > 0:
                print(f"[PropositionChunker] Skipping {skipped} documents already processed.")
            if not raw_docs:
                print("[PropositionChunker] No remaining documents to process. Resume finished.")
                return chunks

        # Separate empty and non-empty documents
        empty_docs = [doc for doc in raw_docs if not doc.text or not doc.text.strip()]
        non_empty_docs = [doc for doc in raw_docs if doc.text and doc.text.strip()]

        if empty_docs:
            print(f"[PropositionChunker] Found {len(empty_docs)} documents with empty text. Creating empty chunks.")

            # Create empty chunks for empty documents
            empty_chunks = []
            for doc in empty_docs:
                chunk = Chunk(
                    doc_id=doc.doc_id,
                    chunk_id=f"{doc.doc_id}-Chunk-0",
                    text=""
                )
                empty_chunks.append(chunk)

            # Save empty chunks immediately
            if self._sink and empty_chunks:
                with self._write_lock:
                    self._sink.write_batch(empty_chunks)
                    if self.resume:
                        self.processed_doc_ids.update(chunk.doc_id for chunk in empty_chunks)
                    print(f"Saved {len(empty_chunks)} empty chunks")

            chunks.extend(empty_chunks)

        # Process only non-empty documents
        if not non_empty_docs:
            print("[PropositionChunker] No non-empty documents to process.")
            return chunks


        # Split documents into batches
        batches = [non_empty_docs[i:i + self.batch_size]
                   for i in range(0, len(non_empty_docs), self.batch_size)]

        batches_to_process = list(enumerate(batches))

        print(
            f"Processing {len(non_empty_docs)} documents in {len(batches)} batches using {self.num_parallel_batches} parallel threads")

        try:
            # Process batches with threads
            with ThreadPoolExecutor(max_workers=self.num_parallel_batches) as executor:
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
                                    # Update processed doc IDs for resume functionality
                                    if self.resume:
                                        self.processed_doc_ids.update(chunk.doc_id for chunk in batch_chunks)
                                    print(f"Batch {batch_idx}: Saved {len(batch_chunks)} chunks")

                            chunks.extend(batch_chunks)
                            pbar.update(1)

                        except Exception as e:
                            print(f"Error retrieving results for batch {batch_idx}: {e}")
                            failed_batches.append(batch_idx)
                            pbar.update(1)

            if failed_batches:
                print(f"WARNING: {len(failed_batches)} batches failed: {failed_batches}")

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

    # Create chunker with resume enabled
    chunker = PropositionChunker(
        gen_backbone="gemini",
        batch_size=100,
        llm_workers=8,
        generative_model_name="gemini-2.5-flash-lite",
        chunk_sink_path="llm_multi_processing.jsonl",
        resume=True  # Enable resume functionality
    )

    # Process
    chunks = chunker.chunk(test_docs)
    print(f"Created {len(chunks)} chunks from {len(test_docs)} documents")