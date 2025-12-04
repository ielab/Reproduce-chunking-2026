from typing import List, Dict, Set, Union, Literal
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

Input: Title: Ēostre. Section: Theories and interpretations, Connection to Easter Hares. Content: The earliest evidence for the Easter Hare (Osterhase) was recorded in south-west Germany in 1678 by the professor of medicine Georg Franck von Franckenau, but it remained unknown in other parts of Germany until the 18th century. Scholar Richard Sermon writes that "hares were frequently seen in gardens in spring, and thus may have served as a convenient explanation for the origin of the colored eggs hidden there for children. Alternatively, there is a European tradition that hares laid eggs, since a hare’s scratch or form and a lapwing’s nest look very similar, and both occur on grassland and are first seen in the spring. In the nineteenth century the influence of Easter cards, toys, and books was to make the Easter Hare/Rabbit popular throughout Europe. German immigrants then exported the custom to Britain and America where it evolved into the Easter Bunny."
Output: [ "The earliest evidence for the Easter Hare was recorded in south-west Germany in 1678 by Georg Franck von Franckenau.", "Georg Franck von Franckenau was a professor of medicine.", "The evidence for the Easter Hare remained unknown in other parts of Germany until the 18th century.", "Richard Sermon was a scholar.", "Richard Sermon writes a hypothesis about the possible explanation for the connection between hares and the tradition during Easter", "Hares were frequently seen in gardens in spring.", "Hares may have served as a convenient explanation for the origin of the colored eggs hidden in gardens for children.", "There is a European tradition that hares laid eggs.", "A hare’s scratch or form and a lapwing’s nest look very similar.", "Both hares and lapwing’s nests occur on grassland and are first seen in the spring.", "In the nineteenth century the influence of Easter cards, toys, and books was to make the Easter Hare/Rabbit popular throughout Europe.", "German immigrants exported the custom of the Easter Hare/Rabbit to Britain and America.", "The custom of the Easter Hare/Rabbit evolved into the Easter Bunny in Britain and America." ]"""


@dataclass
class DocumentGenTracker:
    doc_id: str
    parent_chunk_id: str | None  # None for Document inputs, set for Chunk inputs
    paragraphs: List[str]
    propositions: List[str]


@dataclass
class ProcessingUnit:
    """Unified representation for both Document and Chunk inputs"""
    doc_id: str
    parent_chunk_id: str | None  # None for Documents, actual chunk_id for Chunks
    text: str

    @classmethod
    def from_document(cls, doc: Document) -> 'ProcessingUnit':
        return cls(doc_id=doc.doc_id, parent_chunk_id=None, text=doc.text)

    @classmethod
    def from_chunk(cls, chunk: Chunk) -> 'ProcessingUnit':
        return cls(doc_id=chunk.doc_id, parent_chunk_id=chunk.chunk_id, text=chunk.text)


@CHUNKER_REG.register("Proposition")
class PropositionChunker(BaseChunker):
    """
    Thread-based version of PropositionChunker that avoids multiprocessing pickling issues.

    Uses ThreadPoolExecutor instead of ProcessPoolExecutor:
    - Pros: No pickling issues, simpler error handling, shared memory
    - Cons: Limited by Python's GIL for CPU-bound tasks
    - Best for: I/O-bound tasks like API calls (which this is!)

    Since most time is spent waiting for API responses, threads work well here.

    Accepts both List[Document] and List[Chunk] as input:
    - For List[Document]: Creates chunks with IDs like {doc_id}-Chunk-{N}
    - For List[Chunk]: Creates propositions that inherit the parent chunk_id
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
        self.raw_input_type: Literal["Document", "Chunk"] = "Document"
        self.chunk_sink_path = chunk_sink_path

        self.resume = _safe_bool(kwargs.get("resume"), False)
        # For resume: track by a composite key (doc_id, parent_chunk_id)
        self.processed_units: Set[tuple] = set()

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

    def _load_processed_units(self, path: str) -> Set[tuple]:
        """Load processed units (doc_id, parent_chunk_id tuples) from existing chunk file."""
        units: Set[tuple] = set()

        if not os.path.exists(path):
            return units

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

                    chunk_id = record.get("chunk_id")
                    doc_id = record.get("doc_id")

                    if doc_id is not None:
                        if self.raw_input_type == "Document":
                            units.add((doc_id, None))
                        else:
                            units.add((doc_id, chunk_id))

        except Exception as exc:
            print(f"[PropositionChunker] Failed to read existing chunks for resume: {exc}")

        return units

    @staticmethod
    def _segment_paragraph(text: str) -> List[str]:
        pattern = re.compile(r'\n')
        return [x.strip() for x in pattern.split(text) if x.strip()]

    def process_batch(self, batch_units: List[ProcessingUnit]) -> List[Chunk]:
        """
        Process a batch of processing units (converted from Documents or Chunks).
        This method runs in a thread, so no pickling needed!
        """
        try:
            # Create generator instance for this thread
            gen_cls = GENERATOR_REG.get(self.gen_backbone)
            generation_model: BaseGenerator = gen_cls(model=self.generative_model_name)

            # Create tracker dict - use (doc_id, parent_chunk_id) as key
            tracker_dict: Dict[tuple, DocumentGenTracker] = {}

            for unit in batch_units:
                # if self.raw_input_type == "Document":
                paragraphs = self._segment_paragraph(unit.text)
                # else:
                #     paragraphs = [unit.text]
                if not paragraphs:
                    print(f"⚠️ Unit {unit.doc_id}/{unit.parent_chunk_id} has no paragraphs (empty/whitespace)")

                key = (unit.doc_id, unit.parent_chunk_id)
                tracker_dict[key] = DocumentGenTracker(
                    doc_id=unit.doc_id,
                    parent_chunk_id=unit.parent_chunk_id,
                    paragraphs=paragraphs,
                    propositions=[]
                )

            # Request LLM
            prompts = []
            paragraph_list = []
            unit_key_list = []

            # Pack prompts
            for key, tracker in tracker_dict.items():
                for paragraph in tracker.paragraphs:
                    new_prompt = self.task_instruction + "\n\n" + f"Input: {paragraph}" + "\nOutput:"
                    prompts.append(new_prompt)
                    unit_key_list.append(key)
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

            assert len(unit_key_list) == len(responses) == len(paragraph_list)

            # Process responses
            for unit_key, response, paragraph in zip(unit_key_list, responses, paragraph_list):
                if response:
                    try:
                        if isinstance(response, str):
                            propositions = ast.literal_eval(response)['items']
                        else:
                            propositions = response
                        tracker_dict[unit_key].propositions.extend(propositions)
                    except Exception as e:
                        # Fallback to sentence segmentation
                        sentences = self._segment_sentence(paragraph)
                        tracker_dict[unit_key].propositions.extend(sentences)
                else:
                    sentences = self._segment_sentence(paragraph)
                    tracker_dict[unit_key].propositions.extend(sentences)

            # Create chunks
            chunks: List[Chunk] = []
            for key, tracker in tracker_dict.items():
                doc_id = tracker.doc_id
                parent_chunk_id = tracker.parent_chunk_id

                if parent_chunk_id is None:
                    # From Document: generate sequential chunk IDs
                    chunk_counter = count()
                    for proposition in tracker.propositions:
                        chunk_id = f"{doc_id}-Chunk-{next(chunk_counter)}"
                        chunk = Chunk(
                            doc_id=doc_id,
                            chunk_id=chunk_id,
                            text=proposition
                        )
                        chunks.append(chunk)
                else:
                    # From Chunk: use parent_chunk_id directly for all propositions
                    for proposition in tracker.propositions:
                        chunk = Chunk(
                            doc_id=doc_id,
                            chunk_id=parent_chunk_id,  # Use parent's chunk_id directly
                            text=proposition
                        )
                        chunks.append(chunk)

            return chunks

        except Exception as e:
            import traceback
            print(f"Error processing batch: {str(e)}\n{traceback.format_exc()}")
            return []

    def chunk(self, raw_docs: Union[List[Document], List[Chunk]]):
        """
        Main chunking method using ThreadPoolExecutor for parallel processing.

        Args:
            raw_docs: Either List[Document] or List[Chunk]

        Returns:
            List[Chunk]: Proposition chunks
        """
        chunks: List[Chunk] = []
        failed_batches = []

        # Convert input to unified ProcessingUnit format
        if not raw_docs:
            return chunks

        # Detect input type and convert
        if isinstance(raw_docs[0], Document):
            processing_units = [ProcessingUnit.from_document(doc) for doc in raw_docs]
            self.raw_input_type = "Document"
            print(f"[PropositionChunker] Processing {len(processing_units)} Documents")
        elif isinstance(raw_docs[0], Chunk):
            processing_units = [ProcessingUnit.from_chunk(chunk) for chunk in raw_docs]
            self.raw_input_type = "Chunk"
            print(f"[PropositionChunker] Processing {len(processing_units)} Chunks")
        else:
            raise TypeError(f"Input must be List[Document] or List[Chunk], got {type(raw_docs[0])}")


        if self.resume and self.chunk_sink_path:
            if os.path.exists(self.chunk_sink_path):
                self.processed_units = self._load_processed_units(self.chunk_sink_path)
                print(f"[PropositionChunker] Loaded {len(self.processed_units)} completed units for resume.")
            else:
                print(
                    f"[PropositionChunker] Resume requested but no existing chunk file at {self.chunk_sink_path}. Starting fresh.")
                self.resume = False

        if self._sample is not None:
            processing_units = processing_units[:self._sample]

        # Resume functionality: filter out already-processed units
        if self.resume and self.processed_units:
            original_count = len(processing_units)
            processing_units = [
                unit for unit in processing_units
                if (unit.doc_id, unit.parent_chunk_id) not in self.processed_units
            ]
            skipped = original_count - len(processing_units)
            if skipped > 0:
                print(f"[PropositionChunker] Skipping {skipped} units already processed.")
            if not processing_units:
                print("[PropositionChunker] No remaining units to process. Resume finished.")
                return chunks

        # Separate empty and non-empty units
        empty_units = [unit for unit in processing_units if not unit.text or not unit.text.strip()]
        non_empty_units = [unit for unit in processing_units if unit.text and unit.text.strip()]

        if empty_units:
            print(f"[PropositionChunker] Found {len(empty_units)} units with empty text. Creating empty chunks.")

            # Create empty chunks for empty units
            empty_chunks = []
            for unit in empty_units:
                if unit.parent_chunk_id is None:
                    # From Document: use doc_id-Chunk-0 format
                    chunk_id = f"{unit.doc_id}-Chunk-0"
                else:
                    # From Chunk: use parent_chunk_id directly
                    chunk_id = unit.parent_chunk_id

                chunk = Chunk(
                    doc_id=unit.doc_id,
                    chunk_id=chunk_id,
                    text=""
                )
                empty_chunks.append(chunk)

            # Save empty chunks immediately
            if self._sink and empty_chunks:
                with self._write_lock:
                    self._sink.write_batch(empty_chunks)
                    if self.resume:
                        self.processed_units.update(
                            (chunk.doc_id, unit.parent_chunk_id)
                            for chunk, unit in zip(empty_chunks, empty_units)
                        )
                    print(f"Saved {len(empty_chunks)} empty chunks")

            chunks.extend(empty_chunks)

        # Process only non-empty units
        if not non_empty_units:
            print("[PropositionChunker] No non-empty units to process.")
            return chunks

        # Split units into batches
        batches = [non_empty_units[i:i + self.batch_size]
                   for i in range(0, len(non_empty_units), self.batch_size)]

        batches_to_process = list(enumerate(batches))

        print(
            f"Processing {len(non_empty_units)} units in {len(batches)} batches using {self.num_parallel_batches} parallel threads")

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
                                    # Update processed units for resume functionality
                                    if self.resume:
                                        # Extract unique (doc_id, parent_chunk_id) from the batch
                                        # We need to identify the original processing units from the output
                                        processed_in_batch = set()
                                        if self.raw_input_type == "Document":
                                            processed_in_batch = {(c.doc_id, None) for c in batch_chunks}
                                        else:
                                            processed_in_batch = {(c.doc_id, c.chunk_id) for c in batch_chunks}


                                        self.processed_units.update(processed_in_batch)

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
    from src.types import Document, Chunk

    # Test with Documents
    test_docs = [
        Document(
            doc_id=f"doc_{i}",
            text="This is a test sentence. It has multiple parts. Each part should be separated."
        )
        for i in range(5)
    ]

    docs2 = [
        Document(
            doc_id=f"doc_{i}",
            text="Please call the model, then you will get the LLM result. This is another sentence."
        )
        for i in range(5, 10)
    ]

    test_docs = test_docs + docs2

    # Test with Chunks (simulating chunks from the same document)
    test_chunks = [
        Chunk(
            doc_id="doc_100",
            chunk_id=f"doc_100-Chunk-{i}",
            text=f"This is chunk {i} content. It should be further decomposed into propositions. Each proposition keeps the same chunk_id as the parent."
        )
        for i in range(3)
    ]

    # Add more chunks from another document
    test_chunks.extend([
        Chunk(
            doc_id="doc_101",
            chunk_id=f"doc_101-Chunk-{i}",
            text=f"Document 101 chunk {i}. This will also be decomposed. All propositions will share this chunk_id."
        )
        for i in range(2)
    ])

    # Add empty chunks to test edge cases
    test_chunks.append(
        Chunk(
            doc_id="doc_102",
            chunk_id="doc_102-Chunk-0",
            text=""
        )
    )

    # Create chunker with resume enabled
    chunker = PropositionChunker(
        gen_backbone="gemini",
        batch_size=100,
        llm_workers=8,
        generative_model_name="gemini-2.5-flash-lite",
        chunk_sink_path="llm_multi_processing.jsonl",
        resume=True  # Enable resume functionality
    )

    # Process Documents
    print("\n" + "=" * 80)
    print("=== Processing Documents ===")
    print("=" * 80)
    doc_chunks = chunker.chunk(test_docs)
    print(f"\nCreated {len(doc_chunks)} proposition chunks from {len(test_docs)} documents")
    print(f"\nSample output (first 5):")
    for i, chunk in enumerate(doc_chunks[:5], 1):
        print(f"  {i}. doc_id: {chunk.doc_id}, chunk_id: {chunk.chunk_id}")
        print(f"     text: {chunk.text[:60]}...")

    # Process Chunks
    # print("\n" + "=" * 80)
    # print("=== Processing Chunks ===")
    # print("=" * 80)
    # prop_chunks = chunker.chunk(test_chunks)
    # print(f"\nCreated {len(prop_chunks)} proposition chunks from {len(test_chunks)} input chunks")
    # print(f"\nSample output (first 8):")
    # for i, chunk in enumerate(prop_chunks[:8], 1):
    #     print(f"  {i}. doc_id: {chunk.doc_id}, chunk_id: {chunk.chunk_id}")
    #     print(f"     text: {chunk.text[:60]}...")
    #
    # # Demonstrate that propositions from the same parent chunk share the same chunk_id
    # print("\n" + "=" * 80)
    # print("=== Grouping by chunk_id (showing propositions from same parent) ===")
    # print("=" * 80)
    # from collections import defaultdict
    #
    # grouped = defaultdict(list)
    # for chunk in prop_chunks:
    #     grouped[chunk.chunk_id].append(chunk.text)
    #
    # for chunk_id, texts in list(grouped.items())[:3]:
    #     print(f"\nParent chunk_id: {chunk_id}")
    #     print(f"  Number of propositions: {len(texts)}")
    #     for i, text in enumerate(texts, 1):
    #         print(f"  {i}. {text[:70]}...")