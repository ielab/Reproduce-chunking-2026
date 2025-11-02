import gzip
import json
import os
import re
from typing import List, Dict, Set, Literal
from itertools import count
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from tqdm import tqdm
from transformers import AutoTokenizer

from src.chunkers.base_chunker import BaseChunker
from src.types import Document, Chunk
from src.registry import CHUNKER_REG, GENERATOR_REG
from src.io.sink import JsonlSink
from src.models.generator.base_generator import BaseGenerator

paragraph_task_instruction = """You will receive as input an english document with paragraphs identified by 'ID XXXX: <text>'.

Task: Find the first paragraph (not the first one) where the content clearly changes compared to the previous paragraphs.

Output: Return the ID of the paragraph with the content shift as in the exemplified format: 'Answer: ID XXXX'.

Additional Considerations: Avoid very long groups of paragraphs. Aim for a good balance between identifying content shifts and keeping groups manageable."""

sentence_task_instruction = """You will receive as input an english document with sentences identified by 'ID XXXX: <text>'.

Task: Find the first sentence (not the first one) where the content clearly changes compared to the previous sentences.

Output: Return the ID of the sentence with the content shift as in the exemplified format: 'Answer: ID XXXX'.

Additional Considerations: Avoid very long groups of sentences. Aim for a good balance between identifying content shifts and keeping groups manageable."""


@dataclass
class DocumentSplitTracker:
    doc_id: str
    segments: List[str]
    prefixed_segments: List[str]
    splitter_list: List[int] = field(default_factory=lambda: [0])
    processed_segments: int = 0


@CHUNKER_REG.register("LumberChunker")
class LumberChunker(BaseChunker):
    """
    Optimized LumberChunker with multi-threading support.

    Uses ThreadPoolExecutor to process multiple document batches in parallel.
    Each thread handles a complete lumber_chunking_pipeline for its batch.
    """

    def __init__(self,
                 gen_backbone: str,
                 granularity: Literal["sentence", "paragraph"] = "sentence",
                 batch_size: int = 20000,
                 chunk_sink_path: str | None = None,
                 **kwargs
                 ):

        # Store model configuration for thread-based instantiation
        self.gen_backbone = gen_backbone
        self.generative_model_name = kwargs.get("generative_model_name")

        if self.generative_model_name is None:
            raise KeyError("Generative model name not found")

        # chunking hyperparameters
        self.max_tokens = kwargs.get('max_tokens') or 550
        self.buffer_size = 5

        self.granularity = granularity
        self.batch_size = batch_size or 20000

        tokenizer_name = kwargs.get("tokenizer_name") or "jinaai/jina-embeddings-v2-base-en"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

        if self.granularity == 'sentence':
            self.segment_function = self._segment_sentence
            self.task_instruction = sentence_task_instruction

        elif self.granularity == 'paragraph':
            self.segment_function = self._segment_paragraph
            self.task_instruction = paragraph_task_instruction

        else:
            raise ValueError(f'granularity must be "sentence" or "paragraph"')

        self._sink = JsonlSink(chunk_sink_path) if chunk_sink_path else None
        self._sample = kwargs.get("sample")

        # Thread-safe lock for sink writes and doc_id tracking
        self._write_lock = threading.Lock()

        def _safe_int(value, default):
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

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

        workers_candidate = (
                kwargs.get("llm_workers")
                or kwargs.get("num_workers")
                or kwargs.get("llm_max_workers")
        )
        self.llm_max_workers = max(_safe_int(workers_candidate, 1), 1)

        prompt_batch_candidate = (
                kwargs.get("llm_batch_size")
                or kwargs.get("prompt_batch_size")
        )
        prompt_batch_size = _safe_int(prompt_batch_candidate, None)
        if prompt_batch_size is None:
            self.llm_prompt_batch_size = None
        else:
            self.llm_prompt_batch_size = max(prompt_batch_size, 1)

        self.use_batch_api = _safe_bool(kwargs.get("use_batch_api"), True)
        self.show_paragraph_progress = _safe_bool(kwargs.get("show_paragraph_progress"), True)
        # self.batch_api_threshold = _safe_int(kwargs.get("batch_api_threshold"), 10)
        self.batch_api_threshold = 80

        # Multi-threading configuration
        self.num_parallel_batches = _safe_int(kwargs.get("num_parallel_batches"), 4)

        self.resume = _safe_bool(kwargs.get("resume"), False)
        self.processed_doc_ids: Set[str] = set()

        if self.resume and chunk_sink_path:
            if os.path.exists(chunk_sink_path):
                self.processed_doc_ids = self._load_processed_doc_ids(chunk_sink_path)
                print(f"[LumberChunker] Loaded {len(self.processed_doc_ids)} completed documents for resume.")
            else:
                print(
                    f"[LumberChunker] Resume requested but no existing chunk file at {chunk_sink_path}. Starting fresh.")
                self.resume = False

    def _load_processed_doc_ids(self, path: str) -> Set[str]:
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
            print(f"[LumberChunker] Failed to read existing chunks for resume: {exc}")

        return doc_ids

    @staticmethod
    def _segment_sentence(text: str) -> List[str]:

        pattern = re.compile(r"(?<!\w\.\w.)(?<![A-Z]\.)(?<![A-Z][a-z]\.)(?<! [a-z]\.)(?<![A-Z][a-z][a-z]\.)("
                             r"?<=\.|\?|!)\"*\s*\s*(?:\W*)([A-Z])")

        find_list = pattern.split(text)

        sentences = find_list[:1]
        for ids in range(1, len(find_list), 2):
            sentences.append(find_list[ids] + find_list[ids + 1])

        sentences = [s for s in sentences if s.strip()]

        return sentences

    @staticmethod
    def _segment_paragraph(text: str) -> List[str]:

        pattern = re.compile(r'\n')

        return [x.strip() for x in pattern.split(text) if x.strip()]

    def add_prefix(self, text_list: List[str]) -> List[str]:
        """
        add prefix for each segment (paragraph or sentence)
        """

        prefix_text_list = []

        for idx, text in enumerate(text_list):
            prefix_text_list.append(f"ID {idx}: {text}")

        return prefix_text_list

    def _segment_text_list(self, text_list: List[str], max_tokens: int = 550) -> List[str]:

        token_count = 0

        seg_list = []

        for text in text_list:

            tokens = self.tokenizer.encode(text)
            if token_count == 0 and len(tokens) > max_tokens:
                seg_list.append(text)
                break

            if len(tokens) + token_count > max_tokens:
                break

            else:
                token_count += len(tokens)
                seg_list.append(text)

        return seg_list

    def request_generative_llm(self,
                               generation_model: BaseGenerator,
                               tracker_dict: Dict[str, DocumentSplitTracker],
                               still_active_id_list: List,
                               segment_pbar=None):

        prompts = []
        doc_order = []
        seg_list_lengths = []  # Track segment list lengths for each document

        for doc_id in still_active_id_list:
            tracker = tracker_dict[doc_id]

            splitter = tracker.splitter_list[-1]
            seg_list = self._segment_text_list(tracker.prefixed_segments[splitter:], max_tokens=self.max_tokens)

            # If only one segment fits, advance by 1 (nothing to compare)
            if len(seg_list) == 1:
                next_boundary = splitter + 1
                if next_boundary <= len(tracker.segments):
                    tracker.splitter_list.append(next_boundary)
                    if segment_pbar:
                        tracker.processed_segments += 1
                        segment_pbar.update(1)
                continue

            prompt = self.task_instruction + '\nDocument:\n' + '\n'.join(seg_list)

            prompts.append(prompt)
            doc_order.append(doc_id)
            seg_list_lengths.append(len(seg_list))

        if not prompts:
            return

        try:
            # Determine if batch API should be used
            in_batch = len(prompts) > self.batch_api_threshold

            llm_output = generation_model.generate(
                prompts=prompts,
                temperature=0,
                top_k=1,
                display_name=f"Generate lumber chunk",
                in_batch=in_batch
            )
            responses = llm_output["responses"]

            # Validate response count matches prompt count
            if len(responses) != len(prompts):
                print(f"Warning: Expected {len(prompts)} responses but got {len(responses)}")
                # Pad with None if we got fewer responses
                responses = [None] * len(prompts)

            doc_responses = list(zip(doc_order, responses, seg_list_lengths))

        except Exception as e:
            print(f"Error in generating llm: {e}")
            doc_responses = [(doc_id, None, seg_len) for doc_id, seg_len in zip(doc_order, seg_list_lengths)]

        # Process the results
        for doc_id, response, seg_list_length in doc_responses:
            tracker = tracker_dict[doc_id]
            current_splitter = tracker.splitter_list[-1]

            # Try to extract valid boundary from response
            valid_boundary = None
            error_msg = None

            if not response:
                error_msg = "no answer"
            else:
                match = re.search(r"Answer: ID (\d+)", response)
                if not match:
                    error_msg = "no valid ID match in response"
                else:
                    next_boundary = int(match.group(1))
                    if 0 < next_boundary <= len(tracker.segments):
                        valid_boundary = next_boundary
                    else:
                        error_msg = f"invalid boundary {next_boundary} (max: {len(tracker.segments)})"

            # Apply valid boundary or fallback
            if valid_boundary is not None:
                tracker.splitter_list.append(valid_boundary)
                if segment_pbar:
                    bounded_boundary = min(valid_boundary, len(tracker.segments))
                    progress_delta = max(bounded_boundary - tracker.processed_segments, 0)
                    if progress_delta:
                        tracker.processed_segments += progress_delta
                        segment_pbar.update(progress_delta)
            else:
                print(f"doc_id: {doc_id} {error_msg}")
                fallback_boundary = current_splitter + seg_list_length
                if fallback_boundary <= len(tracker.segments):
                    tracker.splitter_list.append(fallback_boundary)
                    if segment_pbar:
                        progress_delta = seg_list_length
                        tracker.processed_segments += progress_delta
                        segment_pbar.update(progress_delta)
                else:
                    print(f"fallback_boundary: {fallback_boundary}, length_segments: {len(tracker.segments)}")

    def lumber_chunking_pipeline(self, batch_document: List[Document], batch_idx: int = 0) -> List[Chunk]:
        """
        Process a batch of documents. This method runs in a thread.
        Each thread creates its own generator instance to avoid sharing issues.
        """

        segment_pbar = None

        try:
            # Create generator instance for this thread
            gen_cls = GENERATOR_REG.get(self.gen_backbone)
            generation_model: BaseGenerator = gen_cls(model=self.generative_model_name)

            tracker_dict: Dict[str, DocumentSplitTracker] = {}

            for doc in batch_document:
                segments = self.segment_function(doc.text)

                # Skip documents with no segments
                if not segments:
                    print(f"Warning: Document {doc.doc_id} produced no segments, skipping")
                    continue

                tracker_dict[doc.doc_id] = DocumentSplitTracker(
                    doc_id=doc.doc_id,
                    segments=segments,
                    prefixed_segments=self.add_prefix(segments),
                    splitter_list=[0]
                )

            active_doc_ids = list(tracker_dict.keys())

            if self.show_paragraph_progress:
                total_segments = sum(len(tracker.segments) for tracker in tracker_dict.values())
                if total_segments > 0:
                    # Use modulo to prevent position overflow on terminal
                    safe_position = (batch_idx % 10) + 1
                    segment_pbar = tqdm(total=total_segments,
                                        desc=f"Batch {batch_idx} segments",
                                        leave=False,
                                        position=safe_position)

            # request the llm recursively in a batch
            while True:

                # find still active docs
                still_active_id_list = []

                for doc_id in active_doc_ids:
                    chunking_context = tracker_dict[doc_id]
                    if chunking_context.splitter_list[-1] + self.buffer_size <= len(chunking_context.segments):
                        still_active_id_list.append(doc_id)
                if len(still_active_id_list) <= 0:
                    break

                self.request_generative_llm(generation_model, tracker_dict, still_active_id_list,
                                            segment_pbar=segment_pbar)

            # chunking

            chunks: List[Chunk] = []

            for doc_id, tracker in tracker_dict.items():

                splitter_list = tracker.splitter_list

                splitter_list.append(len(tracker.segments))

                if segment_pbar and tracker.processed_segments < len(tracker.segments):
                    remaining = len(tracker.segments) - tracker.processed_segments
                    if remaining > 0:
                        segment_pbar.update(remaining)
                        tracker.processed_segments += remaining

                chunk_counter = count()

                for i in range(1, len(splitter_list)):
                    start = splitter_list[i - 1]
                    end = splitter_list[i]

                    if tracker.segments[start:end]:
                        chunk_text = "\n".join(tracker.segments[start:end])
                        chunk = Chunk(
                            doc_id=doc_id,
                            chunk_id=f"{doc_id}-Chunk-{next(chunk_counter)}",
                            text=chunk_text
                        )

                        chunks.append(chunk)

            return chunks

        except Exception as e:
            import traceback
            print(f"Error processing batch {batch_idx}: {str(e)}\n{traceback.format_exc()}")
            return []

        finally:
            # Clean up resources
            if segment_pbar:
                segment_pbar.close()

    def chunk(self, raw_docs: List[Document]):

        print("run LumberChunker with multi-threading")

        chunks = []
        failed_batches = []

        if self._sample is not None:
            raw_docs = raw_docs[:self._sample]

        # Sort by total number of characters in ascending order
        raw_docs = sorted(raw_docs, key=lambda doc: len(doc.text))

        if self.resume and self.processed_doc_ids:
            original_count = len(raw_docs)
            raw_docs = [doc for doc in raw_docs if doc.doc_id not in self.processed_doc_ids]
            skipped = original_count - len(raw_docs)
            if skipped > 0:
                print(f"[LumberChunker] Skipping {skipped} documents already processed.")
            if not raw_docs:
                print("[LumberChunker] No remaining documents to process. Resume finished.")
                return chunks

        # Split documents into batches
        batches = [raw_docs[i:i + self.batch_size]
                   for i in range(0, len(raw_docs), self.batch_size)]

        print(
            f"Processing {len(raw_docs)} documents in {len(batches)} batches using {self.num_parallel_batches} parallel threads")

        try:
            # Process batches with threads
            with ThreadPoolExecutor(max_workers=self.num_parallel_batches) as executor:
                # Submit all batches
                future_to_batch = {
                    executor.submit(self.lumber_chunking_pipeline, batch, idx): idx
                    for idx, batch in enumerate(batches)
                }

                # Collect results as they complete
                with tqdm(total=len(batches), desc="Processing batches", position=0) as pbar:
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
                                    print(f"\nBatch {batch_idx}: Saved {len(batch_chunks)} chunks")

                            chunks.extend(batch_chunks)
                            pbar.update(1)

                        except Exception as e:
                            print(f"\nError retrieving results for batch {batch_idx}: {e}")
                            failed_batches.append(batch_idx)
                            pbar.update(1)

            if failed_batches:
                print(f"\nWARNING: {len(failed_batches)} batches failed: {failed_batches}")

            return chunks

        finally:
            # Always close the sink
            if self._sink:
                self._sink.close()