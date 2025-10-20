import gzip
import json
import os
import re
from typing import List, Dict, Union, Set
from itertools import count
from dataclasses import dataclass, field

from tqdm import tqdm
from transformers import AutoTokenizer

from src.chunkers.base_chunker import BaseChunker
from src.types import Document, Chunk
from src.registry import CHUNKER_REG, GENERATOR_REG
from src.io.sink import JsonlSink
from src.models.generator.base_generator import BaseGenerator

paragraph_system_prompt = """You will receive as input an english document with paragraphs identified by 'ID XXXX: <text>'.

Task: Find the first paragraph (not the first one) where the content clearly changes compared to the previous paragraphs.

Output: Return the ID of the paragraph with the content shift as in the exemplified format: 'Answer: ID XXXX'.

Additional Considerations: Avoid very long groups of paragraphs. Aim for a good balance between identifying content shifts and keeping groups manageable."""

sentence_system_prompt = """You will receive as input an english document with sentences identified by 'ID XXXX: <text>'.

Task: Find the first sentence (not the first one) where the content clearly changes compared to the previous sentences.

Output: Return the ID of the sentence with the content shift as in the exemplified format: 'Answer: ID XXXX'.

Additional Considerations: Avoid very long groups of sentences. Aim for a good balance between identifying content shifts and keeping groups manageable."""


@dataclass
class DocumentSplitTracker:
    doc_id: str
    paragraphs: List[str]
    prefixed_paragraphs: List[str]
    splitter_list: List[int] = field(default_factory=lambda: [0])
    processed_paragraphs: int = 0


@CHUNKER_REG.register("LumberChunker")
class LumberChunker(BaseChunker):

    def __init__(self,
                 gen_backbone: str,
                 granularity: str = Union["sentence", "paragraph"],
                 batch_size: int = 20000,
                 chunk_sink_path: str | None = None,
                 **kwargs
                 ):

        # gen model
        gen_cls = GENERATOR_REG.get(gen_backbone)
        generative_model_name = kwargs.get("generative_model_name")

        if generative_model_name is None:
            raise KeyError("Generative model name not found")

        gen_backbone_kwargs = {"model": generative_model_name}
        self.generation_model: BaseGenerator = gen_cls(**gen_backbone_kwargs or {})

        # chunking hyperparameters
        self.max_tokens = kwargs.get('max_tokens') or 550
        self.buffer_size = 5

        self.granularity = granularity
        self.batch_size = batch_size or 20000

        tokenizer_name = kwargs.get("tokenizer_name") or "jinaai/jina-embeddings-v2-base-en"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        # self.tokenizer = tiktoken.encoding_for_model("gpt-4.1")

        if self.granularity == 'sentence':
            self.segment_function = self._segment_sentence
            self.system_instruction = sentence_system_prompt

        elif self.granularity == 'paragraph':
            self.segment_function = self._segment_paragraph
            self.system_instruction = paragraph_system_prompt

        else:
            raise ValueError(f'granularity must be "sentence" or "paragraph"')

        self._sink = JsonlSink(chunk_sink_path) if chunk_sink_path else None
        self._sample = kwargs.get("sample")

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

        self.use_batch_api = _safe_bool(kwargs.get("use_batch_api"), False)
        self.show_paragraph_progress = _safe_bool(kwargs.get("show_paragraph_progress"), True)

        self.resume = _safe_bool(kwargs.get("resume"), False)
        self.processed_doc_ids: Set[str] = set()

        if self.resume and chunk_sink_path:
            if os.path.exists(chunk_sink_path):
                self.processed_doc_ids = self._load_processed_doc_ids(chunk_sink_path)
                print(f"[LumberChunker] Loaded {len(self.processed_doc_ids)} completed documents for resume.")
            else:
                print(f"[LumberChunker] Resume requested but no existing chunk file at {chunk_sink_path}. Starting fresh.")
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
        add prefix for each paragraph/sentence
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
                               tracker_dict: Dict[str, DocumentSplitTracker],
                               still_active_id_list: List,
                               paragraph_pbar=None):

        prompts = []
        doc_order = []

        for doc_id in still_active_id_list:
            tracker = tracker_dict[doc_id]

            splitter = tracker.splitter_list[-1]
            seg_list = self._segment_text_list(tracker.prefixed_paragraphs[splitter:], max_tokens=self.max_tokens)

            prompt = f'\nDocument:\n' + '\n'.join(seg_list)

            prompts.append(prompt)
            doc_order.append(doc_id)

        try:

            doc_responses = []

            if not prompts:
                return

            if self.use_batch_api:
                in_batch = False if len(prompts) <= 10 else True

                llm_output = self.generation_model.generate(
                    prompts=prompts,
                    system_instruction=self.system_instruction,
                    temperature=0,
                    in_batch=in_batch
                )
                responses = llm_output["responses"]

                print(f"batch_size from {self.batch_size} to {len(prompts)}, in_batch = {in_batch}")
                doc_responses.extend(zip(doc_order, responses))

            else:
                batch_size = self.llm_prompt_batch_size or len(prompts)

                for start_idx in range(0, len(prompts), batch_size):
                    end_idx = start_idx + batch_size
                    batch_prompts = prompts[start_idx:end_idx]
                    batch_doc_ids = doc_order[start_idx:end_idx]

                    llm_output = self.generation_model.generate(
                        prompts=batch_prompts,
                        system_instruction=self.system_instruction,
                        temperature=0,
                        in_batch=False,
                        max_workers=self.llm_max_workers
                    )
                    responses = llm_output["responses"]

                    if len(responses) != len(batch_doc_ids):
                        print("Warning: response size mismatch, padding with None")
                        responses = (responses or []) + [None] * (len(batch_doc_ids) - len(responses or []))
                        responses = responses[:len(batch_doc_ids)]

                    doc_responses.extend(zip(batch_doc_ids, responses))

                print(f"Processed {len(prompts)} prompts with max_workers={self.llm_max_workers}"
                      f"{'' if self.llm_prompt_batch_size is None else f', prompt_batch_size={self.llm_prompt_batch_size}'}")

        except Exception as e:
            print(f"Error in generating llm: {e}")
            doc_responses = [(doc_id, None) for doc_id in doc_order]

        # clip the result

        for doc_id, response in doc_responses:

            if response:
                match = re.search(r"Answer: ID (\d+)", response)
                if match:

                    next_boundary = int(match.group(1))
                    tracker = tracker_dict[doc_id]
                    tracker.splitter_list.append(next_boundary)

                    if paragraph_pbar:
                        total_paragraphs = len(tracker.paragraphs)
                        bounded_boundary = min(next_boundary, total_paragraphs)
                        progress_delta = max(bounded_boundary - tracker.processed_paragraphs, 0)
                        if progress_delta:
                            tracker.processed_paragraphs += progress_delta
                            paragraph_pbar.set_description(f"{doc_id} paragraphs")
                            paragraph_pbar.update(progress_delta)
                else:
                    print('repeat this one')
            else:
                print('repeat this one')

    def lumber_chunking_pipeline(self, batch_document: List[Document]) -> List[Chunk]:

        tracker_dict: Dict[str, DocumentSplitTracker] = {}

        for doc in batch_document:

            paragraphs = self.segment_function(doc.text)

            tracker_dict[doc.doc_id] = DocumentSplitTracker(
                doc_id=doc.doc_id,
                paragraphs=paragraphs,
                prefixed_paragraphs=self.add_prefix(paragraphs),
                splitter_list=[0]
            )

        active_doc_ids = list(tracker_dict.keys())

        paragraph_pbar = None
        if self.show_paragraph_progress:
            total_paragraphs = sum(len(tracker.paragraphs) for tracker in tracker_dict.values())
            if total_paragraphs > 0:
                paragraph_pbar = tqdm(total=total_paragraphs,
                                      desc="Paragraphs processed",
                                      leave=False)

        try:
            # request the llm recursively in a batch
            while True:

                # find still active docs
                still_active_id_list = []

                for doc_id in active_doc_ids:
                    chunking_context = tracker_dict[doc_id]
                    if chunking_context.splitter_list[-1] + self.buffer_size <= len(chunking_context.paragraphs):
                        still_active_id_list.append(doc_id)
                if len(still_active_id_list) <= 0:
                    break

                self.request_generative_llm(tracker_dict, still_active_id_list, paragraph_pbar=paragraph_pbar)
            # chunking

            chunks: List[Chunk] = []

            for doc_id, tracker in tracker_dict.items():

                splitter_list = tracker.splitter_list

                splitter_list.append(len(tracker.paragraphs))

                if paragraph_pbar and tracker.processed_paragraphs < len(tracker.paragraphs):
                    remaining = len(tracker.paragraphs) - tracker.processed_paragraphs
                    if remaining > 0:
                        paragraph_pbar.set_description(f"{doc_id} paragraphs")
                        paragraph_pbar.update(remaining)
                        tracker.processed_paragraphs += remaining

                chunk_counter = count()

                for i in range(len(splitter_list)):
                    start = 0 if i == 0 else splitter_list[i - 1]
                    end = splitter_list[i]

                    if tracker.paragraphs[start:end]:

                        chunk_text = "\n".join(tracker.paragraphs[start:end])
                        chunk = Chunk(
                            doc_id=doc_id,
                            chunk_id=f"{doc_id}-Chunk-{next(chunk_counter)}",
                            text=chunk_text
                        )

                        chunks.append(chunk)

            return chunks
        finally:
            if paragraph_pbar:
                paragraph_pbar.close()

    def chunk(self, raw_docs: List[Document]):

        print("run LumberChunker")

        chunks = []

        if self._sample is not None:
            raw_docs = raw_docs[:self._sample]

        if self.resume and self.processed_doc_ids:
            original_count = len(raw_docs)
            raw_docs = [doc for doc in raw_docs if doc.doc_id not in self.processed_doc_ids]
            skipped = original_count - len(raw_docs)
            if skipped > 0:
                print(f"[LumberChunker] Skipping {skipped} documents already processed.")
            if not raw_docs:
                print("[LumberChunker] No remaining documents to process. Resume finished.")
                return chunks

        for i in tqdm(range(0, len(raw_docs), self.batch_size)):

            batch_documents = raw_docs[i:i + self.batch_size]

            batch_chunks = self.lumber_chunking_pipeline(batch_documents)

            if self._sink:
                self._sink.write_batch(batch_chunks)

            chunks.extend(batch_chunks)
            if self.resume:
                self.processed_doc_ids.update(chunk.doc_id for chunk in batch_chunks)

        return chunks
