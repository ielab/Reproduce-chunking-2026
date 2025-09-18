import re
from typing import List
from itertools import count
from copy import deepcopy

import tiktoken

from src.chunkers.base_chunker import BaseChunker
from src.types import Document, Chunk
from src.registry import CHUNKER_REG, GENERATOR_REG
from src.io.sink import JsonlSink



paragraph_system_prompt = """You will receive as input an english document with paragraphs identified by 'ID XXXX: <text>'.

Task: Find the first paragraph (not the first one) where the content clearly changes compared to the previous paragraphs.

Output: Return the ID of the paragraph with the content shift as in the exemplified format: 'Answer: ID XXXX'.

Additional Considerations: Avoid very long groups of paragraphs. Aim for a good balance between identifying content shifts and keeping groups manageable."""


sentence_system_prompt = """You will receive as input an english document with sentences identified by 'ID XXXX: <text>'.

Task: Find the first sentence (not the first one) where the content clearly changes compared to the previous sentences.

Output: Return the ID of the sentence with the content shift as in the exemplified format: 'Answer: ID XXXX'.

Additional Considerations: Avoid very long groups of sentences. Aim for a good balance between identifying content shifts and keeping groups manageable."""



# @CHUNKER_REG.register("LumberChunker")
class LumberChunker(BaseChunker):

    def __init__(self,
                 granularity: str = 'sentence',
                 **kwargs):

        self.max_tokens = kwargs.get('max_tokens', 8192)

        self.granularity = granularity

        self.tokenizer = tiktoken.encoding_for_model("text-embedding-ada-002")


        if self.granularity == 'sentence':
            self.segment_function = self._segment_sentence
            self.system_prompt = sentence_system_prompt

        elif self.granularity == 'paragraph':
            self.segment_function = self._segment_paragraph
            self.system_prompt = paragraph_system_prompt

        else:
            raise ValueError(f'granularity must be "sentence" or "paragraph"')

        chunk_sink_path = kwargs.get('chunk_sink_path')

        self._sink = JsonlSink(chunk_sink_path) if chunk_sink_path else None

        generator_name = kwargs.get('generator_name')
        generator_cls = GENERATOR_REG.get('generator_name')
        self.generator = generator_cls({})



    @staticmethod
    def _segment_sentence(text: str) -> List[str]:

        pattern = re.compile(r"(?<!\w\.\w.)(?<![A-Z]\.)(?<![A-Z][a-z]\.)(?<! [a-z]\.)(?<![A-Z][a-z][a-z]\.)("
                             r"?<=\.|\?|!)\"*\s*\s*(?:\W*)([A-Z])")

        find_list = pattern.split(text)

        sentences = find_list[:1]
        for ids in range(1, len(find_list), 2):
            sentences.append(find_list[ids] + find_list[ids+1])

        sentences = [s for s in sentences if s.strip()]

        return sentences


    @staticmethod
    def _segment_paragraph(text: str) -> List[str]:

        pattern = re.compile(r'\n')

        return [x.strip() for x in pattern.split(text) if x.strip()]


    @staticmethod
    def _add_prefix_ids(text_list: List[str]) -> List[str]:

        copy_text_list = deepcopy(text_list)

        new_text_list = []

        for idx, text in enumerate(copy_text_list):

            text = f"ID {idx}: {text}"
            new_text_list.append(text)

        return new_text_list


    def get_seg_list(self, text_list: List[str], start_idx:int):

        token_count = 0

        seg_list = []

        for idx in range(start_idx, len(text_list)):

            text = text_list[idx]
            tokens = self.tokenizer.encode(text)
            if token_count == 0 and len(tokens) > self.max_tokens:
                seg_list.append(text)
                break

            if len(tokens) + token_count > self.max_tokens:
                break

            else:
                token_count += len(tokens)
                seg_list.append(text)
        # print(token_count)
        return seg_list


    def llm_generation(self, prompt):

        # import random
        #
        # text = prompt.split('Document:')[-1].strip()
        #
        # text_list = text.split('\n')
        #
        # text_list = [t.lstrip('ID ') for t in text_list]
        #
        # id_list = [int(t.split(':')[0]) for t in text_list]
        #
        # idx = random.choice(id_list)
        # print(id_list)


        response = self.generator.generate(prompt)


        return response


    def chunk(self, raw_docs: List[Document]):

        # input: Document:
        # each document should be a book in GutenQA or a passage in Beir

        # granularity
        # GutenQA, the smallest granularity is paragraph level
        # Beir, the smallest granularity is sentence_level

        # hyperparameter:
        # - max_tokens

        chunks = []

        for document in raw_docs:

            text_list: List[str] = self.segment_function(document.text)
            add_prefix_text_list = self._add_prefix_ids(text_list)

            start_idx = 0
            chunk_idx_list = []
            while start_idx < len(add_prefix_text_list) - 5:

                seg_list = self.get_seg_list(add_prefix_text_list, start_idx)
                if len(seg_list) == 1:
                    chunk_idx_list.append(start_idx+1)
                    start_idx += 1
                    continue

                prompt = self.system_prompt + f'\nDocument:\n' + '\n'.join(seg_list)

                gpt_output = self.llm_generation(prompt)

                if gpt_output is not None:
                    pattern = r"Answer: ID \d+"
                    match = re.search(pattern, gpt_output)
                else:
                    match = None

                if match is None:
                    print(f'repeat this one')
                else:
                    gpt_output1 = match.group(0)
                    pattern = r'\d+'
                    match = re.search(pattern, gpt_output1)
                    chunk_splitter = int(match.group())
                    chunk_idx_list.append(chunk_splitter)
                    start_idx = chunk_splitter + 1
                    print('splitter', chunk_splitter)



            # # fix bug, when new_id_list == 0
            # if len(new_id_list) > 0 and new_id_list[0] == 0:
            #     new_id_list = new_id_list[1:]

            # # add last chunk to the list
            chunk_idx_list.append(len(text_list))

            chunk_counter = count()

            for i in range(len(chunk_idx_list)):

                start_idx = 0 if i == 0 else chunk_idx_list[i-1]
                end_idx = chunk_idx_list[i]

                chunk_text = '\n'.join(text_list[start_idx:end_idx])

                chunk = Chunk(
                    doc_id=document.doc_id,
                    chunk_id=f'{document.doc_id}-Chunk-{next(chunk_counter)}',
                    text=chunk_text
                )

                chunks.append(chunk)

                if self._sink is not None:
                    self._sink.write_batch([chunk])

        return chunks


if __name__ == '__main__':

    # case
    # 1. when tokens > max_tokens
    # 2. when tokens < max_tokens
    #   - tokens + next_tokens > max_tokens
    #   - tokens + next_tokens + ... > max_tokens
    #   - tokens + ... + last_tokens < max_tokens

    paragraph_list = ['I '* 3, 'And '* 10, 'You ' * 5, 'Repeat '* 2, 'Kim '*3, 'Sent '* 3]


    tokenizer = tiktoken.encoding_for_model("text-embedding-ada-002")

    tokens_list =[len(tokenizer.encode(x)) for x in paragraph_list]

    print(tokens_list)

    text = '\n'.join(['I ' * 3, 'And ' * 10, 'You ' * 5, 'Repeat ' * 2, 'Kim ' * 3, 'Sent ' * 3])
    documents = Document(doc_id=str(1), text=text)
    # print(documents)

    # # init LumberChunker
    #
    chunker = LumberChunker(
        granularity='paragraph',
        max_tokens=25,
    )
    #
    chunks = chunker.chunk([documents])

    # seg_list = chunker.get_seg_list(paragraph_list, 4)
    # print(seg_list)

