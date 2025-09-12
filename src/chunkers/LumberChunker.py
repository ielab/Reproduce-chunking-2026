import re
from typing import List
from itertools import count

from src.chunkers.base_chunker import BaseChunker
from src.types import Document, Chunk
from src.registry import CHUNKER_REG
from src.utils.docid_utils import get_book_range


@CHUNKER_REG.register("LumberChunker")
class LumberChunker(BaseChunker):

    def __init__(self):

        self.system_prompt = """You will receive as input an english document with paragraphs identified by 'ID XXXX: <text>'.

Task: Find the first paragraph (not the first one) where the content clearly changes compared to the previous paragraphs.

Output: Return the ID of the paragraph with the content shift as in the exemplified format: 'Answer: ID XXXX'.

Additional Considerations: Avoid very long groups of paragraphs. Aim for a good balance between identifying content shifts and keeping groups manageable."""

        self.max_words = 550

        self.llm = ''


    @staticmethod
    def _check_doc_id_format(s: str) -> bool:
        pattern = r"^Book-(\d+)-Paragraph-(\d+)$"
        return bool(re.match(pattern, s))

    @staticmethod
    def count_word(input_string):
        words = input_string.split()
        return round(1.2 * len(words))

    def get_chunk_document(self, paragraph_list: List[Document]):
        """
        merge chunks into document backward until count words are larger than max_words.
        :return:
            - final_document: merged document
            - next_chunk_index: next unprocessed chunk index
            - word_count: the count of words
        """
        pass

        # word_count = 0
        # i = 0
        # total_chunks = len(id_chunks)
        #
        # # merge chunks
        # while word_count < self.max_words and (start_index + i) < total_chunks - 1:
        #     i += 1
        #     end_index = start_index + i
        #     # chunk_slice = id_chunks.loc[start_index: end_index-1, 'Chunk']
        #     chunk_slice = id_chunks['Chunk'].iloc[start_index:end_index]
        #     final_document = "\n".join(chunk_slice)
        #     word_count = self.count_word(final_document)
        #
        # # if only one paragraph, use it; otherwise roll back 1
        # if i == 1:
        #     next_chunk_index = start_index + i
        # else:
        #     next_chunk_index = start_index + i - 1
        #
        # # chunk_slice = id_chunks.loc[start_index: next_chunk_index - 1, 'Chunk']
        # chunk_slice = id_chunks['Chunk'].iloc[start_index:next_chunk_index]
        # final_document = "\n".join(chunk_slice)
        # word_count = self.count_word(final_document)
        #
        # return final_document, next_chunk_index, word_count


    def segment_method(self, paragraph_list: List[Document]):
        pass

    def chunk(self, raw_docs: List[Document]):
        pass

        # chunk_counter = count()
        #
        # book_range = get_book_range([doc.doc_id for doc in raw_docs])
        #
        # if raw_docs and self._check_doc_id_format(raw_docs[0].doc_id):
        #
        #     for book_id, position_dict in book_range.items():
        #         start_idx = position_dict["start"]
        #         end_idx = position_dict["end"]
        #
        #
        #
        #
        # new_id_list = []
        # chunk_number = 0
        # word_count_aux = []
        # while chunk_number < len(id_chunks) - 5:
        #
        #     final_document, next_chunk_index, word_count = get_chunk_document(id_chunks, chunk_number)
        #
        #     word_count_aux.append(word_count)
        #     chunk_number = next_chunk_index
        #
        #     prompt = system_prompt + f"\nDocument:\n{final_document}"
        #
        #     # request LLM, get response
        #     gpt_output = llm_generation(prompt)
        #     # print(gpt_output)
        #     if gpt_output is not None:
        #         pattern = r"Answer: ID \d+"
        #         match = re.search(pattern, gpt_output)
        #     else:
        #         match = None
        #
        #     if match is None:
        #         print(f"repeat this one: {gpt_output}")
        #     else:
        #         gpt_output1 = match.group(0)
        #         logger.debug(f"AI Answer: {gpt_output1}")
        #         pattern = r'\d+'
        #         match = re.search(pattern, gpt_output1)
        #         chunk_number = int(match.group())
        #         new_id_list.append(chunk_number)
        #         if new_id_list[-1] == chunk_number:
        #             chunk_number = chunk_number + 1
        #         # break
        #
        # # fix bug, when new_id_list == 0
        # if len(new_id_list) > 0 and new_id_list[0] == 0:
        #     new_id_list = new_id_list[1:]
        #
        # # add last chunk to the list
        # new_id_list.append(len(id_chunks))
        #
        # # Remove IDs as they no longer make sense here.
        # id_chunks['Chunk'] = id_chunks['Chunk'].str.replace(r'^ID \d+:\s*', '', regex=True)
        #
        # # Create final dataframe from chunks
        # new_final_chunks = []
        # chapter_chunk = []
        #
        # for i in range(len(new_id_list)):
        #
        #     # calculate the start and end indices of each chunk
        #     start_idx = 0 if i == 0 else new_id_list[i - 1]
        #     end_idx = new_id_list[i]
        #     new_final_chunks.append('\n'.join(id_chunks.iloc[start_idx: end_idx, 0]))
        #
        #     # update chapters, sometimes text from different chapters
        #     # print(f"start_idx: {start_idx}, end_idx: {end_idx}")
        #     if paragraph_chunks['Chapter'][start_idx] != paragraph_chunks['Chapter'][end_idx - 1]:
        #         chapter_chunk.append(
        #             f"{paragraph_chunks['Chapter'][start_idx]} and {paragraph_chunks['Chapter'][end_idx - 1]}")
        #     else:
        #         chapter_chunk.append(paragraph_chunks['Chapter'][start_idx])