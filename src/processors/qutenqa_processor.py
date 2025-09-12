import os
import pandas as pd
from typing import List, Dict

from tqdm import tqdm

from src.processors.base_processor import BaseProcessor
from src.types import Document, Query
from src.registry import PROCESSOR_REG
from src.io.sink import JsonlSink


@PROCESSOR_REG.register('GutenQA')
class GutenQAProcessor(BaseProcessor):

    def __init__(self,
                 dataset_name: str,
                 data_folder: str,
                 # sample: str|None=None,
                 test: str|None=None):

        self.corpus_path = os.path.join(data_folder, dataset_name, 'GutenQA_paragraphs.parquet')
        self.query_path = os.path.join(data_folder, dataset_name, 'questions.parquet')
        # self.sample = sample
        self.test = test


    @staticmethod
    def _generate_id(book_id, paragraph_id):
        """
        For GutenQA dataset, Document.doc_id = f'Book-{Book ID}-Paragraph-{Chunk ID}',
        when do chunking, we will get another chunk_id, chunk_id != Chunk ID, Chunk ID is just the original ID in GutenQA.
        :return:
        """

        # return f'Book-{book_id}-Paragraph-{paragraph_id}'
        pass

    @staticmethod
    def _get_book_range(docs: List[Document]) -> Dict[str, Dict[str, int]]:
        doc_id_list = [d.doc_id for d in docs]
        from src.utils.docid_utils import get_book_range
        return get_book_range(doc_id_list)



    def load_corpus(self):


        df = pd.read_parquet(self.corpus_path)


        book_name_list = df['Book Name'].unique().tolist()

        docs = []

        new_df = df.groupby(['Book Name', 'Book ID'], as_index=False).agg(
            {'Chunk': lambda x: '\n'.join(x)}
        )


        for idx, row in new_df.iterrows():

            docs.append(Document(
                doc_id=f'Book-{row["Book ID"]}',
                text=row['Chunk']
            ))

        # for book_name in tqdm(book_name_list):
        #
        #     per_book_df = df[df['Book Name'] == book_name]
        #
        #     for idx, row in per_book_df.iterrows():
        #
        #         docs.append(Document(
        #             doc_id=f'Book-{row['Book ID']}-Paragraph-{row['Chunk ID']}',
        #             text=row['Chunk'],
        #         ))
        # print('Sample:', self.sample)
        # if self.sample is not None:
        #     return docs[:self.sample]

        # selecting one book for testing
        if self.test is not None:
            for book_id, position in self._get_book_range(docs).items():
                start_idx, end_idx = position['start'], position['end']
                docs = docs[start_idx:end_idx]
                break

        return docs


    def load_query(self, sink_path: str|None=None):


        df = pd.read_parquet(self.query_path)

        book_name_list = df['Book Name'].unique().tolist()

        queries = []

        for book_name in book_name_list:

            per_book_df = df[df['Book Name'] == book_name]

            for idx, row in per_book_df.iterrows():

                queries.append(Query(
                    query_id=f'Book-{row['Book ID']}-Query-{idx}',
                    text=row['Question'],
                    chunk_must_Contain=row['Chunk Must Contain']
                ))

        if sink_path is not None:
            _sink = JsonlSink(sink_path)
            _sink.write_batch(queries)

        return queries

    # @staticmethod
    # def save_obj(queries: List[Query], sink_path: str):
    #
    #     _sink = JsonlSink(sink_path)
    #     _sink.write_batch(queries)




if __name__ == "__main__":

    para_path = "/src/data/GutenQA"

    c_path = f'{para_path}/GutenQA_paragraphs.parquet'
    q_path = f'{para_path}/questions.parquet'

    processor = GutenQAProcessor(c_path, q_path)
    # processor.load_corpus()
    processor.load_query()