import os
from typing import List, Dict

import json
from tqdm import tqdm
import pandas as pd

from src.io.sink import JsonlSink
from src.types import Document, Query
from src.processors.base_processor import BaseProcessor
from src.registry import PROCESSOR_REG


@PROCESSOR_REG.register("beir")
class BEIRProcessor(BaseProcessor):

    def __init__(self,
                 dataset_name: str,
                 data_folder: str):

        self.corpus_file = os.path.join(data_folder, dataset_name, 'corpus.jsonl')
        self.query_file = os.path.join(data_folder, dataset_name, 'queries.jsonl')

        self.qrels_file = os.path.join(data_folder, dataset_name, 'qrels/test.tsv')


    def load_corpus(self):

        docs: List[Document] = []

        with open(self.corpus_file, encoding="utf8") as fIn:
            for line in tqdm(fIn):
                line = json.loads(line)

                text = line['title'] + ' ' + line['text']

                doc = Document(
                    doc_id=line['_id'],
                    text=text.strip()
                )

                docs.append(doc)

        return docs

    def load_query(self, sink_path: str|None=None) ->List[Query]:

        queries: List[Query] = []

        # create qrels
        query_id2qrels = {}
        qrels_df = pd.read_csv(self.qrels_file, sep='\t')

        for idx, row in qrels_df.iterrows():

            query_id = str(row['query-id'])
            corpus_id = str(row['corpus-id'])
            score = int(row['score'])

            if query_id not in query_id2qrels:
                query_id2qrels[query_id] = {corpus_id: score}

            else:
                query_id2qrels[query_id][corpus_id] = score


        # create Query
        with open(self.query_file, encoding="utf8") as fIn:

            for line in fIn:
                line = json.loads(line)

                qrels = query_id2qrels.get(line['_id'])

                if qrels is not None:

                    query = Query(
                        query_id=line['_id'],
                        text=line['text'],
                        qrels=qrels
                    )

                    queries.append(query)

        if sink_path is not None:

            _sink = JsonlSink(sink_path)
            _sink.write_batch(queries)


        return queries




