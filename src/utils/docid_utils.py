import re
from typing import List
from collections import defaultdict


def get_book_range(doc_id_list: List[str]):
    """
    :param doc_id_list:  List['Book-id-Paragraph-id']
    :return:
    """

    book_range = {}

    for idx, doc_id in enumerate(doc_id_list):
        pattern = re.compile(r'-Paragraph-|-Query-|-Chunk-')
        book_id = pattern.split(doc_id)[0]

        if book_id not in book_range:
            book_range[book_id] = {'start': idx, 'end': idx+1}

        else:
            book_range[book_id]['end'] = idx+1

    return book_range



if __name__ == '__main__':
    doc_ids = [
        'Book-1-Paragraph-1',
        'Book-1-Paragraph-2',
        'Book-1-Paragraph-3',
        'Book-2-Paragraph-1',
        'Book-2-Paragraph-2',
        'Book-3-Paragraph-1',
        'Book-3-Paragraph-2',
        'Book-3-Paragraph-3',
        'Book-3-Query-4'
    ]

    print(get_book_range(doc_ids))