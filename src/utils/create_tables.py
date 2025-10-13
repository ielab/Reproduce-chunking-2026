import os
import pandas as pd

# Define the base path for the results
SOURCE_PATH = "src/chunked_output"

# Define the datasets, models, and chunkers
DATASETS = [
    "GutenQA",
    "fiqa",
    "nfcorpus",
    "scifact",
    "arguana",
    "scidocs",
]

MODELS = [
    "jina-embeddings-v2-small-en",
    "jina-embeddings-v3",
    "nomic-embed-text-v1",
    "multilingual-e5-large-instruct",
]

CHUNKERS = [
    "ParagraphChunker",
    "SentenceChunker",
    "FixedSizeChunker",
    "SemanticChunker",
    "LumberChunker",
    "Proposition",
]

def parse_eval_file(dataset, chunker, encoder, model):
    """
    Parses the nDCG@10.eval file and returns the average score.
    """
    model_name_clean = model.split('/')[-1]
    chunk_embedding_run_id = f"{encoder}-{model_name_clean}"
    eval_path = os.path.join(SOURCE_PATH, dataset, 'results', chunker, chunk_embedding_run_id, 'nDCG@10.eval')

    if not os.path.exists(eval_path):
        return None

    with open(eval_path, 'r') as f:
        lines = f.readlines()
        if not lines:
            return None
        last_line = lines[-1]
        if last_line.startswith('average'):
            return float(last_line.split()[-1])
    return None

def generate_chunking_table():
    """
    Generates the LaTeX table for the evaluation of different chunking methods.
    """
    pass

def generate_late_vs_regular_table():
    """
    Generates the LaTeX table for the comparison of late-chunking vs. regular-chunking.
    """
    pass

if __name__ == "__main__":
    # This is where we will call the functions to generate the tables
    pass
