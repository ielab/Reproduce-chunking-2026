# Reproduce-chunking-2026
reproducibility paper of chunking method 2026 ecir 

# Paper Additional Result:

Below is the additional result for Table 2 of the paper show in github due to page constraint;

| Dataset | Method | Pre-C Orig (Jina-v3) | Pre-C Repro (Jina-v3) | Con-C Orig (Jina-v3) | Con-C Repro (Jina-v3) | Pre-C Orig (Nomic) | Pre-C Repro (Nomic) | Con-C Orig (Nomic) | Con-C Repro (Nomic) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SciFact | Fixed-size | 0.718 | 0.717 | 0.732 | 0.730 | 0.707 | 0.703 | 0.706 | 0.707 |
| | Sentence | 0.714 | 0.716 | 0.732 | 0.734 | 0.713 | 0.715 | 0.714 | 0.712 |
| | Semantic | 0.712 | 0.710 | 0.724 | 0.723 | 0.704 | 0.704 | 0.705 | 0.705 |
| NFCorpus | Fixed-size | 0.356 | 0.355 | 0.367 | 0.368 | 0.353 | 0.348 | 0.353 | 0.351 |
| | Sentence | 0.358 | 0.357 | 0.366 | 0.367 | 0.347 | 0.350 | 0.355 | 0.355 |
| | Semantic | 0.361 | 0.360 | 0.366 | 0.367 | 0.353 | 0.351 | 0.303 | 0.353 |
| FiQA | Fixed-size | 0.333 | 0.468 | 0.338 | 0.479 | 0.370 | 0.386 | 0.383 | 0.387 |
| | Sentence | 0.304 | 0.433 | 0.339 | 0.480 | 0.351 | 0.362 | 0.377 | 0.380 |
| | Semantic | 0.303 | 0.440 | 0.337 | 0.322 | 0.348 | 0.356 | 0.369 | 0.266 |
| TRECCOVID | Fixed-size | 0.730 | 0.739 | 0.772 | 0.766 | 0.729 | 0.758 | 0.750 | 0.750 |
| | Sentence | 0.724 | 0.714 | 0.765 | 0.769 | 0.742 | 0.747 | 0.768 | 0.779 |
| | Semantic | 0.747 | 0.747 | 0.762 | 0.699 | 0.743 | 0.743 | 0.761 | 0.730 |

Next, for investigating RQ4, the impact of chunk size; we have also included the coorlation with respect to other three models tested in the paper:




# 📚 Dataset

This project uses the following datasets for chunking and embedding task:

- **Narrative Dataset (GutenQA):** [GutenQA_Paragraphs](https://huggingface.co/datasets/LumberChunker/GutenQA_Paragraphs) 
- **BEIR Dataset:** [beir](https://github.com/beir-cellar/beir)
  - trec-covid
  - nfcorpus
  - fiqa
  - arguana
  - scidocs
  - scifact


### 🔽 Download Instructions

Download the datasets and place them in the folder `src/data/`.  
- For **GutenQA**, create a subfolder named `GutenQA`.  
- For **BEIR**, unzip the dataset into the same directory.  

# Installation

Install dependencies with:
```bash
pip install -r requirements.txt
```


# Project Guide

This repository provides three core modules for document processing and evaluation:  
- **Chunker**: Splits documents into manageable pieces.  
- **Encoder**: Transforms chunks into embeddings.  
- **Evaluator**: Benchmarks chunking and encoding strategies.  

You can run each module individually or together using the provided shell scripts.  

---

# ▶️ Quick Start

Run all modules from the command line using the provided scripts. Logs are automatically redirected.


#### Chunker

```bash
nohup ./run_chunker.sh > run_chunker.log 2>&1 < /dev/null &
```

#### Encoder

⚠️ **Important:**
Before running, update `QUERY_ID_BY_DATASET.` This ID is generated in the Chunker module, and each dataset corresponds to a unique query ID.

```bash
nohup ./run_encoder.sh > run_encoder.log 2>&1 < /dev/null &
```

#### Evaluator

⚠️ **Note:**
Most parameters in the Evaluator depend on the outputs from the Encoder.
Please verify encoder run IDs and configurations before execution.

```bash
nohup ./run_evaluator.sh > run_evaluator.log 2>&1 < /dev/null &
```




[//]: # (#  Code structure)

[//]: # ()
[//]: # (The codebase is organized into four core modules:)

[//]: # ()
[//]: # (---)

[//]: # ()
[//]: # (### 1. Processor)

[//]: # ()
[//]: # (The **Processor** handles dataset loading and outputs a standardized `Document` format: )

[//]: # (```python)

[//]: # (Document&#40;doc_id='', text='', metadata=''&#41;)

[//]: # (```)

[//]: # ()
[//]: # (- Currently supports **GutenQA** and **BEIR** datasets.)

[//]: # (- To add new datasets, implement a custom processor class inheriting from the base processor, and **register it** in the `processors/__init__.py`.)

[//]: # ()
[//]: # ()
[//]: # (### 2. Chunker)

[//]: # ()
[//]: # (  The Chunker splits documents/passages into smaller chunks for downstream encoding and retrieval.)

[//]: # (Available methods include:)

[//]: # (- FixedSizeChunker)

[//]: # (- ParagraphChunker)

[//]: # (- SemanticChunker)

[//]: # (- SentenceChunker)

[//]: # (- LumberChunker)

[//]: # (- Proposition)

[//]: # ()
[//]: # ()
[//]: # (### 3. Encoder)

[//]: # ()
[//]: # (The Encoder transforms chunks into vector embeddings. Two strategies are supported:)

[//]: # (- **RegularEncoder**: Encodes chunks individually.)

[//]: # (- **LateEncoder**: Concatenates chunks, encodes them jointly, and splits embeddings afterward.)

[//]: # ()
[//]: # (You can choose from different embedding models, such as:)

[//]: # ()
[//]: # (- `jinaai/jina-embeddings-v2-small-en`)

[//]: # (- `Qwen/Qwen3-Embedding-0.6B`)

[//]: # ()
[//]: # ()
[//]: # (Custom encoders can be added by creating a class under `src/encoders` &#40;inheriting from `BaseEncoder`&#41;.)

[//]: # (Similarly, custom embedding models can be added under `src/models/embedding` &#40;inheriting from `BaseEmbeddingModel`&#41;.)

[//]: # ()
[//]: # (### 4. Evaluator)

[//]: # ()
[//]: # (The Evaluator computes ranking-based metrics &#40;nDCG, Recall&#41; to measure the performance of different chunking strategies and encoder configurations)

[//]: # ()
[//]: # ()
[//]: # (# How to run)

[//]: # ()
[//]: # (We can run these modules separately with command line, for more details please look other sections below. Here, we provide a shell to run all the parameters.)





# ✂️ Chunker

The **Chunker** module splits documents/passages into smaller units for efficient encoding and retrieval.
It supports multiple strategies, including those from [LumberChunker](https://arxiv.org/abs/2406.17526) and [Late Chunking](https://arxiv.org/abs/2409.04701)

---

## 🚀 Run Example

Run the **ParagraphChunker** on the nfcorpus dataset:

```bash
python -m src.runner chunker \
--processor_name beir \
--dataset_name nfcorpus \
--data_folder src/data \
--sample 10 \
--chunker ParagraphChunker \
--output_folder src/outputs \
--query
```

## ⚙️ Key Arguments

- `--processor`: Data processor (e.g., `GutenQA`, `beir`).

- `--dataset_name`: Dataset for processing.

- `--data_folder`: Dataset folder.

- `--chunker`: Chunking strategy (e.g., `ParagraphChunker`, `LumberChunker`).

- `--output`: Output directory for processed data.

- `--query`: Enables query mode, which first runs the chunker and then saves the queries.



# 🪄 Encoder

The **Encoder** module transforms text chunks generated by the Chunker into vector embeddings.

-------------------------------

## 🚀 Run Example

Run the **RegularEncoder** with backbone `jinaiV2` and embedding model `jinaai/jina-embeddings-v2-small-en`:  

```bash
python -m src.runner encoder \
--encoder_name RegularEncoder \
--dataset_name nfcorpus \
--chunk_run_id SentenceChunker \
--backbone JinaaiV2 \
--model_name jinaai/jina-embeddings-v2-small-en \
--batch_size 10 \
--output_folder src/test_outputs \
--query \
--query_run_id 20250921-183217-beir-8f3497a6
```

## ⚙️ Arguments

- `--encoder_name`: Encoder class (e.g., `RegularEncoder`, `LateEncoder`).
- `--dataset_name`: Dataset name (e.g., `nfcorpus`, `fiqa`)
- `--chunk_run_id`: The ID of the chunking run whose outputs will be encoded.
- `--backbone`: Embedding backbone (e.g., `openai`, `Qwen3`).  
- `--model_name`: The embedding model to use (e.g., `text-embedding-ada-002`, `Qwen/Qwen3-Embedding-0.6B`).
- `--batch_size`: Number of texts processed per batch.
- `--output`: Output directory where embeddings will be stored. default: `src/outputs`  
- `--query`: Enables query encoding mode.  which first runs the encoder and then saves the query embeddings.
- `--query_run_id`: The ID of the query run to be encoded. 

If you want to customize the encoder, you can define your own **Encoder class** under `src/encoder`, 
inheriting from the base class **BaseEncoder**.

If you want to use another embedding model, you can define a custom **embedding model** under `src/models/embeddings`,
inheriting from the base class **BaseEmbeddingModel**.


# 📊 Evaluator

The **Evaluator** module is to measure the performance of different **chunking strategies** and **encoder configurations**.  
It computes ranking-based metrics such as **DCG** and **Recall**, 
allowing you to compare how well various combinations perform in retrieval tasks.  

---------

## 🚀 Run Example

Run the evaluator with a given chunking run, query run, and their corresponding embeddings:

```bash
python -m src.runner eval \
  --chunk_run_id 20250902-171846-ParagraphChunker-GutenQA-c78b6f37 \
  --query_run_id 20250902-171849-GutenQA-ed7846b6 \
  --chunk_embedding_run_id 20250902-175213-RegularEncoder-Qwen3-Qwen3-Embedding-0.6B-77072742 \
  --query_embedding_run_id 20250902-175652-RegularEncoder-Qwen3-13d17bce \
  --dataset_name QutenQA \
  --scope document \
  --source_path src/test_outputs
```

## ⚙️ Arguments

- `--chunk_run_id`: The ID of the chunking run.
- `--query_run_id`: The ID of the query run.  
- `--chunk_embedding_run_id`: The ID of the chunk embedding run.  
- `--query_embedding_run_id`: The ID of the query embedding run.  
- `--dataset_name`: The dataset used for evaluation (e.g., `QutenQA`).  
- `--scope`: The evaluation scope:
  - `document` -> query retrieval within one document.
  - `corpus` -> query retrieval across the full corpus.
- `--source_path`: Path to the evaluation source data.  default is `src/outputs`
