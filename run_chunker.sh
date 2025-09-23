#!/bin/bash

# ===== User-tunable knobs =====
DATA_FOLDER="src/data"
OUTPUT_FOLDER="src/test_outputs3"
#SAMPLE=10

# Processor → datasets mapping
PROCESSORS=("GutenQA" "beir")
#PROCESSORS=("beir")
DATASETS_GutenQA=("GutenQA")
DATASETS_beir=(
  "fiqa"
  "nfcorpus"
  "scifact"
  "trec-covid"
  "arguana"
  "scidocs"
)

# Chunkers
CHUNKERS=(
  "ParagraphChunker"
  "FixedSizeChunker"
  "SentenceChunker"
  "SemanticChunker"
)

# ==============================


for PROCESSOR in "${PROCESSORS[@]}"; do
    # Pick the dataset list depending on processor
    if [ "$PROCESSOR" = "GutenQA" ]; then
        DATASETS=("${DATASETS_GutenQA[@]}")
    else
        DATASETS=("${DATASETS_beir[@]}")
    fi

    for DATASET in "${DATASETS[@]}"; do

        echo ">>> Running processor=$PROCESSOR | dataset=$DATASET (with query once)"

        first_chunker=1

        for CHUNKER in "${CHUNKERS[@]}"; do
            echo ">>> Running processor=$PROCESSOR | dataset=$DATASET | chunker=$CHUNKER"


            CMD=(
              python -m src.runner chunker
              --processor_name "$PROCESSOR"
              --dataset_name "$DATASET"
              --data_folder "$DATA_FOLDER"
#              --sample "$SAMPLE"
              --chunker "$CHUNKER"
              --output_folder "$OUTPUT_FOLDER"
            )

            # Only the first chunker per dataset includes --query
            if [[ $first_chunker -eq 1 ]]; then
                CMD+=( --query )
                first_chunker=0
                echo ">>> (with --query)"
            fi

            echo "${CMD[@]}"
            "${CMD[@]}"

            echo
        done
    done
done