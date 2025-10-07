#!/usr/bin/env bash

# ===== User-tunable knobs =====
OUTPUT_FOLDER="src/chunked_output"

# if output folder doesn't exist, create it
mkdir -p "$OUTPUT_FOLDER"

dataset_index=$1

# Datasets
DATASETS=(
  "GutenQA"
  "fiqa"
  "nfcorpus"
  "scifact"
  "trec-covid"
  "arguana"
  "scidocs"
)

DATASET="${DATASETS[$dataset_index]:-GutenQA}"

# Encoders
ENCODERS=(
  "RegularEncoder"
  "LateEncoder"
)

# Backbones and their model names (pair format BACKBONE|MODEL_NAME)
BACKBONES_MODELS=(
  "JinaaiV2|jinaai/jina-embeddings-v2-small-en|cosine"
  "JinaaiV3|jinaai/jina-embeddings-v3|cosine"
  "Normic|nomic-ai/nomic-embed-text-v1|cosine"
  "IntFloatE5|intfloat/multilingual-e5-large-instruct|cosine"
)

# Chunk run IDs to iterate
CHUNK_RUN_IDS=(
  "ParagraphChunker"
  #"SentenceChunker"
  #"FixedSizeChunker"
  #"SemanticChunker"
  #"LumberChunker"
  #"Proposition"
)

# ==============================

# Dry run: set to 1 to only echo commands without executing
DRYRUN=${DRYRUN:-0}

timestamp() { date +"%Y-%m-%d %H:%M:%S"; }

# Find the query run ID dynamically
QUERY_RUN_FOLDER=$(find "$OUTPUT_FOLDER/$DATASET/queries/" -mindepth 1 -maxdepth 1 -type d | head -n 1)
if [ -z "$QUERY_RUN_FOLDER" ]; then
  echo "Error: No query run folder found for dataset $DATASET in $OUTPUT_FOLDER/$DATASET/queries/"
  exit 1
fi
QUERY_RUN_ID=$(basename "$QUERY_RUN_FOLDER")
echo "Found Query Run ID for $DATASET: $QUERY_RUN_ID"

    # Scope options: GutenQA => document + corpus; others => corpus only
if [[ "$DATASET" == "GutenQA" || "$DATASET" == "gutenqa" ]]; then
  SCOPE_OPTIONS=("document" "corpus")
else
  SCOPE_OPTIONS=("corpus")
fi

for CHUNK_RUN_ID in "${CHUNK_RUN_IDS[@]}"; do

    for ENCODER in "${ENCODERS[@]}"; do

        for bm in "${BACKBONES_MODELS[@]}"; do

            IFS="|" read -r BACKBONE MODEL SIMILARITY <<<"$bm"
            MODEL_SUFFIX="${MODEL##*/}"
            CHUNK_EMBEDDING_ID="${ENCODER}-${MODEL_SUFFIX}"
            # Query embeddings are based on the model only, not the encoder type
            QUERY_EMBEDDING_ID="${MODEL_SUFFIX}"

            # Construct the expected output file path
            METRIC_FILE="$OUTPUT_FOLDER/$DATASET/results/$CHUNK_RUN_ID/$CHUNK_EMBEDDING_ID/metric.eval"

            echo "Pair: chunk=$CHUNK_EMBEDDING_ID | query=$QUERY_EMBEDDING_ID"

            for SCOPE in "${SCOPE_OPTIONS[@]}"; do
                # Check if the metric file already exists
                if [ -f "$METRIC_FILE" ]; then
                    echo ">>> [$(timestamp)] Skipping: Metric file already exists at $METRIC_FILE"
                    continue
                fi

                CMD=(
                  python -m src.runner evaluator
                  --dataset_name "$DATASET"
                  --chunk_run_id "$CHUNK_RUN_ID"
                  --query_run_id "$QUERY_RUN_ID"
                  --chunk_embedding_run_id "$CHUNK_EMBEDDING_ID"
                  --query_embedding_run_id "$QUERY_EMBEDDING_ID"
                  --source_path "$OUTPUT_FOLDER"
                  --similarity "$SIMILARITY"
                  --scope "$SCOPE"
                  --top_k 100
                )


                echo ">>> [$(timestamp)] dataset=$DATASET | encoder=$ENCODER | backbone=$BACKBONE | model=$MODEL_NAME | chunk=$CHUNK_RUN_ID | query=$QUERY_RUN_ID"
                echo "${CMD[@]}"
                if [[ "$DRYRUN" -eq 0 ]]; then
                  "${CMD[@]}"
                fi
            done
        done
    done
done

echo "[$(timestamp)] All evaluator runs completed."
