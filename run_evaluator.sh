#!/usr/bin/env bash

# ===== User-tunable knobs =====
OUTPUT_FOLDER="src/test_outputs3"

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

# One query_run_id per dataset (fill these in!)
declare -A QUERY_ID_BY_DATASET=(
  ["GutenQA"]="20250923-160046-GutenQA-2266af06"
  ["fiqa"]="20250923-161628-beir-cb5a96e3"
  ["nfcorpus"]="20250923-161654-beir-8f3497a6"
  ["scifact"]="20250923-161721-beir-a3c16da6"
  ["trec-covid"]="20250923-161747-beir-a25ad74b"
  ["arguana"]="20250923-161815-beir-70fb1edb"
  ["scidocs"]="20250923-161839-beir-f4e9a453"
)


# Encoders
ENCODERS=(
  "RegularEncoder"
  "LateEncoder"
)

# Backbones and their model names (pair format BACKBONE|MODEL_NAME)
BACKBONES_MODELS=(
  "JinaaiV2|jinaai/jina-embeddings-v2-small-en"
  "JinaaiV3|jinaai/jina-embeddings-v3"
  "Qwen3|Qwen/Qwen3-Embedding-0.6B"
  "Normic|nomic-ai/nomic-embed-text-v1"
)

# Chunk run IDs to iterate
CHUNK_RUN_IDS=(
  "ParagraphChunker"
  "SentenceChunker"
  "FixedSizeChunker"
  "SemanticChunker"
  "LumberChunker"
  "Proposition"
)

# ==============================


# Generate embedding run IDs
EMBEDDING_RUN_IDS=()
for ENC in "${ENCODERS[@]}"; do
  for MODEL in "${BACKBONES_MODELS[@]}"; do

    # If model contains '/', keep only the part after it
    MODEL_NAME="${MODEL##*/}"
    EMBEDDING_RUN_IDS+=( "${ENC}-${MODEL_NAME}" )
  done
done

# Dry run: set to 1 to only echo commands without executing
DRYRUN=${DRYRUN:-0}

timestamp() { date +"%Y-%m-%d %H:%M:%S"; }

echo "[$(timestamp)] Starting encoder sweep…"
echo "Datasets: ${DATASETS[*]}"
echo "Encoders: ${ENCODERS[*]}"
echo "Backbones/Models:"
for bm in "${BACKBONES_MODELS[@]}"; do
  IFS="|" read -r BB MN <<<"$bm"
  echo "  - $BB | $MN"
done


# Matrix sweep
for DATASET in "${DATASETS[@]}"; do
    QUERY_RUN_ID="${QUERY_ID_BY_DATASET[$DATASET]:-}"

        # Scope options: GutenQA => document + corpus; others => corpus only
    if [[ "$DATASET" == "GutenQA" || "$DATASET" == "gutenqa" ]]; then
      SCOPE_OPTIONS=("document" "corpus")
    else
      SCOPE_OPTIONS=("corpus")
    fi

    for CHUNK_RUN_ID in "${CHUNK_RUN_IDS[@]}"; do

        for ENCODER in "${ENCODERS[@]}"; do

            for bm in "${BACKBONES_MODELS[@]}"; do

                IFS="|" read -r BACKBONE MODEL <<<"$bm"
                MODEL_SUFFIX="${MODEL##*/}"
                CHUNK_EMBEDDING_ID="${ENCODER}-${MODEL_SUFFIX}"
                QUERY_EMBEDDING_ID="${MODEL_SUFFIX}"

                echo "Pair: chunk=$CHUNK_EMBEDDING_ID | query=$QUERY_EMBEDDING_ID"

                for SCOPE in "${SCOPE_OPTIONS[@]}"; do
                    CMD=(
                      python -m src.runner evaluator
                      --dataset_name "$DATASET"
                      --chunk_run_id "$CHUNK_RUN_ID"
                      --query_run_id "$QUERY_RUN_ID"
                      --chunk_embedding_run_id "$CHUNK_EMBEDDING_ID"
                      --query_embedding_run_id "$QUERY_EMBEDDING_ID"
                      --source_path "$OUTPUT_FOLDER"
                      --scope "$SCOPE"
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
done

echo "[$(timestamp)] All evaluator runs completed."
