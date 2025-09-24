#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --gres=gpu:1
#SBATCH --qos=express
#SBATCH --account=OD-236007
#SBATCH --job-name=grpo-bool
#SBATCH --output=slurm_logs/print-chunk.txt
#SBATCH --error=slurm_logs/error-chunk.txt

# === Load Modules and Activate Environment ===
module load miniconda3
module load cuda
source activate "/scratch3/wan458/chunking-reproduce/envs"


PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

dataset_index=$1

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

DATASET="${DATASETS[$dataset_index]:-GutenQA}"

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

# Chunk run IDs to iterate (first one gets --query for that dataset)
CHUNK_RUN_IDS=(
  "ParagraphChunker"
  "SentenceChunker"
  "FixedSizeChunker"
  "SemanticChunker"
  "LumberChunker"
  "Proposition"
)

# ==============================


# Dry run: set to 1 to only echo commands without executing
DRYRUN=${DRYRUN:-0}

timestamp() { date +"%Y-%m-%d %H:%M:%S"; }

echo "[$(timestamp)] Starting encoder sweep…"
echo "Dataset: $DATASET"
echo "Encoders: ${ENCODERS[*]}"
echo "Backbones/Models:"
for bm in "${BACKBONES_MODELS[@]}"; do
  IFS="|" read -r BB MN <<<"$bm"
  echo "  - $BB | $MN"
done
#echo "Chunk|Query pairs:"
#for cq in "${CHUNK_QUERY_PAIRS[@]}"; do
#  IFS="|" read -r CH QR <<<"$cq"
#  echo "  - $CH | $QR"
#done
#echo


# Matrix sweep



QUERY_RUN_ID="${QUERY_ID_BY_DATASET[$DATASET]:-}"

first_chunk=1

for ENCODER in "${ENCODERS[@]}"; do

    for bm in "${BACKBONES_MODELS[@]}"; do

        IFS="|" read -r BACKBONE MODEL_NAME <<<"$bm"
        # if it's jina-embeddings-v2-small-en backbone, batch size use 12, otherwise 512
        if [[ "$MODEL_NAME" == "jinaai/jina-embeddings-v2-small-en" ]]; then
          BATCH_SIZE=12
        else
          BATCH_SIZE=512
        fi

        for CHUNK_RUN_ID in "${CHUNK_RUN_IDS[@]}"; do

            CMD=(
              python -m src.runner encoder
              --dataset_name "$DATASET"
              --chunk_run_id "$CHUNK_RUN_ID"
              --encoder "$ENCODER"
              --backbone "$BACKBONE"
              --model_name "$MODEL_NAME"
              --batch_size "$BATCH_SIZE"
              --output_folder "$OUTPUT_FOLDER"
            )

            if [[ $first_chunk -eq 1 ]]; then
                CMD+=( --query --query_run_id "$QUERY_RUN_ID" )
                first_chunk=0
            fi

            echo ">>> [$(timestamp)] dataset=$DATASET | encoder=$ENCODER | backbone=$BACKBONE | model=$MODEL_NAME | chunk=$CHUNK_RUN_ID | query=$QUERY_RUN_ID"
            echo "${CMD[@]}"
            if [[ "$DRYRUN" -eq 0 ]]; then
              "${CMD[@]}"
            fi
            echo
        done
    done
done

echo "[$(timestamp)] All encoder runs completed."
