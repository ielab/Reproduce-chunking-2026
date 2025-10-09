#!/bin/bash
#SBATCH --time=24:00:00
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


#PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

dataset_index=$1

# ===== User-tunable knobs =====
OUTPUT_FOLDER="src/chunked_output"


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

# Find the query run ID dynamically
QUERY_RUN_FOLDER=$(find "$OUTPUT_FOLDER/$DATASET/queries/" -mindepth 1 -maxdepth 1 -type d | head -n 1)
if [ -z "$QUERY_RUN_FOLDER" ]; then
  echo "Error: No query run folder found for dataset $DATASET in $OUTPUT_FOLDER/$DATASET/queries/"
  exit 1
fi
QUERY_RUN_ID=$(basename "$QUERY_RUN_FOLDER")
echo "Found Query Run ID for $DATASET: $QUERY_RUN_ID"



# Encoders
ENCODERS=(
  "RegularEncoder"
  "LateEncoder"
)

# Backbones and their model names (pair format BACKBONE|MODEL_NAME)
BACKBONES_MODELS=(
  "JinaaiV2|jinaai/jina-embeddings-v2-small-en"
  "JinaaiV3|jinaai/jina-embeddings-v3"
  "Normic|nomic-ai/nomic-embed-text-v1"
  "IntFloatE5|intfloat/multilingual-e5-large-instruct"
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
for ENCODER in "${ENCODERS[@]}"; do

    for bm in "${BACKBONES_MODELS[@]}"; do

        IFS="|" read -r BACKBONE MODEL_NAME <<<"$bm"
        MODEL_NAME_CLEAN="${MODEL_NAME##*/}"

        # if it's jina-embeddings-v2-small-en backbone, batch size use 12, otherwise 512
        if [[ "$MODEL_NAME" == "jinaai/jina-embeddings-v2-small-en" ]]; then
          BATCH_SIZE=12
        else
          BATCH_SIZE=256
        fi

        # --- Handle Query Encoding ---
        # As per build_query_embedding_run_id in run_ids.py, the ID is just the model name.
        QUERY_EMBED_RUN_ID="${MODEL_NAME_CLEAN}"
        QUERY_ENCODED_FOLDER="$OUTPUT_FOLDER/$DATASET/query_embeddings/${QUERY_RUN_ID}/${QUERY_EMBED_RUN_ID}"
        echo "Checking for query embeddings: $QUERY_ENCODED_FOLDER"

        if [[ ! -d "$QUERY_ENCODED_FOLDER" ]]; then
          echo "Query embeddings not found for $MODEL_NAME. Encoding queries."
          # The runner needs a chunk_run_id as a placeholder, but it will only encode queries
          # because of the --query flag and the logic in runner.py.
          FIRST_CHUNK_RUN_ID="${CHUNK_RUN_IDS[0]}"

          CMD=(
            python -m src.runner encoder
            --dataset_name "$DATASET"
            --chunk_run_id "$FIRST_CHUNK_RUN_ID"
            --encoder_name "$ENCODER"
            --backbone "$BACKBONE"
            --model_name "$MODEL_NAME"
            --batch_size "$BATCH_SIZE"
            --output_folder "$OUTPUT_FOLDER"
            --query --query_run_id "$QUERY_RUN_ID"
          )
          echo ">>> [$(timestamp)] ENCODING QUERIES | dataset=$DATASET | model=$MODEL_NAME"
          echo "${CMD[@]}"
          if [[ "$DRYRUN" -eq 0 ]]; then
            "${CMD[@]}"
          fi
          echo
        else
          echo "Query embeddings already exist for $MODEL_NAME."
        fi


        # --- Handle Document Chunk Encoding ---
        for CHUNK_RUN_ID in "${CHUNK_RUN_IDS[@]}"; do
            DOC_EMBEDDINGS_FOLDER="$OUTPUT_FOLDER/$DATASET/embeddings/${CHUNK_RUN_ID}/${ENCODER}-${MODEL_NAME_CLEAN}"
            echo "Checking for document embeddings: $DOC_EMBEDDINGS_FOLDER"
            if [[ -d "$DOC_EMBEDDINGS_FOLDER" ]]; then
              echo "Skipping existing document embeddings for chunk $CHUNK_RUN_ID."
              continue
            fi

            CMD=(
              python -m src.runner encoder
              --dataset_name "$DATASET"
              --chunk_run_id "$CHUNK_RUN_ID"
              --encoder_name "$ENCODER"
              --backbone "$BACKBONE"
              --model_name "$MODEL_NAME"
              --batch_size "$BATCH_SIZE"
              --output_folder "$OUTPUT_FOLDER"
            )

            echo ">>> [$(timestamp)] ENCODING DOCS | dataset=$DATASET | model=$MODEL_NAME | chunk=$CHUNK_RUN_ID"
            echo "${CMD[@]}"
            if [[ "$DRYRUN" -eq 0 ]]; then
              "${CMD[@]}"
            fi
            echo
        done
    done
done

echo "[$(timestamp)] All encoder runs completed."
