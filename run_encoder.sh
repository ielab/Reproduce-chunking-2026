#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256g
#SBATCH --gres=gpu:1
#SBATCH --qos=express
#SBATCH --account=OD-236007
#SBATCH --job-name=doc-encode
#SBATCH --output=slurm_logs/doc-encode-%A_%a.txt
#SBATCH --error=slurm_logs/doc-encode-error-%A_%a.txt

# === Load Modules and Activate Environment ===
module load miniconda3
module load cuda
source activate "/scratch3/wan458/chunking-reproduce/envs"


PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ===== Script Arguments =====
DATASET=$1
ENCODER=$2
BACKBONE=$3
MODEL_NAME=$4
CHUNK_RUN_ID=$5

if [ -z "$DATASET" ] || [ -z "$ENCODER" ] || [ -z "$BACKBONE" ] || [ -z "$MODEL_NAME" ] || [ -z "$CHUNK_RUN_ID" ]; then
  echo "Usage: $0 <DATASET> <ENCODER> <BACKBONE> <MODEL_NAME> <CHUNK_RUN_ID>"
  exit 1
fi

# ===== Fixed knobs =====
OUTPUT_FOLDER="src/chunked_output"
DRYRUN=${DRYRUN:-0}

timestamp() { date +"%Y-%m-%d %H:%M:%S"; }

echo "[$(timestamp)] Starting document encoder job…"
echo "Dataset: $DATASET"
echo "Encoder: $ENCODER"
echo "Backbone: $BACKBONE"
echo "Model Name: $MODEL_NAME"
echo "Chunk Run ID: $CHUNK_RUN_ID"

MODEL_NAME_CLEAN="${MODEL_NAME##*/}"

# Set batch size based on model
if [[ "$MODEL_NAME" == "jinaai/jina-embeddings-v2-small-en" ]]; then
  BATCH_SIZE=4
else
  BATCH_SIZE=4
fi

# --- Handle Document Chunk Encoding ---
DOC_EMBEDDINGS_FOLDER="$OUTPUT_FOLDER/$DATASET/embeddings/${CHUNK_RUN_ID}/${ENCODER}-${MODEL_NAME_CLEAN}"
echo "Checking for document embeddings: $DOC_EMBEDDINGS_FOLDER"
if [[ -d "$DOC_EMBEDDINGS_FOLDER" ]]; then
  echo "Skipping existing document embeddings for chunk $CHUNK_RUN_ID."
else
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
fi

echo "[$(timestamp)] Document encoder job finished."
