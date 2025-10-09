#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --qos=express
#SBATCH --account=OD-236007
#SBATCH --job-name=evaluate
#SBATCH --output=slurm_logs/evaluate-%A_%a.txt
#SBATCH --error=slurm_logs/evaluate-error-%A_%a.txt

# === Load Modules and Activate Environment ===
module load miniconda3
module load cuda
source activate "/scratch3/wan458/chunking-reproduce/envs"

# ===== Script Arguments =====
DATASET=$1
CHUNK_RUN_ID=$2
ENCODER=$3
MODEL_NAME=$4
SCOPE=$5

if [ -z "$DATASET" ] || [ -z "$CHUNK_RUN_ID" ] || [ -z "$ENCODER" ] || [ -z "$MODEL_NAME" ] || [ -z "$SCOPE" ]; then
  echo "Usage: $0 <DATASET> <CHUNK_RUN_ID> <ENCODER> <MODEL_NAME> <SCOPE>"
  exit 1
fi

# ===== Fixed knobs =====
SOURCE_PATH="src/chunked_output"
DRYRUN=${DRYRUN:-0}

timestamp() { date +"%Y-%m-%d %H:%M:%S"; }

echo "[$(timestamp)] Starting single evaluation job…"
echo "Dataset: $DATASET"
echo "Chunk Run ID: $CHUNK_RUN_ID"
echo "Encoder: $ENCODER"
echo "Model Name: $MODEL_NAME"
echo "Scope: $SCOPE"

# --- Path and ID Construction ---
MODEL_NAME_CLEAN="${MODEL_NAME##*/}"

# Find the query run ID dynamically
QUERY_RUN_FOLDER=$(find "$SOURCE_PATH/$DATASET/queries/" -mindepth 1 -maxdepth 1 -type d | head -n 1)
if [ -z "$QUERY_RUN_FOLDER" ]; then
  echo "Error: No query run folder found for dataset $DATASET in $SOURCE_PATH/$DATASET/queries/"
  exit 1
fi
QUERY_RUN_ID=$(basename "$QUERY_RUN_FOLDER")

# Construct IDs for embeddings
CHUNK_EMBEDDING_RUN_ID="${ENCODER}-${MODEL_NAME_CLEAN}"
QUERY_EMBEDDING_RUN_ID="${MODEL_NAME_CLEAN}"

# --- Existence Checks ---
QUERY_EMBED_PATH="$SOURCE_PATH/$DATASET/query_embeddings/${QUERY_RUN_ID}/${QUERY_EMBEDDING_RUN_ID}/embeddings.pkl"
CHUNK_EMBED_PATH="$SOURCE_PATH/$DATASET/embeddings/${CHUNK_RUN_ID}/${CHUNK_EMBEDDING_RUN_ID}/embeddings.pkl"

if [ ! -f "$QUERY_EMBED_PATH" ]; then
    echo "Error: Query embeddings not found at $QUERY_EMBED_PATH"
    exit 1
fi

if [ ! -f "$CHUNK_EMBED_PATH" ]; then
    echo "Error: Document embeddings not found at $CHUNK_EMBED_PATH"
    exit 1
fi

echo "All required embedding files found."

# --- Run Evaluation ---
CMD=(
  python -m src.runner evaluator
  --dataset_name "$DATASET"
  --chunk_run_id "$CHUNK_RUN_ID"
  --query_run_id "$QUERY_RUN_ID"
  --chunk_embedding_run_id "$CHUNK_EMBEDDING_RUN_ID"
  --query_embedding_run_id "$QUERY_EMBEDDING_RUN_ID"
  --scope "$SCOPE"
  --source_path "$SOURCE_PATH"
)

echo ">>> [$(timestamp)] RUNNING EVALUATION"
echo "${CMD[@]}"
if [[ "$DRYRUN" -eq 0 ]]; then
  "${CMD[@]}"
fi

echo "[$(timestamp)] Evaluation job finished."