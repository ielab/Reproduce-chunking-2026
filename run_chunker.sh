#!/bin/bash
#SBATCH --time=20:00:00
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



# ===== User-tunable knobs =====
DATA_FOLDER="src/data"
OUTPUT_FOLDER="src/chunked_output"
#SAMPLE=10
# if output folder doesn't exist, create it
mkdir -p "$OUTPUT_FOLDER"

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

            chunk_run_id="$CHUNKER"
            chunker_kwargs='{}'

            case "$CHUNKER" in
              "FixedSizeChunker")
                chunk_run_id="FixedSizeChunker-256"
                chunker_kwargs='{"fixed_size":256}'
                ;;
              "ParagraphChunker"|"SentenceChunker"|"SemanticChunker")
                chunk_run_id="$CHUNKER"
                ;;
            esac

            chunks_dir="$OUTPUT_FOLDER/$DATASET/chunks/$chunk_run_id"
            if [[ -d "$chunks_dir" ]]; then
                echo ">>> Skipping $chunk_run_id (already exists at $chunks_dir)"
                if [[ $first_chunker -eq 1 ]]; then
                    first_chunker=0
                fi
                continue
            fi

            CMD=(
              python -m src.runner chunker
              --processor_name "$PROCESSOR"
              --dataset_name "$DATASET"
              --data_folder "$DATA_FOLDER"
#             --sample "$SAMPLE"
              --chunker "$CHUNKER"
              --output_folder "$OUTPUT_FOLDER"
              --chunk_run_id "$chunk_run_id"
            )

            if [[ "$chunker_kwargs" != "{}" ]]; then
                CMD+=( --chunker_kwargs "$chunker_kwargs" )
            fi

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
