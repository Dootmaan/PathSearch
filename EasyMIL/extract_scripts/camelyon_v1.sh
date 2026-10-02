#!/usr/bin/env bash
# Camelyon extraction wrapper (uses PATHSEARCH_* env vars or repo-relative defaults)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

root_dir="${PATHSEARCH_ROOT_DIR:-$REPO_ROOT/extract_scripts}"
tasks="CAMELYON16 CAMELYON17"

# models to run
models="conch_v1_5"
skip_partial="no"

declare -A gpus
gpus["conch_v1_5"]=5

for model in $models; do
        for task in $tasks; do
                DIR_TO_COORDS="${PATHSEARCH_DIR_TO_COORDS:-$REPO_ROOT/Pathology/Patches/$task}"
                CSV_FILE_NAME="${PATHSEARCH_CSV_FILE_NAME:-$REPO_ROOT/EasyMIL/dataset_csv/camelyon.csv}"
                FEATURES_DIRECTORY="${PATHSEARCH_FEATURES_DIRECTORY:-$REPO_ROOT/Pathology/Patches/$task}"

                echo "$model, GPU is: ${gpus[$model]}"
                export CUDA_VISIBLE_DEVICES=${gpus[$model]}

                nohup python3 extract_features_fp_from_patch.py \
                        --data_h5_dir "$DIR_TO_COORDS" \
                        --csv_path "$CSV_FILE_NAME" \
                        --feat_dir "$FEATURES_DIRECTORY" \
                        --batch_size 64 \
                        --model "$model" \
                        --skip_partial "$skip_partial" > "$root_dir/logs/${task}_log_${model}.log" 2>&1 &
        done
done
