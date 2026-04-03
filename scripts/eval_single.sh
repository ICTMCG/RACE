#!/bin/bash

# This script runs single-GPU evaluation using a config file and a model checkpoint.

# --- GPU Configuration ---
GPU_ID=${3:-0}  # Default to GPU 0 if not provided
export CUDA_VISIBLE_DEVICES=$GPU_ID

# --- Script Arguments ---
CONFIG_FILE=$1
CHECKPOINT_PATH=$2

if [[ -z "$CONFIG_FILE" ]]; then
    echo "Error: Please provide a path to a config file as the first argument."
    echo "Usage: $0 path/to/your_config.json path/to/your_model.pth [gpu_id]"
    exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Config file not found at: $CONFIG_FILE"
    exit 1
fi

if [[ -z "$CHECKPOINT_PATH" ]]; then
    echo "Error: Please provide a path to a model checkpoint as the second argument."
    echo "Usage: $0 path/to/your_config.json path/to/your_model.pth"
    exit 1
fi

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
    echo "Error: Checkpoint file not found at: $CHECKPOINT_PATH"
    exit 1
fi

echo "--- Starting Single-GPU Evaluation ---"
echo "Loading configuration from: $CONFIG_FILE"
echo "Using checkpoint: $CHECKPOINT_PATH"

python test.py \
    --config "$CONFIG_FILE" \
    --checkpoint_path "$CHECKPOINT_PATH"

echo "--- Evaluation Finished ---"
