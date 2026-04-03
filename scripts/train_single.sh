#!/bin/bash

# This script runs single-GPU training by loading all settings from a config file.

# --- GPU Configuration ---
GPU_ID=${2:-0}  # Default to GPU 0 if not provided

# --- Config File ---
# The first argument to the script should be the path to the config file.
CONFIG_FILE=$1

if [[ -z "$CONFIG_FILE" ]]; then
    echo "Error: Please provide a path to a config file as the first argument."
    echo "Usage: $0 path/to/your_config.json"
    exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Config file not found at: $CONFIG_FILE"
    exit 1
fi

echo "--- Starting Single-GPU Training ---"
echo "Loading configuration from: $CONFIG_FILE, using GPU ID: $GPU_ID"

export CUDA_VISIBLE_DEVICES=$GPU_ID
python train.py --config "$CONFIG_FILE"

echo "--- Training Finished ---"
