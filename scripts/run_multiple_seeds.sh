#!/bin/bash

# =================================================================================
# run_multiple_seeds.sh
#
# Description:
#   This script runs the training process multiple times with different random
#   seeds to evaluate the stability and statistical significance of the results.
#
#   It takes a base configuration file, dynamically updates the 'seed' and
#   'output_dir' for each run, and then launches the training. All artifacts
#   for a given run are stored in a dedicated directory.
#
# Usage:
#   bash scripts/run_multiple_seeds.sh <path_to_config_json> <seeds>
#
# Parameters:
#   - <path_to_config_json>: The path to the base JSON configuration file.
#   - <seeds>: EITHER a comma-separated list of seeds (e.g., "42,100,2024")
#              OR a single integer specifying the number of runs with random seeds.
#
# Example (specific seeds):
#   bash scripts/run_multiple_seeds.sh cconfigs/RACE.json "3407,42,2025,0,1"
#
# Example (random seeds):
#   bash scripts/run_multiple_seeds.sh configs/RACE.json 5
# =================================================================================

# --- 1. Input Validation ---
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <path_to_config_json> <seeds> [gpu_id]"
    echo "  <seeds> can be a comma-separated list (e.g., '42,100') or a number for random runs (e.g., 5)."
    exit 1
fi

CONFIG_FILE=$1
SEEDS_ARG=$2
GPU_ID=${3:-0}

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found at '$CONFIG_FILE'"
    exit 1
fi

# Check for jq dependency
if ! command -v jq &> /dev/null; then
    echo "Error: 'jq' is not installed. Please install it to proceed."
    echo "e.g., sudo apt-get install jq"
    exit 1
fi

SEED_ARRAY=()
# --- 2. Seed Interpretation ---
if [[ "$SEEDS_ARG" == *","* ]]; then
    # Argument is a comma-separated list of seeds
    echo "Interpreting seeds as a specific list: $SEEDS_ARG"
    OLD_IFS=$IFS
    IFS=','
    read -ra SEED_ARRAY <<< "$SEEDS_ARG"
    IFS=$OLD_IFS
else
    # Argument is a number of runs
    if ! [[ "$SEEDS_ARG" =~ ^[0-9]+$ ]] || [ "$SEEDS_ARG" -le 0 ]; then
        echo "Error: For random runs, the second argument must be a positive integer."
        exit 1
    fi
    echo "Interpreting seeds as $SEEDS_ARG runs with random seeds."
    for i in $(seq 1 "$SEEDS_ARG"); do
        SEED_ARRAY+=($((RANDOM % 10000)))
    done
fi

NUM_RUNS=${#SEED_ARRAY[@]}

# --- 3. Base Directory Setup ---
# The output will be grouped in a parent directory named after the config file.
CONFIG_BASENAME=$(basename "$CONFIG_FILE" .json)
BASE_OUTPUT_DIR="results/multi_seed_runs/${CONFIG_BASENAME}"

echo "=================================================="
echo "Starting multiple seed training run..."
echo "  - Config File:    $CONFIG_FILE"
echo "  - Total Runs:     $NUM_RUNS"
echo "  - Using GPU ID:   $GPU_ID"
echo "  - Seeds to run:   ${SEED_ARRAY[*]}"
echo "  - Base Output Dir:  $BASE_OUTPUT_DIR"
echo "=================================================="

# --- 4. Main Loop ---
RUN_COUNT=0
for SEED in "${SEED_ARRAY[@]}"; do
    RUN_COUNT=$((RUN_COUNT + 1))
    RUN_OUTPUT_DIR="${BASE_OUTPUT_DIR}/seed_${SEED}"
    
    mkdir -p "$RUN_OUTPUT_DIR"
    
    # Create a temporary config file for this specific run
    TEMP_CONFIG_FILE="${RUN_OUTPUT_DIR}/config.json"
    
    echo "--------------------------------------------------"
    echo "Run ${RUN_COUNT}/${NUM_RUNS}: Starting training with Seed=${SEED}"
    echo "  - Output directory: ${RUN_OUTPUT_DIR}"
    
    # Use jq to update the seed and output directory in the config
    jq --argjson seed "$SEED" --arg out_dir "$RUN_OUTPUT_DIR" \
       '.seed = $seed | .output_dir = $out_dir' \
       "$CONFIG_FILE" > "$TEMP_CONFIG_FILE"
       
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create temporary config file with jq for seed ${SEED}."
        continue # Skip to the next run
    fi

    # Launch the training script with the temporary config
    bash scripts/train_single.sh "$TEMP_CONFIG_FILE" "$GPU_ID"
    
    if [ $? -ne 0 ]; then
        echo "Error: Training run ${RUN_COUNT} with seed ${SEED} failed."
    else
        echo "Run ${RUN_COUNT}/${NUM_RUNS} with Seed=${SEED} completed."
    fi
done

echo "=================================================="
echo "All runs completed."
echo "Results are stored in: ${BASE_OUTPUT_DIR}"
echo "=================================================="
