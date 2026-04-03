#!/bin/bash

# ============================================================================
# RACE Data Preparation Pipeline
# ============================================================================
# This script automates the full data preparation workflow for the Hart dataset.
# It executes 4 sequential steps:
#   1. Data Unification: Parse Hart raw JSON → Schema A (grouped JSONL)
#   2. RST Parsing: Add RST tree structures to each text
#   3. Data Flattening: Convert grouped data → per-article flat JSONL (Schema B)
#   4. Dataset Splitting: Split into train/val/test sets
#
# Usage:
#   bash scripts/prepare_data.sh [hart_data_path]
#
# Arguments:
#   hart_data_path  (optional) Path to the Hart dataset directory.
#                   Default: data/truth-mirror/benchmark/hart
#
# Prerequisites:
#   - Python environment with required packages (isanlp_rst, pandas, sklearn, etc.)
#   - RST parser model path set in .env file (RST_PARSER_PATH=...)
# ============================================================================

set -e  # Exit immediately if any command fails

HART_PATH=${1:-"data/truth-mirror/benchmark/hart"}

echo "============================================"
echo " RACE Data Preparation Pipeline"
echo "============================================"
echo "Hart data path: $HART_PATH"
echo ""

# --- Step 1: Data Unification ---
echo "[Step 1/4] Running Data Unification Pipeline..."
python3 utils/data_unification_pipeline.py --output "data/master_grouped.jsonl"

if [ ! -f "data/master_grouped.jsonl" ]; then
    echo "Error: Data unification failed. 'data/master_grouped.jsonl' not found."
    exit 1
fi
echo "[Step 1/4] Done. Output: data/master_grouped.jsonl"
echo "---------------------------------"

# --- Step 2: RST Parsing ---
echo "[Step 2/4] Running RST Parsing Pipeline..."
echo "  (This step may take a long time depending on dataset size)"
python3 utils/rst_parser_pipeline.py \
    --input "data/master_grouped.jsonl" \
    --output "data/master_rst_parsed.jsonl"

if [ ! -f "data/master_rst_parsed.jsonl" ]; then
    echo "Error: RST parsing failed. 'data/master_rst_parsed.jsonl' not found."
    exit 1
fi
echo "[Step 2/4] Done. Output: data/master_rst_parsed.jsonl"
echo "---------------------------------"

# --- Step 3: Data Flattening ---
echo "[Step 3/4] Running Data Flattening Pipeline..."
python3 utils/flatten_data_pipeline.py \
    --input_file "data/master_rst_parsed.jsonl" \
    --output_file "data/flattened_articles.jsonl"

if [ ! -f "data/flattened_articles.jsonl" ]; then
    echo "Error: Data flattening failed. 'data/flattened_articles.jsonl' not found."
    exit 1
fi
echo "[Step 3/4] Done. Output: data/flattened_articles.jsonl"
echo "---------------------------------"

# --- Step 4: Dataset Splitting ---
echo "[Step 4/4] Running Dataset Splitting..."
python3 utils/split_dataset.py

if [ ! -f "data/hart_split/train_graph.jsonl" ]; then
    echo "Error: Dataset splitting failed. 'data/hart_split/train_graph.jsonl' not found."
    exit 1
fi
echo "[Step 4/4] Done. Output: data/hart_split/"
echo ""
echo "============================================"
echo " Data preparation completed successfully!"
echo " Output files:"
echo "   data/hart_split/train_graph.jsonl"
echo "   data/hart_split/val_graph.jsonl"
echo "   data/hart_split/test_graph.jsonl"
echo "============================================"
