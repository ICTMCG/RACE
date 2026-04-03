# RACE: Rhetorical Analysis for Creator-Editor Modeling

This repository contains the implementation of **RACE** (_Beyond the Final Actor: Modeling the Dual Roles of Creator and Editor for Fine-Grained LLM-Generated Text Detection_).

RACE leverages Rhetorical Structure Theory (RST) to construct discourse-level graphs from documents, then applies Relational Graph Convolutional Networks (RGCN) to learn structure-aware representations for fine-grained AI-generated text detection.

## 1. Installation

### Requirements

- **Python**: 3.8+
- **PyTorch & PyTorch Geometric**:
  Please install these first, following the official guides for your CUDA version to ensure GPU support:
  - [PyTorch](https://pytorch.org/get-started/locally/)
  - [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)

### Install Dependencies

Install the remaining Python dependencies using `pip`:

```bash
pip install -r requirements.txt
```

**Note**: The script `scripts/run_multiple_seeds.sh` requires `jq` (a command-line JSON processor).

- Ubuntu/Debian: `sudo apt-get install jq`
- MacOS: `brew install jq`

## 2. Data Preparation

This codebase expects processed data in `.jsonl` format. By default, the configuration looks for data in a `data/hart_split/` directory.

### Data Format

Each line in the `.jsonl` file is a JSON object with the following fields:

| Field            | Type   | Description                                                                           |
| ---------------- | ------ | ------------------------------------------------------------------------------------- |
| `item_id`        | string | Unique identifier for the article                                                     |
| `group_id`       | string | Group identifier (e.g., same source document)                                         |
| `source_dataset` | string | Origin dataset name                                                                   |
| `article`        | string | Full text content of the document                                                     |
| `label`          | string | Category label (`human_written`, `ai_generated`, `human_ai_polished`, `ai_humanized`) |
| `model_name`     | string | Name of the AI model (if applicable)                                                  |
| `rst_structure`  | object | Pre-parsed RST tree structure                                                         |

A sample data file is provided in `data/example/sample_data.jsonl` for reference.

### Full Preprocessing Pipeline (Recommended)

A one-command script is provided to process the Hart dataset end-to-end:

```bash
bash scripts/prepare_data.sh
```

This runs the following 4 steps automatically:

| Step           | Script                               | Input → Output                                                         |
| -------------- | ------------------------------------ | ---------------------------------------------------------------------- |
| 1. Unification | `utils/data_unification_pipeline.py` | Hart raw JSON → `data/master_grouped.jsonl`                            |
| 2. RST Parsing | `utils/rst_parser_pipeline.py`       | `master_grouped.jsonl` → `master_rst_parsed.jsonl`                     |
| 3. Flattening  | `utils/flatten_data_pipeline.py`     | `master_rst_parsed.jsonl` → `flattened_articles.jsonl`                 |
| 4. Splitting   | `utils/split_dataset.py`             | `flattened_articles.jsonl` → `hart_split/{train,val,test}_graph.jsonl` |

> **Note**: Step 2 (RST Parsing) requires the `isanlp_rst` parser. Set the `RST_PARSER_PATH` in your `.env` file before running.

### Run Steps Individually

You can also run each step separately:

```bash
# Step 1: Parse and unify Hart dataset
python utils/data_unification_pipeline.py --output data/master_grouped.jsonl

# Step 2: RST parsing (requires isanlp_rst)
python utils/rst_parser_pipeline.py --input data/master_grouped.jsonl --output data/master_rst_parsed.jsonl

# Step 3: Flatten to per-article format
python utils/flatten_data_pipeline.py --input_file data/master_rst_parsed.jsonl --output_file data/flattened_articles.jsonl

# Step 4: Split into train/val/test
python utils/split_dataset.py
```

### Expected Output Structure

```
RACE/
├── data/
│   └── hart_split/
│       ├── train_graph.jsonl
│       ├── val_graph.jsonl
│       ├── test_graph.jsonl
│       └── stats.md           # Split statistics
└── ...
```

> **Note**: You can change the data paths in `configs/RACE.json` to point to any location on your system.

## 3. Usage

### Training (Single GPU)

To train the model using the provided configuration:

```bash
bash scripts/train_single.sh configs/RACE.json
```

You can optionally specify a GPU ID (default is 0):

```bash
# Run on GPU 1
bash scripts/train_single.sh configs/RACE.json 1
```

### Evaluation

To evaluate a trained model checkpoint on the test set:

```bash
bash scripts/eval_single.sh configs/RACE.json path/to/your/checkpoint.pth
```

You can optionally specify a GPU ID (default is 0) as the third argument:

```bash
# Evaluate on GPU 1
bash scripts/eval_single.sh configs/RACE.json path/to/your/checkpoint.pth 1
```

### Multiple Seeds (Reproducibility)

To run the training loop multiple times with different random seeds:

```bash
# Run 5 experiments with random seeds
bash scripts/run_multiple_seeds.sh configs/RACE.json 5
```

## 4. Configuration

The core configuration is located in `configs/RACE.json`. Key settings include:

| Category               | Parameters                                        |
| ---------------------- | ------------------------------------------------- |
| **Data Paths**         | `data_path`, `val_data_path`, `test_data_path`    |
| **Model Backbone**     | `backbone_model_path` (e.g., `roberta-base`)      |
| **Training**           | `lr`, `batch_size`, `epochs`, `weight_decay`      |
| **GNN**                | `feature_dim`, `gnn_hidden_dim`, `num_gnn_layers` |
| **Graph Construction** | Under the `graph` key (see below)                 |
| **Loss**               | `loss.use_supcon` (`true` / `false`)              |

### Graph Configuration

The `graph` object controls RST-based graph construction:

```json
{
  "graph": {
    "graph_builder_type": "rst",
    "gnn_layer_type": "RGCN",
    "rst_use_nuclearity": false,
    "use_distinct_reverse_edges": false,
    "rgcn_num_bases": 10,
    "pooling_strategy": "root"
  }
}
```

## 5. Model Variants

| Model                     | Config `model_type` | Description                        |
| ------------------------- | ------------------- | ---------------------------------- |
| **RACEModel**             | `"gnn"` (default)   | Full RACE model with RST-based GNN |
| **FlexibleBaselineModel** | `"baseline"`        | Ablation: backbone only, no GNN    |

## 6. Directory Structure

```
RACE/
├── configs/          # Model and training configurations
├── data/             # Dataset directory (user-provided)
│   └── example/      # Sample data for reference
├── models/           # Core PyTorch model definitions
├── scripts/          # Shell scripts for training and evaluation
├── utils/            # Data loading, graph building, metrics, and data prep
├── train.py          # Main training entry point
├── test.py           # Main evaluation entry point
└── requirements.txt  # Python dependencies
```
