# RACE: Rhetorical Analysis for Creator-Editor Modeling

[![arXiv](https://img.shields.io/badge/arXiv-2604.04932-b31b1b.svg)](https://arxiv.org/abs/2604.04932)
[![ACL-2026](https://img.shields.io/badge/ACL-2026--Main-red.svg)](https://aclanthology.org/2026.acl-long.235/)

This repository contains the implementation of **RACE** (_Beyond the Final Actor: Modeling the Dual Roles of Creator and Editor for Fine-Grained LLM-Generated Text Detection_), which is accepted by **ACL 2026 Main Conference**.

> We are currently cleaning up the codebase, and there might be some omissions during this process. Please feel free to raise an issue if you meet any problems when using this code.

RACE leverages Rhetorical Structure Theory (RST) to construct logical graphs from documents, then applies Relational Graph Convolutional Networks (RGCN) to learn structure-aware representations for fine-grained AI-generated text detection.

## News

- 2026-7-21: Since the data partitioning is also random, we have uploaded both the data partition used in the paper and the model weights used in the experiments; these can be downloaded via Google Drive: [data-and-model](https://drive.google.com/drive/u/1/folders/1goJ9F5uXhBTabURr9q23KhbPxiYNFK-8)

## 1. Installation

### Requirements

- **Python**: 3.8+
- **PyTorch & PyTorch Geometric**:
  Please install these first, following the official guides for your CUDA version to ensure GPU support:
  - [PyTorch](https://pytorch.org/get-started/locally/)
  - [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
- [isanlp_rst](https://github.com/tchewik/isanlp_rst)

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

1. Place the [HART dataset](https://github.com/baoguangsheng/truth-mirror) into the `data/` directory to form the `data/truth-mirror/` structure.
2. Run the one-command script to process it end-to-end and generate the training-ready `data/hart_split/` directory:

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

> **Note**: Step 2 (RST Parsing) requires the [`isanlp_rst`](https://github.com/tchewik/isanlp_rst) parser. Set the `RST_PARSER_PATH` in your `.env` file before running.

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
│   ├── truth-mirror/          # Raw dataset from HART
│   │   ├── benchmark/
│   │   │   ├── hart/
│   │   │   └── raid/
│   │   └── assets/
│   └── hart_split/            # Generated training-ready graphs
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

## 7. Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{li-etal-2026-beyond-final,
    title = "Beyond the Final Actor: Modeling the Dual Roles of Creator and Editor for Fine-Grained {LLM}-Generated Text Detection",
    author = "Li, Yang  and
      Sheng, Qiang  and
      Wang, Zhengjia  and
      Yang, Yehan  and
      Wang, Danding  and
      Cao, Juan",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-long.235/",
    pages = "5188--5203",
    ISBN = "979-8-89176-390-6"
}
```
