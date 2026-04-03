import argparse
import json
import logging
import os

import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

# Local imports
from models.flexible_model import RACEModel, FlexibleBaselineModel
from utils.flexible_dataset import FlexibleGraphDataset
from utils.metrics import compute_classification_metrics


def get_args_parser():
    """
    Parses command-line arguments for the testing script.
    """
    parser = argparse.ArgumentParser("RACE Testing Script", add_help=False)

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the JSON configuration file for testing.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Full path to the trained model checkpoint.",
    )
    # Optional override for the number of workers, can also be in the config.
    parser.add_argument("--num_workers", default=None, type=int)

    return parser


def main(config):
    """
    Main function to run the evaluation.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # --- Setup ---
    if not os.path.exists(config.checkpoint_path):
        logger.error(f"Checkpoint not found at {config.checkpoint_path}. Aborting.")
        return

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Load Configuration from Checkpoint ---
    checkpoint = torch.load(config.checkpoint_path, map_location="cpu")
    train_args = checkpoint.get("args")  # This is a dict of the original training args
    train_graph_config = checkpoint.get("config")  # This is the graph_config dict

    if not train_args or not train_graph_config:
        logger.error(
            "Training arguments ('args') or graph config ('config') not found in checkpoint."
        )
        return

    logger.info("Loading model and data configuration from checkpoint...")

    # The test config file can provide a 'graph' object for overrides
    graph_config_from_test = config.graph if hasattr(config, "graph") else {}
    # Merge, with test-time config taking precedence
    final_graph_config = {**train_graph_config, **graph_config_from_test}
    logger.info(
        f"Final Graph Configuration: {json.dumps(final_graph_config, indent=2)}"
    )

    # --- Data Loading ---
    if not hasattr(config, "test_data_path"):
        logger.error("`test_data_path` not specified in the test config file.")
        return
    test_data_path = config.test_data_path

    dataset_test = FlexibleGraphDataset(
        file_path=test_data_path, config=final_graph_config
    )
    sampler_test = SequentialSampler(dataset_test)
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=getattr(config, "batch_size", 16),
        sampler=sampler_test,
        collate_fn=FlexibleGraphDataset.collate_fn,
        num_workers=getattr(config, "num_workers", 0),
    )
    logger.info(f"Test data loaded from: {test_data_path}")

    # --- Model Preparation ---
    # Load metadata from the checkpoint to ensure consistency
    metadata = checkpoint.get("metadata")
    if not metadata:
        logger.error("Metadata not found in checkpoint. Aborting.")
        return

    logger.info("Metadata loaded from checkpoint.")

    model_type = train_args.get("model_type", "gnn")
    if model_type == "baseline":
        logger.info("Initializing FlexibleBaselineModel.")
        model = FlexibleBaselineModel(
            feature_dim=train_args["feature_dim"],
            gnn_hidden_dim=train_args["gnn_hidden_dim"],
            num_heads=train_args["num_heads"],
            num_classes=train_args["num_classes"],
            config=final_graph_config,
            metadata=metadata,
            output_features=True,
        )
    else:
        logger.info(
            f"Initializing RACEModel for model_type='{model_type}'."
        )
        model = RACEModel(
            feature_dim=train_args["feature_dim"],
            gnn_hidden_dim=train_args["gnn_hidden_dim"],
            num_heads=train_args["num_heads"],
            num_classes=train_args["num_classes"],
            config=final_graph_config,
            metadata=metadata,
            output_features=True,
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info(f"Model loaded successfully from {config.checkpoint_path}")

    # --- Evaluation Loop ---
    all_logits = []
    all_labels = []
    all_features = []
    all_ids = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader_test, desc="Evaluating")):
            outputs = model(batch, batch_idx=batch_idx)

            if isinstance(outputs, dict):
                logits = outputs["logits"]
                if "features" in outputs:
                    all_features.append(outputs["features"].cpu())
            else:
                logits = outputs

            all_logits.append(logits.cpu())
            all_labels.append(batch["labels"].cpu())
            all_ids.extend(batch["id"])

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    final_features = torch.cat(all_features) if all_features else None

    # --- Results ---
    test_metrics = compute_classification_metrics(all_logits, all_labels)
    formatted_metrics = {
        k: f"{v:.4f}" for k, v in test_metrics.items() if k != "classification_report"
    }
    logger.info(
        f"--- Test Results --- \n{test_metrics.get('classification_report', '')}"
    )
    logger.info(f"Summary Metrics: {json.dumps(formatted_metrics, indent=2)}")

    # --- Save Predictions & Metrics ---
    output_dir = os.path.dirname(config.checkpoint_path)

    predictions = []
    for i, group_id in enumerate(all_ids):
        pred_entry = {
            "id": group_id,
            "logits": all_logits[i].tolist(),
            "prediction": torch.argmax(all_logits[i]).item(),
        }
        if final_features is not None:
            pred_entry["features"] = final_features[i].tolist()
        predictions.append(pred_entry)

    predictions_path = os.path.join(output_dir, "test_predictions.json")
    metrics_path = os.path.join(output_dir, "test_metrics.json")

    with open(predictions_path, "w") as f:
        json.dump(predictions, f, indent=2)
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)

    logger.info(f"Saved test predictions to {predictions_path}")
    logger.info(f"Saved test metrics to {metrics_path}")


if __name__ == "__main__":
    parser = get_args_parser()
    cli_args = parser.parse_args()

    # Load config from the specified JSON file
    with open(cli_args.config, "r") as f:
        config_dict = json.load(f)

    # Create a namespace object for easy access
    config = argparse.Namespace(**config_dict)

    # Add/override config with CLI arguments
    config.checkpoint_path = cli_args.checkpoint_path
    if cli_args.num_workers is not None:
        config.num_workers = cli_args.num_workers
    if not hasattr(config, "device"):
        config.device = "cuda"

    main(config)
