import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


def compute_classification_metrics(preds: torch.Tensor, labels: torch.Tensor) -> dict:
    """
    Computes multi-class classification metrics, including AUROC and TPR @ 1% FPR.

    Args:
        preds (torch.Tensor): The raw logits output from the model. Shape: [batch_size, num_classes].
        labels (torch.Tensor): The ground truth labels. Shape: [batch_size].

    Returns:
        dict: A dictionary containing accuracy, macro precision, recall, f1-score, AUROC, and TPR @ 1% FPR.
    """
    # --- Standard Metrics ---
    pred_indices = torch.argmax(preds, dim=1).cpu().numpy()
    labels_np = labels.cpu().numpy()

    accuracy = accuracy_score(labels_np, pred_indices)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels_np, pred_indices, average="macro", zero_division=0
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        labels_np, pred_indices, average="micro", zero_division=0
    )

    metrics = {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,
    }

    # --- Per-class Metrics ---
    # Calculate metrics for each class individually
    p_class, r_class, f_class, _ = precision_recall_fscore_support(
        labels_np, pred_indices, average=None, zero_division=0
    )

    for i in range(len(p_class)):
        metrics[f"precision_class_{i}"] = p_class[i]
        metrics[f"recall_class_{i}"] = r_class[i]
        metrics[f"f1_class_{i}"] = f_class[i]

    # --- Advanced Metrics (require probabilities) ---
    if preds.shape[1] > 1:  # Ensure it's a classification task with logits
        probs = torch.softmax(preds, dim=1).cpu().numpy()
        num_classes = preds.shape[1]

        # --- AUROC (One-vs-Rest) ---
        try:
            auroc = roc_auc_score(labels_np, probs, multi_class="ovr", average="macro")
            metrics["auroc_macro"] = auroc
        except ValueError as e:
            # This can happen if a class is missing in a batch
            metrics["auroc_macro"] = 0.0
            # print(f"Could not compute AUROC: {e}")

        # --- AUROC (Micro) ---
        try:
            # One-hot encode labels for micro-average calculation
            y_true_one_hot = np.eye(num_classes)[labels_np]
            auroc_micro = roc_auc_score(y_true_one_hot, probs, average="micro")
            metrics["auroc_micro"] = auroc_micro
        except ValueError as e:
            metrics["auroc_micro"] = 0.0

        # --- TPR @ 1% FPR & 5% FPR (Micro) ---
        try:
            # Flatten for micro-average
            y_true_flat = y_true_one_hot.ravel()
            probs_flat = probs.ravel()
            fpr_micro, tpr_micro, _ = roc_curve(y_true_flat, probs_flat)

            # TPR @ 1% FPR (Micro)
            target_fpr_1 = 0.01
            if np.any(fpr_micro <= target_fpr_1):
                idx = np.where(fpr_micro <= target_fpr_1)[0][-1]
                metrics["tpr_at_1_fpr_micro"] = tpr_micro[idx]
            else:
                metrics["tpr_at_1_fpr_micro"] = 0.0

            # TPR @ 5% FPR (Micro)
            target_fpr_5 = 0.05
            if np.any(fpr_micro <= target_fpr_5):
                idx = np.where(fpr_micro <= target_fpr_5)[0][-1]
                metrics["tpr_at_5_fpr_micro"] = tpr_micro[idx]
            else:
                metrics["tpr_at_5_fpr_micro"] = 0.0
        except Exception as e:
            metrics["tpr_at_1_fpr_micro"] = 0.0
            metrics["tpr_at_5_fpr_micro"] = 0.0

        # --- TPR @ 1% FPR (One-vs-Rest) ---
        tpr_at_fpr_list = []
        target_fpr = 0.01
        for i in range(num_classes):
            # Create binary labels for the current class
            y_true_binary = (labels_np == i).astype(int)

            # Skip if a class is not present in the labels
            if np.sum(y_true_binary) == 0:
                tpr_at_fpr_list.append(0.0)  # Append 0 if class not present
                continue

            y_score_binary = probs[:, i]
            fprs, tprs, _ = roc_curve(y_true_binary, y_score_binary)

            # Find the TPR at the highest FPR that is <= target_fpr
            if np.any(fprs <= target_fpr):
                # Find the index of the last FPR that is <= target_fpr
                idx = np.where(fprs <= target_fpr)[0][-1]
                tpr_at_fpr_list.append(tprs[idx])
            else:
                tpr_at_fpr_list.append(0.0)  # No point meets the criteria

        # Store per-class TPR values
        for i, tpr_val in enumerate(tpr_at_fpr_list):
            metrics[f"tpr_at_1_fpr_class_{i}"] = tpr_val

        if tpr_at_fpr_list:
            metrics["tpr_at_1_fpr_macro"] = np.mean(tpr_at_fpr_list)
        else:
            metrics["tpr_at_1_fpr_macro"] = 0.0

        # --- TPR @ 5% FPR (One-vs-Rest) ---
        tpr_at_fpr_5_list = []
        target_fpr_5 = 0.05
        for i in range(num_classes):
            # Create binary labels for the current class
            y_true_binary = (labels_np == i).astype(int)

            # Skip if a class is not present in the labels
            if np.sum(y_true_binary) == 0:
                tpr_at_fpr_5_list.append(0.0)
                continue

            y_score_binary = probs[:, i]
            fprs, tprs, _ = roc_curve(y_true_binary, y_score_binary)

            # Find the TPR at the highest FPR that is <= target_fpr_5
            if np.any(fprs <= target_fpr_5):
                idx = np.where(fprs <= target_fpr_5)[0][-1]
                tpr_at_fpr_5_list.append(tprs[idx])
            else:
                tpr_at_fpr_5_list.append(0.0)

        # Store per-class TPR values
        for i, tpr_val in enumerate(tpr_at_fpr_5_list):
            metrics[f"tpr_at_5_fpr_class_{i}"] = tpr_val

        if tpr_at_fpr_5_list:
            metrics["tpr_at_5_fpr_macro"] = np.mean(tpr_at_fpr_5_list)
        else:
            metrics["tpr_at_5_fpr_macro"] = 0.0

    classification_report = ""
    for k, v in metrics.items():
        classification_report += f"{k}: {v:.4f}\n"
    metrics["classification_report"] = classification_report

    return metrics


if __name__ == "__main__":
    # --- DEMONSTRATION ---
    print("--- Demonstrating metric computation ---")

    # Example for a 3-class classification problem
    NUM_SAMPLES = 100
    NUM_CLASSES = 3

    # Dummy model outputs (logits)
    dummy_logits = torch.randn(NUM_SAMPLES, NUM_CLASSES)

    # Dummy ground truth labels
    dummy_labels = torch.randint(0, NUM_CLASSES, (NUM_SAMPLES,))

    # Make some predictions correct to get non-zero metrics
    for i in range(NUM_SAMPLES):
        # Make the correct class have a higher logit
        if torch.rand(1) > 0.3:  # 70% accuracy
            dummy_logits[i, dummy_labels[i]] += 5

    print(f"Sample logits shape: {dummy_logits.shape}")
    print(f"Sample labels shape: {dummy_labels.shape}")

    # Compute metrics
    calculated_metrics = compute_classification_metrics(dummy_logits, dummy_labels)

    print("\n--- Calculated Metrics ---")
    for name, value in calculated_metrics.items():
        if isinstance(value, float):
            print(f"{name}: {value:.4f}")
        else:
            print(f"{name}: {value}")
    print("-" * 20)
