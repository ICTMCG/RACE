import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (SupCon).
    Pulls together samples with the same label in the embedding space,
    while pushing apart samples with different labels.

    Reference: Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020.
    """

    def __init__(self, temperature=0.07):
        """
        Args:
            temperature (float): Temperature coefficient for the contrastive loss.
                Lower values make the loss more sensitive to differences.
        """
        super().__init__()
        self.temperature = temperature
        self.epsilon = 1e-8

    def forward(self, features, labels):
        """
        Computes the supervised contrastive loss.

        Args:
            features (Tensor): [Batch_Size, Dim] L2-normalized feature vectors.
            labels (Tensor): [Batch_Size] Category labels for each sample.
        Returns:
            loss (Tensor): Scalar contrastive loss value.
        """
        device = features.device
        batch_size = features.shape[0]

        # 1. Calculate Similarity Matrix (Cosine Similarity)
        # Features must be pre-normalized (F.normalize); otherwise, normalization is needed here.
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # 2. Construct Base Masks (N x N)
        # Diagonal mask (exclude self-contrast)
        mask_diag = torch.eye(batch_size, device=device).bool()

        # Reshape labels to [N, 1] for broadcasting
        labels = labels.view(-1, 1)

        # Positive mask: same label, excluding self
        pos_mask = torch.eq(labels, labels.T) & (~mask_diag)

        # Denominator mask: all non-self samples
        mask_denominator = ~mask_diag

        # 3. Compute supervised contrastive loss
        loss = self._compute_sup_con_loss(similarity_matrix, pos_mask, mask_denominator)

        return loss

    def _compute_sup_con_loss(self, logits, pos_mask, den_mask):
        """
        General helper function to compute Supervised Contrastive Loss.

        Args:
            logits: [N, N] Similarity matrix (scaled by temperature).
            pos_mask: [N, N] Boolean mask where True indicates a positive sample pair.
            den_mask: [N, N] Boolean mask for samples included in the denominator
                      (usually excludes diagonal / self-contrast).
        Returns:
            loss (Tensor): Scalar loss value.
        """
        # Subtract the max value per row for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # 1. Compute Denominator Term (Log-Sum-Exp over denominator set)
        # Only compute exp where den_mask is True; others are set to 0 (effectively ignored in sum)
        exp_logits = torch.exp(logits) * den_mask.float()
        log_prob_denom = torch.log(exp_logits.sum(1, keepdim=True) + self.epsilon)

        # 2. Compute Numerator Term (Log-Prob over positive set)
        # log(exp(sim) / sum(exp(sim))) = sim - log_sum_exp
        log_probs = logits - log_prob_denom

        # 3. Aggregate Loss
        # Only take log_prob at positive sample positions
        # Formula: - (1/|P(i)|) * sum_{p in P(i)} log_prob(i, p)

        # Count how many positive pairs exist for each anchor
        pos_counts = pos_mask.sum(1, keepdim=True)

        # Mask summation: only add log_prob of positive samples
        log_probs_sum = (log_probs * pos_mask.float()).sum(1, keepdim=True)

        # Avoid division by zero (loss is set to 0 for anchors with no positive pairs)
        # This might happen if the batch size is small or a specific label has only one sample.
        loss = -(log_probs_sum / (pos_counts + self.epsilon))

        # Average only over anchors that actually have positive samples
        has_pos = (pos_counts > 0).float()
        final_loss = (loss * has_pos).sum() / (has_pos.sum() + self.epsilon)

        return final_loss
