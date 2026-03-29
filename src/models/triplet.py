# triplet.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchHardTripletLoss(nn.Module):
    """
    Batch-hard triplet loss (anchor is each sample in batch).
    Requires >=2 samples per class in the batch to create positives.

    embeddings: (B, D)
    labels: (B,)
    Uses cosine distance: d = 1 - cos
    """
    def __init__(self, margin: float = 0.2, normalize: bool = True, eps: float = 1e-12):
        super().__init__()
        self.margin = margin
        self.normalize = normalize
        self.eps = eps

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be (B, D), got {tuple(embeddings.shape)}")
        if labels.ndim != 1:
            labels = labels.view(-1)
        if labels.shape[0] != embeddings.shape[0]:
            raise ValueError("labels length must equal batch size")

        B = embeddings.size(0)
        if B < 2:
            return embeddings.sum() * 0.0

        x = embeddings
        if self.normalize:
            x = F.normalize(x, p=2, dim=1, eps=self.eps)

        sim = x @ x.t()
        dist = 1.0 - sim  # (B, B)

        labels = labels.to(device=embeddings.device)
        same = labels.unsqueeze(0) == labels.unsqueeze(1)
        eye = torch.eye(B, device=embeddings.device, dtype=torch.bool)
        pos_mask = same & ~eye
        neg_mask = ~same

        pos_dist = dist.masked_fill(~pos_mask, float("-inf"))
        neg_dist = dist.masked_fill(~neg_mask, float("inf"))

        hardest_pos, _ = pos_dist.max(dim=1)
        hardest_neg, _ = neg_dist.min(dim=1)

        valid = pos_mask.any(dim=1) & neg_mask.any(dim=1)
        if not valid.any():
            return embeddings.sum() * 0.0

        loss = F.relu(hardest_pos[valid] - hardest_neg[valid] + self.margin)
        return loss.mean()
