import torch
from torch import nn
import torch.nn.functional as F


class FeedForwardNet(nn.Module):
    def __init__(self, n_mfcc: int, max_frames: int, num_classes: int):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(n_mfcc * max_frames, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        x = self.flatten(x)
        return self.net(x)


class CNN1DNET(nn.Module):
    """Input: x(B, T, F) where F = n_mels (or n_mfcc).
        Treat bins as channels: (B, F, T) for Conv1d."""
    def __init__(self, n_feats: int, num_classes: int, emb_dim: int = 192, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_feats, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)  # global average over time
        self.emb = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, emb_dim)
        )
        # classifier head (used for training with CrossEntropyLoss)
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> torch.Tensor:
        x = x.transpose(1, 2)   # (B, F, T)
        x = self.features(x)               # (B, C, T') C=256
        x = self.pool(x).squeeze(-1)       # (B, C) C=256

        e = self.emb(x)                    # (B, emb_dim)
        e = F.normalize(e, p=2, dim=1)     # L2 normalize for cosine

        if return_embedding:
            return e

        logits = self.classifier(e)        # (B, num_classes)
        return logits
