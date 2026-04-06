import torch
from torch import nn
import torch.nn.functional as F


class CNN1DNET(nn.Module):
    """Input: x (B, T, F), where B = batch size, T = time steps(frames), F = feature dimension.
    Treat feature bins as channels by transposing to (B, F, T) for Conv1d."""
    def __init__(self, n_feats: int, emb_dim: int = 192, dropout: float = 0.3):
        super().__init__()
        self.emb_dim = emb_dim
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

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.emb = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, emb_dim)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = features.transpose(1, 2)   # (B, F, T)
        features = self.features(features)               # (B, C, T') C=256
        features = self.pool(features).squeeze(-1)       # (B, C) C=256

        e = self.emb(features)                    # (B, emb_dim)
        e = F.normalize(e, p=2, dim=1)     # L2 normalize for cosine

        return e
