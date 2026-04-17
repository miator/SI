from typing import Optional

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


class FeedForwardModule(nn.Module):
    def __init__(self, d_model: int, ff_mult: int = 4, dropout: float = 0.4):
        super().__init__()
        hidden_dim = d_model * ff_mult
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StatisticsPooling(nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1)
        std = torch.sqrt(x.var(dim=1, unbiased=False).clamp_min(self.eps))
        return torch.cat([mean, std], dim=1)


class ConformerConvModule(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("conv_kernel_size must be odd for same-length padding.")

        self.norm = nn.LayerNorm(d_model)
        self.pointwise_in = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.depthwise = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()
        self.pointwise_out = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)                # (B, T, D)
        x = x.transpose(1, 2)           # (B, D, T)
        x = F.glu(self.pointwise_in(x), dim=1)
        x = self.depthwise(x)           # (B, D, T)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_out(x)
        x = self.dropout(x)
        return x.transpose(1, 2)        # (B, T, D)


class ConformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        ff_mult: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ffn1 = FeedForwardModule(d_model=d_model, ff_mult=ff_mult, dropout=dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.conv = ConformerConvModule(
            d_model=d_model,
            kernel_size=conv_kernel_size,
            dropout=dropout,
        )
        self.ffn2 = FeedForwardModule(d_model=d_model, ff_mult=ff_mult, dropout=dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ffn1(x)      # (B, T, D)

        attn_input = self.attn_norm(x)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + self.attn_dropout(attn_output)

        x = x + self.conv(x)            # (B, T, D)
        x = x + 0.5 * self.ffn2(x)      # (B, T, D)

        return self.final_norm(x)


class ConformerEmbeddingNet(nn.Module):
    """Input: x (B, T, F). Output: L2-normalized speaker embeddings (B, emb_dim)."""

    def __init__(
        self,
        n_feats: int,
        emb_dim: int = 192,
        d_model: int = 144,
        num_heads: int = 4,
        ff_mult: int = 4,
        conv_kernel_size: int = 31,
        num_blocks: int = 3,
        dropout: float = 0.4,
        max_len: int = 301,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.max_len = max_len
        self.input_proj = nn.Linear(n_feats, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model))
        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                ConformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    ff_mult=ff_mult,
                    conv_kernel_size=conv_kernel_size,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.pool_norm = nn.LayerNorm(d_model)
        self.stats_pool = StatisticsPooling()
        self.emb = nn.Linear(2 * d_model, emb_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(features)   # (B, T, D)
        if x.size(1) > self.max_len:
            raise ValueError(
                f"Sequence length {x.size(1)} exceeds conformer max_len={self.max_len}"
            )
        x = x + self.pos_emb[:, :x.size(1)]
        x = self.input_dropout(x)

        for block in self.blocks:
            x = block(x)                # (B, T, D)

        x = self.pool_norm(x)
        x = self.stats_pool(x)          # (B, 2D)

        e = self.emb(x)                 # (B, emb_dim)
        e = F.normalize(e, p=2, dim=1)  # (B, emb_dim)
        return e


def build_embedding_model(
    model_name: str,
    *,
    n_feats: int,
    emb_dim: int,
    dropout: float,
    conformer_d_model: int = 144,
    conformer_num_heads: int = 4,
    conformer_ff_mult: int = 4,
    conformer_conv_kernel_size: int = 31,
    conformer_num_blocks: int = 4,
    conformer_dropout: Optional[float] = None,
) -> nn.Module:
    model_name = model_name.lower()

    if model_name == "cnn":
        return CNN1DNET(
            n_feats=n_feats,
            emb_dim=emb_dim,
            dropout=dropout,
        )

    if model_name == "conformer":
        effective_conformer_dropout = dropout if conformer_dropout is None else conformer_dropout
        return ConformerEmbeddingNet(
            n_feats=n_feats,
            emb_dim=emb_dim,
            d_model=conformer_d_model,
            num_heads=conformer_num_heads,
            ff_mult=conformer_ff_mult,
            conv_kernel_size=conformer_conv_kernel_size,
            num_blocks=conformer_num_blocks,
            dropout=effective_conformer_dropout,
        )

    raise ValueError(f"Unsupported MODEL_NAME: {model_name}")
