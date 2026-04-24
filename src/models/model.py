from typing import Optional
import math

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


class ResidualBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)

        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        return self.relu(out + residual)


class ResCNN1DNET(nn.Module):
    """Input: x (B, T, F). Output: L2-normalized speaker embeddings (B, emb_dim)."""

    def __init__(self, n_feats: int, emb_dim: int = 192, dropout: float = 0.3):
        super().__init__()
        self.emb_dim = emb_dim

        self.stem = nn.Sequential(
            nn.Conv1d(n_feats, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.stage1 = nn.Sequential(
            ResidualBlock1D(64, 64),
            ResidualBlock1D(64, 64),
        )
        self.pool1 = nn.MaxPool1d(2)
        self.stage2 = nn.Sequential(
            ResidualBlock1D(64, 128),
            ResidualBlock1D(128, 128),
        )
        self.pool2 = nn.MaxPool1d(2)
        self.stage3 = nn.Sequential(
            ResidualBlock1D(128, 256),
            ResidualBlock1D(256, 256),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.emb = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, emb_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = features.transpose(1, 2)    # (B, F, T)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.pool1(x)
        x = self.stage2(x)
        x = self.pool2(x)
        x = self.stage3(x)
        x = self.pool(x).squeeze(-1)    # (B, 256)

        e = self.emb(x)                 # (B, emb_dim)
        e = F.normalize(e, p=2, dim=1)
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


class SEModule(nn.Module):
    def __init__(self, channels: int, bottleneck: int = 128):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(bottleneck, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x)


class ECAPARes2Block(nn.Module):
    """Res2Net TDNN block with squeeze-excitation, adapted for cached log-mels."""

    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int,
        dilation: int,
        scale: int = 8,
        se_bottleneck: int = 128,
    ):
        super().__init__()
        if channels % scale != 0:
            raise ValueError("ecapa_channels must be divisible by ecapa_scale.")

        width = channels // scale
        padding = math.floor(kernel_size / 2) * dilation
        self.width = width
        self.num_branches = scale - 1

        self.conv1 = nn.Conv1d(channels, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    width,
                    width,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=padding,
                )
                for _ in range(self.num_branches)
            ]
        )
        self.bns = nn.ModuleList([nn.BatchNorm1d(width) for _ in range(self.num_branches)])
        self.conv3 = nn.Conv1d(width * scale, channels, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.se = SEModule(channels, bottleneck=se_bottleneck)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        splits = torch.split(out, self.width, dim=1)
        branch_outputs = []
        running = None
        for idx in range(self.num_branches):
            running = splits[idx] if running is None else running + splits[idx]
            running = self.convs[idx](running)
            running = self.relu(running)
            running = self.bns[idx](running)
            branch_outputs.append(running)

        out = torch.cat([*branch_outputs, splits[self.num_branches]], dim=1)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)
        out = self.se(out)
        return out + residual


class AttentiveStatisticsPooling(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        attention_channels: int = 256,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.eps = eps
        self.attention = nn.Sequential(
            nn.Conv1d(3 * channels, attention_channels, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(attention_channels),
            nn.Tanh(),
            nn.Conv1d(attention_channels, channels, kernel_size=1),
            nn.Softmax(dim=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time_steps = x.size(2)
        mean = x.mean(dim=2, keepdim=True).repeat(1, 1, time_steps)
        std = torch.sqrt(x.var(dim=2, keepdim=True, unbiased=False).clamp_min(self.eps))
        std = std.repeat(1, 1, time_steps)

        weights = self.attention(torch.cat([x, mean, std], dim=1))
        pooled_mean = torch.sum(x * weights, dim=2)
        pooled_std = torch.sqrt(
            (torch.sum((x ** 2) * weights, dim=2) - pooled_mean ** 2).clamp_min(self.eps)
        )
        return torch.cat([pooled_mean, pooled_std], dim=1)


class ECAPATDNNEmbeddingNet(nn.Module):
    """ECAPA-TDNN encoder for precomputed log-mel features shaped (B, T, F)."""

    def __init__(
        self,
        n_feats: int,
        emb_dim: int = 192,
        channels: int = 512,
        mfa_channels: int = 1536,
        attention_channels: int = 256,
        scale: int = 8,
        se_bottleneck: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.conv1 = nn.Conv1d(n_feats, channels, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(channels)

        self.layer1 = ECAPARes2Block(
            channels,
            kernel_size=3,
            dilation=2,
            scale=scale,
            se_bottleneck=se_bottleneck,
        )
        self.layer2 = ECAPARes2Block(
            channels,
            kernel_size=3,
            dilation=3,
            scale=scale,
            se_bottleneck=se_bottleneck,
        )
        self.layer3 = ECAPARes2Block(
            channels,
            kernel_size=3,
            dilation=4,
            scale=scale,
            se_bottleneck=se_bottleneck,
        )
        self.mfa = nn.Conv1d(3 * channels, mfa_channels, kernel_size=1)
        self.pool = AttentiveStatisticsPooling(
            mfa_channels,
            attention_channels=attention_channels,
        )
        self.bn_pool = nn.BatchNorm1d(2 * mfa_channels)
        self.dropout = nn.Dropout(dropout)
        self.emb = nn.Linear(2 * mfa_channels, emb_dim)
        self.bn_emb = nn.BatchNorm1d(emb_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = features.transpose(1, 2)    # (B, F, T)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)

        x = self.mfa(torch.cat([x1, x2, x3], dim=1))
        x = self.relu(x)
        x = self.pool(x)
        x = self.bn_pool(x)
        x = self.dropout(x)

        e = self.emb(x)
        e = self.bn_emb(e)
        e = F.normalize(e, p=2, dim=1)
        return e


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
    ecapa_channels: int = 512,
    ecapa_mfa_channels: int = 1536,
    ecapa_attention_channels: int = 256,
    ecapa_scale: int = 8,
    ecapa_se_bottleneck: int = 128,
    ecapa_dropout: Optional[float] = None,
) -> nn.Module:
    model_name = model_name.lower().replace("-", "_")

    if model_name == "cnn":
        return CNN1DNET(
            n_feats=n_feats,
            emb_dim=emb_dim,
            dropout=dropout,
        )

    if model_name == "rescnn":
        return ResCNN1DNET(
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

    if model_name in {"ecapa", "ecapa_tdnn"}:
        effective_ecapa_dropout = dropout if ecapa_dropout is None else ecapa_dropout
        return ECAPATDNNEmbeddingNet(
            n_feats=n_feats,
            emb_dim=emb_dim,
            channels=ecapa_channels,
            mfa_channels=ecapa_mfa_channels,
            attention_channels=ecapa_attention_channels,
            scale=ecapa_scale,
            se_bottleneck=ecapa_se_bottleneck,
            dropout=effective_ecapa_dropout,
        )

    raise ValueError(f"Unsupported MODEL_NAME: {model_name}")
