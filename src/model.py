import torch
from torch import nn


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
    