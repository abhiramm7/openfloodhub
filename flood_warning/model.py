"""Two-branch 1D CNN for hourly streamflow forecasting.

Architecture:
  past branch:   3 channels × 24 hours -> 1D convs -> features
  future branch: 1 channel × 12 hours -> 1D convs -> features
  concat -> FC -> 12-step flow forecast (normalized space)

Small enough to train per-gauge in minutes on CPU. Large enough to capture
storm response patterns when given hourly precip.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FloodCNN(nn.Module):
    def __init__(self, n_past_features: int = 4, n_future_features: int = 1,
                 past_steps: int = 24, future_steps: int = 12,
                 hidden: int = 32, kernel: int = 3, dropout: float = 0.1):
        super().__init__()
        pad = kernel // 2

        # Past stream: (B, 3, 24) -> (B, hidden, 24) -> pool -> (B, hidden, 12)
        self.past_cnn = nn.Sequential(
            nn.Conv1d(n_past_features, hidden, kernel_size=kernel, padding=pad),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=kernel, padding=pad),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=kernel, padding=pad),
            nn.ReLU(),
        )

        # Future precip stream: (B, 1, 12) -> (B, hidden, 12)
        self.future_cnn = nn.Sequential(
            nn.Conv1d(n_future_features, hidden, kernel_size=kernel, padding=pad),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=kernel, padding=pad),
            nn.ReLU(),
        )

        # Heads
        feat_size = hidden * past_steps + hidden * future_steps
        self.head = nn.Sequential(
            nn.Linear(feat_size, hidden * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, future_steps),
        )

    def forward(self, past: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        h_past = self.past_cnn(past).flatten(1)
        h_fut = self.future_cnn(future).flatten(1)
        z = torch.cat([h_past, h_fut], dim=1)
        return self.head(z)


def nse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Negative Nash-Sutcliffe efficiency, averaged across the batch.
    Operates in the model's normalized space — works fine as a minimization
    target since translation/scale of the standardized series is consistent.
    """
    mean_t = target.mean(dim=1, keepdim=True)
    num = ((pred - target) ** 2).sum(dim=1)
    den = ((target - mean_t) ** 2).sum(dim=1) + 1e-6
    return (num / den).mean()
