"""LSTM forecaster: last hidden state + temporal mean pool -> 24-step vector."""

from __future__ import annotations

import torch
import torch.nn as nn

from src import config


class LSTMForecaster(nn.Module):
    def __init__(
        self,
        in_dim: int = config.FEATURE_DIM,
        hidden: int = config.HIDDEN_SIZE,
        layers: int = config.LSTM_LAYERS,
        horizon: int = config.HORIZON,
        dropout: float = config.DROPOUT,
        use_sequence_pool: bool = config.USE_SEQUENCE_POOL,
    ):
        super().__init__()
        self.use_sequence_pool = use_sequence_pool
        lstm_dropout = dropout if layers > 1 else 0.0
        self.lstm = nn.LSTM(
            in_dim,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        head_in = hidden * (2 if use_sequence_pool else 1)
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, (h_n, _) = self.lstm(x)
        last = h_n[-1]
        if self.use_sequence_pool:
            pooled = lstm_out.mean(dim=1)
            fused = torch.cat([last, pooled], dim=1)
            return self.head(fused)
        return self.head(last)
