"""Temporal motion-history encoders."""

from typing import Optional

import torch
from torch import nn


class GRUHistoryEncoder(nn.Module):
    """Encode normalized state history, optionally exposing validity to the GRU."""

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        use_mask: bool = True,
    ) -> None:
        super().__init__()
        self.use_mask = use_mask
        input_dim = state_dim + int(use_mask)
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.projection = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

    def forward(self, history: torch.Tensor, history_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return one context embedding per batch item."""
        if self.use_mask:
            if history_mask is None:
                history_mask = torch.ones(history.shape[:2], device=history.device, dtype=history.dtype)
            history = torch.cat([history, history_mask.unsqueeze(-1).to(history.dtype)], dim=-1)
        output, _ = self.gru(history)
        final = output[:, -1]
        return self.projection(final)
