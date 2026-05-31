"""Feature standardization for UAV motion histories."""

from typing import Any, Dict, Iterable

import torch


class StateNormalizer:
    """Standardize input history state vectors using training-set statistics."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-6) -> None:
        self.mean = mean.float()
        self.std = std.float().clamp_min(eps)
        self.eps = eps

    @classmethod
    def fit(cls, histories: Iterable[torch.Tensor], state_dim: int = 16) -> "StateNormalizer":
        """Compute stable per-feature statistics from a stream of histories."""
        count = 0
        total = torch.zeros(state_dim, dtype=torch.float64)
        squared = torch.zeros(state_dim, dtype=torch.float64)
        for history in histories:
            values = history.reshape(-1, state_dim).double()
            count += values.shape[0]
            total += values.sum(dim=0)
            squared += (values * values).sum(dim=0)
        if count == 0:
            raise ValueError("Cannot fit StateNormalizer on an empty dataset.")
        mean = total / count
        variance = squared / count - mean * mean
        return cls(mean.float(), variance.clamp_min(0.0).sqrt().float())

    @classmethod
    def fit_dataset(cls, dataset: Any, state_dim: int = 16) -> "StateNormalizer":
        """Fit from any dataset yielding a ``history`` tensor."""
        return cls.fit((dataset[index]["history"] for index in range(len(dataset))), state_dim)

    def transform(self, history: torch.Tensor) -> torch.Tensor:
        """Normalize history features."""
        return (history - self.mean.to(history.device)) / self.std.to(history.device)

    def inverse_transform(self, history: torch.Tensor) -> torch.Tensor:
        """Undo history normalization."""
        return history * self.std.to(history.device) + self.mean.to(history.device)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Return serializable statistics."""
        return {"mean": self.mean.cpu(), "std": self.std.cpu()}

    @classmethod
    def from_state_dict(cls, state: Dict[str, torch.Tensor]) -> "StateNormalizer":
        """Restore normalizer statistics from a checkpoint."""
        return cls(state["mean"], state["std"])
