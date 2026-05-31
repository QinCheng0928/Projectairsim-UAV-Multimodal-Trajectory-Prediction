"""Metrics for multi-modal 3D trajectory forecasts."""

from typing import Dict, Optional

import torch


def trajectory_metrics(
    trajectories: torch.Tensor,
    target: torch.Tensor,
    scores: Optional[torch.Tensor] = None,
    miss_threshold: float = 2.0,
) -> Dict[str, torch.Tensor]:
    """Compute minADE/FDE, miss rate, and highest-scored mode errors."""
    distances = torch.linalg.vector_norm(trajectories - target.unsqueeze(1), dim=-1)
    ade = distances.mean(dim=-1)
    fde = distances[:, :, -1]
    if scores is None:
        top1_index = torch.zeros(trajectories.shape[0], dtype=torch.long, device=trajectories.device)
    else:
        top1_index = scores.argmax(dim=-1)
    batch_index = torch.arange(trajectories.shape[0], device=trajectories.device)
    min_fde = fde.min(dim=-1).values
    return {
        "minADE": ade.min(dim=-1).values.mean(),
        "minFDE": min_fde.mean(),
        "MR": (min_fde > miss_threshold).float().mean(),
        "top1_ADE": ade[batch_index, top1_index].mean(),
        "top1_FDE": fde[batch_index, top1_index].mean(),
    }


def average_nll_placeholder(*_: torch.Tensor) -> None:
    """Reserved hook for future calibrated probabilistic likelihood metrics."""
    return None
