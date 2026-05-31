"""Best-of-K trajectory learning objectives."""

from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn


class BestOfKTrajectoryLoss(nn.Module):
    """Winner-takes-all regression plus mode classification and endpoint loss."""

    def __init__(self, lambda_cls: float = 0.2, lambda_goal: float = 1.0, miss_threshold: float = 2.0) -> None:
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_goal = lambda_goal
        self.miss_threshold = miss_threshold

    def forward(self, prediction: Dict[str, torch.Tensor], target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Calculate loss and detached-friendly training metrics."""
        trajectories = prediction["trajectories"]
        distances = torch.linalg.vector_norm(trajectories - target.unsqueeze(1), dim=-1)
        ade_per_mode = distances.mean(dim=-1)
        fde_per_mode = distances[:, :, -1]
        best_index = ade_per_mode.argmin(dim=-1)
        batch_index = torch.arange(target.shape[0], device=target.device)
        best_trajectory = trajectories[batch_index, best_index]
        best_goal = prediction["goals"][batch_index, best_index]
        reg_loss = F.smooth_l1_loss(best_trajectory, target)
        cls_loss = F.cross_entropy(prediction["score_logits"], best_index)
        goal_loss = torch.linalg.vector_norm(best_goal - target[:, -1], dim=-1).mean()
        min_ade = ade_per_mode[batch_index, best_index].mean()
        min_fde = fde_per_mode.min(dim=-1).values.mean()
        miss_rate = (fde_per_mode.min(dim=-1).values > self.miss_threshold).float().mean()
        loss = reg_loss + self.lambda_cls * cls_loss + self.lambda_goal * goal_loss
        return {
            "loss": loss,
            "reg_loss": reg_loss,
            "cls_loss": cls_loss,
            "goal_loss": goal_loss,
            "minADE": min_ade,
            "minFDE": min_fde,
            "MR": miss_rate,
        }
