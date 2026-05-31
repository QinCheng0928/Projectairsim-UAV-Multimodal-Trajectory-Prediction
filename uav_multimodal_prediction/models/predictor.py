"""Dense-goal inspired multi-modal 3D UAV trajectory predictor."""

from typing import Dict, Optional

import torch
from torch import nn

from uav_multimodal_prediction.config import ModelConfig
from uav_multimodal_prediction.utils.geometry import to_absolute

from .encoder import GRUHistoryEncoder


class UAVDenseGoalPredictor(nn.Module):
    """Predict K endpoint-conditioned future displacement trajectories.

    The model's native trajectory and goal outputs are displacements relative
    to the latest observed position. When ``current_position`` is provided,
    absolute NED-coordinate variants are included in the output dictionary.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_dim
        self.encoder = GRUHistoryEncoder(
            config.state_dim, hidden, config.num_layers, config.dropout, config.use_mask
        )
        self.goal_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, config.num_modes * 3),
        )
        self.goal_embedding = nn.Sequential(nn.Linear(3, hidden), nn.ReLU())
        self.trajectory_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden * 2, config.future_len * 3),
        )
        self.score_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        history: torch.Tensor,
        history_mask: Optional[torch.Tensor] = None,
        current_position: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Generate score-sorted trajectory displacement hypotheses."""
        batch_size = history.shape[0]
        context = self.encoder(history, history_mask)
        goals = self.goal_head(context).view(batch_size, self.config.num_modes, 3)
        expanded_context = context.unsqueeze(1).expand(-1, self.config.num_modes, -1)
        conditioned = torch.cat([expanded_context, self.goal_embedding(goals)], dim=-1)
        # residual = self.trajectory_head(conditioned).view(
        #     batch_size, self.config.num_modes, self.config.future_len, 3
        # )
        # alpha = torch.linspace(0.0, 1.0, self.config.future_len, device=history.device, dtype=history.dtype)
        # alpha = alpha.view(1, 1, self.config.future_len, 1)
        # residual = residual - alpha * residual[:, :, -1:, :]
        # trajectories = alpha * goals.unsqueeze(-2) + residual
        raw_trajectories = self.trajectory_head(conditioned).view(
            batch_size, self.config.num_modes, self.config.future_len, 3
        )
        alpha = torch.linspace(0.0, 1.0, self.config.future_len, device=history.device, dtype=history.dtype)
        alpha = alpha.view(1, 1, self.config.future_len, 1)
        endpoint_error = raw_trajectories[:, :, -1:, :] - goals.unsqueeze(-2)
        trajectories = raw_trajectories - alpha * endpoint_error
        
        score_logits = self.score_head(conditioned).squeeze(-1)
        scores = torch.softmax(score_logits, dim=-1)

        order = torch.argsort(scores, dim=-1, descending=True)
        gather_trajectory = order[..., None, None].expand(-1, -1, self.config.future_len, 3)
        gather_goal = order[..., None].expand(-1, -1, 3)
        trajectories = torch.gather(trajectories, 1, gather_trajectory)
        goals = torch.gather(goals, 1, gather_goal)
        score_logits = torch.gather(score_logits, 1, order)
        scores = torch.gather(scores, 1, order)
        output = {
            "trajectories": trajectories,
            "scores": scores,
            "score_logits": score_logits,
            "goals": goals,
        }
        if current_position is not None:
            output["absolute_trajectories"] = to_absolute(trajectories, current_position)
            output["absolute_goals"] = goals + current_position.unsqueeze(1)
        return output


UAVMultiModalTrajectoryPredictor = UAVDenseGoalPredictor
