"""Geometry helpers for displacement and absolute NED trajectories."""

import torch


def to_displacements(positions: torch.Tensor, current_position: torch.Tensor) -> torch.Tensor:
    """Convert absolute positions to offsets from the latest observed position."""
    return positions - current_position.unsqueeze(-2)


def to_absolute(displacements: torch.Tensor, current_position: torch.Tensor) -> torch.Tensor:
    """Convert future offsets back to absolute positions."""
    if displacements.ndim == current_position.ndim + 2:
        current_position = current_position.unsqueeze(-2)
    return displacements + current_position.unsqueeze(-2)
