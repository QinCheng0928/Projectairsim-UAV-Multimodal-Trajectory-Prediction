"""Checkpoint persistence."""

from pathlib import Path
from typing import Any, Dict

import torch


def save_checkpoint(path: str, payload: Dict[str, Any]) -> None:
    """Write a checkpoint, creating its parent directory as needed."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)


def load_checkpoint(path: str, device: str = "cpu", weights_only: bool = True) -> Dict[str, Any]:
    """Load a checkpoint onto the selected device.
    """
    return torch.load(path, map_location=torch.device(device), weights_only=weights_only)
