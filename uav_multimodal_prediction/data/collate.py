"""Batch collation with metadata preservation."""

from typing import Any, Dict, List

import torch


def trajectory_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate tensors and retain non-tensor metadata as a list."""
    result: Dict[str, Any] = {}
    for key in batch[0]:
        values = [sample[key] for sample in batch]
        if isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values)
        else:
            result[key] = values
    return result
