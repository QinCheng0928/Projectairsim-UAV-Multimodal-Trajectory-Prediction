"""Cold-start history storage for immediate online prediction."""

from typing import Any, Dict, Optional, Tuple

import numpy as np


STATE_KEYS = ("position", "orientation", "linear_velocity", "angular_velocity", "linear_acceleration")


def state_dict_to_vector(state: Dict[str, Any]) -> np.ndarray:
    """Flatten a unified motion-state dictionary into the 16-D model vector."""
    parts = [
        np.asarray(state.get("position", [0.0, 0.0, 0.0]), dtype=np.float32),
        np.asarray(state.get("orientation", [0.0, 0.0, 0.0, 1.0]), dtype=np.float32),
        np.asarray(state.get("linear_velocity", [0.0, 0.0, 0.0]), dtype=np.float32),
        np.asarray(state.get("angular_velocity", [0.0, 0.0, 0.0]), dtype=np.float32),
        np.asarray(state.get("linear_acceleration", [0.0, 0.0, 0.0]), dtype=np.float32),
    ]
    return np.concatenate(parts, axis=0)


class HistoryBuffer:
    """Maintain virtual warm-up observations followed by real sliding history."""

    def __init__(self, history_len: int, state_dim: int = 16) -> None:
        self.history_len = history_len
        self.state_dim = state_dim
        self._history = np.zeros((history_len, state_dim), dtype=np.float32)
        self._mask = np.zeros(history_len, dtype=np.float32)
        self.real_count = 0

    def reset(self, start_state: Dict[str, Any]) -> None:
        """Initialize all frames from a stationary virtual state at the start pose."""
        virtual_state = {
            "position": start_state.get("position", [0.0, 0.0, 0.0]),
            "orientation": start_state.get("orientation", [0.0, 0.0, 0.0, 1.0]),
            "linear_velocity": [0.0, 0.0, 0.0],
            "angular_velocity": [0.0, 0.0, 0.0],
            "linear_acceleration": [0.0, 0.0, 0.0],
        }
        vector = state_dict_to_vector(virtual_state)
        if vector.shape[0] != self.state_dim:
            raise ValueError(f"State vector has {vector.shape[0]} dimensions, expected {self.state_dim}.")
        self._history[:] = vector
        self._mask[:] = 0.0
        self.real_count = 0

    def update(self, real_state: Dict[str, Any]) -> None:
        """Append one real state, replacing the oldest virtual or real observation."""
        vector = state_dict_to_vector(real_state)
        self._history[:-1] = self._history[1:]
        self._history[-1] = vector
        self._mask[:-1] = self._mask[1:]
        self._mask[-1] = 1.0
        self.real_count = min(self.real_count + 1, self.history_len)

    @property
    def is_fully_warmed(self) -> bool:
        """Whether all frames in the window are real observations."""
        return self.real_count >= self.history_len

    def get_history(self, return_mask: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Return a copy of buffered states and, optionally, the validity mask."""
        mask = self._mask.copy() if return_mask else None
        return self._history.copy(), mask
