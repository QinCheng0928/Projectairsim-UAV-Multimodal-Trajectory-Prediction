"""Dataset adapter for fixed-format ProjectAirSim UAV trajectory episodes."""

from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from uav_multimodal_prediction.config import DataConfig
from uav_multimodal_prediction.utils.logging import get_logger

LOGGER = get_logger(__name__)
RAW_STATE_DIM = 17


def describe_dataset(dataset: Any) -> None:
    """Log available splits, features, and one sample's visible schema."""
    split_names = list(dataset.keys())
    LOGGER.info("Dataset splits: %s", split_names)
    for split_name in split_names:
        split = dataset[split_name]
        features = getattr(split, "features", None)
        LOGGER.info("Split '%s': rows=%s, features=%s", split_name, len(split), features)
        if len(split):
            LOGGER.info("Split '%s' first sample keys/types: %s", split_name, _schema_summary(split[0]))


def select_dataset_splits(dataset: Any, validation_ratio: float, seed: int) -> Tuple[Any, Any, Optional[Any]]:
    """Return train, validation, and test splits, creating validation when absent."""
    names = set(dataset.keys())
    if "train" not in names:
        raise ValueError(f"Dataset does not include a train split. Found: {sorted(names)}")
    test = dataset["test"] if "test" in names else None
    validation = dataset["validation"] if "validation" in names else dataset.get("val")
    train = dataset["train"]
    if validation is None:
        divided = train.train_test_split(test_size=validation_ratio, seed=seed)
        train, validation = divided["train"], divided["test"]
        LOGGER.info("No validation split found; created %.1f%% split from train.", validation_ratio * 100)
    return train, validation, test


def _schema_summary(value: Any, depth: int = 0) -> Any:
    if depth > 2:
        return type(value).__name__
    if isinstance(value, Mapping):
        return {key: _schema_summary(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        detail = f"<{_schema_summary(value[0], depth + 1)}>" if value else ""
        return f"{type(value).__name__}[{len(value)}]{detail}"
    return type(value).__name__


def _parse_episode_states(row: Mapping[str, Any], row_index: int) -> np.ndarray:
    """Convert one fixed-format episode ``states`` matrix to internal 16-D states.

    Dataset storage order:
        [t, x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz, ax, ay, az]

    Internal model order:
        [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz, ax, ay, az]
    """
    if "states" not in row:
        raise ValueError(f"Row {row_index} does not contain the required 'states' field.")
    raw_states = np.asarray(row["states"], dtype=np.float32)
    if raw_states.ndim != 2 or raw_states.shape[1] != RAW_STATE_DIM:
        raise ValueError(
            f"Row {row_index} field 'states' must have shape [T, {RAW_STATE_DIM}], got {raw_states.shape}."
        )
    if raw_states.shape[0] == 0:
        raise ValueError(f"Row {row_index} contains an empty 'states' sequence.")
    position = raw_states[:, 1:4]
    quaternion_xyzw = raw_states[:, [5, 6, 7, 4]]
    motion = raw_states[:, 8:17]
    return np.concatenate((position, quaternion_xyzw, motion), axis=-1)


class UAVTrajectoryWindowDataset(Dataset):
    """Convert fixed-format episodes into full-history and cold-start windows."""

    def __init__(self, split: Any, config: DataConfig, split_name: str = "train") -> None:
        self.config = config
        self.split_name = split_name
        self.trajectories = self._adapt_trajectories(split)
        self.windows: List[Tuple[int, int, int]] = []
        needed = config.history_len + config.future_len
        for trajectory_index, trajectory in enumerate(self.trajectories):
            trajectory_length = trajectory["states"].shape[0]
            cold_start_max = min(config.history_len - 1, trajectory_length - config.future_len)
            for real_history_len in range(1, cold_start_max + 1):
                self.windows.append((trajectory_index, 0, real_history_len))
            for start in range(0, trajectory_length - needed + 1, config.stride):
                self.windows.append((trajectory_index, start, config.history_len))
        LOGGER.info(
            "Adapted split '%s': trajectories=%d, windows=%d (including cold-start), "
            "fixed state format=[t,pos,quat_wxyz,vel,omega,acc]",
            split_name,
            len(self.trajectories),
            len(self.windows),
        )
        if not self.windows:
            raise ValueError(
                f"Split '{split_name}' has no valid full-history or cold-start windows for "
                f"history_len={config.history_len}, future_len={config.future_len}, stride={config.stride}."
            )

    def _adapt_trajectories(self, split: Any) -> List[Dict[str, Any]]:
        trajectories: List[Dict[str, Any]] = []
        for row_index in range(len(split)):
            row = split[row_index]
            states = _parse_episode_states(row, row_index)
            metadata = {key: value for key, value in row.items() if key != "states"}
            metadata["source_row"] = row_index
            trajectories.append({"states": states, "metadata": metadata})
        return trajectories

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        trajectory_index, start, real_history_len = self.windows[index]
        states = self.trajectories[trajectory_index]["states"]
        history_end = start + real_history_len
        future_end = history_end + self.config.future_len
        real_history = states[start:history_end]
        virtual_count = self.config.history_len - real_history_len
        if virtual_count:
            virtual_state = states[0].copy()
            virtual_state[7:] = 0.0
            virtual_history = np.repeat(virtual_state[None, :], virtual_count, axis=0)
            history_array = np.concatenate((virtual_history, real_history), axis=0)
        else:
            history_array = real_history
        history = torch.from_numpy(history_array).float()
        future_absolute = torch.from_numpy(states[history_end:future_end, :3]).float()
        current_position = history[-1, :3].clone()
        future = future_absolute - current_position if self.config.use_relative_future else future_absolute
        history_mask = torch.cat(
            (
                torch.zeros(virtual_count, dtype=torch.float32),
                torch.ones(real_history_len, dtype=torch.float32),
            )
        )
        return {
            "history": history,
            "history_mask": history_mask,
            "future": future,
            "future_absolute": future_absolute,
            "current_position": current_position,
            "metadata": {
                **self.trajectories[trajectory_index]["metadata"],
                "window_start": start,
                "real_history_len": real_history_len,
                "is_cold_start": virtual_count > 0,
            },
        }
