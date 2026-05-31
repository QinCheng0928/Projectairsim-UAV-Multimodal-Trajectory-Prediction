"""Configuration dataclasses for training, inference, and online prediction."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class DataConfig:
    """Dataset adaptation and sliding-window options."""

    dataset_name: str = "qincheng037/ProjectAirSim-UAV-Kinematic-Trajectories"
    history_len: int = 20
    future_len: int = 30
    stride: int = 1
    state_dim: int = 16
    use_relative_future: bool = True
    validation_ratio: float = 0.2


@dataclass
class ModelConfig:
    """Neural network architecture options."""

    history_len: int = 20
    future_len: int = 30
    state_dim: int = 16
    num_modes: int = 6
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    use_mask: bool = True


@dataclass
class TrainingConfig:
    """End-to-end training configuration."""

    dataset_name: str = "qincheng037/ProjectAirSim-UAV-Kinematic-Trajectories"
    output_dir: str = "outputs/uav_dense_goal"
    history_len: int = 20
    future_len: int = 30
    num_modes: int = 6
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    stride: int = 1
    num_workers: int = 0
    device: str = "auto"
    seed: int = 42
    save_every: int = 1
    eval_every: int = 1
    validation_ratio: float = 0.1
    lambda_cls: float = 0.2
    lambda_goal: float = 1.0
    miss_threshold: float = 2.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def data_config(self) -> DataConfig:
        return DataConfig(
            dataset_name=self.dataset_name,
            history_len=self.history_len,
            future_len=self.future_len,
            stride=self.stride,
            validation_ratio=self.validation_ratio,
        )

    def model_config(self) -> ModelConfig:
        return ModelConfig(
            history_len=self.history_len,
            future_len=self.future_len,
            num_modes=self.num_modes,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )


@dataclass
class OnlineConfig:
    """ProjectAirSim online prediction settings."""

    checkpoint: str = ""
    scene_config: str = "scene_basic_drone.jsonc"
    drone_name: str = "Drone1"
    frequency_hz: float = 10.0
    log_path: Optional[str] = "outputs/online_predictions.jsonl"
    max_steps: Optional[int] = None
    delay_after_load_sec: float = 2.0


def resolve_device(device: str) -> str:
    """Resolve ``auto`` to a usable PyTorch device name."""
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"
