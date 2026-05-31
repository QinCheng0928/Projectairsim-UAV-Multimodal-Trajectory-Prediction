"""Visualize one true history/future trajectory and all predicted modes."""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from datasets import load_dataset

from uav_multimodal_prediction.config import DataConfig, ModelConfig, resolve_device
from uav_multimodal_prediction.data.dataset import UAVTrajectoryWindowDataset, describe_dataset
from uav_multimodal_prediction.data.normalizer import StateNormalizer
from uav_multimodal_prediction.models.predictor import UAVDenseGoalPredictor
from uav_multimodal_prediction.utils.checkpoint import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to a training checkpoint.")
    parser.add_argument("--dataset_name", default=None, help="Override dataset name saved in the checkpoint.")
    parser.add_argument("--split", default="test", help="Dataset split to visualize.")
    parser.add_argument(
        "--sample_index",
        type=int,
        default=0,
        help="Index among full-history visualization samples by default.",
    )
    parser.add_argument(
        "--include_cold_start",
        action="store_true",
        help="Allow selecting cold-start windows. By default only full-history windows are visualized.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="outputs/trajectory_visualization.png")
    parser.add_argument("--show", action="store_true", help="Display the matplotlib window after saving.")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--figsize", type=float, nargs=2, default=(9.0, 7.0))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = torch.device(resolve_device(args.device))
    checkpoint = load_checkpoint(args.checkpoint, str(device))
    model_config = ModelConfig(**checkpoint["model_config"])
    data_config = _load_data_config(checkpoint, model_config, args.dataset_name)

    raw_dataset = load_dataset(data_config.dataset_name)
    describe_dataset(raw_dataset)
    selected_split = args.split if args.split in raw_dataset else ("validation" if "validation" in raw_dataset else "train")
    dataset = UAVTrajectoryWindowDataset(raw_dataset[selected_split], data_config, selected_split)
    dataset_index = _resolve_dataset_index(dataset, args.sample_index, model_config.history_len, args.include_cold_start)
    sample = dataset[dataset_index]

    normalizer = StateNormalizer.from_state_dict(checkpoint["normalizer"])
    model = UAVDenseGoalPredictor(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        prediction = model(
            normalizer.transform(sample["history"].unsqueeze(0).to(device)),
            sample["history_mask"].unsqueeze(0).to(device),
            sample["current_position"].unsqueeze(0).to(device),
        )

    figure = plt.figure(figsize=tuple(args.figsize))
    axes = figure.add_subplot(111, projection="3d")
    _plot_sample(axes, sample, prediction, model_config)
    figure.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=args.dpi)
    print(f"Saved trajectory visualization to {output}")
    print(f"Dataset index: {dataset_index}, metadata: {sample['metadata']}")
    if args.show:
        plt.show()
    plt.close(figure)


def _load_data_config(checkpoint: Dict[str, Any], model_config: ModelConfig, dataset_name: str = None) -> DataConfig:
    data_values = checkpoint.get("data_config", {})
    valid_data_values = {key: value for key, value in data_values.items() if key in DataConfig.__dataclass_fields__}
    if valid_data_values:
        data_config = DataConfig(**valid_data_values)
    else:
        data_config = DataConfig(history_len=model_config.history_len, future_len=model_config.future_len)
    if dataset_name is not None:
        data_config.dataset_name = dataset_name
    return data_config


def _resolve_dataset_index(
    dataset: UAVTrajectoryWindowDataset,
    sample_index: int,
    history_len: int,
    include_cold_start: bool,
) -> int:
    if include_cold_start:
        if sample_index < 0 or sample_index >= len(dataset):
            raise IndexError(f"sample_index={sample_index} is outside dataset length {len(dataset)}.")
        return sample_index
    full_history_indices: List[int] = [
        index for index, (_, _, real_history_len) in enumerate(dataset.windows) if real_history_len == history_len
    ]
    if not full_history_indices:
        raise ValueError("No full-history windows are available for visualization.")
    if sample_index < 0 or sample_index >= len(full_history_indices):
        raise IndexError(
            f"sample_index={sample_index} is outside full-history sample count {len(full_history_indices)}."
        )
    return full_history_indices[sample_index]


def _plot_sample(
    axes: Any,
    sample: Dict[str, Any],
    prediction: Dict[str, torch.Tensor],
    model_config: ModelConfig,
) -> None:
    history_positions = sample["history"][:, :3].cpu().numpy()
    future_positions = sample["future_absolute"].cpu().numpy()
    true_trajectory = np.concatenate((history_positions, future_positions), axis=0)
    current_position = sample["current_position"].cpu().numpy()
    predicted_trajectories = prediction["absolute_trajectories"][0].cpu().numpy()
    scores = prediction["scores"][0].cpu().numpy()

    axes.plot(
        true_trajectory[:, 0],
        true_trajectory[:, 1],
        true_trajectory[:, 2],
        color="black",
        linewidth=2.5,
        marker="o",
        markersize=3,
        label=f"true trajectory ({model_config.history_len}+{model_config.future_len})",
    )
    axes.plot(
        history_positions[:, 0],
        history_positions[:, 1],
        history_positions[:, 2],
        color="tab:blue",
        linewidth=3.0,
        marker="o",
        markersize=4,
        label="history",
    )

    colors = plt_colormap(len(predicted_trajectories))
    for mode, trajectory in enumerate(predicted_trajectories):
        trajectory_with_start = np.concatenate((current_position[None, :], trajectory), axis=0)
        axes.plot(
            trajectory_with_start[:, 0],
            trajectory_with_start[:, 1],
            trajectory_with_start[:, 2],
            color=colors[mode],
            linewidth=1.8,
            linestyle="--",
            marker=".",
            markersize=3,
            label=f"mode {mode + 1}: p={scores[mode]:.3f}",
        )

    axes.set_xlabel("North / X")
    axes.set_ylabel("East / Y")
    axes.set_zlabel("Down / Z")
    axes.set_title("UAV Multi-Modal Trajectory Prediction")
    axes.legend(loc="best", fontsize=8)
    _set_equal_3d_axes(axes, [true_trajectory, *predicted_trajectories])


def plt_colormap(count: int) -> List[Tuple[float, float, float, float]]:
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab10" if count <= 10 else "viridis")
    return [cmap(index / max(count - 1, 1)) for index in range(count)]


def _set_equal_3d_axes(axes: Any, arrays: List[np.ndarray]) -> None:
    points = np.concatenate([array.reshape(-1, 3) for array in arrays], axis=0)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float((maxs - mins).max() / 2.0)
    radius = max(radius, 1e-3)
    axes.set_xlim(center[0] - radius, center[0] + radius)
    axes.set_ylim(center[1] - radius, center[1] + radius)
    axes.set_zlim(center[2] - radius, center[2] + radius)


if __name__ == "__main__":
    main()


# python visualize_trajectory.py --checkpoint outputs/uav_dense_goal/checkpoint_epoch_0014.pt --split test --sample_index 1 --output outputs/trajectory_visualization.png