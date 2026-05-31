"""Run offline trajectory forecasts from a trained checkpoint."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from uav_multimodal_prediction.config import DataConfig, ModelConfig, resolve_device
from uav_multimodal_prediction.data.collate import trajectory_collate
from uav_multimodal_prediction.data.dataset import UAVTrajectoryWindowDataset, describe_dataset
from uav_multimodal_prediction.data.normalizer import StateNormalizer
from uav_multimodal_prediction.models.predictor import UAVDenseGoalPredictor
from uav_multimodal_prediction.utils.checkpoint import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="outputs/offline_predictions.json")
    parser.add_argument("--format", choices=("json", "npz"), default="json")
    parser.add_argument("--plot_index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(resolve_device(args.device))
    checkpoint = load_checkpoint(args.checkpoint, str(device))
    model_config = ModelConfig(**checkpoint["model_config"])
    data_values = checkpoint.get("data_config", {})
    dataset_name = args.dataset_name or data_values.get(
        "dataset_name", "qincheng037/ProjectAirSim-UAV-Kinematic-Trajectories"
    )
    valid_data_values = {key: value for key, value in data_values.items() if key in DataConfig.__dataclass_fields__}
    data_config = DataConfig(**valid_data_values) if valid_data_values else DataConfig(
        dataset_name=dataset_name, history_len=model_config.history_len, future_len=model_config.future_len
    )
    data_config.dataset_name = dataset_name
    raw = load_dataset(dataset_name)
    describe_dataset(raw)
    selected_split = args.split if args.split in raw else ("validation" if "validation" in raw else "train")
    dataset = UAVTrajectoryWindowDataset(raw[selected_split], data_config, selected_split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=trajectory_collate)
    normalizer = StateNormalizer.from_state_dict(checkpoint["normalizer"])
    model = UAVDenseGoalPredictor(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    outputs: List[Dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            prediction = model(
                normalizer.transform(batch["history"].to(device)),
                batch["history_mask"].to(device),
                batch["current_position"].to(device),
            )
            for index in range(prediction["scores"].shape[0]):
                outputs.append(
                    {
                        "trajectories": prediction["absolute_trajectories"][index].cpu().tolist(),
                        "scores": prediction["scores"][index].cpu().tolist(),
                        "goals": prediction["absolute_goals"][index].cpu().tolist(),
                        "ground_truth": batch["future_absolute"][index].tolist(),
                        "metadata": batch["metadata"][index],
                    }
                )
                if len(outputs) >= args.num_samples:
                    break
            if len(outputs) >= args.num_samples:
                break
    _save_outputs(outputs, args.output, args.format)
    if args.plot_index is not None:
        _plot_prediction(outputs[args.plot_index], args.plot_index)
    print(f"Saved {len(outputs)} predictions to {args.output}")


def _save_outputs(outputs: List[Dict[str, Any]], path: str, output_format: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        target.write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    else:
        np.savez_compressed(
            target,
            trajectories=np.asarray([item["trajectories"] for item in outputs], dtype=np.float32),
            scores=np.asarray([item["scores"] for item in outputs], dtype=np.float32),
            goals=np.asarray([item["goals"] for item in outputs], dtype=np.float32),
            ground_truth=np.asarray([item["ground_truth"] for item in outputs], dtype=np.float32),
        )


def _plot_prediction(prediction: Dict[str, Any], index: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Install matplotlib to use --plot_index.") from exc
    figure = plt.figure()
    axes = figure.add_subplot(111, projection="3d")
    ground_truth = np.asarray(prediction["ground_truth"])
    axes.plot(ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2], label="ground truth", linewidth=3)
    for mode, trajectory in enumerate(np.asarray(prediction["trajectories"])):
        axes.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label=f"mode {mode}")
    axes.set_title(f"Prediction {index} (NED coordinates)")
    axes.legend()
    plt.show()


if __name__ == "__main__":
    main()
