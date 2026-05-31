"""Train a dense-goal inspired UAV multi-modal trajectory predictor."""

import argparse
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader

from uav_multimodal_prediction.config import TrainingConfig, resolve_device
from uav_multimodal_prediction.data.collate import trajectory_collate
from uav_multimodal_prediction.data.dataset import UAVTrajectoryWindowDataset, describe_dataset, select_dataset_splits
from uav_multimodal_prediction.data.normalizer import StateNormalizer
from uav_multimodal_prediction.models.losses import BestOfKTrajectoryLoss
from uav_multimodal_prediction.models.metrics import trajectory_metrics
from uav_multimodal_prediction.models.predictor import UAVDenseGoalPredictor
from uav_multimodal_prediction.utils.checkpoint import save_checkpoint
from uav_multimodal_prediction.utils.logging import get_logger

LOGGER = get_logger("train")


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default, cast in (
        ("dataset_name", TrainingConfig.dataset_name, str),
        ("output_dir", TrainingConfig.output_dir, str),
        ("history_len", TrainingConfig.history_len, int),
        ("future_len", TrainingConfig.future_len, int),
        ("num_modes", TrainingConfig.num_modes, int),
        ("batch_size", TrainingConfig.batch_size, int),
        ("epochs", TrainingConfig.epochs, int),
        ("learning_rate", TrainingConfig.learning_rate, float),
        ("weight_decay", TrainingConfig.weight_decay, float),
        ("hidden_dim", TrainingConfig.hidden_dim, int),
        ("num_layers", TrainingConfig.num_layers, int),
        ("dropout", TrainingConfig.dropout, float),
        ("stride", TrainingConfig.stride, int),
        ("num_workers", TrainingConfig.num_workers, int),
        ("device", TrainingConfig.device, str),
        ("seed", TrainingConfig.seed, int),
        ("save_every", TrainingConfig.save_every, int),
        ("eval_every", TrainingConfig.eval_every, int),
        ("validation_ratio", TrainingConfig.validation_ratio, float),
        ("lambda_cls", TrainingConfig.lambda_cls, float),
        ("lambda_goal", TrainingConfig.lambda_goal, float),
        ("miss_threshold", TrainingConfig.miss_threshold, float),
    ):
        parser.add_argument(f"--{name}", type=cast, default=default)
    values = vars(parser.parse_args())
    return TrainingConfig(**values)


def set_seed(seed: int) -> None:
    """Set CPU and accelerator random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: UAVDenseGoalPredictor,
    loader: DataLoader,
    normalizer: StateNormalizer,
    criterion: BestOfKTrajectoryLoss,
    device: torch.device,
    optimizer: Optional[AdamW] = None,
) -> Dict[str, float]:
    """Train or evaluate for one epoch and average batch metrics."""
    training = optimizer is not None
    model.train(training)
    totals: Dict[str, float] = {}
    samples = 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            history = normalizer.transform(batch["history"].to(device))
            mask = batch["history_mask"].to(device)
            target = batch["future"].to(device)
            prediction = model(history, mask, batch["current_position"].to(device))
            losses = criterion(prediction, target)
            metrics = trajectory_metrics(prediction["trajectories"], target, prediction["scores"], criterion.miss_threshold)
            values = {**losses, **{f"metric_{key}": value for key, value in metrics.items()}}
            if training:
                optimizer.zero_grad()
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            batch_size = history.shape[0]
            samples += batch_size
            for key, value in values.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu()) * batch_size
    return {key: value / max(samples, 1) for key, value in totals.items()}


def main() -> None:
    config = parse_args()
    set_seed(config.seed)
    device = torch.device(resolve_device(config.device))
    LOGGER.info("Using device: %s", device)
    raw_dataset = load_dataset(config.dataset_name)
    describe_dataset(raw_dataset)
    train_split, validation_split, _ = select_dataset_splits(raw_dataset, config.validation_ratio, config.seed)
    data_config = config.data_config()
    train_dataset = UAVTrajectoryWindowDataset(train_split, data_config, "train")
    validation_dataset = UAVTrajectoryWindowDataset(validation_split, data_config, "validation")
    normalizer = StateNormalizer.fit_dataset(train_dataset, data_config.state_dim)
    LOGGER.info("State normalizer mean=%s std=%s", normalizer.mean.tolist(), normalizer.std.tolist())
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=trajectory_collate,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=trajectory_collate,
    )
    model_config = config.model_config()
    model = UAVDenseGoalPredictor(model_config).to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = BestOfKTrajectoryLoss(config.lambda_cls, config.lambda_goal, config.miss_threshold)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(model, train_loader, normalizer, criterion, device, optimizer)
        LOGGER.info("Epoch %d/%d train: %s", epoch, config.epochs, _format_metrics(train_metrics))
        if epoch % config.eval_every == 0:
            validation_metrics = run_epoch(model, validation_loader, normalizer, criterion, device)
            LOGGER.info("Epoch %d/%d validation: %s", epoch, config.epochs, _format_metrics(validation_metrics))
        if epoch % config.save_every == 0 or epoch == config.epochs:
            save_checkpoint(
                str(output_dir / f"checkpoint_epoch_{epoch:04d}.pt"),
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "training_config": config.to_dict(),
                    "model_config": asdict(model_config),
                    "data_config": asdict(data_config),
                    "normalizer": normalizer.state_dict(),
                },
            )


def _format_metrics(metrics: Dict[str, float]) -> str:
    keys = ("loss", "reg_loss", "cls_loss", "goal_loss", "minADE", "minFDE", "MR", "metric_top1_ADE", "metric_top1_FDE")
    return ", ".join(f"{key}={metrics[key]:.4f}" for key in keys if key in metrics)


if __name__ == "__main__":
    main()


# python train.py --dataset_name qincheng037/ProjectAirSim-UAV-Kinematic-Trajectories --output_dir outputs/uav_dense_goal --history_len 20 --future_len 30 --num_modes 6 --batch_size 64 --epochs 20 --hidden_dim 16 --stride 1 --device auto