"""Online fixed-rate multi-modal prediction loop for ProjectAirSim."""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from uav_multimodal_prediction.config import ModelConfig, resolve_device
from uav_multimodal_prediction.data.normalizer import StateNormalizer
from uav_multimodal_prediction.models.predictor import UAVDenseGoalPredictor
from uav_multimodal_prediction.online.history_buffer import HistoryBuffer
from uav_multimodal_prediction.online.projectairsim_adapter import ProjectAirSimAdapter
from uav_multimodal_prediction.utils.checkpoint import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scene_config", default="scene_basic_drone.jsonc")
    parser.add_argument("--drone_name", default="Drone1")
    parser.add_argument("--frequency_hz", type=float, default=10.0)
    parser.add_argument("--log_path", default="outputs/online_predictions.jsonl")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--delay_after_load_sec", type=float, default=2.0)
    return parser.parse_args()


async def predict_loop(args: argparse.Namespace) -> None:
    """Connect, warm-start from virtual frames, and forecast at fixed frequency."""
    device = torch.device(resolve_device(args.device))
    checkpoint = load_checkpoint(args.checkpoint, str(device))
    model_config = ModelConfig(**checkpoint["model_config"])
    model = UAVDenseGoalPredictor(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    normalizer = StateNormalizer.from_state_dict(checkpoint["normalizer"])
    adapter = ProjectAirSimAdapter(args.scene_config, args.drone_name, args.delay_after_load_sec)
    buffer = HistoryBuffer(model_config.history_len, model_config.state_dim)
    log_file: Optional[Any] = None
    if args.log_path:
        target = Path(args.log_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        log_file = target.open("a", encoding="utf-8")
    try:
        adapter.connect()
        initial_state = adapter.get_current_state()
        buffer.reset(initial_state)
        interval = 1.0 / args.frequency_hz
        step = 0
        while args.max_steps is None or step < args.max_steps:
            state = adapter.get_current_state()
            buffer.update(state)
            history, mask = buffer.get_history()
            history_tensor = torch.from_numpy(history).unsqueeze(0).to(device)
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(device)
            current_position = history_tensor[:, -1, :3]
            with torch.no_grad():
                prediction = model(normalizer.transform(history_tensor), mask_tensor, current_position)
            record: Dict[str, Any] = {
                "timestamp": state["timestamp"],
                "real_count": buffer.real_count,
                "is_fully_warmed": buffer.is_fully_warmed,
                "trajectories": prediction["absolute_trajectories"][0].cpu().tolist(),
                "scores": prediction["scores"][0].cpu().tolist(),
                "goals": prediction["absolute_goals"][0].cpu().tolist(),
                "coordinate_frame": "NED",
            }
            ranked = [
                f"{rank + 1}: p={score:.3f}, goal={goal}"
                for rank, (score, goal) in enumerate(zip(record["scores"], record["goals"]))
            ]
            print(f"step={step} warm={buffer.real_count}/{buffer.history_len} | " + " | ".join(ranked))
            if log_file is not None:
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
            step += 1
            await asyncio.sleep(interval)
    finally:
        if log_file is not None:
            log_file.close()
        adapter.disconnect()


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(predict_loop(args))
    except KeyboardInterrupt:
        print("Online prediction stopped.")


if __name__ == "__main__":
    main()
