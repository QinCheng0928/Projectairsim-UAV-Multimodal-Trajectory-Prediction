"""Thin, tolerant adapter over the official ProjectAirSim Python client."""

import time
from typing import Any, Dict, List, Optional, Sequence

from uav_multimodal_prediction.utils.logging import get_logger

LOGGER = get_logger(__name__)


def _lookup(value: Any, paths: Sequence[str], default: Any = None) -> Any:
    for path in paths:
        current = value
        valid = True
        for name in path.split("."):
            if isinstance(current, dict):
                current = current.get(name)
            else:
                current = getattr(current, name, None)
            if current is None:
                valid = False
                break
        if valid:
            return current
    return default


def _vector3(value: Any, default: Sequence[float] = (0.0, 0.0, 0.0)) -> List[float]:
    if value is None:
        return list(default)
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value[:3]]
    return [
        float(_lookup(value, ("x",), default[0])),
        float(_lookup(value, ("y",), default[1])),
        float(_lookup(value, ("z",), default[2])),
    ]


def _quaternion(value: Any) -> List[float]:
    if value is None:
        return [0.0, 0.0, 0.0, 1.0]
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value[:4]]
    return [
        float(_lookup(value, ("x", "qx"), 0.0)),
        float(_lookup(value, ("y", "qy"), 0.0)),
        float(_lookup(value, ("z", "qz"), 0.0)),
        float(_lookup(value, ("w", "qw"), 1.0)),
    ]


class ProjectAirSimAdapter:
    """Connect to a ProjectAirSim drone and expose unified kinematic state.

    ProjectAirSim publishes SI-unit state in NED coordinates. No ENU or degree
    conversion is applied here so predicted trajectories remain simulator-native.
    """

    def __init__(
        self,
        scene_config: str = "scene_basic_drone.jsonc",
        drone_name: str = "Drone1",
        delay_after_load_sec: float = 2.0,
    ) -> None:
        self.scene_config = scene_config
        self.drone_name = drone_name
        self.delay_after_load_sec = delay_after_load_sec
        self.client: Optional[Any] = None
        self.world: Optional[Any] = None
        self.drone: Optional[Any] = None

    def connect(self) -> None:
        """Create the official client, load a scene, and obtain the drone handle."""
        try:
            from projectairsim import Drone, ProjectAirSimClient, World
        except ImportError as exc:
            raise ImportError(
                "ProjectAirSim Python client is not installed. Install it from "
                "ProjectAirSim/client/python/projectairsim before online prediction."
            ) from exc
        self.client = ProjectAirSimClient()
        self.client.connect()
        self.world = World(self.client, self.scene_config, delay_after_load_sec=self.delay_after_load_sec)
        self.drone = Drone(self.client, self.world, self.drone_name)
        LOGGER.info("Connected to ProjectAirSim drone '%s' in scene '%s'.", self.drone_name, self.scene_config)

    def get_current_state(self) -> Dict[str, Any]:
        """Read ground-truth kinematics using documented and compatible key forms."""
        if self.drone is None:
            raise RuntimeError("Adapter is not connected.")
        kinematics = self.drone.get_ground_truth_kinematics()
        pose = _lookup(kinematics, ("pose",), None)
        if pose is None and hasattr(self.drone, "get_ground_truth_pose"):
            pose = self.drone.get_ground_truth_pose()
        position = _lookup(pose, ("position", "translation"), _lookup(kinematics, ("position",), None))
        orientation = _lookup(pose, ("orientation", "rotation"), _lookup(kinematics, ("orientation",), None))
        return {
            "position": _vector3(position),
            "orientation": _quaternion(orientation),
            "linear_velocity": _vector3(_lookup(kinematics, ("twist.linear", "linear_velocity", "velocity"))),
            "angular_velocity": _vector3(_lookup(kinematics, ("twist.angular", "angular_velocity"))),
            "linear_acceleration": _vector3(_lookup(kinematics, ("accels.linear", "linear_acceleration", "acceleration"))),
            "timestamp": _lookup(kinematics, ("time_stamp", "timestamp"), time.time()),
            "coordinate_frame": "NED",
        }

    def disconnect(self) -> None:
        """Disconnect from the simulator when a client connection exists."""
        if self.client is not None:
            self.client.disconnect()
            LOGGER.info("Disconnected from ProjectAirSim.")
        self.client = None
        self.world = None
        self.drone = None
