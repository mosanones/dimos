# Copyright 2025-2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Camera calibration utilities for eye-in-hand and external calibration."""

from __future__ import annotations

import json
from pathlib import Path

from dimos.msgs.geometry_msgs import Quaternion, Transform, Vector3
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

# Default calibration directory (relative to project root)
CALIBRATION_DIR = Path(__file__).parent.parent.parent.parent.parent / "hardware" / "camera"


def load_eye_in_hand_calibration(
    calibration_path: str | Path | None = None,
    robot_type: str = "xarm6",
    camera_type: str = "realsense",
) -> Transform:
    """Load eye-in-hand calibration from JSON file.

    Args:
        calibration_path: Path to calibration JSON file. If None, uses default
            path based on robot_type and camera_type.
        robot_type: Robot type for default path (e.g., "xarm6", "ur5")
        camera_type: Camera type for default path (e.g., "realsense", "zed")

    Returns:
        Transform from end-effector to camera frame

    Raises:
        FileNotFoundError: If calibration file not found
    """
    if calibration_path is None:
        calibration_path = (
            CALIBRATION_DIR / camera_type / f"eye_in_hand_calibration_{robot_type}.json"
        )

    calibration_path = Path(calibration_path)
    if not calibration_path.exists():
        raise FileNotFoundError(
            f"Calibration file not found: {calibration_path}\n"
            f"Run eye-in-hand calibration or provide a valid calibration file."
        )

    with open(calibration_path) as f:
        data = json.load(f)

    # Extract translation
    trans = data.get("translation_m", {})
    translation = Vector3(
        x=float(trans.get("x", 0.0)),
        y=float(trans.get("y", 0.0)),
        z=float(trans.get("z", 0.0)),
    )

    # Extract rotation (prefer quaternion wxyz format from calibration)
    quat = data.get("rotation_quat_wxyz", {})
    if quat:
        rotation = Quaternion(
            x=float(quat.get("x", 0.0)),
            y=float(quat.get("y", 0.0)),
            z=float(quat.get("z", 0.0)),
            w=float(quat.get("w", 1.0)),
        )
    else:
        # Fallback to RPY using dimos utilities
        rpy = data.get("rotation_rpy_rad", {})
        if rpy:
            euler = Vector3(
                x=float(rpy.get("roll", 0.0)),
                y=float(rpy.get("pitch", 0.0)),
                z=float(rpy.get("yaw", 0.0)),
            )
            rotation = Quaternion.from_euler(euler)
        else:
            rotation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

    transform = Transform(
        translation=translation,
        rotation=rotation,
        frame_id=data.get("parent_frame", "eef"),
        child_frame_id=data.get("child_frame", "camera_link"),
    )

    logger.info(f"Loaded eye-in-hand calibration from {calibration_path}")
    return transform


__all__ = ["load_eye_in_hand_calibration", "CALIBRATION_DIR"]
