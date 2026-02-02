# Copyright 2026 Dimensional Inc.
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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(slots=True)
class RobotStateMsg:
    """Canonical robot state for policy/runtime logic (vendor-agnostic)."""

    stamp_s: float = 0.0
    robot_type: str = "g1"
    joint_names: list[str] = field(default_factory=list)

    # Base state
    imu_quat_wxyz: NDArray[np.floating] = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    )
    base_ang_vel: NDArray[np.floating] = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    base_lin_vel_world: NDArray[np.floating] = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )

    # Joint state (order matches joint_names if provided)
    q: NDArray[np.floating] = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    dq: NDArray[np.floating] = field(default_factory=lambda: np.zeros(0, dtype=np.float32))

    # Optional vendor-specific extras (e.g., raw LowState fields)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RobotCommandMsg:
    """Canonical robot command for policy/runtime logic (vendor-agnostic)."""

    stamp_s: float = 0.0
    robot_type: str = "g1"
    joint_names: list[str] = field(default_factory=list)

    enabled: bool = False
    estop: bool = False

    # Joint targets (order matches joint_names if provided)
    q: NDArray[np.floating] = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    dq: NDArray[np.floating] = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    kp: NDArray[np.floating] = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    kd: NDArray[np.floating] = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    tau: NDArray[np.floating] = field(default_factory=lambda: np.zeros(0, dtype=np.float32))

    extra: dict[str, Any] = field(default_factory=dict)
