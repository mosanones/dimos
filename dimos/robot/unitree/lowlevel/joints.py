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

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable

# Canonical motor ordering for Unitree G1 in low-level LowCmd/LowState arrays.
# This should be the single source of truth for motor index semantics across:
# - MuJoCo Unitree DDS bridge/mirror
# - Unitree DDS RobotIO
# - Policy adapters mapping by joint name
G1_LOWLEVEL_MOTOR_JOINT_NAMES: list[str] = [
    # Left leg (0-5)
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    # Right leg (6-11)
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # Waist (12-14)
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    # Left arm (15-21)
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    # Right arm (22-28)
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def name_to_index(names: Iterable[str]) -> dict[str, int]:
    return {n: i for i, n in enumerate(list(names))}


def make_reorder(src_names: list[str], dst_names: list[str]) -> np.ndarray:
    """Return indices so that `src[src_to_dst]` aligns with `dst`.

    Specifically returns an array `src_index_for_dst[i]` such that:
      dst[i] == src[src_index_for_dst[i]]
    """
    src_map = name_to_index(src_names)
    idx: list[int] = []
    for n in dst_names:
        if n not in src_map:
            raise KeyError(f"Name '{n}' not found in source list")
        idx.append(int(src_map[n]))
    return np.array(idx, dtype=np.int32)
