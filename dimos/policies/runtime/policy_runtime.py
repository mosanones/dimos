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

from dataclasses import dataclass
import json
import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from dimos.msgs.robot_msgs import RobotCommandMsg, RobotStateMsg
from dimos.utils.logging_config import setup_logger

from .types import CommandContext, JointTargets, RobotState

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .adapter import PolicyAdapter

logger = setup_logger()


def _quat_to_projected_gravity(quat_wxyz: NDArray[np.floating]) -> NDArray[np.floating]:
    w, x, y, z = quat_wxyz
    gx = -2.0 * (x * z - w * y)
    gy = -2.0 * (y * z + w * x)
    gz = -1.0 + 2.0 * (x * x + y * y)
    return np.array([gx, gy, gz], dtype=np.float32)


def _world_to_body_velocity(
    world_vel: NDArray[np.floating], quat_wxyz: NDArray[np.floating]
) -> NDArray[np.floating]:
    w, x, y, z = quat_wxyz
    vx = (
        (1 - 2 * (y * y + z * z)) * world_vel[0]
        + 2 * (x * y + w * z) * world_vel[1]
        + 2 * (x * z - w * y) * world_vel[2]
    )
    vy = (
        2 * (x * y - w * z) * world_vel[0]
        + (1 - 2 * (x * x + z * z)) * world_vel[1]
        + 2 * (y * z + w * x) * world_vel[2]
    )
    vz = (
        2 * (x * z + w * y) * world_vel[0]
        + 2 * (y * z - w * x) * world_vel[1]
        + (1 - 2 * (x * x + y * y)) * world_vel[2]
    )
    return np.array([vx, vy, vz], dtype=np.float32)


@dataclass
class PolicyRuntimeComputeConfig:
    robot_type: str = "g1"


class PolicyRuntimeCompute:
    """Pure compute runtime: RobotStateMsg -> RobotCommandMsg."""

    def __init__(self, *, adapter: PolicyAdapter, config: PolicyRuntimeComputeConfig) -> None:
        self.adapter = adapter
        self.config = config

        self._enabled = False
        self._estop = False
        self._ctx = CommandContext()

        self._index_policy_from_state: NDArray[np.int32] | None = None
        self._last_q_policy: NDArray[np.floating] | None = None

        # Hold gains in policy order.
        self._hold_kp_policy = np.zeros(len(self.adapter.joint_names), dtype=np.float32)
        self._hold_kd_policy = np.zeros(len(self.adapter.joint_names), dtype=np.float32)
        self._hold_tau_policy = np.zeros(len(self.adapter.joint_names), dtype=np.float32)

        # Seed hold gains from adapter defaults if available.
        try:
            default_kp_policy = getattr(self.adapter, "default_kp", None)
            default_kd_policy = getattr(self.adapter, "default_kd", None)
            if default_kp_policy is not None:
                self._hold_kp_policy[:] = np.asarray(default_kp_policy, dtype=np.float32)
            if default_kd_policy is not None:
                self._hold_kd_policy[:] = np.asarray(default_kd_policy, dtype=np.float32)
        except Exception:
            pass

        self._data_lock = threading.Lock()

    def reset(self) -> None:
        self.adapter.reset()
        with self._data_lock:
            self._index_policy_from_state = None
            self._last_q_policy = None

    def set_enabled(self, enabled: bool) -> None:
        with self._data_lock:
            self._enabled = bool(enabled)

    def set_estop(self, estop: bool) -> None:
        with self._data_lock:
            self._estop = bool(estop)
            if self._estop:
                self._enabled = False

    def set_cmd_vel(self, vx: float, vy: float, wz: float) -> None:
        with self._data_lock:
            self._ctx.cmd_vel[0] = float(vx)
            self._ctx.cmd_vel[1] = float(vy)
            self._ctx.cmd_vel[2] = float(wz)

    def set_policy_params_json(self, params_json: str) -> None:
        """Set adapter-specific params from a JSON blob."""
        try:
            data = json.loads(params_json) if params_json else {}
        except Exception:
            data = {}
        with self._data_lock:
            if isinstance(data, dict):
                if "stand" in data:
                    self._ctx.stand = int(bool(data["stand"]))
                if "base_height" in data:
                    try:
                        self._ctx.base_height = float(data["base_height"])
                    except Exception:
                        pass
                if (
                    "waist_rpy" in data
                    and isinstance(data["waist_rpy"], (list, tuple))
                    and len(data["waist_rpy"]) == 3
                ):
                    self._ctx.waist_rpy[:] = np.array(data["waist_rpy"], dtype=np.float32)
                if "kp_scale" in data:
                    try:
                        self._ctx.kp_scale = float(data["kp_scale"])
                    except Exception:
                        pass
                if (
                    "ee_left_xyz" in data
                    and isinstance(data["ee_left_xyz"], (list, tuple))
                    and len(data["ee_left_xyz"]) == 3
                ):
                    self._ctx.ee_left_xyz[:] = np.array(data["ee_left_xyz"], dtype=np.float32)
                if (
                    "ee_right_xyz" in data
                    and isinstance(data["ee_right_xyz"], (list, tuple))
                    and len(data["ee_right_xyz"]) == 3
                ):
                    self._ctx.ee_right_xyz[:] = np.array(data["ee_right_xyz"], dtype=np.float32)
                if "ee_yaw_deg" in data:
                    try:
                        self._ctx.ee_yaw_deg = float(data["ee_yaw_deg"])
                    except Exception:
                        pass
                self._ctx.extra = data

    def _ensure_index_mapping(self, joint_names: list[str]) -> None:
        if self._index_policy_from_state is not None:
            return
        name_to_idx = {n: i for i, n in enumerate(joint_names)}
        idx: list[int] = []
        for name in self.adapter.joint_names:
            if name not in name_to_idx:
                raise ValueError(f"Joint '{name}' not found in RobotStateMsg.joint_names")
            idx.append(name_to_idx[name])
        self._index_policy_from_state = np.array(idx, dtype=np.int32)

    def step(self, state_msg: RobotStateMsg) -> RobotCommandMsg | None:
        with self._data_lock:
            enabled = bool(self._enabled)
            estop = bool(self._estop)
            ctx = self._ctx

        if not state_msg.joint_names:
            raise ValueError("RobotStateMsg.joint_names must be provided for policy runtime")
        self._ensure_index_mapping(state_msg.joint_names)
        assert self._index_policy_from_state is not None

        q_policy = state_msg.q[self._index_policy_from_state].astype(np.float32, copy=False)
        dq_policy = state_msg.dq[self._index_policy_from_state].astype(np.float32, copy=False)

        base_lin_vel_body = _world_to_body_velocity(
            state_msg.base_lin_vel_world, state_msg.imu_quat_wxyz
        )
        proj_gravity = _quat_to_projected_gravity(state_msg.imu_quat_wxyz)

        state = RobotState(
            t_wall_s=time.time(),
            base_lin_vel=base_lin_vel_body,
            base_ang_vel=state_msg.base_ang_vel,
            imu_quat_wxyz=state_msg.imu_quat_wxyz,
            projected_gravity=proj_gravity,
            q=q_policy.copy(),
            dq=dq_policy.copy(),
        )

        # Reset detection for history-buffered policies.
        if self._last_q_policy is not None:
            try:
                max_jump = float(np.max(np.abs(state.q - self._last_q_policy)))
                if max_jump > 1.0:
                    self.adapter.reset()
            except Exception:
                pass
        self._last_q_policy = state.q.copy()

        if estop:
            return RobotCommandMsg(
                stamp_s=time.time(),
                robot_type=state_msg.robot_type,
                joint_names=list(self.adapter.joint_names),
                enabled=False,
                estop=True,
                q=q_policy.copy(),
                dq=np.zeros_like(q_policy),
                kp=np.zeros_like(q_policy),
                kd=np.zeros_like(q_policy),
                tau=np.zeros_like(q_policy),
            )

        if not enabled:
            return RobotCommandMsg(
                stamp_s=time.time(),
                robot_type=state_msg.robot_type,
                joint_names=list(self.adapter.joint_names),
                enabled=True,
                estop=False,
                q=q_policy.copy(),
                dq=np.zeros_like(q_policy),
                kp=self._hold_kp_policy.copy(),
                kd=self._hold_kd_policy.copy(),
                tau=self._hold_tau_policy.copy(),
            )

        targets = self.adapter.step(state, ctx)
        return RobotCommandMsg(
            stamp_s=time.time(),
            robot_type=state_msg.robot_type,
            joint_names=list(self.adapter.joint_names),
            enabled=True,
            estop=False,
            q=targets.q_target.astype(np.float32, copy=False),
            dq=(
                targets.dq_target.astype(np.float32, copy=False)
                if targets.dq_target is not None
                else np.zeros_like(q_policy)
            ),
            kp=(
                targets.kp.astype(np.float32, copy=False)
                if targets.kp is not None
                else np.zeros_like(q_policy)
            ),
            kd=(
                targets.kd.astype(np.float32, copy=False)
                if targets.kd is not None
                else np.zeros_like(q_policy)
            ),
            tau=(
                targets.tau_ff.astype(np.float32, copy=False)
                if targets.tau_ff is not None
                else np.zeros_like(q_policy)
            ),
        )
