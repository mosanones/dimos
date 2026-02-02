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

# pyright: reportMissingImports=false
# pyright: reportMissingModuleSource=false
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import onnxruntime as ort

from dimos.utils.logging_config import setup_logger

from ..runtime.types import CommandContext, JointTargets, RobotState

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = setup_logger()


def _parse_csv_floats(s: str) -> NDArray[np.floating]:
    return np.array([float(x) for x in s.split(",")], dtype=np.float32)


def _parse_csv_strings(s: str) -> list[str]:
    return [x.strip() for x in s.split(",")]


@dataclass
class MjlabMetadata:
    joint_names: list[str]
    joint_stiffness: NDArray[np.floating]
    joint_damping: NDArray[np.floating]
    default_joint_pos: NDArray[np.floating]
    action_scale: NDArray[np.floating]


def _load_mjlab_metadata(session: ort.InferenceSession) -> MjlabMetadata:
    meta = session.get_modelmeta().custom_metadata_map
    return MjlabMetadata(
        joint_names=_parse_csv_strings(meta["joint_names"]),
        joint_stiffness=_parse_csv_floats(meta["joint_stiffness"]),
        joint_damping=_parse_csv_floats(meta["joint_damping"]),
        default_joint_pos=_parse_csv_floats(meta["default_joint_pos"]),
        action_scale=_parse_csv_floats(meta["action_scale"]),
    )


class MjlabVelocityAdapter:
    """Adapter for MJLab-exported velocity policies (99-dim obs, residual joint targets)."""

    def __init__(self, *, policy_path: str, providers: list[str] | None = None) -> None:
        self.session = ort.InferenceSession(
            policy_path, providers=providers or ort.get_available_providers()
        )
        self.input_name = self.session.get_inputs()[0].name
        self.metadata = _load_mjlab_metadata(self.session)

        self._prev_actions = np.zeros(len(self.metadata.joint_names), dtype=np.float32)
        # Expose defaults for runtime hold behavior (policy joint order).
        self.default_kp = self.metadata.joint_stiffness.astype(np.float32)
        self.default_kd = self.metadata.joint_damping.astype(np.float32)

        logger.info(
            "MjlabVelocityAdapter loaded",
            policy_path=policy_path,
            providers=self.session.get_providers(),
            joints=len(self.metadata.joint_names),
        )

    @property
    def joint_names(self) -> list[str]:
        return list(self.metadata.joint_names)

    def reset(self) -> None:
        self._prev_actions[:] = 0.0

    def step(self, state: RobotState, ctx: CommandContext) -> JointTargets:
        # MJLab obs layout (matches the legacy policy runner):
        # [base_lin_vel(3), base_ang_vel(3), projected_gravity(3),
        #  joint_pos_rel(N), joint_vel(N), last_action(N), command(3)]
        q_rel = (state.q - self.metadata.default_joint_pos).astype(np.float32, copy=False)
        obs = np.concatenate(
            [
                state.base_lin_vel.astype(np.float32, copy=False),
                state.base_ang_vel.astype(np.float32, copy=False),
                state.projected_gravity.astype(np.float32, copy=False),
                q_rel,
                state.dq.astype(np.float32, copy=False),
                self._prev_actions.astype(np.float32, copy=False),
                ctx.cmd_vel.astype(np.float32, copy=False),
            ],
            dtype=np.float32,
        )
        obs = np.clip(obs, -100.0, 100.0).reshape(1, -1)

        actions = self.session.run(None, {self.input_name: obs})[0][0].astype(np.float32)
        actions = np.clip(actions, -10.0, 10.0)
        self._prev_actions = actions.copy()

        q_target = (self.metadata.default_joint_pos + actions * self.metadata.action_scale).astype(
            np.float32
        )
        kp = self.metadata.joint_stiffness.astype(np.float32) * float(ctx.kp_scale)
        kd = self.metadata.joint_damping.astype(np.float32) * float(ctx.kp_scale)

        return JointTargets(q_target=q_target, kp=kp, kd=kd)
