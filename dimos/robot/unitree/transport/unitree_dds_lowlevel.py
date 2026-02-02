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
import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

from dimos.msgs.robot_msgs import RobotCommandMsg, RobotStateMsg
from dimos.robot.unitree.lowlevel.joints import G1_LOWLEVEL_MOTOR_JOINT_NAMES, make_reorder
from dimos.utils.logging_config import setup_logger

from .lowcmd_builder import UnitreeLowCmdBuilder, UnitreeLowCmdBuilderConfig

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = setup_logger()

TOPIC_LOWCMD = "rt/lowcmd"
TOPIC_LOWSTATE = "rt/lowstate"
TOPIC_SPORTMODESTATE = "rt/sportmodestate"


@dataclass
class UnitreeDDSRobotIOConfig:
    domain_id: int = 1
    interface: str = "lo0"
    robot_type: str = "g1"
    mode_pr: int = 0


class UnitreeDDSRobotIO:
    """Host-side Unitree DDS boundary (rt/lowstate <-> rt/lowcmd)."""

    def __init__(self, config: UnitreeDDSRobotIOConfig) -> None:
        self.config = config

        if config.robot_type in ("g1", "h1_2"):
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (
                LowCmd_ as LowCmdHg,
                LowState_ as LowStateHg,
            )

            self._LowCmd = LowCmdHg
            self._LowState = LowStateHg
        else:
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import (
                LowCmd_ as LowCmdGo,
                LowState_ as LowStateGo,
            )

            self._LowCmd = LowCmdGo
            self._LowState = LowStateGo

        self._cmd_pub = None
        self._state_sub = None
        self._sport_sub = None

        self._data_lock = threading.Lock()
        self._state_received = False

        self._mode_machine: int = 0
        self._joint_pos_motor = np.zeros(self._num_motors, dtype=np.float32)
        self._joint_vel_motor = np.zeros(self._num_motors, dtype=np.float32)
        self._imu_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._base_ang_vel = np.zeros(3, dtype=np.float32)
        self._base_lin_vel_world = np.zeros(3, dtype=np.float32)

        self._cmd_builder = UnitreeLowCmdBuilder(
            UnitreeLowCmdBuilderConfig(robot_type=config.robot_type, mode_pr=config.mode_pr)
        )
        self._cached_reorder: tuple[tuple[str, ...], NDArray[np.int32]] | None = None

        self._init_dds()

    @property
    def _num_motors(self) -> int:
        if self.config.robot_type == "g1":
            return len(G1_LOWLEVEL_MOTOR_JOINT_NAMES)
        raise NotImplementedError(
            f"Robot motor layout not implemented for {self.config.robot_type}"
        )

    @property
    def joint_names(self) -> list[str]:
        if self.config.robot_type == "g1":
            return list(G1_LOWLEVEL_MOTOR_JOINT_NAMES)
        raise NotImplementedError(f"Robot joint names not implemented for {self.config.robot_type}")

    def _init_dds(self) -> None:
        ChannelFactoryInitialize(self.config.domain_id, self.config.interface)

        self._cmd_pub = ChannelPublisher(TOPIC_LOWCMD, self._LowCmd)
        self._cmd_pub.Init()

        self._state_sub = ChannelSubscriber(TOPIC_LOWSTATE, self._LowState)
        self._state_sub.Init(self._lowstate_callback, 10)

        self._sport_sub = ChannelSubscriber(TOPIC_SPORTMODESTATE, SportModeState_)
        self._sport_sub.Init(self._sportstate_callback, 10)

        logger.info(
            "Unitree DDS RobotIO initialized",
            robot_type=self.config.robot_type,
            domain_id=self.config.domain_id,
            interface=self.config.interface,
        )

    def _lowstate_callback(self, msg: Any) -> None:
        with self._data_lock:
            for i in range(len(self._joint_pos_motor)):
                self._joint_pos_motor[i] = msg.motor_state[i].q
                self._joint_vel_motor[i] = msg.motor_state[i].dq

            if hasattr(msg, "mode_machine"):
                try:
                    self._mode_machine = int(msg.mode_machine)
                except Exception:
                    pass

            self._imu_quat[0] = msg.imu_state.quaternion[0]
            self._imu_quat[1] = msg.imu_state.quaternion[1]
            self._imu_quat[2] = msg.imu_state.quaternion[2]
            self._imu_quat[3] = msg.imu_state.quaternion[3]

            self._base_ang_vel[0] = msg.imu_state.gyroscope[0]
            self._base_ang_vel[1] = msg.imu_state.gyroscope[1]
            self._base_ang_vel[2] = msg.imu_state.gyroscope[2]

            self._state_received = True

    def _sportstate_callback(self, msg: SportModeState_) -> None:
        with self._data_lock:
            self._base_lin_vel_world[0] = msg.velocity[0]
            self._base_lin_vel_world[1] = msg.velocity[1]
            self._base_lin_vel_world[2] = msg.velocity[2]

    def get_state(self) -> RobotStateMsg | None:
        with self._data_lock:
            if not self._state_received:
                return None
            state = RobotStateMsg(
                stamp_s=time.time(),
                robot_type=self.config.robot_type,
                joint_names=self.joint_names,
                imu_quat_wxyz=self._imu_quat.copy(),
                base_ang_vel=self._base_ang_vel.copy(),
                base_lin_vel_world=self._base_lin_vel_world.copy(),
                q=self._joint_pos_motor.copy(),
                dq=self._joint_vel_motor.copy(),
                extra={
                    "mode_machine": int(self._mode_machine),
                    "mode_pr": int(self.config.mode_pr),
                },
            )
        return state

    def _reorder_to_motor(self, cmd: RobotCommandMsg) -> NDArray[np.int32] | None:
        if not cmd.joint_names:
            return None
        key = tuple(cmd.joint_names)
        if self._cached_reorder and self._cached_reorder[0] == key:
            return self._cached_reorder[1]
        idx = make_reorder(cmd.joint_names, self.joint_names)
        self._cached_reorder = (key, idx)
        return idx

    def write_command(self, cmd: RobotCommandMsg) -> None:
        if self._cmd_pub is None:
            return

        with self._data_lock:
            mode_machine = int(self._mode_machine)

        reorder = self._reorder_to_motor(cmd)
        q = cmd.q if reorder is None else cmd.q[reorder]
        dq = cmd.dq if reorder is None else cmd.dq[reorder]
        kp = cmd.kp if reorder is None else cmd.kp[reorder]
        kd = cmd.kd if reorder is None else cmd.kd[reorder]
        tau = cmd.tau if reorder is None else cmd.tau[reorder]

        lowcmd = self._cmd_builder.build(
            mode_machine=mode_machine,
            enabled=bool(cmd.enabled) and not bool(cmd.estop),
            q=q,
            dq=dq,
            kp=kp,
            kd=kd,
            tau=tau,
        )
        self._cmd_pub.Write(lowcmd)
