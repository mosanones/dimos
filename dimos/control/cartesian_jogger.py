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

"""Pygame-based cartesian jogger module for CartesianIK control."""

import os
import threading
import time

import numpy as np
import pygame

from dimos.core import Module, Out, rpc
from dimos.msgs.geometry_msgs import PoseStamped, Quaternion, Vector3
from dimos.utils.data import get_data

os.environ["SDL_VIDEODRIVER"] = "x11"

LINEAR_SPEED = 0.05
ANGULAR_SPEED = 0.5
X_LIMITS = (-0.5, 0.5)
Y_LIMITS = (-0.5, 0.5)
Z_LIMITS = (-0.2, 0.6)
TASK_NAME = "cartesian_ik_arm"


def _clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


def _get_home_pose() -> tuple[float, ...]:
    import pinocchio  # type: ignore[import-untyped]

    model_path = str(
        get_data("piper_description") / "mujoco_model/piper_no_gripper_description.xml"
    )
    model = pinocchio.buildModelFromMJCF(model_path)
    data = model.createData()
    pinocchio.forwardKinematics(model, data, np.zeros(model.nq))
    ee_pose = data.oMi[6]
    rpy = pinocchio.rpy.matrixToRpy(ee_pose.rotation)
    return tuple(float(x) for x in (*ee_pose.translation, *rpy))


class CartesianJogger(Module):
    """Pygame-based cartesian jogger for arm end-effector control."""

    cartesian_command: Out[PoseStamped]

    _stop_event: threading.Event
    _thread: threading.Thread | None = None
    _pose: list[float]
    _home: tuple[float, ...] | None = None

    def __init__(self) -> None:
        super().__init__()
        self._stop_event = threading.Event()
        self._pose = [0.0] * 6

    @rpc
    def start(self) -> bool:
        super().start()
        self._home = _get_home_pose()
        self._pose = list(self._home)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._pygame_loop, daemon=True)
        self._thread.start()
        return True

    @rpc
    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(2)
        super().stop()

    def _publish_pose(self) -> None:
        pose = PoseStamped(
            ts=time.time(),
            frame_id=TASK_NAME,
            position=Vector3(self._pose[0], self._pose[1], self._pose[2]),
            orientation=Quaternion.from_euler(Vector3(self._pose[3], self._pose[4], self._pose[5])),
        )
        self.cartesian_command.publish(pose)

    def _pygame_loop(self) -> None:
        pygame.init()
        screen = pygame.display.set_mode((600, 400), pygame.SWSURFACE)
        pygame.display.set_caption("Cartesian Jogger")
        font = pygame.font.Font(None, 28)
        clock = pygame.time.Clock()

        self._publish_pose()
        last_time = time.perf_counter()

        while not self._stop_event.is_set():
            dt = time.perf_counter() - last_time
            last_time = time.perf_counter()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._stop_event.set()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self._stop_event.set()
                    elif event.key == pygame.K_SPACE and self._home:
                        self._pose = list(self._home)

            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]:
                self._pose[0] += LINEAR_SPEED * dt
            if keys[pygame.K_s]:
                self._pose[0] -= LINEAR_SPEED * dt
            if keys[pygame.K_a]:
                self._pose[1] -= LINEAR_SPEED * dt
            if keys[pygame.K_d]:
                self._pose[1] += LINEAR_SPEED * dt
            if keys[pygame.K_q]:
                self._pose[2] += LINEAR_SPEED * dt
            if keys[pygame.K_e]:
                self._pose[2] -= LINEAR_SPEED * dt
            if keys[pygame.K_r]:
                self._pose[3] += ANGULAR_SPEED * dt
            if keys[pygame.K_f]:
                self._pose[3] -= ANGULAR_SPEED * dt
            if keys[pygame.K_t]:
                self._pose[4] += ANGULAR_SPEED * dt
            if keys[pygame.K_g]:
                self._pose[4] -= ANGULAR_SPEED * dt
            if keys[pygame.K_y]:
                self._pose[5] += ANGULAR_SPEED * dt
            if keys[pygame.K_h]:
                self._pose[5] -= ANGULAR_SPEED * dt

            self._pose[0] = _clamp(self._pose[0], *X_LIMITS)
            self._pose[1] = _clamp(self._pose[1], *Y_LIMITS)
            self._pose[2] = _clamp(self._pose[2], *Z_LIMITS)

            self._publish_pose()

            screen.fill((30, 30, 30))
            screen.blit(font.render("Cartesian Jogger", True, (255, 255, 255)), (200, 20))
            screen.blit(
                font.render(
                    f"X={self._pose[0]:.3f}  Y={self._pose[1]:.3f}  Z={self._pose[2]:.3f}",
                    True,
                    (100, 255, 100),
                ),
                (50, 70),
            )
            screen.blit(
                font.render(
                    f"R={np.degrees(self._pose[3]):.1f}  P={np.degrees(self._pose[4]):.1f}  Y={np.degrees(self._pose[5]):.1f}",
                    True,
                    (100, 200, 255),
                ),
                (50, 100),
            )
            screen.blit(
                font.render("W/S: X  A/D: Y  Q/E: Z  |  R/F T/G Y/H: RPY", True, (150, 150, 150)),
                (50, 150),
            )
            screen.blit(font.render("SPACE: Home  ESC: Quit", True, (150, 150, 150)), (50, 180))
            pygame.display.flip()
            clock.tick(50)

        pygame.quit()


cartesian_jogger = CartesianJogger.blueprint

__all__ = ["CartesianJogger", "cartesian_jogger"]
