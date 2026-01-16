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

"""GraspGen Module: Docker-based grasp generation service.

Model checkpoints are managed via Git LFS in data/graspgen/. Use the build
script to prepare the Docker image:
    cd dimos/grasping/docker_context && ./build.sh
"""

from __future__ import annotations

import base64
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import requests

from dimos.core.module import Module, ModuleConfig, rpc
from dimos.core.stream import Out
from dimos.grasping.data_paths import SUPPORTED_GRIPPERS, ensure_graspgen_data
from dimos.msgs.geometry_msgs import PoseArray
from dimos.msgs.sensor_msgs import PointCloud2
from dimos.msgs.std_msgs import Header
from dimos.utils.logging_config import setup_logger
from dimos.utils.transform_utils import matrix_to_pose

logger = setup_logger()


@dataclass
class GraspGenConfig(ModuleConfig):
    """Configuration for GraspGen module."""

    docker_image: str = "dimos-graspgen"
    container_name: str = "dimos_graspgen_service"
    service_port: int = 8094
    service_url: str = "http://localhost:8094"
    startup_timeout: int = 60
    gripper_type: str = "robotiq_2f_140"
    num_grasps: int = 400
    topk_num_grasps: int = 100
    grasp_threshold: float = -1.0
    filter_collisions: bool = False
    save_visualization_data: bool = False
    visualization_output_path: str = "/tmp/grasp_visualization.json"


class GraspGenModule(Module):
    """Docker-based grasp generation service.

    Manages a Docker container running GraspGen and provides RPC-based
    grasp generation. Container starts lazily on first request.
    """

    grasps: Out[PoseArray]

    _container_running: bool = False

    def __init__(self, config: GraspGenConfig | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.config = config or GraspGenConfig()

        # Validate gripper type
        if self.config.gripper_type not in SUPPORTED_GRIPPERS:
            raise ValueError(
                f"Unsupported gripper: {self.config.gripper_type}. "
                f"Supported: {', '.join(sorted(SUPPORTED_GRIPPERS))}"
            )

    @rpc
    def start(self) -> None:
        """Initialize module (Docker starts lazily on first grasp request)."""
        super().start()
        logger.info("GraspGenModule ready (Docker starts on first request)")

    @rpc
    def stop(self) -> None:
        """Stop the Docker container."""
        if self._container_running:
            try:
                subprocess.run(
                    ["docker", "stop", self.config.container_name],
                    check=True,
                    timeout=10,
                )
                self._container_running = False
                logger.info("GraspGen container stopped")
            except subprocess.CalledProcessError as e:
                logger.error(f"Error stopping container: {e}")
            except subprocess.TimeoutExpired:
                subprocess.run(
                    ["docker", "rm", "-f", self.config.container_name],
                    check=False,
                )
                self._container_running = False

        super().stop()

    @rpc
    def get_status(self) -> dict[str, Any]:
        """Get service status."""
        status: dict[str, Any] = {
            "container_running": self._container_running,
            "service_url": self.config.service_url,
        }

        if self._container_running:
            try:
                resp = requests.get(f"{self.config.service_url}/health", timeout=2)
                status["healthy"] = resp.status_code == 200
            except Exception:
                status["healthy"] = False

        return status

    @rpc
    def generate_grasps(
        self,
        pointcloud: PointCloud2,
        scene_pointcloud: PointCloud2 | None = None,
    ) -> PoseArray | None:
        """Generate grasps for a point cloud.

        Args:
            pointcloud: Object point cloud
            scene_pointcloud: Optional scene point cloud for collision filtering.
                              If provided and filter_collisions=True, grasps will be
                              filtered against the scene.

        Returns:
            PoseArray with grasp poses sorted by quality, or None on failure
        """
        if not self._ensure_container_running():
            return None

        try:
            points = self._extract_points(pointcloud)
            if len(points) < 10:
                logger.warning(f"Too few points: {len(points)}")
                return None

            # Call service
            payload = {
                "point_cloud_b64": base64.b64encode(
                    points.astype(np.float32).tobytes()
                ).decode(),
                "gripper_type": self.config.gripper_type,
                "num_grasps": self.config.num_grasps,
                "topk_num_grasps": self.config.topk_num_grasps,
                "grasp_threshold": self.config.grasp_threshold,
                "filter_collisions": self.config.filter_collisions,
            }

            # Add scene point cloud for collision filtering if provided
            if self.config.filter_collisions and scene_pointcloud is not None:
                scene_points = self._extract_points(scene_pointcloud)
                if len(scene_points) > 0:
                    payload["scene_pc_b64"] = base64.b64encode(
                        scene_points.astype(np.float32).tobytes()
                    ).decode()
                    logger.info(
                        f"Including scene pointcloud for collision filtering: "
                        f"{len(scene_points)} points"
                    )

            response = self._call_service(payload)
            if response is None:
                return None

            grasps_data = response["grasps"]
            if self.config.filter_collisions:
                total_grasps = len(grasps_data)
                grasps_data = [g for g in grasps_data if g.get("collision_free", True)]
                logger.info(
                    f"Collision filtering: {len(grasps_data)}/{total_grasps} "
                    "grasps are collision-free"
                )

            poses = self._parse_grasps(grasps_data, pointcloud.frame_id)
            self.grasps.publish(poses)

            logger.info(f"Generated {len(poses.poses)} grasps")

            # Save visualization data for Open3D visualization (only collision-free grasps)
            if self.config.save_visualization_data:
                self._save_visualization_data(points, grasps_data, pointcloud.frame_id)

            return poses

        except Exception as e:
            logger.error(f"Grasp generation failed: {e}")
            return None

    def _check_docker_image_exists(self) -> bool:
        """Check if the Docker image exists locally."""
        try:
            result = subprocess.run(
                ["docker", "image", "inspect", self.config.docker_image],
                capture_output=True,
                check=False,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _ensure_container_running(self) -> bool:
        """Start Docker container if needed."""
        if self._container_running:
            return True

        # Check if Docker image exists
        if not self._check_docker_image_exists():
            logger.error(
                f"Docker image '{self.config.docker_image}' not found. "
                "Build it with:\n"
                "  cd dimos/grasping/docker_context && ./build.sh"
            )
            return False

        logger.info("Starting GraspGen Docker container...")

        # Remove existing container
        subprocess.run(
            ["docker", "rm", "-f", self.config.container_name],
            capture_output=True,
            check=False,
        )

        # Start new container
        try:
            cmd = [
                "docker", "run",
                "--gpus", "all",
                "-d", "--rm",
                "-p", f"{self.config.service_port}:8094",
                "-e", f"DEFAULT_GRIPPER={self.config.gripper_type}",
                "--name", self.config.container_name,
                self.config.docker_image,
            ]
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start container: {e}")
            return False

        # Wait for service
        if not self._wait_for_service():
            return False

        self._container_running = True
        logger.info("GraspGen service ready")
        return True

    def _wait_for_service(self) -> bool:
        """Wait for service to become ready."""
        start = time.time()
        while time.time() - start < self.config.startup_timeout:
            try:
                resp = requests.get(f"{self.config.service_url}/health", timeout=2)
                if resp.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)

        logger.error("Service startup timeout")
        return False

    def _call_service(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        """Call grasp generation service with retry."""
        for attempt in range(3):
            try:
                resp = requests.post(
                    f"{self.config.service_url}/generate",
                    json=payload,
                    timeout=30,
                )
                if resp.status_code == 200:
                    return resp.json()

                if resp.status_code == 500 and "CUDA" in resp.text:
                    logger.warning(f"CUDA error, retry {attempt + 1}/3")
                    time.sleep(2)
                    continue

                logger.error(f"Service error: {resp.status_code}")
                return None

            except Exception as e:
                logger.error(f"Service call failed: {e}")
                return None

        return None

    def _extract_points(self, msg: PointCloud2) -> np.ndarray:
        """Extract Nx3 points from PointCloud2."""
        points = msg.points().numpy()
        if not np.isfinite(points).all():
            raise ValueError("Point cloud contains NaN/Inf")
        return points

    def _save_visualization_data(
        self,
        points: np.ndarray,
        grasps_data: list[dict[str, Any]],
        frame_id: str,
    ) -> None:
        """Save grasp data for offline visualization with Open3D.

        Args:
            points: Nx3 point cloud
            grasps_data: List of grasp dicts with transforms and scores
            frame_id: Coordinate frame ID
        """
        try:
            transforms = [g["transform"] for g in grasps_data]
            scores = [g.get("score", 1.0) for g in grasps_data]

            data = {
                "point_cloud": points.tolist(),
                "grasps": transforms,
                "scores": scores,
                "frame_id": frame_id,
                "timestamp": time.time(),
                "num_grasps": len(transforms),
            }

            output_path = Path(self.config.visualization_output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as f:
                json.dump(data, f)

            logger.debug(f"Saved visualization to: {output_path}")

        except Exception as e:
            logger.warning(f"Failed to save visualization data: {e}")

    def _parse_grasps(
        self, grasps_data: list[dict[str, Any]], frame_id: str
    ) -> PoseArray:
        """Convert grasp transforms to PoseArray."""
        poses = []
        for grasp in grasps_data:
            transform = np.array(grasp["transform"]).reshape(4, 4)
            pose = matrix_to_pose(transform)
            poses.append(pose)

        return PoseArray(header=Header(frame_id), poses=poses)


def graspgen(**kwargs: Any) -> GraspGenModule:
    """Create GraspGen module blueprint."""
    return GraspGenModule.blueprint(config=GraspGenConfig(**kwargs))


__all__ = ["GraspGenModule", "GraspGenConfig", "graspgen"]
