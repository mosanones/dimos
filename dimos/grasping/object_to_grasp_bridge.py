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

"""Object-to-Grasp Bridge Module.

Bridges the perception pipeline to GraspGen by extracting individual
object point clouds and sending them for grasp generation.

This module provides RPC calls for downstream modules (e.g., IK solver)
to generate grasp poses for detected objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dimos.core import Out, rpc
from dimos.core.module import Module, ModuleConfig
from dimos.msgs.geometry_msgs import PoseArray
from dimos.msgs.sensor_msgs import PointCloud2
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


@dataclass
class ObjectToGraspBridgeConfig(ModuleConfig):
    """Configuration for ObjectToGraspBridge module."""

    min_points: int = 100


class ObjectToGraspBridgeModule(Module):
    """Bridges perception to GraspGen for grasp generation.

    Provides RPC interface for downstream modules (e.g., IK solver) to:
    - Generate grasp poses for objects by name or object_id
    - Retrieve the latest generated grasps
    - List available detected objects

    The actual grasping skill (IK solving + motion execution) should be
    implemented in the robot arm module that calls these RPCs.
    """

    rpc_calls = [
        "ObjectSceneRegistrationModule.get_object_pointcloud_by_name",
        "ObjectSceneRegistrationModule.get_object_pointcloud_by_object_id",
        "ObjectSceneRegistrationModule.get_detected_objects",
        "ObjectSceneRegistrationModule.get_full_scene_pointcloud",
        "GraspGenModule.generate_grasps",
    ]

    grasps: Out[PoseArray]

    def __init__(
        self,
        config: ObjectToGraspBridgeConfig | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.config = config or ObjectToGraspBridgeConfig()
        self._latest_grasps: PoseArray | None = None
        self._latest_object_id: str | None = None

    @rpc
    def start(self) -> None:
        """Initialize the bridge module."""
        super().start()
        logger.info("ObjectToGraspBridgeModule started")

    @rpc
    def stop(self) -> None:
        """Stop the bridge module."""
        self._latest_grasps = None
        self._latest_object_id = None
        super().stop()
        logger.info("ObjectToGraspBridgeModule stopped")

    @rpc
    def get_latest_grasps(self) -> PoseArray | None:
        """Get the most recent generated grasps.

        Returns:
            PoseArray with grasp poses sorted by quality (best first),
            or None if no grasps have been generated.
        """
        return self._latest_grasps

    @rpc
    def get_detected_objects(self) -> list[dict[str, Any]]:
        """Get list of currently detected objects.

        Returns:
            List of dicts with keys: 'name', 'object_id', 'confidence'
        """
        try:
            get_detected = self.get_rpc_calls(
                "ObjectSceneRegistrationModule.get_detected_objects"
            )
            return get_detected() or []
        except Exception:
            logger.warning("Could not get detected objects from perception")
            return []

    @rpc
    def generate_grasps_for_object(self, identifier: str) -> PoseArray | None:
        """Generate grasp poses for an object.

        This is the main RPC for downstream modules (e.g., IK solver) to request
        grasp generation. Returns structured PoseArray for programmatic use.

        Args:
            identifier: Object name (e.g., "cup") or object_id from detection

        Returns:
            PoseArray with grasp poses sorted by quality (best first),
            or None if object not found or grasp generation failed.
        """
        pointcloud, object_id = self._get_pointcloud_with_id(identifier)
        if pointcloud is None:
            logger.warning(f"Object not found: '{identifier}'")
            self._log_available_objects()
            return None

        num_points = len(pointcloud) if hasattr(pointcloud, "__len__") else 0
        if num_points < self.config.min_points:
            logger.warning(
                f"Point cloud too small for '{identifier}': "
                f"{num_points} points (min: {self.config.min_points})"
            )
            return None

        scene_pointcloud = self._get_scene_pointcloud(object_id)

        try:
            generate_grasps = self.get_rpc_calls("GraspGenModule.generate_grasps")
        except Exception:
            logger.error("GraspGen module not connected")
            return None

        grasps = generate_grasps(pointcloud, scene_pointcloud)
        if grasps is None or len(grasps.poses) == 0:
            logger.warning(f"No grasps generated for '{identifier}'")
            return None

        # Store for later retrieval
        self._latest_grasps = grasps
        self._latest_object_id = object_id

        # Publish to output stream
        self.grasps.publish(grasps)

        logger.info(
            f"Generated {len(grasps.poses)} grasps for '{identifier}' "
            f"(object_id={object_id})"
        )
        return grasps

    def _get_pointcloud_with_id(
        self, identifier: str
    ) -> tuple[PointCloud2 | None, str | None]:
        """Get pointcloud and object_id by identifier (object_id or name).

        Args:
            identifier: Object name (e.g., "cup") or object_id

        Returns:
            Tuple of (pointcloud, object_id) or (None, None) if not found
        """
        try:
            get_by_id, get_by_name, get_detected = self.get_rpc_calls(
                "ObjectSceneRegistrationModule.get_object_pointcloud_by_object_id",
                "ObjectSceneRegistrationModule.get_object_pointcloud_by_name",
                "ObjectSceneRegistrationModule.get_detected_objects",
            )
        except Exception:
            return None, None

        pointcloud = get_by_id(identifier)
        if pointcloud is not None:
            return pointcloud, identifier

        pointcloud = get_by_name(identifier)
        if pointcloud is not None:
            detected = get_detected()
            for obj in detected:
                if obj.get("name") == identifier:
                    return pointcloud, obj.get("object_id")
            return pointcloud, None

        return None, None

    def _get_scene_pointcloud(self, exclude_object_id: str | None) -> PointCloud2 | None:
        """Get the FULL scene pointcloud excluding the target object.

        Args:
            exclude_object_id: Object ID to exclude from scene (for collision filtering)

        Returns:
            PointCloud2 of the full scene including table, or None if not available
        """
        try:
            get_full_scene_pointcloud = self.get_rpc_calls(
                "ObjectSceneRegistrationModule.get_full_scene_pointcloud"
            )
        except Exception:
            logger.warning("Could not get full scene pointcloud RPC")
            return None

        scene_pc = get_full_scene_pointcloud(exclude_object_id)
        if scene_pc is not None:
            num_points = len(scene_pc) if hasattr(scene_pc, "__len__") else 0
            logger.debug(f"Scene pointcloud: {num_points} points (excluding {exclude_object_id})")
        return scene_pc

    def _log_available_objects(self) -> None:
        """Log available detected objects for debugging."""
        try:
            get_detected = self.get_rpc_calls(
                "ObjectSceneRegistrationModule.get_detected_objects"
            )
            detected = get_detected()
            if detected:
                obj_list = [f"{o['name']} (id='{o['object_id']}')" for o in detected]
                logger.debug(f"Available objects: {', '.join(obj_list)}")
        except Exception:
            pass


def object_to_grasp_bridge(
    min_points: int = 100,
) -> ObjectToGraspBridgeModule:
    """Create ObjectToGraspBridge module blueprint.

    Args:
        min_points: Minimum points required in object pointcloud

    Returns:
        Module blueprint
    """
    return ObjectToGraspBridgeModule.blueprint(
        config=ObjectToGraspBridgeConfig(
            min_points=min_points,
        )
    )


__all__ = ["ObjectToGraspBridgeModule", "ObjectToGraspBridgeConfig", "object_to_grasp_bridge"]
