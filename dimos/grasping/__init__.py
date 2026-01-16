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

"""Grasping Module.

Provides GraspGen grasp generation integrated into the Dimos architecture.
The GraspGenModule manages a Docker container running the GraspGen service,
while the ObjectToGraspBridgeModule coordinates perception and grasp generation.

Data Management:
    Model checkpoints and sample data are stored in data/graspgen/ and
    managed via Git LFS. Use get_graspgen_data_dir() to access the data.

For visualization (debug only):
    python -m dimos.grasping.visualize_grasps
"""

from dimos.grasping.data_paths import (
    SUPPORTED_GRIPPERS,
    ensure_graspgen_data,
    get_checkpoints_dir,
    get_graspgen_data_dir,
    get_gripper_checkpoint_paths,
    get_gripper_config_path,
    get_sample_data_dir,
    get_sample_meshes_dir,
    get_sample_object_pc_dir,
    get_sample_scene_pc_dir,
    list_available_grippers,
)
from dimos.grasping.graspgen_module import GraspGenConfig, GraspGenModule, graspgen
from dimos.grasping.object_to_grasp_bridge import (
    ObjectToGraspBridgeConfig,
    ObjectToGraspBridgeModule,
    object_to_grasp_bridge,
)

__all__ = [
    # Core grasp generation service
    "GraspGenModule",
    "GraspGenConfig",
    "graspgen",
    # Perception-to-grasp bridge
    "ObjectToGraspBridgeModule",
    "ObjectToGraspBridgeConfig",
    "object_to_grasp_bridge",
    # Data path utilities
    "SUPPORTED_GRIPPERS",
    "ensure_graspgen_data",
    "get_graspgen_data_dir",
    "get_checkpoints_dir",
    "get_gripper_config_path",
    "get_gripper_checkpoint_paths",
    "get_sample_data_dir",
    "get_sample_meshes_dir",
    "get_sample_object_pc_dir",
    "get_sample_scene_pc_dir",
    "list_available_grippers",
]
