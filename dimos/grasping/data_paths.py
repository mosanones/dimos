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

"""GraspGen Data Path Management.

Provides centralized path management for GraspGen model checkpoints and sample data.
Uses the standard dimos LFS data loading pattern via `get_data`.

Data is stored in Git LFS as `data/.lfs/graspgen.tar.gz` and extracted to `data/graspgen/`.

    from dimos.grasping.data_paths import (
        get_graspgen_data_dir,
        get_checkpoints_dir,
        get_gripper_config_path,
        get_sample_data_dir,
        ensure_graspgen_data,
    )

    # Ensure data is downloaded
    ensure_graspgen_data()

    # Get specific paths
    config_path = get_gripper_config_path("robotiq_2f_140")
    sample_meshes = get_sample_data_dir() / "meshes"
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

from dimos.utils.data import get_data
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

# Data directory name in the LFS archive
GRASPGEN_DATA_NAME = "graspgen"

# Supported gripper types
SUPPORTED_GRIPPERS = frozenset({
    "robotiq_2f_140",
    "franka_panda",
    "single_suction_cup_30mm",
})


@cache
def get_graspgen_data_dir() -> Path:
    """Get the root directory for GraspGen data.

    Downloads from LFS if not already present.

    Returns:
        Path to data/graspgen/ directory
    """
    return get_data(GRASPGEN_DATA_NAME)


@cache
def get_checkpoints_dir() -> Path:
    """Get the directory containing model checkpoints.

    Returns:
        Path to data/graspgen/checkpoints/ directory
    """
    return get_graspgen_data_dir() / "checkpoints"


def get_gripper_config_path(gripper_name: str) -> Path:
    """Get path to a gripper's configuration YAML file.

    Args:
        gripper_name: Name of gripper (e.g., "robotiq_2f_140")

    Returns:
        Path to the gripper's .yml configuration file

    Raises:
        ValueError: If gripper_name is not supported
        FileNotFoundError: If config file doesn't exist
    """
    if gripper_name not in SUPPORTED_GRIPPERS:
        raise ValueError(
            f"Unsupported gripper: {gripper_name}. "
            f"Supported grippers: {', '.join(sorted(SUPPORTED_GRIPPERS))}"
        )

    config_path = get_checkpoints_dir() / f"graspgen_{gripper_name}.yml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Gripper config not found: {config_path}. "
            f"Run `get_data('graspgen')` to download checkpoints."
        )

    return config_path


def get_gripper_checkpoint_paths(gripper_name: str) -> dict[str, Path]:
    """Get paths to all checkpoint files for a gripper.

    Args:
        gripper_name: Name of gripper (e.g., "robotiq_2f_140")

    Returns:
        Dict with keys 'config', 'generator', 'discriminator' mapping to Paths
    """
    checkpoints_dir = get_checkpoints_dir()
    prefix = f"graspgen_{gripper_name}"

    return {
        "config": checkpoints_dir / f"{prefix}.yml",
        "generator": checkpoints_dir / f"{prefix}_gen.pth",
        "discriminator": checkpoints_dir / f"{prefix}_dis.pth",
    }


@cache
def get_sample_data_dir() -> Path:
    """Get the directory containing sample data for testing.

    Returns:
        Path to data/graspgen/sample_data/ directory
    """
    return get_graspgen_data_dir() / "sample_data"


def get_sample_meshes_dir() -> Path:
    """Get the directory containing sample mesh files.

    Returns:
        Path to data/graspgen/sample_data/meshes/ directory
    """
    return get_sample_data_dir() / "meshes"


def get_sample_object_pc_dir() -> Path:
    """Get the directory containing sample object point clouds.

    Returns:
        Path to data/graspgen/sample_data/real_object_pc/ directory
    """
    return get_sample_data_dir() / "real_object_pc"


def get_sample_scene_pc_dir() -> Path:
    """Get the directory containing sample scene point clouds.

    Returns:
        Path to data/graspgen/sample_data/real_scene_pc/ directory
    """
    return get_sample_data_dir() / "real_scene_pc"


def ensure_graspgen_data() -> Path:
    """Ensure GraspGen data is downloaded and available.

    This function explicitly triggers the LFS download if needed
    and verifies that the checkpoints are present.

    Returns:
        Path to the graspgen data directory

    Raises:
        FileNotFoundError: If data could not be downloaded or is incomplete
    """
    data_dir = get_graspgen_data_dir()

    # Verify checkpoints exist
    checkpoints_dir = data_dir / "checkpoints"
    if not checkpoints_dir.exists():
        raise FileNotFoundError(
            f"GraspGen checkpoints directory not found: {checkpoints_dir}. "
            "Data may not have been downloaded correctly."
        )

    # Check for at least one complete gripper checkpoint set
    found_gripper = False
    for gripper in SUPPORTED_GRIPPERS:
        paths = get_gripper_checkpoint_paths(gripper)
        if all(p.exists() for p in paths.values()):
            found_gripper = True
            logger.debug(f"Found complete checkpoint set for gripper: {gripper}")
            break

    if not found_gripper:
        raise FileNotFoundError(
            "No complete gripper checkpoint set found. "
            "Expected at least one of: "
            + ", ".join(sorted(SUPPORTED_GRIPPERS))
        )

    logger.info(f"GraspGen data verified at: {data_dir}")
    return data_dir


def list_available_grippers() -> list[str]:
    """List all grippers with available checkpoints.

    Returns:
        List of gripper names that have complete checkpoint sets
    """
    available = []
    try:
        checkpoints_dir = get_checkpoints_dir()
        for gripper in SUPPORTED_GRIPPERS:
            paths = get_gripper_checkpoint_paths(gripper)
            if all(p.exists() for p in paths.values()):
                available.append(gripper)
    except FileNotFoundError:
        pass

    return sorted(available)


__all__ = [
    "GRASPGEN_DATA_NAME",
    "SUPPORTED_GRIPPERS",
    "get_graspgen_data_dir",
    "get_checkpoints_dir",
    "get_gripper_config_path",
    "get_gripper_checkpoint_paths",
    "get_sample_data_dir",
    "get_sample_meshes_dir",
    "get_sample_object_pc_dir",
    "get_sample_scene_pc_dir",
    "ensure_graspgen_data",
    "list_available_grippers",
]
