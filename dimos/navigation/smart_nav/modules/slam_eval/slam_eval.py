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

"""SLAM evaluation harness for DimOS modules.

Replays sensor data through a SLAM algorithm, compares estimated trajectory
against ground truth, and outputs accuracy metrics as JSON.

Works with any SLAM backend that implements the ``SlamBackend`` protocol:
feed it (rotation, translation, timestamp, point_cloud) per frame and
read back estimated poses.

Example usage with PGO::

    from dimos.navigation.smart_nav.modules.slam_eval import slam_eval
    from dimos.navigation.smart_nav.modules.slam_eval.slam_eval import PGOBackend

    results = slam_eval(PGOBackend(), dataset="tum_fr1_desk")
    # results is a dict with ATE, RPE, timing — also written to metrics.json
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import tarfile
import time
from typing import Any, Protocol, runtime_checkable

import cv2
import numpy as np
import requests
from scipy.spatial.transform import Rotation

from dimos.core.module import Module
from dimos.core.stream import In, Out, Transport
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.nav_msgs.Odometry import Odometry
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2

logger = logging.getLogger(__name__)

# Maximum time difference (seconds) for matching timestamps between
# estimated and ground truth poses, or between RGB and depth frames.
_MAX_TIMESTAMP_DIFF = 0.1

# ─── Dataset cache ──────────────────────────────────────────────────────────

DATASET_CACHE_DIR = Path(
    os.environ.get(
        "DIMOS_SLAM_EVAL_CACHE",
        Path.home() / ".cache" / "dimos" / "slam_eval" / "datasets",
    )
)

# ─── Types ──────────────────────────────────────────────────────────────────


@dataclass
class TrajectoryPose:
    """A single timestamped pose in a trajectory."""

    timestamp: float
    position: np.ndarray  # (3,)
    rotation: np.ndarray  # (3, 3)


@dataclass
class SensorFrame:
    """A single sensor frame from a dataset."""

    timestamp: float
    position: np.ndarray  # (3,) ground truth
    rotation: np.ndarray  # (3, 3) ground truth
    point_cloud: np.ndarray  # (N, 3) in body frame
    rgb_image: np.ndarray | None = None  # (H, W, 3) uint8, optional
    depth_image: np.ndarray | None = None  # (H, W) float32 meters, optional


@dataclass
class NoiseProfile:
    """Noise model for corrupting odometry input to a SLAM backend.

    Simulates sensor imperfections by adding noise to the ground truth
    odometry before feeding it to the SLAM algorithm. This tests how
    well the backend handles realistic input quality.

    Presets are calibrated against published sensor noise specs:
    - ``none``: Zero noise (baseline)
    - ``lidar_mild``: Typical LiDAR odometry (e.g. Ouster OS1 + LOAM)
    - ``lidar_harsh``: LiDAR in degraded conditions (dust, narrow corridors)
    - ``visual_mild``: Good visual odometry (stereo, textured environment)
    - ``visual_harsh``: Visual odometry in challenging conditions (low texture, motion blur)
    - ``wheel_odom``: Wheel encoder odometry (high drift, no rotation noise)
    """

    # Per-step translational noise std (meters)
    translation_noise_std: float = 0.0
    # Per-step rotational noise std (degrees)
    rotation_noise_std_deg: float = 0.0
    # Cumulative translational drift per meter traveled (meters/meter)
    translation_drift_rate: float = 0.0
    # Cumulative rotational drift per meter traveled (degrees/meter)
    rotation_drift_rate_deg: float = 0.0
    # Point cloud noise std (meters, applied to each point independently)
    point_cloud_noise_std: float = 0.0
    # Point cloud outlier ratio (fraction of points replaced with random outliers)
    point_cloud_outlier_ratio: float = 0.0
    # Random seed for reproducibility (None = non-deterministic)
    seed: int | None = 42
    # Human-readable label
    label: str = "custom"

    @classmethod
    def none(cls) -> NoiseProfile:
        """Zero noise baseline."""
        return cls(label="none")

    @classmethod
    def lidar_mild(cls) -> NoiseProfile:
        """Typical LiDAR odometry noise (good conditions)."""
        return cls(
            translation_noise_std=0.005,
            rotation_noise_std_deg=0.05,
            translation_drift_rate=0.002,
            rotation_drift_rate_deg=0.01,
            point_cloud_noise_std=0.02,
            point_cloud_outlier_ratio=0.001,
            label="lidar_mild",
        )

    @classmethod
    def lidar_harsh(cls) -> NoiseProfile:
        """LiDAR odometry in degraded conditions (dust, featureless)."""
        return cls(
            translation_noise_std=0.02,
            rotation_noise_std_deg=0.2,
            translation_drift_rate=0.01,
            rotation_drift_rate_deg=0.05,
            point_cloud_noise_std=0.05,
            point_cloud_outlier_ratio=0.01,
            label="lidar_harsh",
        )

    @classmethod
    def visual_mild(cls) -> NoiseProfile:
        """Good visual odometry (stereo, textured environment)."""
        return cls(
            translation_noise_std=0.01,
            rotation_noise_std_deg=0.1,
            translation_drift_rate=0.005,
            rotation_drift_rate_deg=0.02,
            point_cloud_noise_std=0.03,
            point_cloud_outlier_ratio=0.005,
            label="visual_mild",
        )

    @classmethod
    def visual_harsh(cls) -> NoiseProfile:
        """Visual odometry in challenging conditions (low texture, motion blur)."""
        return cls(
            translation_noise_std=0.05,
            rotation_noise_std_deg=0.5,
            translation_drift_rate=0.02,
            rotation_drift_rate_deg=0.1,
            point_cloud_noise_std=0.08,
            point_cloud_outlier_ratio=0.02,
            label="visual_harsh",
        )

    @classmethod
    def wheel_odom(cls) -> NoiseProfile:
        """Wheel encoder odometry (high translational drift, no direct rotation noise)."""
        return cls(
            translation_noise_std=0.03,
            rotation_noise_std_deg=0.3,
            translation_drift_rate=0.03,
            rotation_drift_rate_deg=0.15,
            point_cloud_noise_std=0.0,
            point_cloud_outlier_ratio=0.0,
            label="wheel_odom",
        )


def apply_noise(
    frames: list[SensorFrame],
    noise: NoiseProfile,
) -> list[SensorFrame]:
    """Apply noise to a list of sensor frames, returning corrupted copies.

    Corrupts both odometry (translation + rotation) and point clouds.
    Drift accumulates over the trajectory to simulate real sensor behavior.
    """
    all_zero = (
        noise.translation_noise_std == 0.0
        and noise.rotation_noise_std_deg == 0.0
        and noise.translation_drift_rate == 0.0
        and noise.rotation_drift_rate_deg == 0.0
        and noise.point_cloud_noise_std == 0.0
        and noise.point_cloud_outlier_ratio == 0.0
    )
    if all_zero:
        return list(frames)

    rng = np.random.default_rng(noise.seed)
    noisy_frames = []
    cumulative_t_drift = np.zeros(3)
    cumulative_r_drift = np.zeros(3)  # euler angles accumulator

    for i, frame in enumerate(frames):
        # Compute distance traveled from previous frame for drift scaling
        if i > 0:
            step_dist = float(np.linalg.norm(frame.position - frames[i - 1].position))
        else:
            step_dist = 0.0

        # Accumulate drift
        cumulative_t_drift += rng.normal(0, noise.translation_drift_rate * step_dist, 3)
        cumulative_r_drift += rng.normal(0, noise.rotation_drift_rate_deg * step_dist, 3)

        # Per-step noise
        t_noise = rng.normal(0, noise.translation_noise_std, 3)
        r_noise_deg = rng.normal(0, noise.rotation_noise_std_deg, 3)

        # Apply to position
        noisy_pos = frame.position + t_noise + cumulative_t_drift

        # Apply to rotation
        total_rot_noise_deg = r_noise_deg + cumulative_r_drift
        r_perturbation = Rotation.from_euler("xyz", total_rot_noise_deg, degrees=True).as_matrix()
        noisy_rot = frame.rotation @ r_perturbation

        # Apply point cloud noise
        noisy_cloud = frame.point_cloud.copy()
        if noise.point_cloud_noise_std > 0:
            noisy_cloud += rng.normal(0, noise.point_cloud_noise_std, noisy_cloud.shape).astype(
                np.float32
            )

        # Replace some points with outliers
        if noise.point_cloud_outlier_ratio > 0 and len(noisy_cloud) > 0:
            n_outliers = int(len(noisy_cloud) * noise.point_cloud_outlier_ratio)
            if n_outliers > 0:
                outlier_idx = rng.choice(len(noisy_cloud), n_outliers, replace=False)
                noisy_cloud[outlier_idx] = rng.uniform(-10, 10, (n_outliers, 3)).astype(np.float32)

        noisy_frames.append(
            SensorFrame(
                timestamp=frame.timestamp,
                position=noisy_pos,
                rotation=noisy_rot,
                point_cloud=noisy_cloud,
            )
        )

    return noisy_frames


@dataclass
class EvalDataset:
    """A loaded dataset ready for replay."""

    name: str
    frames: list[SensorFrame] = field(default_factory=list)


# ─── SLAM Backend Protocol ─────────────────────────────────────────────────


@runtime_checkable
class SlamBackend(Protocol):
    """Protocol for pluggable SLAM backends.

    Any SLAM module that implements these three methods can be evaluated.
    """

    def reset(self) -> None:
        """Reset internal state for a fresh run."""
        ...

    def process_frame(
        self,
        rotation: np.ndarray,
        translation: np.ndarray,
        timestamp: float,
        point_cloud: np.ndarray,
    ) -> bool:
        """Feed one sensor frame. Returns True if a keyframe was added."""
        ...

    def get_trajectory(self) -> list[TrajectoryPose]:
        """Return the estimated trajectory so far."""
        ...


# ─── Module Backend (wraps any DimOS Module) ─────────────────────────────


class ModuleBackend:
    """SlamBackend that wraps any DimOS Module with odometry streams.

    Automatically detects ``In[Odometry]``, ``In[PointCloud2]``, and
    ``Out[Odometry]`` streams on the module, instantiates it, and replays
    data through the module's actual callbacks.

    Usage::

        from dimos.navigation.smart_nav.modules.pgo.pgo import PGO
        results = slam_eval(ModuleBackend(PGO))

    Or more concisely — ``slam_eval`` accepts a Module class directly::

        results = slam_eval(PGO, dataset="tum_fr1_desk")
    """

    def __init__(self, module_class: type, **config_overrides: Any) -> None:
        if not (isinstance(module_class, type) and issubclass(module_class, Module)):
            raise TypeError(f"Expected a DimOS Module class, got {type(module_class)}")

        self._module_class = module_class
        self._config_overrides = config_overrides
        self._module: Any = None
        self._trajectory: list[TrajectoryPose] = []

        # Discover stream names from type annotations
        self._odom_in_name: str | None = None
        self._scan_in_name: str | None = None
        self._odom_out_name: str | None = None

        # Annotations may be strings (from __future__ annotations) or types.
        # Match both forms.
        hints: dict[str, Any] = {}
        for cls in reversed(module_class.__mro__):
            if hasattr(cls, "__annotations__"):
                hints.update(cls.__annotations__)

        for attr_name, annotation in hints.items():
            ann_str = annotation if isinstance(annotation, str) else repr(annotation)
            if "In[" in ann_str and "Odometry" in ann_str:
                self._odom_in_name = attr_name
            elif "In[" in ann_str and "PointCloud2" in ann_str:
                self._scan_in_name = attr_name
            elif "Out[" in ann_str and "Odometry" in ann_str and self._odom_out_name is None:
                self._odom_out_name = attr_name

        if self._odom_in_name is None:
            raise ValueError(f"{module_class.__name__} has no In[Odometry] stream")
        if self._scan_in_name is None:
            raise ValueError(f"{module_class.__name__} has no In[PointCloud2] stream")
        if self._odom_out_name is None:
            raise ValueError(f"{module_class.__name__} has no Out[Odometry] stream")

        self._init_module()

    def _init_module(self) -> None:
        """Instantiate and start the module."""
        self._trajectory = []
        self._module = self._module_class(**self._config_overrides)

        # Create a minimal local transport for In streams so subscribe() works.
        # Normally the coordinator sets this up, but we're running standalone.
        class _LocalTransport(Transport):  # type: ignore[type-arg]
            def __init__(self) -> None:
                self._callbacks: list[Any] = []

            def subscribe(self, callback: Any, selfstream: Any = None) -> Any:
                self._callbacks.append(callback)

                def unsub() -> None:
                    self._callbacks.remove(callback)

                return unsub

            def broadcast(self, selfstream: Any, value: Any) -> None:
                for cb in list(self._callbacks):
                    cb(value)

            def start(self) -> None:
                pass

            def stop(self) -> None:
                pass

        # Assign local transports to all In/Out streams on the module
        for attr_name in dir(self._module):
            attr = getattr(self._module, attr_name, None)
            if isinstance(attr, (In, Out)) and attr._transport is None:
                attr._transport = _LocalTransport()

        # Subscribe to the output odometry stream to capture trajectory
        out_stream = getattr(self._module, self._odom_out_name)
        out_stream.subscribe(self._on_output_odom)

        # Start the module (registers its internal callbacks on In streams)
        self._module.start()

    def _on_output_odom(self, msg: Any) -> None:
        """Capture output odometry into trajectory."""
        q = [
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ]
        r = Rotation.from_quat(q).as_matrix()
        t = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self._trajectory.append(TrajectoryPose(timestamp=msg.ts, position=t, rotation=r))

    def reset(self) -> None:
        if self._module is not None:
            try:
                self._module.stop()
            except Exception:
                pass
        self._init_module()

    def process_frame(
        self,
        rotation: np.ndarray,
        translation: np.ndarray,
        timestamp: float,
        point_cloud: np.ndarray,
    ) -> bool:
        assert self._module is not None

        # Build Odometry message
        q = Rotation.from_matrix(rotation).as_quat()  # [x,y,z,w]
        odom = Odometry(
            ts=timestamp,
            frame_id="odom",
            child_frame_id="body",
            pose=Pose(position=translation.tolist(), orientation=q.tolist()),
        )

        # Build PointCloud2 from numpy
        cloud = PointCloud2.from_numpy(point_cloud, frame_id="world", timestamp=timestamp)

        # Feed data through the module's In stream transports
        odom_in = getattr(self._module, self._odom_in_name)  # type: ignore[arg-type]
        scan_in = getattr(self._module, self._scan_in_name)  # type: ignore[arg-type]

        # Publish odometry first (module needs it before processing scan)
        odom_in._transport.broadcast(odom_in, odom)

        # Publish scan (triggers keyframe detection, loop closure, etc.)
        n_before = len(self._trajectory)
        scan_in._transport.broadcast(scan_in, cloud)

        return len(self._trajectory) > n_before

    def get_trajectory(self) -> list[TrajectoryPose]:
        return list(self._trajectory)


# ─── Dataset Loaders ───────────────────────────────────────────────────────

# TUM RGB-D benchmark: the standard benchmark for visual SLAM evaluation.
# Full .tgz archives with RGB + depth images + ground truth.
# Reference: Sturm et al., "A Benchmark for the Evaluation of RGB-D SLAM
# Systems", IROS 2012.

_TUM_BASE_URL = "https://cvg.cit.tum.de/rgbd/dataset"

# Map short names to (freiburg_num, sequence_name, approx_size_mb)
_TUM_SEQUENCES: dict[str, tuple[int, str, int]] = {
    # ── Static scenes (baseline benchmarks, most cited) ──
    "tum_fr1_xyz": (1, "xyz", 470),
    "tum_fr1_rpy": (1, "rpy", 420),
    "tum_fr1_desk": (1, "desk", 360),
    "tum_fr1_desk2": (1, "desk2", 370),
    "tum_fr1_room": (1, "room", 830),
    "tum_fr2_xyz": (2, "xyz", 2390),
    "tum_fr2_desk": (2, "desk", 2010),
    "tum_fr3_long_office_household": (3, "long_office_household", 1580),
    # ── Dynamic scenes (robustness testing) ──
    "tum_fr3_walking_xyz": (3, "walking_xyz", 540),
    "tum_fr3_walking_static": (3, "walking_static", 480),
    "tum_fr3_walking_halfsphere": (3, "walking_halfsphere", 650),
    "tum_fr3_walking_rpy": (3, "walking_rpy", 540),
    "tum_fr3_sitting_xyz": (3, "sitting_xyz", 790),
    "tum_fr3_sitting_static": (3, "sitting_static", 440),
    # ── Structure vs texture (diagnostic) ──
    "tum_fr3_nostructure_texture_far": (3, "nostructure_texture_far", 200),
    "tum_fr3_structure_texture_near": (3, "structure_texture_near", 600),
}

# TUM camera intrinsics (recommended defaults from the benchmark website).
# Using the ROS default is recommended by TUM because undistorting the
# pre-registered depth images is non-trivial.
_TUM_INTRINSICS: dict[str, tuple[float, float, float, float]] = {
    # (fx, fy, cx, cy)
    "default": (525.0, 525.0, 319.5, 239.5),
    "freiburg1": (517.3, 516.5, 318.6, 255.3),
    "freiburg2": (520.9, 521.0, 325.1, 249.7),
    "freiburg3": (535.4, 539.2, 320.1, 247.6),
}

_TUM_DEPTH_FACTOR = 5000.0  # pixel value / 5000 = meters


_DOWNLOAD_TIMEOUT_SECONDS = 600
_DOWNLOAD_CHUNK_SIZE = 8192


def _download_file(url: str, dest: Path, expected_size_mb: int = 0) -> None:
    """Download a file with progress logging, timeout, and integrity checks."""
    logger.info("Downloading (~%d MB) from %s", expected_size_mb, url)
    response = requests.get(url, stream=True, timeout=_DOWNLOAD_TIMEOUT_SECONDS)
    response.raise_for_status()

    sha256 = hashlib.sha256()
    downloaded = 0
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK_SIZE):
            f.write(chunk)
            sha256.update(chunk)
            downloaded += len(chunk)

    actual_mb = downloaded / (1024 * 1024)
    digest = sha256.hexdigest()
    logger.info("Download complete: %s (%.1f MB, sha256=%s)", dest.name, actual_mb, digest[:16])


def _download_tum_sequence(sequence: str) -> Path:
    """Download a full TUM RGB-D sequence (.tgz) if not already cached.

    Downloads the complete archive with RGB images, depth images, and
    ground truth trajectory. Verifies integrity via SHA-256 checksum
    when available. Extracts to the cache directory.
    """
    if sequence not in _TUM_SEQUENCES:
        available = list(_TUM_SEQUENCES.keys())
        raise ValueError(f"Unknown TUM sequence: {sequence}. Available: {available}")

    fr_num, seq_name, size_mb = _TUM_SEQUENCES[sequence]
    tgz_name = f"rgbd_dataset_freiburg{fr_num}_{seq_name}"
    url = f"{_TUM_BASE_URL}/freiburg{fr_num}/{tgz_name}.tgz"

    cache_dir = DATASET_CACHE_DIR / sequence
    extracted_dir = cache_dir / tgz_name

    # Check if already extracted
    if (extracted_dir / "groundtruth.txt").exists():
        logger.info("Dataset %s already cached at %s", sequence, extracted_dir)
        return extracted_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    tgz_path = cache_dir / f"{tgz_name}.tgz"

    if not tgz_path.exists():
        _download_file(url, tgz_path, expected_size_mb=size_mb)

    logger.info("Extracting %s...", tgz_path.name)
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(cache_dir, filter="data")
    logger.info("Extracted to %s", extracted_dir)

    # Clean up .tgz to save disk space
    tgz_path.unlink()

    return extracted_dir


def _load_tum_groundtruth(gt_file: Path) -> list[TrajectoryPose]:
    """Parse a TUM-format groundtruth.txt file.

    Format: timestamp tx ty tz qx qy qz qw
    """
    poses = []
    with open(gt_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            ts = float(parts[0])
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            r = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            poses.append(
                TrajectoryPose(
                    timestamp=ts,
                    position=np.array([tx, ty, tz]),
                    rotation=r,
                )
            )
    return poses


def _load_tum_timestamp_file(path: Path) -> list[tuple[float, str]]:
    """Parse a TUM rgb.txt or depth.txt file.

    Returns list of (timestamp, filename) tuples.
    """
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                entries.append((float(parts[0]), parts[1]))
    return entries


def _associate_timestamps(
    first: list[tuple[float, str]],
    second: list[tuple[float, str]],
    max_diff: float = 0.02,
) -> list[tuple[tuple[float, str], tuple[float, str]]]:
    """Associate two timestamp-filename lists by nearest timestamp.

    Reimplements TUM's associate.py algorithm.
    """
    matches = []
    second_times = np.array([t for t, _ in second])
    for t1, f1 in first:
        idx = int(np.argmin(np.abs(second_times - t1)))
        if abs(t1 - second[idx][0]) < max_diff:
            matches.append(((t1, f1), second[idx]))
    return matches


def _depth_to_pointcloud(
    depth: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    max_depth: float = 8.0,
    subsample: int = 4,
) -> np.ndarray:
    """Convert a depth image to a 3D point cloud in camera frame.

    Args:
        depth: (H, W) float32 depth in meters
        fx, fy, cx, cy: Camera intrinsics
        max_depth: Discard points beyond this depth
        subsample: Take every Nth pixel (reduces point count for efficiency)

    Returns:
        (N, 3) float32 point cloud in camera/body frame
    """
    h, w = depth.shape
    u = np.arange(0, w, subsample)
    v = np.arange(0, h, subsample)
    u, v = np.meshgrid(u, v)
    u = u.flatten()
    v = v.flatten()

    z = depth[v, u]
    valid = (z > 0) & (z < max_depth)
    z = z[valid]
    u = u[valid]
    v = v[valid]

    x = (u.astype(np.float32) - cx) * z / fx
    y = (v.astype(np.float32) - cy) * z / fy

    return np.column_stack([x, y, z]).astype(np.float32)


def _generate_synthetic_cloud(
    n_points: int = 200,
    spread: float = 3.0,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a synthetic point cloud in body frame (centered at origin)."""
    rng = np.random.default_rng(seed)
    return rng.normal(0, spread, (n_points, 3)).astype(np.float32)


def load_dataset(
    name: str,
    *,
    max_frames: int | None = None,
    cloud_points: int = 200,
) -> EvalDataset:
    """Load a dataset for SLAM evaluation.

    Supports:
    - ``"synthetic_circle"``: Synthetic circular trajectory with generated point clouds
    - ``"synthetic_figure8"``: Synthetic figure-8 with loop closure opportunity
    - ``"tum_*"``: TUM RGB-D benchmark (auto-downloaded). Downloads the full
      sequence with real RGB + depth images, converts depth to point clouds.

    TUM sequences available (auto-downloaded on first use):

    **Static (baseline):** ``tum_fr1_xyz`` (470MB), ``tum_fr1_desk`` (360MB),
    ``tum_fr1_desk2`` (370MB), ``tum_fr1_room`` (830MB), ``tum_fr2_desk`` (2GB),
    ``tum_fr3_long_office_household`` (1.6GB)

    **Dynamic:** ``tum_fr3_walking_xyz`` (540MB), ``tum_fr3_walking_static`` (480MB),
    ``tum_fr3_walking_halfsphere`` (650MB), ``tum_fr3_sitting_xyz`` (790MB),
    ``tum_fr3_sitting_static`` (440MB)

    **Smallest (CI-friendly):** ``tum_fr3_nostructure_texture_far`` (200MB)

    Args:
        name: Dataset name
        max_frames: Limit number of frames (None = all)
        cloud_points: Number of points per synthetic cloud (synthetic datasets only)
    """
    if name == "synthetic_circle":
        return _make_synthetic_circle(max_frames or 200, cloud_points)
    elif name == "synthetic_figure8":
        return _make_synthetic_figure8(max_frames or 300, cloud_points)
    elif name.startswith("tum_"):
        return _load_tum_dataset(name, max_frames=max_frames)
    else:
        available = ["synthetic_circle", "synthetic_figure8", *_TUM_SEQUENCES.keys()]
        raise ValueError(f"Unknown dataset: {name}. Available: {available}")


def _make_synthetic_circle(
    n_frames: int,
    cloud_points: int,
) -> EvalDataset:
    """Generate a circular trajectory dataset."""
    frames = []
    radius = 5.0
    for i in range(n_frames):
        angle = 2.0 * math.pi * i / n_frames
        t = float(i) * 0.1
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        z = 0.0
        # Heading tangent to circle
        yaw = angle + math.pi / 2
        r = Rotation.from_euler("z", yaw).as_matrix()
        cloud = _generate_synthetic_cloud(cloud_points, seed=i)
        frames.append(
            SensorFrame(
                timestamp=t,
                position=np.array([x, y, z]),
                rotation=r,
                point_cloud=cloud,
            )
        )
    return EvalDataset(name="synthetic_circle", frames=frames)


def _make_synthetic_figure8(
    n_frames: int,
    cloud_points: int,
) -> EvalDataset:
    """Generate a figure-8 trajectory (has natural loop closure)."""
    frames = []
    scale = 5.0
    for i in range(n_frames):
        angle = 2.0 * math.pi * i / n_frames
        t = float(i) * 0.1
        x = scale * math.sin(angle)
        y = scale * math.sin(angle) * math.cos(angle)
        z = 0.0
        # Heading from finite difference of position
        angle_next = 2.0 * math.pi * (i + 1) / n_frames
        x_next = scale * math.sin(angle_next)
        y_next = scale * math.sin(angle_next) * math.cos(angle_next)
        dx = x_next - x
        dy = y_next - y
        yaw = math.atan2(dy, dx) if (abs(dx) + abs(dy)) > 1e-9 else 0.0
        r = Rotation.from_euler("z", yaw).as_matrix()
        cloud = _generate_synthetic_cloud(cloud_points, seed=i)
        frames.append(
            SensorFrame(
                timestamp=t,
                position=np.array([x, y, z]),
                rotation=r,
                point_cloud=cloud,
            )
        )
    return EvalDataset(name="synthetic_figure8", frames=frames)


def _load_tum_dataset(
    name: str,
    *,
    max_frames: int | None = None,
) -> EvalDataset:
    """Load a TUM RGB-D dataset with real RGB + depth images.

    Downloads the full .tgz archive on first use, extracts it, and loads
    the actual sensor data. Depth images are converted to 3D point clouds
    using the camera intrinsics. Each SensorFrame contains:
    - ground truth pose (from groundtruth.txt)
    - point_cloud (from depth image projection)
    - rgb_image (640x480 uint8)
    - depth_image (640x480 float32 meters)
    """
    seq_dir = _download_tum_sequence(name)

    # Determine intrinsics based on freiburg camera
    fr_num = _TUM_SEQUENCES[name][0]
    intrinsics_key = f"freiburg{fr_num}"
    fx, fy, cx, cy = _TUM_INTRINSICS.get(intrinsics_key, _TUM_INTRINSICS["default"])

    # Load ground truth, RGB index, and depth index
    gt_poses = _load_tum_groundtruth(seq_dir / "groundtruth.txt")
    rgb_entries = _load_tum_timestamp_file(seq_dir / "rgb.txt")
    depth_entries = _load_tum_timestamp_file(seq_dir / "depth.txt")

    # Associate RGB and depth by timestamp
    rgb_depth_pairs = _associate_timestamps(rgb_entries, depth_entries, max_diff=0.02)

    if not rgb_depth_pairs:
        raise RuntimeError(f"No RGB-depth pairs found in {seq_dir}. Dataset may be corrupt.")

    # Associate paired frames with ground truth
    gt_times = np.array([p.timestamp for p in gt_poses])
    frames = []
    n_loaded = 0

    for (rgb_ts, rgb_file), (_depth_ts, depth_file) in rgb_depth_pairs:
        if max_frames is not None and n_loaded >= max_frames:
            break

        # Find nearest ground truth pose
        frame_ts = rgb_ts
        gt_idx = int(np.argmin(np.abs(gt_times - frame_ts)))
        if abs(gt_poses[gt_idx].timestamp - frame_ts) > _MAX_TIMESTAMP_DIFF:
            continue  # no close enough ground truth

        gt_pose = gt_poses[gt_idx]

        # Load images
        rgb_path = seq_dir / rgb_file
        depth_path = seq_dir / depth_file

        if not rgb_path.exists() or not depth_path.exists():
            continue

        rgb_img = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb_img is None:
            continue
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            continue
        depth_meters = depth_raw.astype(np.float32) / _TUM_DEPTH_FACTOR

        # Convert depth to point cloud
        cloud = _depth_to_pointcloud(depth_meters, fx, fy, cx, cy)

        frames.append(
            SensorFrame(
                timestamp=frame_ts,
                position=gt_pose.position.copy(),
                rotation=gt_pose.rotation.copy(),
                point_cloud=cloud,
                rgb_image=rgb_img,
                depth_image=depth_meters,
            )
        )
        n_loaded += 1

    logger.info(
        "Loaded %d frames from %s (intrinsics: %s, fx=%.1f fy=%.1f)",
        len(frames),
        name,
        intrinsics_key,
        fx,
        fy,
    )
    return EvalDataset(name=name, frames=frames)


# ─── Metrics Computation ──────────────────────────────────────────────────


def _align_trajectories(
    estimated: list[TrajectoryPose],
    ground_truth: list[TrajectoryPose],
) -> list[tuple[TrajectoryPose, TrajectoryPose]]:
    """Align estimated and ground truth trajectories by nearest timestamp.

    Returns pairs of (estimated, ground_truth) poses.

    Note: multiple estimated poses may match the same ground truth pose
    if the estimated trajectory has higher frequency than ground truth.
    """
    if not estimated or not ground_truth:
        return []

    gt_times = np.array([p.timestamp for p in ground_truth])
    pairs = []
    for est in estimated:
        idx = int(np.argmin(np.abs(gt_times - est.timestamp)))
        time_diff = abs(est.timestamp - ground_truth[idx].timestamp)
        # Only match if within 0.1s
        if time_diff < _MAX_TIMESTAMP_DIFF:
            pairs.append((est, ground_truth[idx]))
    return pairs


def compute_ate(pairs: list[tuple[TrajectoryPose, TrajectoryPose]]) -> dict[str, float]:
    """Compute Absolute Trajectory Error (ATE).

    ATE measures the global consistency of the trajectory by comparing
    estimated and ground truth positions. No Umeyama SE(3) alignment is
    applied — the SLAM backend is expected to output poses in the same
    coordinate frame as the ground truth.

    Returns dict with rmse, mean, median, std, max.
    """
    if not pairs:
        return {
            "rmse": float("nan"),
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "max": float("nan"),
            "n_pairs": 0,
        }

    # Compute translation errors (without Umeyama alignment for simplicity —
    # SLAM modules should already output in the same frame as ground truth)
    errors = []
    for est, gt in pairs:
        err = np.linalg.norm(est.position - gt.position)
        errors.append(err)

    errors_arr = np.array(errors)
    return {
        "rmse": float(np.sqrt(np.mean(errors_arr**2))),
        "mean": float(np.mean(errors_arr)),
        "median": float(np.median(errors_arr)),
        "std": float(np.std(errors_arr)),
        "max": float(np.max(errors_arr)),
        "n_pairs": len(pairs),
    }


def compute_rpe(
    pairs: list[tuple[TrajectoryPose, TrajectoryPose]],
    delta: int = 1,
) -> dict[str, float]:
    """Compute Relative Pose Error (RPE).

    RPE measures the local accuracy by comparing relative transforms between
    consecutive pose pairs. Tests odometry drift.

    Args:
        pairs: Aligned (estimated, ground_truth) pairs
        delta: Frame step for relative comparison

    Returns dict with trans_rmse, trans_mean, rot_rmse, rot_mean (rotation in degrees).
    """
    if len(pairs) < delta + 1:
        return {
            "trans_rmse": float("nan"),
            "trans_mean": float("nan"),
            "rot_rmse": float("nan"),
            "rot_mean": float("nan"),
            "n_pairs": 0,
        }

    trans_errors = []
    rot_errors = []

    for i in range(len(pairs) - delta):
        est_i, gt_i = pairs[i]
        est_j, gt_j = pairs[i + delta]

        # Ground truth relative transform in local frame: T_i^{-1} * T_j
        gt_rel_r = gt_i.rotation.T @ gt_j.rotation
        gt_rel_t = gt_i.rotation.T @ (gt_j.position - gt_i.position)

        # Estimated relative transform in local frame: T_i^{-1} * T_j
        est_rel_r = est_i.rotation.T @ est_j.rotation
        est_rel_t = est_i.rotation.T @ (est_j.position - est_i.position)

        # Translation error
        trans_err = np.linalg.norm(est_rel_t - gt_rel_t)
        trans_errors.append(trans_err)

        # Rotation error (angle of the difference rotation)
        r_diff = gt_rel_r.T @ est_rel_r
        angle = np.arccos(np.clip((np.trace(r_diff) - 1.0) / 2.0, -1.0, 1.0))
        rot_errors.append(math.degrees(angle))

    trans_arr = np.array(trans_errors)
    rot_arr = np.array(rot_errors)
    return {
        "trans_rmse": float(np.sqrt(np.mean(trans_arr**2))),
        "trans_mean": float(np.mean(trans_arr)),
        "rot_rmse": float(np.sqrt(np.mean(rot_arr**2))),
        "rot_mean": float(np.mean(rot_arr)),
        "n_pairs": len(trans_errors),
    }


def compute_drift(
    pairs: list[tuple[TrajectoryPose, TrajectoryPose]],
) -> dict[str, float]:
    """Compute trajectory drift — error growth over distance traveled.

    Returns drift as a percentage: (final_error / total_distance) * 100.
    """
    if len(pairs) < 2:
        return {"drift_pct": float("nan"), "total_distance": 0.0, "final_error": float("nan")}

    # Total ground truth distance
    total_dist = 0.0
    for i in range(1, len(pairs)):
        total_dist += float(np.linalg.norm(pairs[i][1].position - pairs[i - 1][1].position))

    if total_dist < 1e-6:
        return {"drift_pct": float("nan"), "total_distance": 0.0, "final_error": float("nan")}

    final_error = float(np.linalg.norm(pairs[-1][0].position - pairs[-1][1].position))
    return {
        "drift_pct": (final_error / total_dist) * 100.0,
        "total_distance": total_dist,
        "final_error": final_error,
    }


def _resolve_noise(noise: NoiseProfile | str | None) -> NoiseProfile | None:
    """Resolve a noise argument to a NoiseProfile or None."""
    if noise is None:
        return None
    if isinstance(noise, NoiseProfile):
        return noise
    preset_map: dict[str, Any] = {
        "none": NoiseProfile.none,
        "lidar_mild": NoiseProfile.lidar_mild,
        "lidar_harsh": NoiseProfile.lidar_harsh,
        "visual_mild": NoiseProfile.visual_mild,
        "visual_harsh": NoiseProfile.visual_harsh,
        "wheel_odom": NoiseProfile.wheel_odom,
    }
    if noise not in preset_map:
        raise ValueError(f"Unknown noise preset: {noise!r}. Available: {list(preset_map.keys())}")
    return preset_map[noise]()


# ─── Main Entry Point ─────────────────────────────────────────────────────


def slam_eval(
    backend: SlamBackend | type,
    dataset: str | EvalDataset = "synthetic_circle",
    *,
    noise: NoiseProfile | str | None = None,
    output_path: str | Path | None = None,
    max_frames: int | None = None,
    cloud_points: int = 200,
    **module_config: Any,
) -> dict[str, Any]:
    """Run SLAM evaluation on a dataset and return metrics.

    Args:
        backend: Either a ``SlamBackend`` instance, or a DimOS Module class
            (e.g. ``PGO``). Module classes are auto-wrapped via ``ModuleBackend``.
        dataset: Dataset name (string) or pre-loaded ``EvalDataset``
        noise: Noise profile to corrupt input odometry. Can be a ``NoiseProfile``
            instance, a preset name (``"none"``, ``"lidar_mild"``, ``"lidar_harsh"``,
            ``"visual_mild"``, ``"visual_harsh"``, ``"wheel_odom"``), or ``None``
            (no noise).
        output_path: Path to write metrics.json (default: module dir)
        max_frames: Limit number of frames
        cloud_points: Points per synthetic cloud (for datasets without real clouds)
        **module_config: Config overrides passed to ``ModuleBackend`` when
            ``backend`` is a Module class.

    Returns:
        Dict with ate, rpe, drift, timing, and dataset metadata.

    Examples::

        # Pass a DimOS module class directly:
        from dimos.navigation.smart_nav.modules.pgo.pgo import PGO
        results = slam_eval(PGO, dataset="tum_fr1_desk")

        # Or with config overrides:
        results = slam_eval(PGO, dataset="tum_fr1_desk",
                            key_pose_delta_trans=0.3)

        # Or pass a custom SlamBackend:
        results = slam_eval(my_backend, dataset="synthetic_circle")
    """
    # Auto-wrap Module classes
    if isinstance(backend, type):
        backend = ModuleBackend(backend, **module_config)
    elif module_config:
        logger.warning(
            "module_config kwargs (%s) are ignored when backend is an instance, not a class",
            list(module_config.keys()),
        )

    # Resolve noise profile
    noise_profile = _resolve_noise(noise)

    # Load dataset
    if isinstance(dataset, str):
        ds = load_dataset(dataset, max_frames=max_frames, cloud_points=cloud_points)
    else:
        ds = dataset

    if not ds.frames:
        raise ValueError(f"Dataset {ds.name} has no frames")

    # Apply noise to input frames (ground truth is preserved separately)
    if noise_profile is not None:
        replay_frames = apply_noise(ds.frames, noise_profile)
    else:
        replay_frames = ds.frames

    # Reset backend
    backend.reset()

    # Replay frames (potentially noisy)
    logger.info(
        "Replaying %d frames from %s (noise=%s)",
        len(ds.frames),
        ds.name,
        noise_profile.label if noise_profile else "none",
    )
    t_start = time.monotonic()
    keyframes_added = 0

    for frame in replay_frames:
        added = backend.process_frame(
            rotation=frame.rotation,
            translation=frame.position,
            timestamp=frame.timestamp,
            point_cloud=frame.point_cloud,
        )
        if added:
            keyframes_added += 1

    t_elapsed = time.monotonic() - t_start

    # Get estimated trajectory
    estimated = backend.get_trajectory()

    # Build ground truth trajectory
    ground_truth = [
        TrajectoryPose(
            timestamp=f.timestamp,
            position=f.position.copy(),
            rotation=f.rotation.copy(),
        )
        for f in ds.frames
    ]

    # Align and compute metrics
    pairs = _align_trajectories(estimated, ground_truth)

    ate = compute_ate(pairs)
    rpe = compute_rpe(pairs)
    drift = compute_drift(pairs)

    results: dict[str, Any] = {
        "dataset": ds.name,
        "n_frames": len(ds.frames),
        "n_keyframes": keyframes_added,
        "n_aligned_pairs": len(pairs),
        "ate": ate,
        "rpe": rpe,
        "drift": drift,
        "timing": {
            "total_seconds": round(t_elapsed, 4),
            "fps": round(len(ds.frames) / t_elapsed, 2) if t_elapsed > 0 else 0.0,
        },
        "backend": type(backend).__name__,
        "noise": {
            "label": noise_profile.label if noise_profile else "none",
            "translation_noise_std": noise_profile.translation_noise_std if noise_profile else 0.0,
            "rotation_noise_std_deg": noise_profile.rotation_noise_std_deg
            if noise_profile
            else 0.0,
            "translation_drift_rate": noise_profile.translation_drift_rate
            if noise_profile
            else 0.0,
            "rotation_drift_rate_deg": noise_profile.rotation_drift_rate_deg
            if noise_profile
            else 0.0,
            "point_cloud_noise_std": noise_profile.point_cloud_noise_std if noise_profile else 0.0,
            "point_cloud_outlier_ratio": noise_profile.point_cloud_outlier_ratio
            if noise_profile
            else 0.0,
        },
    }

    # Write output
    if output_path is None:
        output_path = Path(__file__).parent / "metrics.json"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Metrics written to %s", output_path)

    # Clean up ModuleBackend (stops the module's threads)
    if isinstance(backend, ModuleBackend) and backend._module is not None:
        try:
            backend._module.stop()
        except Exception:
            pass

    return results
