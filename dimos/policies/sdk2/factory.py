from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from dimos.simulation.mujoco.model import load_bundle_json
from dimos.utils.data import get_data
from dimos.utils.logging_config import setup_logger

from .adapters.falcon import FalconLocoManipAdapter
from .adapters.mjlab import MjlabVelocityAdapter
from .runner import PolicyRunner
from .runtime import PolicyRuntime, PolicyRuntimeConfig

logger = setup_logger()


@dataclass
class PolicyFactoryConfig:
    domain_id: int = 1
    interface: str = "lo0"
    control_dt: float = 0.02


class PolicyFactory:
    """Construct PolicyRunner instances from MuJoCo profile bundles."""

    @staticmethod
    def from_profile(profile: str, config: PolicyFactoryConfig) -> PolicyRunner | None:
        bundle_cfg = load_bundle_json(profile)
        if not bundle_cfg:
            logger.warning("No bundle.json found for profile", profile=profile)
            return None

        policy_name = bundle_cfg.get("policy")
        if not policy_name:
            logger.info("SDK2 mode: no policy in bundle.json, waiting for external policy")
            return None

        robot_type = str(bundle_cfg.get("robot_type", "g1"))
        policy_kind = str(bundle_cfg.get("policy_kind", "mjlab_velocity"))
        policy_config_name = bundle_cfg.get("policy_config")

        data_dir = Path(get_data("mujoco_sim"))
        policy_path = PolicyFactory._resolve_path(data_dir, profile, str(policy_name))
        if policy_path is None:
            logger.warning("SDK2 policy not found", policy=policy_name, profile=profile)
            return None

        logger.info(
            "Starting SDK2 policy runner",
            policy=str(policy_path),
            robot_type=robot_type,
            policy_kind=policy_kind,
        )

        if policy_kind == "mjlab_velocity":
            adapter = MjlabVelocityAdapter(policy_path=str(policy_path))
        elif policy_kind == "falcon_loco_manip":
            if not policy_config_name:
                raise RuntimeError("Falcon policy_kind requires bundle.json policy_config (YAML)")
            yaml_path = PolicyFactory._resolve_path(data_dir, profile, str(policy_config_name))
            if yaml_path is None:
                raise RuntimeError(f"Falcon policy_config not found: {policy_config_name}")
            adapter = FalconLocoManipAdapter(
                policy_path=str(policy_path),
                falcon_yaml_path=str(yaml_path),
                policy_action_scale=float(bundle_cfg.get("policy_action_scale", 0.25)),
            )
        else:
            raise RuntimeError(f"Unknown policy_kind '{policy_kind}'")

        rt = PolicyRuntime(
            adapter=adapter,
            config=PolicyRuntimeConfig(
                robot_type=robot_type,
                domain_id=config.domain_id,
                interface=config.interface,
                control_dt=config.control_dt,
                mode_pr=int(bundle_cfg.get("mode_pr", 0)),
            ),
        )
        return PolicyRunner(rt)

    @staticmethod
    def _resolve_path(data_dir: Path, profile: str, name: str) -> Path | None:
        candidate = data_dir / name
        if candidate.exists():
            return candidate
        candidate = data_dir / profile / name
        if candidate.exists():
            return candidate
        return None

