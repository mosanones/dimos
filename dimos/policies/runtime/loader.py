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

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .policy_spec import PolicySpec

if TYPE_CHECKING:
    from dimos.core.global_config import GlobalConfig

_POLICIES_DIR = Path(__file__).resolve().parent.parent.parent / "config" / "policies"


def _resolve_policy_profile(profile: str) -> Path:
    candidate = Path(profile)
    if candidate.exists():
        return candidate

    if not profile.endswith(".yaml"):
        profile = f"{profile}.yaml"
    candidate = _POLICIES_DIR / profile
    if candidate.exists():
        return candidate

    raise FileNotFoundError(f"Policy profile not found: {profile}")


def load_policy_spec(global_config: GlobalConfig) -> PolicySpec | None:
    if global_config.policy_profile:
        profile_path = _resolve_policy_profile(global_config.policy_profile)
        with profile_path.open("r", encoding="utf-8") as f:
            import yaml  # lazy import to avoid hard dependency when not using policy profiles

            data: dict[str, Any] = yaml.safe_load(f) or {}
        return PolicySpec.model_validate(data)

    if not global_config.policy_type or not global_config.policy_path:
        return None

    return PolicySpec(
        policy_type=global_config.policy_type,
        policy_path=global_config.policy_path,
        policy_config=global_config.policy_config,
        policy_action_scale=global_config.policy_action_scale or 0.25,
        mode_pr=global_config.policy_mode_pr or 0,
    )
