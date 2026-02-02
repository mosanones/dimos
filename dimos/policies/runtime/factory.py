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

from typing import TYPE_CHECKING

from dimos.utils.logging_config import setup_logger

from ..adapters.falcon import FalconLocoManipAdapter
from ..adapters.mjlab import MjlabVelocityAdapter

if TYPE_CHECKING:
    from .adapter import PolicyAdapter
    from .policy_spec import PolicySpec

logger = setup_logger()


def build_policy_adapter(spec: PolicySpec) -> PolicyAdapter:
    logger.info(
        "Starting policy runtime",
        policy_type=spec.policy_type,
        policy_path=spec.policy_path,
        robot_type=spec.robot_type,
    )

    if spec.policy_type == "mjlab_velocity":
        return MjlabVelocityAdapter(policy_path=spec.policy_path)

    if spec.policy_type == "falcon_loco_manip":
        return FalconLocoManipAdapter(
            policy_path=spec.policy_path,
            falcon_yaml_path=spec.policy_config,
            policy_action_scale=spec.policy_action_scale,
        )

    raise ValueError(f"Unknown policy_type: {spec.policy_type}")
