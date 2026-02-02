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

from typing import Literal

from pydantic import BaseModel, field_validator


class PolicySpec(BaseModel):
    policy_type: Literal["mjlab_velocity", "falcon_loco_manip"]
    policy_path: str
    robot_type: Literal["g1"] = "g1"
    mode_pr: int = 0

    # Falcon-only
    policy_config: str | None = None
    policy_action_scale: float = 0.25

    @field_validator("policy_config")
    @classmethod
    def _validate_falcon_config(cls, value: str | None, info):  # type: ignore[override]
        if info.data.get("policy_type") == "falcon_loco_manip" and not value:
            raise ValueError("Falcon policy_type requires policy_config")
        return value
