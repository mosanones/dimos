from __future__ import annotations

from .runtime import PolicyRuntime


class PolicyRunner:
    """Thin wrapper around PolicyRuntime with a stable runner interface."""

    def __init__(self, runtime: PolicyRuntime) -> None:
        self._rt = runtime

    def step(self) -> None:
        self._rt.step()

    def set_cmd_vel(self, vx: float, vy: float, wz: float) -> None:
        self._rt.set_cmd_vel(vx, vy, wz)

    def set_command(self, vx: float, vy: float, wz: float) -> None:
        """Back-compat alias for legacy callers."""
        self.set_cmd_vel(vx, vy, wz)

    def set_enabled(self, enabled: bool) -> None:
        self._rt.set_enabled(enabled)

    def set_estop(self, estop: bool) -> None:
        self._rt.set_estop(estop)

    def set_policy_params_json(self, params_json: str) -> None:
        self._rt.set_policy_params_json(params_json)

