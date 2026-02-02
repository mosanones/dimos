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

import argparse
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber

from dimos.utils.logging_config import setup_logger

logger = setup_logger()

TOPIC_LOWCMD = "rt/lowcmd"
TOPIC_LOWSTATE = "rt/lowstate"


def _get_types(robot_type: str):
    if robot_type in ("g1", "h1_2"):
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_

        return LowCmd_, LowState_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_

    return LowCmd_, LowState_


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unitree DDS parity harness (lowstate/lowcmd rates)."
    )
    parser.add_argument("--robot-type", default="g1", help="Robot type (g1, h1_2, go2, ...)")
    parser.add_argument("--domain-id", type=int, default=1, help="DDS domain id")
    parser.add_argument("--interface", default="lo0", help="DDS interface (e.g., lo0, en7)")
    parser.add_argument("--interval", type=float, default=1.0, help="Reporting interval (s)")
    args = parser.parse_args()

    LowCmd_, LowState_ = _get_types(args.robot_type)
    ChannelFactoryInitialize(args.domain_id, args.interface)

    counts = {"lowcmd": 0, "lowstate": 0}

    def _lowcmd_cb(_msg):  # type: ignore[no-untyped-def]
        counts["lowcmd"] += 1

    def _lowstate_cb(_msg):  # type: ignore[no-untyped-def]
        counts["lowstate"] += 1

    sub_cmd = ChannelSubscriber(TOPIC_LOWCMD, LowCmd_)
    sub_state = ChannelSubscriber(TOPIC_LOWSTATE, LowState_)
    sub_cmd.Init(_lowcmd_cb, 10)
    sub_state.Init(_lowstate_cb, 10)

    logger.info(
        "Parity harness started",
        robot_type=args.robot_type,
        domain_id=args.domain_id,
        interface=args.interface,
    )

    last = time.perf_counter()
    last_counts = dict(counts)
    while True:
        time.sleep(args.interval)
        now = time.perf_counter()
        dt = max(now - last, 1e-6)
        rates = {
            "lowcmd_hz": (counts["lowcmd"] - last_counts["lowcmd"]) / dt,
            "lowstate_hz": (counts["lowstate"] - last_counts["lowstate"]) / dt,
        }
        logger.info("DDS rates", **rates)
        last = now
        last_counts = dict(counts)


if __name__ == "__main__":
    main()
