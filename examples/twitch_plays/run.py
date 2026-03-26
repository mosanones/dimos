#!/usr/bin/env python3
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

"""Twitch Plays Go2 — standalone example.

This is equivalent to ``dimos run unitree-go2-twitch`` but can be
customised directly in Python.

Usage::

    export DIMOS_TWITCH_TOKEN=oauth:...
    export DIMOS_CHANNEL_NAME=your_channel
    python examples/twitch_plays/run.py
"""

from dimos.robot.unitree.go2.blueprints.smart.unitree_go2_twitch import (
    unitree_go2_twitch,
)

if __name__ == "__main__":
    unitree_go2_twitch.build().loop()
