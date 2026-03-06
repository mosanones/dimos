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

"""Memory module — record input streams into persistent memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import cv2

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.memory.impl.sqlite import SqliteStore
from dimos.msgs.sensor_msgs.Image import sharpness_barrier
from dimos.utils.logging_config import setup_logger

cv2.setNumThreads(1)

if TYPE_CHECKING:
    from reactivex.observable import Observable

    from dimos.core.stream import In
    from dimos.memory.store import Session
    from dimos.memory.stream import Stream
    from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped

logger = setup_logger()


@dataclass
class RecordSpec:
    """Declares an input stream to record."""

    input_name: str
    stream_name: str
    payload_type: type | None = None
    fps: float = 0
    """Target FPS. If >0, uses sharpness_barrier to select best frame per window."""


@dataclass
class MemoryModuleConfig(ModuleConfig):
    db_path: str = "memory.db"
    world_frame: str = "world"
    robot_frame: str = "base_link"
    records: list[RecordSpec] = field(default_factory=list)


class MemoryModule(Module[MemoryModuleConfig]):
    default_config: type[MemoryModuleConfig] = MemoryModuleConfig

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._store: SqliteStore | None = None
        self._session: Session | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────

    def pose(self) -> PoseStamped | None:
        return self.tf.get_pose(self.config.world_frame, self.config.robot_frame)  # type: ignore[no-any-return]

    @rpc
    def start(self) -> None:
        super().start()
        self._store = SqliteStore(self.config.db_path)
        self._session = self._store.session()
        self._disposables.add(self._session)

        # Auto-record streams declared in config
        for spec in self.config.records:
            input_stream: In[Any] = getattr(self, spec.input_name)
            self.record(
                input_stream,
                spec.stream_name,
                spec.payload_type,
                fps=spec.fps,
            )

        logger.info("MemoryModule started (db=%s)", self.config.db_path)

    def record(
        self,
        input: In[Any],
        name: str,
        payload_type: type | None = None,
        fps: float = 0,
    ) -> Stream[Any]:
        assert self._store is not None, "record() called before start()"
        session = self._store.session()
        self._disposables.add(session)
        stream = session.stream(name, payload_type, pose_provider=self.pose)

        obs: Observable[Any] = input.observable()
        if fps > 0:
            obs = obs.pipe(sharpness_barrier(fps))

        def _on_item(item: Any) -> None:
            stream.append(item, ts=getattr(item, "ts", None))

        self._disposables.add(obs.subscribe(on_next=_on_item))

        return stream

    @rpc
    def stop(self) -> None:
        self._session = None
        super().stop()  # disposes all sessions via CompositeDisposable
        if self._store is not None:
            self._store.close()
            self._store = None

    # ── Public API ────────────────────────────────────────────────────

    @property
    def session(self) -> Session:
        if self._session is None:
            raise RuntimeError("MemoryModule not started")
        return self._session

    @rpc
    def get_stats(self) -> dict[str, int]:
        if self._session is None:
            return {}
        return {s.name: s.count for s in self._session.list_streams()}


memory_module = MemoryModule.blueprint
memory_module = MemoryModule.blueprint
memory_module = MemoryModule.blueprint
