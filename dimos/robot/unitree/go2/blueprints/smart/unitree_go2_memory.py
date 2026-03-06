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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from dimos.core.blueprints import autoconnect
from dimos.core.core import rpc
from dimos.memory.module import MemoryModule, MemoryModuleConfig
from dimos.msgs.sensor_msgs import Image, PointCloud2
from dimos.robot.unitree.go2.blueprints.smart.unitree_go2 import unitree_go2

if TYPE_CHECKING:
    from dimos.core.stream import In


@dataclass
class UnitreeGo2MemoryConfig(MemoryModuleConfig):
    image_fps: float = 5.0
    enable_clip: bool = False


class UnitreeGo2Memory(MemoryModule):
    color_image: In[Image]
    lidar: In[PointCloud2]

    config: UnitreeGo2MemoryConfig  # type: ignore[assignment]
    default_config: type[UnitreeGo2MemoryConfig] = UnitreeGo2MemoryConfig

    @rpc
    def start(self) -> None:
        super().start()
        self._images = self.record(self.color_image, "images", Image, fps=self.config.image_fps)
        if self.lidar._transport is not None:
            self._pointclouds = self.record(self.lidar, "pointclouds", PointCloud2)

        if self.config.enable_clip:
            self._setup_clip_pipeline()

    def _setup_clip_pipeline(self) -> None:
        from dimos.memory.transformer import EmbeddingTransformer
        from dimos.models.embedding.clip import CLIPModel

        clip = CLIPModel()
        clip.start()

        self._embeddings: Any = self._images.transform(EmbeddingTransformer(clip), live=True).store(
            "clip_embeddings"
        )


unitree_go2_memory = autoconnect(
    unitree_go2,
    UnitreeGo2Memory.blueprint(),
).global_config(n_workers=8)

__all__ = ["UnitreeGo2Memory", "unitree_go2_memory"]
