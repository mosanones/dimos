# Copyright 2025 Dimensional Inc.
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

import math
from typing import Optional
from dimos.core import Module, In, Out, rpc
from dimos.msgs.vision_msgs import Detection2DArray
from dimos.msgs.geometry_msgs import PoseStamped, Vector3, Quaternion
from dimos_lcm.sensor_msgs import CameraInfo
from dimos.protocol.skill import skill
from dimos.protocol.tf import TF


class BBoxNavigationModule(Module):
    """Module that converts 2D bbox to navigation goals with tracking capability."""

    detection2d: In[Detection2DArray] = None
    camera_info: In[CameraInfo] = None
    odom: In[PoseStamped] = None
    goal_request: Out[PoseStamped] = None

    tracking_target: Optional[str] = None

    def __init__(self, goal_distance: float = 1.0):
        super().__init__()
        self.goal_distance = goal_distance
        self.camera_intrinsics = None
        self.tracking_target = None
        self.tf = TF()
        self.latest_odom = None

    @rpc
    def start(self):
        self.camera_info.subscribe(
            lambda msg: setattr(self, "camera_intrinsics", [msg.K[0], msg.K[4], msg.K[2], msg.K[5]])
        )
        self.odom.subscribe(lambda msg: setattr(self, "latest_odom", msg))
        self.detection2d.subscribe(self._on_detection)

    def _on_detection(self, det: Detection2DArray):
        if det.detections_length == 0 or not self.camera_intrinsics or not self.latest_odom:
            return

        robot_odom = self.latest_odom

        if self.tracking_target:
            matching = [d for d in det.detections if d.id.lower() == self.tracking_target]
            if not matching:
                return
            # Find largest bbox that matches
            target_detection = max(matching, key=lambda d: d.bbox.size_x * d.bbox.size_y)
        else:
            target_detection = det.detections[0]

        # Calculate angle to bounding box center
        fx, _, cx, _ = self.camera_intrinsics
        center_x = target_detection.bbox.center.position.x

        # Horizontal angle to turn toward the object
        angle_offset = math.atan2((center_x - cx), fx)

        q = robot_odom.orientation
        robot_yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

        # Target angle = robot's current yaw + angle offset
        target_yaw = robot_yaw + angle_offset

        # Calculate goal position: 1 meter forward in the target direction
        goal_x = robot_odom.position.x + self.goal_distance * math.cos(target_yaw)
        goal_y = robot_odom.position.y + self.goal_distance * math.sin(target_yaw)

        # Create quaternion for target orientation
        goal_qw = math.cos(target_yaw / 2.0)
        goal_qz = math.sin(target_yaw / 2.0)

        goal = PoseStamped(
            position=Vector3(goal_x, goal_y, robot_odom.position.z),
            orientation=Quaternion(0, 0, goal_qz, goal_qw),
            frame_id="world",
        )
        self.goal_request.publish(goal)

    @skill()
    def track(self, target: str) -> str:
        """Track object by class name."""
        self.tracking_target = target.lower()
        return f"Tracking {target}"

    @skill()
    def stop_tracking(self) -> str:
        """Stop tracking."""
        self.tracking_target = None
        return "Stopped tracking"
