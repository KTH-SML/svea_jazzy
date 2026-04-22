#! /usr/bin/env python3

import rclpy
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from mocap4r2_msgs.msg import RigidBodies
from tf2_ros import TransformListener, Buffer
from tf2_ros import TransformException
import tf2_geometry_msgs
import time
import numpy as np

from svea_core import rosonic as rx


qos_normal = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
)

qos_subber = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,  # BEST_EFFORT
    history=QoSHistoryPolicy.KEEP_LAST,         # Keep the last N messages
    durability=QoSDurabilityPolicy.VOLATILE,    # Volatile
    depth=10,                                   # Size of the queue
)

class MocapToPose(rx.Node):

    ## Subscribers ##
    mocap_topic = rx.Parameter('/rigid_bodies')
    svea_body_name = rx.Parameter('svea67')
    trailer_body_name = rx.Parameter('trailer')
    charging_station_body_name = rx.Parameter('charging_station')
    svea_pose_topic = rx.Parameter('/svea67/pose')
    trailer_pose_topic = rx.Parameter('/trailer/pose')
    charging_station_pose_topic = rx.Parameter('/charging_station/pose')
    output_frame = rx.Parameter('map')
    initial_pose_topic = rx.Parameter('/set_pose')
    tf_lookup_timeout = rx.Parameter(1.0)
    tf_startup_wait = rx.Parameter(2.0)

    ## Publishers ##
    mocap_svea_pub = rx.Publisher(PoseWithCovarianceStamped, '/mocap/svea/pose', qos_normal)
    mocap_trailer_pub = rx.Publisher(PoseWithCovarianceStamped, '/mocap/trailer/pose', qos_normal)
    mocap_charging_station_pub = rx.Publisher(PoseWithCovarianceStamped, '/mocap/charging_station/pose', qos_normal)
    initialpose_pub = rx.Publisher(PoseWithCovarianceStamped, initial_pose_topic, qos_normal)

    _svea_pose_msg = PoseWithCovarianceStamped()
    _trailer_pose_msg = PoseWithCovarianceStamped()
    _initialpose_msg = PoseWithCovarianceStamped()
    _charging_station_pose_msg = PoseWithCovarianceStamped()

    def on_startup(self):
        """Start the Mocap interface by subscribing to the pose topic."""
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.initial_pose = False
        self.pose_update = False
        self._known_rigid_bodies = set()
        self._missing_body_warnings = set()
        time.sleep(float(self.tf_startup_wait))  # Give TF a moment to populate.
        self.get_logger().info("Starting Mocap interface Node...")
        self.get_logger().info(
            f"Mocap is ready. Listening on {self.mocap_topic} and publishing in the "
            f"{self.output_frame} frame."
        )

        # self.create_timer(0.05, self._timer_callback)

    def _timer_callback(self):
        self.initial_pose = False

    def _transform_pose(self, pose: PoseStamped) -> PoseStamped:
        source_frame = pose.header.frame_id
        target_frame = str(self.output_frame)

        if source_frame == target_frame:
            return pose

        transform = self.tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            Time.from_msg(pose.header.stamp),
            timeout=Duration(seconds=float(self.tf_lookup_timeout)),
        )
        return tf2_geometry_msgs.do_transform_pose_stamped(pose, transform)

    def _build_pose_stamped(self, header, pose) -> PoseStamped:
        pose_stamped = PoseStamped()
        pose_stamped.header = header
        pose_stamped.pose = pose
        return pose_stamped

    def _publish_pose(self, pose_stamped: PoseStamped, out_msg: PoseWithCovarianceStamped, publisher) -> None:
        out_msg.header = pose_stamped.header
        out_msg.pose.pose = pose_stamped.pose
        publisher.publish(out_msg)

    def _transform_and_publish(
        self,
        pose_stamped: PoseStamped,
        out_msg: PoseWithCovarianceStamped,
        publisher,
        body_name: str,
    ) -> PoseStamped | None:
        try:
            pose_transformed = self._transform_pose(pose_stamped)
        except TransformException as ex:
            self.get_logger().warn(
                f"Could not transform {body_name} from "
                f"{pose_stamped.header.frame_id} to {self.output_frame}: {ex}"
            )
            return None

        self._publish_pose(pose_transformed, out_msg, publisher)
        return pose_transformed

    def _is_valid_pose(self, pose) -> bool:
        return (
            not np.isnan(pose.position.x)
            and not np.isnan(pose.position.y)
            and not np.isnan(pose.position.z)
        )

    def _handle_svea_pose(self, pose_stamped: PoseStamped, body_name: str) -> None:
        pose_transformed = self._transform_and_publish(
            pose_stamped,
            self._svea_pose_msg,
            self.mocap_svea_pub,
            body_name,
        )
        if pose_transformed is None:
            return

        self._initialpose_msg.header = pose_transformed.header
        self._initialpose_msg.pose.pose = pose_transformed.pose
        self.pose_update = True

        if not self.initial_pose:
            self.initialpose_pub.publish(self._initialpose_msg)
            self.initial_pose = True

    def _handle_trailer_pose(self, pose_stamped: PoseStamped, body_name: str) -> None:
        self._transform_and_publish(
            pose_stamped,
            self._trailer_pose_msg,
            self.mocap_trailer_pub,
            body_name,
        )

    def _handle_charging_station_pose(self, pose_stamped: PoseStamped, body_name: str) -> None:
        self._transform_and_publish(
            pose_stamped,
            self._charging_station_pose_msg,
            self.mocap_charging_station_pub,
            body_name,
        )

    def _warn_for_missing_body(self, body_name: str) -> None:
        if body_name in self._known_rigid_bodies or body_name in self._missing_body_warnings:
            return

        self._missing_body_warnings.add(body_name)
        seen_names = ", ".join(sorted(self._known_rigid_bodies)) or "none yet"
        self.get_logger().warn(
            f"Configured rigid body '{body_name}' has not been seen. "
            f"Currently seen rigid bodies: {seen_names}"
        )

    @rx.Subscriber(RigidBodies, mocap_topic, qos_profile=qos_subber)
    def _rigid_bodies_pose_cb(self, msg: RigidBodies) -> None:
        if not msg.rigidbodies:
            return

        for rigid_body in msg.rigidbodies:
            self._known_rigid_bodies.add(rigid_body.rigid_body_name)

            if not self._is_valid_pose(rigid_body.pose):
                continue

            if rigid_body.rigid_body_name == str(self.svea_body_name):
                self._handle_svea_pose(
                    self._build_pose_stamped(msg.header, rigid_body.pose),
                    rigid_body.rigid_body_name,
                )

            elif rigid_body.rigid_body_name == str(self.trailer_body_name):
                self._handle_trailer_pose(
                    self._build_pose_stamped(msg.header, rigid_body.pose),
                    rigid_body.rigid_body_name,
                )

            elif rigid_body.rigid_body_name == str(self.charging_station_body_name):
                self._handle_charging_station_pose(
                    self._build_pose_stamped(msg.header, rigid_body.pose),
                    rigid_body.rigid_body_name,
                )

        self._warn_for_missing_body(str(self.svea_body_name))
        self._warn_for_missing_body(str(self.trailer_body_name))
        self._warn_for_missing_body(str(self.charging_station_body_name))

    @rx.Subscriber(PoseStamped, svea_pose_topic, qos_profile=qos_subber)
    def _svea_pose_cb(self, msg: PoseStamped) -> None:
        if self._is_valid_pose(msg.pose):
            self._handle_svea_pose(msg, str(self.svea_body_name))

    @rx.Subscriber(PoseStamped, trailer_pose_topic, qos_profile=qos_subber)
    def _trailer_pose_cb(self, msg: PoseStamped) -> None:
        if self._is_valid_pose(msg.pose):
            self._handle_trailer_pose(msg, str(self.trailer_body_name))

    @rx.Subscriber(PoseStamped, charging_station_pose_topic, qos_profile=qos_subber)
    def _charging_station_pose_cb(self, msg: PoseStamped) -> None:
        if self._is_valid_pose(msg.pose):
            self._handle_charging_station_pose(msg, str(self.charging_station_body_name))

if __name__ == '__main__':
    MocapToPose.main()

    
