#!/usr/bin/env python3

import cv2
import numpy as np
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String

from svea_core import rosonic as rx
from svea_core.interfaces import LocalizationInterface
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
)

#QoS Profile
qos_pubber = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


class LineFollowerLocalizationInterface(LocalizationInterface):
    def _resolve_base_frame(self, odom=None):
        base_frame = str(self.localization.base_frame)
        if base_frame and base_frame != "self/base_link":
            return base_frame

        namespace = self.node.get_namespace().strip("/")
        if namespace:
            return f"{namespace}/base_link"

        if odom is not None and odom.child_frame_id:
            return odom.child_frame_id

        return base_frame

    def transform_odom(
        self,
        odom,
        pose_target=None,
        twist_target=None,
        timeout_s=0.2,
    ):
        resolved_twist_target = (
            twist_target
            if twist_target is not None
            else self._resolve_base_frame(odom)
        )
        return super().transform_odom(
            odom,
            pose_target=pose_target,
            twist_target=resolved_twist_target,
            timeout_s=timeout_s,
        )


class line_follower(rx.Node):
    dt = rx.Parameter(0.05)
    image_topic = rx.Parameter("/svea67/image_raw")
    target_velocity = rx.Parameter(0.4)
    max_velocity = rx.Parameter(0.7)
    stop_on_lost_line = rx.Parameter(True)
    controller_name = rx.Parameter("line_follower")
    active_controller = rx.Parameter('idle')
    steering_cmd_topic = rx.Parameter("line_follower/cmd_steering_rad")
    velocity_cmd_topic = rx.Parameter("line_follower/cmd_velocity_mps")


    publish_debug_image = rx.Parameter(False)
    debug_image_topic = rx.Parameter("line_follower/debug_image")
    debug_publish_every_n = rx.Parameter(3)

    lower_h = rx.Parameter(20)
    lower_s = rx.Parameter(100)
    lower_v = rx.Parameter(100)
    upper_h = rx.Parameter(35)
    upper_s = rx.Parameter(255)
    upper_v = rx.Parameter(255)


    crop_start_ratio = rx.Parameter(0.55)
    min_contour_area = rx.Parameter(120)
    steering_kp = rx.Parameter(1.6)
    steering_ki = rx.Parameter(.3)
    steering_kd = rx.Parameter(0.01)
    steering_limit_rad = rx.Parameter(0.6)
    lost_line_steering_rad = rx.Parameter(0.0)
    velocity_scale_from_error = rx.Parameter(True)

    use_aruco_stop = rx.Parameter(True)
    aruco_distance_topic = rx.Parameter("aruco/distance_m")
    aruco_stop_distance_m = rx.Parameter(1.68)
    aruco_distance_kp = rx.Parameter(10.0)
    aruco_distance_ki = rx.Parameter(0.15)
    aruco_distance_kd = rx.Parameter(0.0)
    aruco_distance_integral_limit = rx.Parameter(1.0)
    aruco_velocity_kp = rx.Parameter(10.0)
    aruco_velocity_ki = rx.Parameter(0.3)
    aruco_velocity_kd = rx.Parameter(0.0)
    aruco_max_backup_velocity = rx.Parameter(0.35)
    aruco_overshoot_deadband_m = rx.Parameter(0.03)

    aruco_distance_integral_limit = rx.Parameter(5.0)

    localizer = LineFollowerLocalizationInterface()

    steering_cmd_pub = rx.Publisher(Float32, steering_cmd_topic)
    velocity_cmd_pub = rx.Publisher(Float32, velocity_cmd_topic)
    line_error_pub = rx.Publisher(Float32, "line_follower/error_px")
    status_pub = rx.Publisher(String, "line_follower/status")
    centroid_pub = rx.Publisher(Point, "line_follower/centroid")
    debug_image_pub = rx.Publisher(Image, debug_image_topic)

    @rx.Subscriber(Float32, aruco_distance_topic)
    def _aruco_distance_callback(self, msg: Float32):
        self.aruco_distance = float(msg.data)

    @rx.Subscriber(String, 'mission/active_controller', qos_pubber)
    def _mission_active(self, msg: String):
        self.active_controller = msg.data

    def on_startup(self):
        self.bridge = CvBridge()
        self.latest_frame = None
        self.latest_mask = None
        self.latest_centroid = None
        self.line_detected = False
        self.aruco_distance = -1.0
        self.debug_publish_counter = 0
        
        self.steering_error_prev = 0.0
        self.steering_error_integral = 0.0

        self.aruco_distance_error_prev = 0.0
        self.aruco_distance_integral = 0.0
        self.aruco_velocity_integral = 0.0
        self.aruco_velocity_error_prev = 0.0
        self.position_prev = 0.0

        self.position_prev = 0.0
        self.vel_error_prev = 0.0

        self.create_subscription(
            Image,
            str(self.image_topic),
            self._image_callback,
            1,
        )

        self.dt_s = max(float(self.dt), 1e-3)
        self.create_timer(self.dt_s, self.loop)
        self.get_logger().info(
            f"Line follower started on image_topic={self.image_topic}, dt={self.dt_s:.3f}s"
        )

    def on_shutdown(self):
        pass

    def _image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"Failed to convert image: {exc}")
            return

        centroid, mask = self._extract_line_centroid(frame)
        self.latest_frame = frame
        self.latest_mask = mask
        self.latest_centroid = centroid
        self.line_detected = centroid is not None

    def _extract_line_centroid(self, frame):
        height, _ = frame.shape[:2]
        crop_start = int(np.clip(float(self.crop_start_ratio), 0.0, 0.95) * height)
        roi = frame[crop_start:, :]

        hsv_image = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower = np.array(
            [int(self.lower_h), int(self.lower_s), int(self.lower_v)],
            dtype=np.uint8,
        )
        upper = np.array(
            [int(self.upper_h), int(self.upper_s), int(self.upper_v)],
            dtype=np.uint8,
        )
        mask = cv2.inRange(hsv_image, lower, upper)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        min_area = float(self.min_contour_area)
        line = None

        for contour in contours:
            moments = cv2.moments(contour)
            if moments["m00"] > min_area:
                line = (
                    int(moments["m10"] / moments["m00"]),
                    int(moments["m01"] / moments["m00"]) + crop_start,
                )

        return line, mask

    def _calculate_steering(self, normalized_error, dt):
        error_i = (normalized_error + self.steering_error_prev) / 2.0 * dt
        error_d = (normalized_error - self.steering_error_prev) / max(dt, 1e-6)
        self.steering_error_integral += error_i
        self.steering_error_integral = float(
            np.clip(
                self.steering_error_integral,
                -float(self.steering_limit_rad) * 1.5,
                float(self.steering_limit_rad) * 1.5,
            )
        )
        self.steering_error_prev = float(normalized_error)

        steering = -(
            float(self.steering_kp) * normalized_error
            + float(self.steering_ki) * self.steering_error_integral
            + float(self.steering_kd) * error_d
        )
        steering = float(
            np.clip(
                steering,
                -float(self.steering_limit_rad),
                float(self.steering_limit_rad),
            )
        )
        return steering

    def _calculate_velocity(self, normalized_error, dt):
        # Base velocity from line following
        if bool(self.velocity_scale_from_error):
            speed_scale = max(0.25, 1.0 - min(abs(normalized_error), 1.0))
        else:
            speed_scale = 1.0
        base_velocity = min(
            float(self.max_velocity),
            float(self.target_velocity) * speed_scale,
        )

        if not bool(self.use_aruco_stop):
            self.aruco_distance_error_prev = 0.0
            self.aruco_distance_integral = 0.0
            self.aruco_velocity_error_prev = 0.0
            self.aruco_velocity_integral = 0.0
            self.position_prev = 0.0
            return base_velocity

        if self.aruco_distance <= 0.0:
            self.aruco_distance_error_prev = 0.0
            self.aruco_distance_integral = 0.0
            self.aruco_velocity_error_prev = 0.0
            self.aruco_velocity_integral = 0.0
            self.position_prev = 0.0
            return base_velocity

        ref_dist = float(self.aruco_stop_distance_m)
        dist = float(self.aruco_distance)
        dist_error = dist - ref_dist
        prev_dist_error = float(self.aruco_distance_error_prev)
        overshoot_deadband = max(float(self.aruco_overshoot_deadband_m), 0.0)

        crossed_target = (
            abs(prev_dist_error) > overshoot_deadband
            and abs(dist_error) > overshoot_deadband
            and np.sign(prev_dist_error) != np.sign(dist_error)
        )
        if crossed_target:
            self.aruco_distance_integral = 0.0
            self.aruco_velocity_integral = 0.0
            self.aruco_velocity_error_prev = 0.0

        # PID control for distance
        error_i = (dist_error + prev_dist_error) / 2.0 * dt
        error_d = (dist_error - prev_dist_error) / max(dt, 1e-6)
        self.aruco_distance_integral += error_i
        self.aruco_distance_integral = float(
            np.clip(
                self.aruco_distance_integral,
                -float(self.aruco_distance_integral_limit),
                float(self.aruco_distance_integral_limit),
            )
        )
        self.aruco_distance_error_prev = float(dist_error)

        desired_velocity = (
            float(self.aruco_distance_kp) * dist_error
            + float(self.aruco_distance_ki) * self.aruco_distance_integral
            + float(self.aruco_distance_kd) * error_d
        )
        backup_velocity_limit = min(
            float(self.max_velocity),
            float(self.aruco_max_backup_velocity),
        )
        desired_velocity = float(
            np.clip(
                desired_velocity,
                -backup_velocity_limit,
                base_velocity,
            )
        )

        _, _, _, vel = self.localizer.get_state()
        vel_error = desired_velocity - vel

        # PID control for velocity
        error_vel_i = (vel_error + self.aruco_velocity_error_prev) / 2.0 * dt
        error_vel_d = (vel_error - self.aruco_velocity_error_prev) / max(dt, 1e-6)
        self.aruco_velocity_integral += error_vel_i
        self.aruco_velocity_integral = float(
            np.clip(
                self.aruco_velocity_integral,
                -float(self.aruco_distance_integral_limit),
                float(self.aruco_distance_integral_limit),
            )
        )

        velocity = (
            float(self.aruco_velocity_kp) * vel_error
            + float(self.aruco_velocity_ki) * self.aruco_velocity_integral
            + float(self.aruco_velocity_kd) * error_vel_d
        )

        self.aruco_velocity_error_prev = float(vel_error)
        self.position_prev = float(dist_error)

        velocity = float(
            np.clip(
                velocity,
                -backup_velocity_limit,
                base_velocity,
            )
        )
        return velocity

    def loop(self):
        if self.active_controller != str(self.controller_name):
            return

        frame = self.latest_frame
        if frame is None:
            return

        _, width = frame.shape[:2]
        image_center_x = width / 2.0

        if self.latest_centroid is None:
            self._publish_status("line_lost")
            self.steering_error_prev = 0.0
            self.steering_error_integral = 0.0
            if bool(self.stop_on_lost_line):
                self.steering_cmd_pub.publish(
                    Float32(data=float(self.lost_line_steering_rad))
                )
                self.velocity_cmd_pub.publish(Float32(data=0.0))
            self._publish_debug_image(frame, None, None)
            return

        cx, cy = self.latest_centroid
        error_px = cx - image_center_x
        normalized_error = error_px / max(image_center_x, 1.0)

        dt = self.dt_s
        steering = self._calculate_steering(normalized_error, dt)
        velocity = self._calculate_velocity(normalized_error, dt)

        self.steering_cmd_pub.publish(Float32(data=float(steering)))
        self.velocity_cmd_pub.publish(Float32(data=float(velocity)))
        self.line_error_pub.publish(Float32(data=float(error_px)))
        self._publish_status(self._get_status_text(velocity))

        centroid_msg = Point()
        centroid_msg.x = float(cx)
        centroid_msg.y = float(cy)
        centroid_msg.z = 0.0
        self.centroid_pub.publish(centroid_msg)

        self._publish_debug_image(frame, (cx, cy), error_px)


    def _publish_status(self, text: str):
        self.status_pub.publish(String(data=text))

    def _get_status_text(self, velocity: float) -> str:
        if not bool(self.use_aruco_stop) or self.aruco_distance <= 0.0 or self.aruco_distance > self.aruco_stop_distance_m + 0.5:
            return "tracking"

        if velocity < 0.0:
            return "backing_up_to_aruco"

        if abs(velocity) < 1e-3:
            return "stopped_at_aruco"

        return "approaching_aruco"

    def _publish_debug_image(self, frame, centroid, error_px):
        if not bool(self.publish_debug_image):
            return

        self.debug_publish_counter += 1
        if self.debug_publish_counter % max(int(self.debug_publish_every_n), 1) != 0:
            return

        debug = frame.copy()
        height, width = debug.shape[:2]
        center_x = width // 2

        cv2.line(debug, (center_x, 0), (center_x, height), (0, 255, 255), 2)

        crop_start = int(np.clip(float(self.crop_start_ratio), 0.0, 0.95) * height)
        cv2.line(debug, (0, crop_start), (width, crop_start), (255, 255, 0), 2)

        if centroid is not None:
            cv2.circle(debug, centroid, 8, (0, 0, 255), -1)
            cv2.putText(
                debug,
                f"error_px={error_px:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            if bool(self.use_aruco_stop) and self.aruco_distance > 0.0:
                cv2.putText(
                    debug,
                    f"aruco_d={self.aruco_distance:.2f} m",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 220, 0),
                    2,
                    cv2.LINE_AA,
                )
        else:
            cv2.putText(
                debug,
                "line lost",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        msg = self.bridge.cv2_to_imgmsg(debug, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        self.debug_image_pub.publish(msg)


if __name__ == "__main__":
    line_follower.main()
