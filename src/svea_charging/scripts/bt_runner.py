#!/usr/bin/env python3

import math

from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import BatteryState
from std_msgs.msg import Float32, Int8, String
from tf_transformations import euler_from_quaternion

from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
battery_qos = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)

from svea_core import rosonic as rx
from svea_charging.behaviourTree.behaviourTree import ChargingMissionTree, MissionBlackboard


qos_pubber = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


class bt_runner(rx.Node):
    tick_hz = rx.Parameter(20.0)
    switch_distance_m = rx.Parameter(3.8)
    dock_distance_m = rx.Parameter(1.6)
    docking_timeout_s = rx.Parameter(10000.0)
    docking_goal_x = rx.Parameter(1.8808068847589485)
    docking_goal_y = rx.Parameter(1.345089355475518)
    docking_goal_yaw_deg = rx.Parameter(0.0)

    dist_to_goal_topic = rx.Parameter("dist_to_goal")
    aruco_distance_topic = rx.Parameter("aruco/distance_m")
    line_status_topic = rx.Parameter("line_follower/status")
    battery_charging_topic = rx.Parameter("/lli/battery/state")
    throttle_topic = rx.Parameter("/lli/ctrl/throttle")
    mocap_pose_topic = rx.Parameter("/mocap/svea/pose")

    active_controller_pub = rx.Publisher(String, "mission/active_controller", qos_pubber)
    phase_pub = rx.Publisher(String, "mission/phase", qos_pubber)
    tree_status_pub = rx.Publisher(String, "mission/tree_status", qos_pubber)

    @rx.Subscriber(Float32, dist_to_goal_topic)
    def _dist_to_goal_cb(self, msg: Float32):
        self.bb.dist_to_station = float(msg.data)

    @rx.Subscriber(Float32, aruco_distance_topic)
    def _aruco_distance_cb(self, msg: Float32):
        distance = float(msg.data)
        if distance > 0.0:
            self.bb.aruco_distance = distance
            self.bb.charger_visible = True
        else:
            self.bb.aruco_distance = None
            self.bb.charger_visible = False

    @rx.Subscriber(String, line_status_topic, qos_pubber)
    def _line_status_cb(self, msg: String):
        status = msg.data
        self.bb.line_visible = status not in {"line_lost", "idle"}

    @rx.Subscriber(BatteryState, battery_charging_topic, battery_qos)
    def _battery_charging_cb(self, msg: BatteryState):
        self.bb.battery_current = float(msg.current)
        self.bb.battery_voltage = float(msg.voltage)
        self.bb.battery_level = float(msg.percentage) * 100
        # self.get_logger().info(f'battery current: {self.bb.battery_current}')

    @rx.Subscriber(Int8, throttle_topic)
    def _throttle_cb(self, msg: Int8):
        throttle_cmd = int(msg.data)
        if getattr(self, "motion_start_time_s", None) is None and throttle_cmd != 0:
            self.motion_start_time_s = self._now_s()
            self.get_logger().info(
                f"Motion timer started from throttle command {throttle_cmd}"
            )

    @rx.Subscriber(PoseWithCovarianceStamped, mocap_pose_topic, qos_pubber)
    def _mocap_pose_cb(self, msg: PoseWithCovarianceStamped):
        self.mocap_pose = msg

    def on_startup(self):
        self.bb = MissionBlackboard(
            switch_distance_m=float(self.switch_distance_m),
            dock_distance_m=float(self.dock_distance_m),
        )
        self.tree = ChargingMissionTree(self.bb)
        self.motion_start_time_s = None
        self.mocap_pose = None
        self.docking_phase_start_time_s = None
        self.docking_timed_out = False
        self.docking_summary_reported = False
        self.last_mission_phase = self.bb.mission_phase
        period = 1.0 / self.tick_hz
        self.create_timer(period, self.loop)
        self.get_logger().info(
            "BT runner started "
            f"(switch={self.bb.switch_distance_m:.2f} m, dock={self.bb.dock_distance_m:.2f} m, "
            f"timeout={float(self.docking_timeout_s):.1f} s)"
        )

    def loop(self):
        now_s = self._now_s()
        if self.docking_timed_out:
            self.bb.active_controller = "idle"
            self.bb.mission_phase = "docking_timeout"
            status = "FAILURE"
        else:
            status = self.tree.tick()
            self._update_docking_timeout(now_s)
            if self.docking_timed_out:
                self.bb.active_controller = "idle"
                self.bb.mission_phase = "docking_timeout"
                status = "FAILURE"

        self._report_docking_summary(now_s)
        self.active_controller_pub.publish(String(data=self.bb.active_controller))
        self.phase_pub.publish(String(data=self.bb.mission_phase))
        self.tree_status_pub.publish(String(data=status))
        self.last_mission_phase = self.bb.mission_phase

    def _update_docking_timeout(self, now_s: float):
        current_phase = self.bb.mission_phase
        if current_phase == "docking" and self.last_mission_phase != "docking":
            self.docking_phase_start_time_s = now_s
            self.get_logger().info(
                f"Docking phase started, timeout is {float(self.docking_timeout_s):.1f} s"
            )

        if current_phase != "docking" or self.docking_phase_start_time_s is None:
            return

        docking_elapsed = now_s - self.docking_phase_start_time_s
        if docking_elapsed < float(self.docking_timeout_s):
            return

        self.docking_timed_out = True
        self.get_logger().warn(
            f"Docking phase timed out after {docking_elapsed:.2f} s"
        )

    def _report_docking_summary(self, now_s: float):
        if self.bb.mission_phase != "docked" or self.docking_summary_reported:
            return

        total_time_s = None
        if self.motion_start_time_s is not None:
            total_time_s = now_s - self.motion_start_time_s

        mocap_state = self._mocap_state()
        if mocap_state is None:
            if total_time_s is None:
                self.get_logger().warn(
                    "Docked, but no non-zero throttle command or mocap pose was available "
                    "for the docking summary."
                )
            else:
                self.get_logger().warn(
                    f"Docked in {total_time_s:.2f} s, but no mocap pose was available "
                    "for the docking error summary."
                )
            self.docking_summary_reported = True
            return

        goal_x = float(self.docking_goal_x)
        goal_y = float(self.docking_goal_y)
        goal_yaw = math.radians(float(self.docking_goal_yaw_deg))

        x, y, yaw = mocap_state
        x_error = goal_x - x
        y_error = goal_y - y
        abs_error = math.hypot(x_error, y_error)
        yaw_error_deg = math.degrees(self._wrap_to_pi(goal_yaw - yaw))

        if total_time_s is None:
            timing_summary = "total docking time unavailable"
        else:
            timing_summary = f"total docking time: {total_time_s:.2f} s"

        self.get_logger().info(
            "Docking summary: "
            f"{timing_summary}, "
            f"absolute error to goal ({goal_x:.3f}, {goal_y:.3f}): {abs_error:.3f} m, "
            f"x-error: {x_error:.3f} m, "
            f"y-error: {y_error:.3f} m, "
            f"yaw-error: {yaw_error_deg:.2f} deg"
        )
        self.docking_summary_reported = True

    def _mocap_state(self):
        if self.mocap_pose is None:
            return None

        pose = self.mocap_pose.pose.pose
        yaw = euler_from_quaternion(
            [
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ]
        )[2]
        return pose.position.x, pose.position.y, yaw

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _wrap_to_pi(self, angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))


if __name__ == "__main__":
    bt_runner.main()
