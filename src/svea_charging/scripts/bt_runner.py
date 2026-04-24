#!/usr/bin/env python3

import math
import time
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from std_msgs.msg import Float32, Int8, String
from sensor_msgs.msg import BatteryState
from tf_transformations import quaternion_from_euler, euler_from_quaternion

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
    switch_distance_m = rx.Parameter(2.5)
    dock_distance_m = rx.Parameter(1.6)
    mpc_target_topic = rx.Parameter("/mpc_target")
    mpc_target_frame = rx.Parameter("map")
    
    mpc_target_yaw = rx.Parameter(0.0)
    goal_x = rx.Parameter(1.8808068847589485)
    goal_y = rx.Parameter(1.345089355475518)
    mpc_target_x = 1.86
    mpc_target_y = goal_y
   
    docking_timeout_sec = rx.Parameter(500.0)

    dist_to_goal_topic = rx.Parameter("dist_to_goal")
    aruco_distance_topic = rx.Parameter("aruco/distance_m")
    battery_charging_topic = rx.Parameter("/lli/battery/state")
    throttle_cmd_topic = rx.Parameter("/lli/ctrl/throttle")

    active_controller_pub = rx.Publisher(String, "mission/active_controller", qos_pubber)
    phase_pub = rx.Publisher(String, "mission/phase", qos_pubber)
    tree_status_pub = rx.Publisher(String, "mission/tree_status", qos_pubber)
    mpc_target_pub = rx.Publisher(PoseStamped, mpc_target_topic, qos_pubber)

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

    @rx.Subscriber(BatteryState, battery_charging_topic, battery_qos)
    def _battery_charging_cb(self, msg: BatteryState):
        self.bb.battery_current = float(msg.current)
        self.bb.battery_voltage = float(msg.voltage)
        self.bb.battery_level = float(msg.percentage) * 100
        # self.get_logger().info(f'battery current: {self.bb.battery_current}')

    @rx.Subscriber(PoseWithCovarianceStamped, '/mocap/svea/pose')
    def _mocap_pose_cb(self, msg: PoseWithCovarianceStamped):
        """Update current mocap pose of the car."""
        self.current_pose = msg.pose.pose

    @rx.Subscriber(Int8, throttle_cmd_topic)
    def _throttle_cmd_cb(self, msg: Int8):
        """Start docking timer when any controller commands non-zero throttle."""
        self._check_movement_started(int(msg.data))

    def _check_movement_started(self, throttle_cmd: int):
        """Start docking timer when the low-level throttle command is non-zero."""
        if throttle_cmd != 0 and not self.docking_timer_started:
            self.docking_timer_started = True
            self.docking_start_time = time.time()
            self.get_logger().info("Non-zero throttle command detected - timer initiated")

    def on_startup(self):
        self.bb = MissionBlackboard(
            switch_distance_m=float(self.switch_distance_m),
            dock_distance_m=float(self.dock_distance_m),
        )
        self.tree = ChargingMissionTree(self.bb)
        self.last_active_controller = str(self.bb.active_controller)
        self.last_mission_phase = str(self.bb.mission_phase)
        
        # Docking timing variables
        self.docking_timer_started = False
        self.docking_start_time = None
        self.current_pose = None
        self.goal_x = float(self.goal_x)
        self.goal_y = float(self.goal_y)
        self.docking_timeout_sec = float(self.docking_timeout_sec)
        
        period = 1.0 / self.tick_hz
        self.create_timer(period, self.loop)
        self.get_logger().info(
            "BT runner started "
            f"(switch={self.bb.switch_distance_m:.2f} m, dock={self.bb.dock_distance_m:.2f} m, "
            f"docking timeout={self.docking_timeout_sec:.1f}s, goal=({self.goal_x:.2f}, {self.goal_y:.2f}))"
        )

    def loop(self):
        status = self.tree.tick()
        if (
            self.bb.active_controller == "mpc"
            and self.last_active_controller != "mpc"
        ):
            self.publish_mpc_target()
        
        # Check for docking phase transitions
        if self.bb.mission_phase == "docked" and self.last_mission_phase != "docked":
            self._on_docking_complete()
        # Check for docking timeout
        if self.bb.mission_phase == "docking" and self.docking_timer_started:
            elapsed_time = time.time() - self.docking_start_time
            if elapsed_time > self.docking_timeout_sec:
                self.get_logger().error(
                    f"DOCKING TIMEOUT: {elapsed_time:.2f}s exceeded {self.docking_timeout_sec}s limit"
                )
                self._on_docking_timeout()
        
        self.active_controller_pub.publish(String(data=self.bb.active_controller))
        self.phase_pub.publish(String(data=self.bb.mission_phase))
        self.tree_status_pub.publish(String(data=status))
        self.last_active_controller = str(self.bb.active_controller)
        self.last_mission_phase = str(self.bb.mission_phase)

    def _on_docking_complete(self):
        """Called when docking phase completes successfully."""
        if self.docking_timer_started:
            elapsed_time = time.time() - self.docking_start_time
            self._calculate_and_print_docking_metrics(elapsed_time, success=True)
        else:
            self.get_logger().warn("Docking completed but timer was not started")
    
    def _on_docking_timeout(self):
        """Called when docking phase times out."""
        elapsed_time = time.time() - self.docking_start_time
        self._calculate_and_print_docking_metrics(elapsed_time, success=False)
    
    def _calculate_and_print_docking_metrics(self, docking_time: float, success: bool):
        """Calculate and print docking metrics including errors from goal."""
        status_str = "SUCCESSFUL" if success else "TIMEOUT"
        
        self.get_logger().info("="*60)
        self.get_logger().info(f"DOCKING COMPLETE ({status_str})")
        self.get_logger().info(f"Total docking time: {docking_time:.2f} seconds")
        
        if self.current_pose is not None:
            # Current position
            current_x = self.current_pose.position.x
            current_y = self.current_pose.position.y
            
            # Extract yaw from quaternion
            orientation = self.current_pose.orientation
            quaternion = (orientation.x, orientation.y, orientation.z, orientation.w)
            euler_angles = euler_from_quaternion(quaternion)
            current_yaw = euler_angles[2]
            
            # Calculate errors
            x_error = current_x - self.goal_x
            y_error = current_y - self.goal_y
            abs_error = math.sqrt(x_error**2 + y_error**2)
            yaw_error = current_yaw
            mpc_target_x = float(self.mpc_target_x)
            mpc_target_y = float(self.mpc_target_y)
            mpc_x_error = current_x - mpc_target_x
            mpc_y_error = current_y - mpc_target_y
            mpc_abs_error = math.sqrt(mpc_x_error**2 + mpc_y_error**2)
            
            # Print metrics
            self.get_logger().info(f"Goal position: (x: {self.goal_x}, y: {self.goal_y})")
            self.get_logger().info(f"MPC target: (x: {mpc_target_x}, y: {mpc_target_y})")
            self.get_logger().info(f"Final position: (x: {current_x:.3f}, y: {current_y:.3f})")
            self.get_logger().info(f"Absolute error to goal: {abs_error:.3f}m")
            self.get_logger().info(f"  x-error: {x_error:.3f}m")
            self.get_logger().info(f"  y-error: {y_error:.3f}m")
            self.get_logger().info(f"Absolute error to /mpc_target: {mpc_abs_error:.3f}m")
            self.get_logger().info(f"  mpc x-error: {mpc_x_error:.3f}m")
            self.get_logger().info(f"  mpc y-error: {mpc_y_error:.3f}m")
            self.get_logger().info(f"  yaw-error: {math.degrees(yaw_error):.2f}° ({yaw_error:.4f} rad)")
        else:
            self.get_logger().warn("No mocap pose available - cannot calculate errors")
        
        self.get_logger().info("="*60)
        
        # Reset timer
        self.docking_timer_started = False
        self.docking_start_time = None
    
    def publish_mpc_target(self):
        msg = PoseStamped()
        msg.header.frame_id = str(self.mpc_target_frame)
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(self.mpc_target_x)
        msg.pose.position.y = float(self.mpc_target_y)

        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, float(self.mpc_target_yaw))
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw

        self.mpc_target_pub.publish(msg)
        self.get_logger().info(
            f"Published /mpc_target: ({msg.pose.position.x:.6f}, {msg.pose.position.y:.6f})"
        )


if __name__ == "__main__":
    bt_runner.main()
