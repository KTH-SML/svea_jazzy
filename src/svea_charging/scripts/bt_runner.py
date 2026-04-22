#!/usr/bin/env python3

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32, String
from sensor_msgs.msg import BatteryState
from tf_transformations import quaternion_from_euler

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
    tick_hz = rx.Parameter(10.0)
    switch_distance_m = rx.Parameter(3.4)
    dock_distance_m = rx.Parameter(1.6)
    mpc_target_topic = rx.Parameter("/mpc_target")
    mpc_target_frame = rx.Parameter("map")
    mpc_target_x = rx.Parameter(1.851350)
    mpc_target_y = rx.Parameter(1.350)
    mpc_target_yaw = rx.Parameter(0.0)

    dist_to_goal_topic = rx.Parameter("dist_to_goal")
    aruco_distance_topic = rx.Parameter("aruco/distance_m")
    battery_charging_topic = rx.Parameter("/lli/battery/state")

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

    def on_startup(self):
        self.bb = MissionBlackboard(
            switch_distance_m=float(self.switch_distance_m),
            dock_distance_m=float(self.dock_distance_m),
        )
        self.tree = ChargingMissionTree(self.bb)
        self.last_active_controller = str(self.bb.active_controller)
        period = 1.0 / self.tick_hz
        self.create_timer(period, self.loop)
        self.get_logger().info(
            "BT runner started "
            f"(switch={self.bb.switch_distance_m:.2f} m, dock={self.bb.dock_distance_m:.2f} m)"
        )

    def loop(self):
        status = self.tree.tick()
        if (
            self.bb.active_controller == "mpc"
            and self.last_active_controller != "mpc"
        ):
            self.publish_mpc_target()
        self.active_controller_pub.publish(String(data=self.bb.active_controller))
        self.phase_pub.publish(String(data=self.bb.mission_phase))
        self.tree_status_pub.publish(String(data=status))
        self.last_active_controller = str(self.bb.active_controller)

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