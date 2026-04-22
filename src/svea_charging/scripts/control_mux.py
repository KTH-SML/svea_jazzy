#!/usr/bin/env python3

from dataclasses import dataclass
import math

from std_msgs.msg import Float32, String

from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)

from svea_core import rosonic as rx
from svea_core.interfaces import ActuationInterface


qos_pubber = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


@dataclass
class ControllerCommand:
    steering: float = 0.0
    velocity: float = 0.0
    stamp_s: float = 0.0


class control_mux(rx.Node):
    controller_timeout_s = rx.Parameter(0.3)
    output_hz = rx.Parameter(20.0)
    active_controller = rx.Parameter("idle")
    line_follower_handover_steering_deg = rx.Parameter(16.0)
    line_follower_handover_timeout_s = rx.Parameter(1.0)

    actuation = ActuationInterface()

    @rx.Subscriber(String, "mission/active_controller", qos_pubber)
    def _active_controller_cb(self, msg: String):
        previous_controller = getattr(self, "previous_controller", str(self.active_controller))
        self.active_controller = msg.data
        if self.active_controller != previous_controller:
            if self.active_controller == "line_follower":
                self.line_follower_switch_time_s = self._now_s()
        self.previous_controller = self.active_controller

    @rx.Subscriber(Float32, "stanley/cmd_steering_rad", qos_pubber)
    def _stanley_steering_cb(self, msg: Float32):
        self.stanley_cmd.steering = float(msg.data)
        self.stanley_cmd.stamp_s = self._now_s()

    @rx.Subscriber(Float32, "stanley/cmd_velocity_mps", qos_pubber)
    def _stanley_velocity_cb(self, msg: Float32):
        self.stanley_cmd.velocity = float(msg.data)
        self.stanley_cmd.stamp_s = self._now_s()

    @rx.Subscriber(Float32, "line_follower/cmd_steering_rad", qos_pubber)
    def _line_steering_cb(self, msg: Float32):
        self.line_cmd.steering = float(msg.data)
        self.line_cmd.stamp_s = self._now_s()

    @rx.Subscriber(Float32, "line_follower/cmd_velocity_mps", qos_pubber)
    def _line_velocity_cb(self, msg: Float32):
        self.line_cmd.velocity = float(msg.data)
        self.line_cmd.stamp_s = self._now_s()

    def on_startup(self):
        self.stanley_cmd = ControllerCommand()
        self.line_cmd = ControllerCommand()
        self.previous_controller = str(self.active_controller)
        self.line_follower_switch_time_s = 0.0
        period = 1.0 / max(float(self.output_hz), 1.0)
        self.create_timer(period, self.loop)
        self.get_logger().info("Control mux started")

    def loop(self):
        cmd = self._get_selected_command()
        self.actuation.send_control(cmd.steering, cmd.velocity)

    def _get_selected_command(self) -> ControllerCommand:
        active = str(self.active_controller)
        if active == "stanley":
            return self._validated_command(self.stanley_cmd)
        if active == "line_follower":
            return self._line_follower_command()
        return ControllerCommand()

    def _validated_command(self, cmd: ControllerCommand) -> ControllerCommand:
        if self._now_s() - cmd.stamp_s > float(self.controller_timeout_s):
            return ControllerCommand()
        return cmd

    def _line_follower_command(self) -> ControllerCommand:
        line_cmd = self._validated_command(self.line_cmd)
        if line_cmd.stamp_s >= self.line_follower_switch_time_s:
            return line_cmd

        if self._now_s() - self.line_follower_switch_time_s <= float(self.line_follower_handover_timeout_s):
            stanley_cmd = self._validated_command(self.stanley_cmd)
            return ControllerCommand(
                steering=math.radians(float(self.line_follower_handover_steering_deg)),
                velocity=max(0.0, stanley_cmd.velocity),
                stamp_s=self._now_s(),
            )

        return line_cmd

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9


if __name__ == "__main__":
    control_mux.main()
