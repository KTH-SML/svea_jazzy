#! /usr/bin/env python3

import numpy as np
from geometry_msgs.msg import Point
from geometry_msgs.msg import PoseArray
from visualization_msgs.msg import Marker

from svea_core.interfaces import LocalizationInterface
from svea_charging.controllers.stanleyController import StanleyController
from svea_core.interfaces import ActuationInterface
from svea_core import rosonic as rx
from std_msgs.msg import Float32
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

class stanley_control(rx.Node):
    DELTA_TIME = 0.1

    endPoint = rx.Parameter('[5.0, 2.5]') #0.5, -1.2 irl
    target_velocity = rx.Parameter(0.4)
    use_aruco_goal = rx.Parameter(True)
    aruco_goal_topic = rx.Parameter("aruco/poses")
    aruco_pose_is_car_in_marker_frame = rx.Parameter(True)
    aruco_goal_offset = rx.Parameter(0.35)  # stop short of marker center [m]
    aruco_goal_blend_alpha = rx.Parameter(0.25)  # EMA gain for marker position in map
    aruco_max_marker_jump = rx.Parameter(0.8)  # [m] reject large frame-to-frame jumps
    aruco_measurement_timeout = rx.Parameter(0.8)  # [s] marker freshness window
    aruco_min_forward_distance = rx.Parameter(0.05)  # [m]
    aruco_max_forward_distance = rx.Parameter(6.0)  # [m]
    aruco_max_lateral_distance = rx.Parameter(2.5)  # [m]
    aruco_goal_update_eps = rx.Parameter(0.03)  # [m] deadband to avoid jitter
    # Interfaces
    actuation = ActuationInterface()
    localizer = LocalizationInterface()
    goal_tolerance = rx.Parameter(0.2) #m

    #Publishers
    goal_pub = rx.Publisher(Marker, 'goal_marker', qos_pubber)
    path_pub = rx.Publisher(Marker, 'path_marker', qos_pubber)
    traj_pub = rx.Publisher(Marker, 'traj_marker', qos_pubber)
    cross_track_error_pub = rx.Publisher(Float32, 'cross_track_error', qos_pubber)
    yaw_error_pub = rx.Publisher(Float32, 'yaw_error', qos_pubber)
    velocity_error_pub = rx.Publisher(Float32, 'velocity_error', qos_pubber)
    dist_to_goal = rx.Publisher(Float32, 'dist_to_goal', qos_pubber)


    def on_startup(self):
        self.reached_goal = False
        self.counter = 0
        self.filtered_marker_map = None
        self.last_marker_update_s = None

        self.controller = StanleyController()
        self.controller.target_velocity = self.target_velocity

        import time
        time.sleep(8) # wait for localization to start up and get first state
        state = self.localizer.get_state()
        x, y, yaw, vel = state

        self.goal = eval(self.endPoint)
        mx = 0.5*(x + self.goal[0])
        my = 0.5*(y + self.goal[1])
        self.waypoints = [[x, y], [mx, my], self.goal]
        
        #publish goal and waypoints
        #self.publish_goal_marker(self.goal)
        #self.publish_waypoints_marker(self.waypoints)

        self.controller.update_traj(state, self.waypoints)
        self.create_timer(self.DELTA_TIME, self.loop)

    @rx.Subscriber(PoseArray, aruco_goal_topic)
    def _aruco_goal_cb(self, msg: PoseArray):
        if not bool(self.use_aruco_goal):
            return
        if len(msg.poses) == 0:
            return

        pose = msg.poses[0]
        marker_x_cam, marker_z_cam = self._marker_in_car_frame_from_pose(pose)
        if not np.isfinite(marker_x_cam) or not np.isfinite(marker_z_cam):
            return
        if marker_z_cam < float(self.aruco_min_forward_distance):
            return
        if marker_z_cam > float(self.aruco_max_forward_distance):
            return
        if abs(marker_x_cam) > float(self.aruco_max_lateral_distance):
            return

        state = self.localizer.get_state()
        car_x, car_y, car_yaw, _ = state
        marker_map = self._car_frame_point_to_map(car_x, car_y, car_yaw, marker_z_cam, marker_x_cam)

        if self.filtered_marker_map is None:
            self.filtered_marker_map = marker_map
        else:
            innovation = np.linalg.norm(marker_map - self.filtered_marker_map)
            if innovation > float(self.aruco_max_marker_jump):
                return
            alpha = float(np.clip(self.aruco_goal_blend_alpha, 0.0, 1.0))
            self.filtered_marker_map = (1.0 - alpha) * self.filtered_marker_map + alpha * marker_map

        self.last_marker_update_s = self._now_s()
        self._update_goal_from_filtered_marker(state)

    def _marker_in_car_frame_from_pose(self, pose):
        x = float(pose.position.x)
        y = float(pose.position.y)
        z = float(pose.position.z)

        if not bool(self.aruco_pose_is_car_in_marker_frame):
            return x, z

        # Input is assumed T_marker_car; invert to get T_car_marker.
        qx = float(pose.orientation.x)
        qy = float(pose.orientation.y)
        qz = float(pose.orientation.z)
        qw = float(pose.orientation.w)
        R = self._quat_to_rot(qx, qy, qz, qw)  # marker <- car
        t_m_c = np.array([x, y, z], dtype=float)
        t_c_m = -R.T @ t_m_c
        return float(t_c_m[0]), float(t_c_m[2])

    def _car_frame_point_to_map(self, car_x, car_y, car_yaw, forward_m, left_m):
        dx = forward_m * np.cos(car_yaw) - left_m * np.sin(car_yaw)
        dy = forward_m * np.sin(car_yaw) + left_m * np.cos(car_yaw)
        return np.array([car_x + dx, car_y + dy], dtype=float)

    def _now_s(self):
        now = self.get_clock().now().nanoseconds
        return float(now) * 1e-9

    def _has_fresh_marker(self):
        if self.filtered_marker_map is None or self.last_marker_update_s is None:
            return False
        return (self._now_s() - self.last_marker_update_s) <= float(self.aruco_measurement_timeout)

    def _update_goal_from_filtered_marker(self, state):
        if self.filtered_marker_map is None:
            return

        car_x, car_y, _, _ = state
        car_xy = np.array([car_x, car_y], dtype=float)
        to_marker = self.filtered_marker_map - car_xy
        marker_dist = np.linalg.norm(to_marker)
        if marker_dist < 1e-6:
            return

        approach_dist = max(marker_dist - float(self.aruco_goal_offset), 0.0)
        goal_xy = car_xy + (to_marker / marker_dist) * approach_dist

        if hasattr(self, "goal"):
            prev_goal = np.array(self.goal, dtype=float)
            if np.linalg.norm(goal_xy - prev_goal) < float(self.aruco_goal_update_eps):
                return

        self.goal = [float(goal_xy[0]), float(goal_xy[1])]
        mid_x = 0.5 * (car_x + self.goal[0])
        mid_y = 0.5 * (car_y + self.goal[1])
        self.waypoints = [[car_x, car_y], [mid_x, mid_y], self.goal]
        self.reached_goal = False

    def _quat_to_rot(self, qx, qy, qz, qw):
        n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if n < 1e-8:
            return np.eye(3)
        qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
        return np.array(
            [
                [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
                [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
                [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
            ],
            dtype=float,
        )


    def loop(self):
        """
        Main loop of the Stanley controller. 
        """
        state = self.localizer.get_state()
        x, y, yaw, vel = state

        if bool(self.use_aruco_goal) and self._has_fresh_marker():
            self._update_goal_from_filtered_marker(state)

        dist = self.distance_to_goal(state)
        if dist <= self.goal_tolerance:
            if not self.reached_goal:
                self.get_logger().info("Reached goal!")
                self.reached_goal = True

        #self.update_goal()
        self.controller.update_traj(state, self.waypoints)

        if not self.reached_goal:
            steering, velocity = self.controller.compute_control(state)
            self.get_logger().info(f"Steering: {steering}, Velocity: {velocity}")
        else:
            steering, velocity = 0.0, 0.0

        self.actuation.send_control(steering, velocity)
        

        if self.counter % 5 == 0: # publish markers every .5 seconds
            self.publish_goal_marker(self.goal)
            self.publish_waypoints_marker(self.waypoints)
            self.publish_trajectory_marker(self.controller.cx, self.controller.cy)
            # Publish errors
            self.publish_errors(x, y, yaw, vel)
            self.dist_to_goal.publish(Float32(data=dist))
        self.counter += 1

    def distance_to_goal(self, state):
        x, y, _, _ = state
        goal_x, goal_y = self.goal
        return np.sqrt((goal_x - x)**2 + (goal_y - y)**2)

    def update_goal(self):

        self.curr += 1
        self.curr %= len(self._points)
        self.goal = self._points[self.curr]
        self.controller.is_finished = False
        # Mark the goal
        self.publish_goal_marker()
    
    def publish_goal_marker(self, goal_xy):
        msg = Marker()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.ns = "stanley_goal"
        msg.id = 0
        msg.type = Marker.SPHERE
        msg.action = Marker.ADD

        msg.pose.position.x = float(goal_xy[0])
        msg.pose.position.y = float(goal_xy[1])
        msg.pose.position.z = 0.2
        msg.pose.orientation.w = 1.0

        msg.scale.x = 0.4
        msg.scale.y = 0.4
        msg.scale.z = 0.4

        msg.color.r = 0.0
        msg.color.g = 0.0
        msg.color.b = 1.0
        msg.color.a = 1.0

        self.goal_pub.publish(msg)

    def publish_waypoints_marker(self, waypoints):
        # Draw straight segments between waypoints
        msg = Marker()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.ns = "stanley_waypoints"
        msg.id = 0
        msg.type = Marker.LINE_STRIP
        msg.action = Marker.ADD

        msg.scale.x = 0.02  # line width

        msg.color.r = 1.0
        msg.color.g = 1.0
        msg.color.b = 0.0
        msg.color.a = 1.0

        msg.points = []
        for wp in waypoints:
            p = Point()
            p.x = float(wp[0])
            p.y = float(wp[1])
            p.z = 0.05
            msg.points.append(p)

        self.path_pub.publish(msg)

    def publish_trajectory_marker(self, cx, cy):
        msg = Marker()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.ns = "stanley_traj"
        msg.id = 0
        msg.type = Marker.LINE_STRIP
        msg.action = Marker.ADD

        msg.scale.x = 0.05
        msg.color.r = 0.0
        msg.color.g = 1.0
        msg.color.b = 0.0
        msg.color.a = 1.0

        msg.points = []
        for x, y in zip(cx, cy):
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.03
            msg.points.append(p)

        self.traj_pub.publish(msg)

    def publish_errors(self, x, y, yaw, vel):
        self.cross_track_error_pub.publish(Float32(data=self.controller.cross_track_error)) #m
        self.yaw_error_pub.publish(Float32(data=np.rad2deg(self.controller.yaw_error))) #degrees

        # Velocity error
        vel_err = self.controller.target_velocity - vel
        self.velocity_error_pub.publish(Float32(data=vel_err))



if __name__ == '__main__':
    stanley_control.main()
