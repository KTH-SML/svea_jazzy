#! /usr/bin/env python3

import numpy as np
from geometry_msgs.msg import Point
from geometry_msgs.msg import PoseArray, PoseWithCovarianceStamped
from visualization_msgs.msg import Marker
import time

from svea_core.interfaces import LocalizationInterface
from svea_charging.controllers.stanleyController import StanleyController
from svea_core import rosonic as rx
from std_msgs.msg import Float32, String
from tf_transformations import euler_from_quaternion
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

    def __init__(self):
        super().__init__('stanley_control')
        self.charging_station_pose = None
        self.charging_station_identified_mocap = False
        self.svea_identified_mocap = False

    DELTA_TIME = 0.05


    endPoint = rx.Parameter('[1.891350, 1.363510]') #x= -1.885,y=  1.348, yaw = 90deg alt x = 1.6
    endPoints = rx.Parameter('[-0.1, 1.363510], [1.6, 1.363510], [1.891350, 1.363510]')

    target_velocity = rx.Parameter(0.5)
    controller_name = rx.Parameter("stanley")
    active_controller = rx.Parameter("idle")

    steering_cmd_topic = rx.Parameter("stanley/cmd_steering_rad")
    velocity_cmd_topic = rx.Parameter("stanley/cmd_velocity_mps")
    
    use_adaptive_speed = rx.Parameter(True)
    use_mocap_goal = rx.Parameter(False)
    use_mocap = rx.Parameter(False)
    aruco_goal_topic = rx.Parameter("aruco/poses")
    aruco_pose_is_car_in_marker_frame = rx.Parameter(True)
    aruco_goal_offset = rx.Parameter(0.0)  # stop short of marker center [m]
    aruco_distance_topic = rx.Parameter("aruco/distance_m") # distance to aruco marker, updated by subscriber
    use_aruco_goal = rx.Parameter(False)

    localizer = LocalizationInterface()
    goal_tolerance = rx.Parameter(0.2) #m

    #Publishers
    steering_cmd_pub = rx.Publisher(Float32, steering_cmd_topic, qos_pubber)
    velocity_cmd_pub = rx.Publisher(Float32, velocity_cmd_topic, qos_pubber)
    goal_pub = rx.Publisher(Marker, 'goal_marker', qos_pubber)
    path_pub = rx.Publisher(Marker, 'path_marker', qos_pubber)
    traj_pub = rx.Publisher(Marker, 'traj_marker', qos_pubber)
    cross_track_error_pub = rx.Publisher(Float32, 'cross_track_error', qos_pubber)
    yaw_error_pub = rx.Publisher(Float32, 'yaw_error', qos_pubber)
    velocity_error_pub = rx.Publisher(Float32, 'velocity_error', qos_pubber)
    dist_to_goal = rx.Publisher(Float32, 'dist_to_goal', qos_pubber)

    #Subscribers
    @rx.Subscriber(PoseWithCovarianceStamped, '/mocap/svea/pose', qos_pubber)
    def _svea67_pose_cb(self, msg: PoseWithCovarianceStamped):
        self.svea_pose = msg
        self.svea_identified_mocap = True

    @rx.Subscriber(PoseWithCovarianceStamped, '/mocap/charging_station/pose', qos_pubber)
    def _charging_station_cb(self, msg: PoseWithCovarianceStamped):
        if not self.charging_station_identified_mocap: #runs once
            self.charging_station_pose = msg
            self.goal = [msg.pose.pose.position.x, msg.pose.pose.position.y]
            pointOne, pointTwo = self.calculate_points()
            self.waypoints = [pointTwo, pointOne, self.goal]
            self.charging_station_identified_mocap = True


    @rx.Subscriber(Float32, aruco_distance_topic)
    def _aruco_distance_cb(self, msg: Float32):
        if not bool(self.use_aruco_goal):
            return
        self.aruco_distance = msg.data

    @rx.Subscriber(String, "mission/active_controller", qos_pubber)
    def _mission_active_cb(self, msg: String):
        self.active_controller = msg.data

    @rx.Subscriber(PoseArray, aruco_goal_topic)
    def _aruco_goal_cb(self, msg: PoseArray):
        if not bool(self.use_aruco_goal):
            return
        if len(msg.poses) == 0:
            #self.get_logger().warn("No Aruco markers detected, cannot update goal")
            return

        pose = msg.poses[0]
        aruco_x = pose.position.x
        aruco_y = pose.position.z
        thetaAruco = pose.orientation.y
        aruco_x_rel = self.aruco_distance * np.cos(thetaAruco)
        aruco_y_rel = self.aruco_distance * np.sin(thetaAruco)

        state = self.localizer.get_state()
        x, y, yaw, vel = state

        aruco_vec = np.array([aruco_x_rel, aruco_y_rel])
        car_vec = np.array([x, y])
        A = np.array([[np.cos(yaw), -np.sin(yaw)],
                      [np.sin(yaw), np.cos(yaw)]])
        aruco_in_map = car_vec + A @ aruco_vec

       
        self.goal = [aruco_in_map[0], aruco_in_map[1]]
        self.waypoints = self.endPoints
        self.reached_goal = False
         

    def on_startup(self):
        self.reached_goal = False
        self.counter = 0
        self.aruco_distance = 5.0 # default value until we get a reading from the subscriber
        self.controller_ready = False
        self.wait_log_throttle = 0
        self.endPoints = eval(self.endPoints)
        self.goal = eval(self.endPoint)
        self.waypoints = self.endPoints
        self.controller = StanleyController(node=self)
        self.controller.target_velocity = self.target_velocity
        self.create_timer(self.DELTA_TIME, self.loop)


    def loop(self):
        """
        Main loop of the Stanley controller. 
        """
        if self.active_controller != str(self.controller_name):
            return

        state = self._get_control_state()
        if state is None:
            return

        x, y, yaw, vel = state

        if bool(self.use_adaptive_speed):
            self.controller.target_velocity = self.target_velocity * (self.aruco_distance / 5.0)

        dist = self.distance_to_goal(state)
        if dist <= self.goal_tolerance:
            if not self.reached_goal:
                self.get_logger().info("Reached goal!")
                self.reached_goal = False # for the line follower implementation
        
        if not self.reached_goal:
            steering, velocity = self.controller.compute_control(state)
            # self.get_logger().info(f"Steering: {steering}, Velocity: {velocity}")
        else:
            steering, velocity = np.deg2rad(16), 0.0
            self.velocity = 0.0

        self.steering_cmd_pub.publish(Float32(data=float(steering)))
        self.velocity_cmd_pub.publish(Float32(data=float(velocity)))
        self.publish_errors()
        self.dist_to_goal.publish(Float32(data=dist))
        

        if self.counter % 10 == 0: # publish markers every 1 seconds
            self.publish_goal_marker(self.goal)
            self.publish_waypoints_marker(self.waypoints)
            self.publish_trajectory_marker(self.controller.cx, self.controller.cy)
            # Publish errors
        self.counter += 1

    def _get_control_state(self):
        if not bool(self.use_mocap):
            state = self.localizer.get_state()
            if not self.controller_ready:
                self.controller.update_traj(state, self.waypoints)
                self.controller_ready = True
            return state

        if bool(self.use_mocap_goal):
            if not (self.charging_station_identified_mocap and self.svea_identified_mocap):
                if self.wait_log_throttle % 20 == 0:
                    self.get_logger().info("Waiting for mocap pose and charging station pose...")
                self.wait_log_throttle += 1
                return None
        elif not self.svea_identified_mocap:
            if self.wait_log_throttle % 20 == 0:
                self.get_logger().info("Waiting for SVEA mocap pose...")
            self.wait_log_throttle += 1
            return None

        self.wait_log_throttle = 0
        state = (
            self.svea_pose.pose.pose.position.x,
            self.svea_pose.pose.pose.position.y,
            euler_from_quaternion(
                [
                    self.svea_pose.pose.pose.orientation.x,
                    self.svea_pose.pose.pose.orientation.y,
                    self.svea_pose.pose.pose.orientation.z,
                    self.svea_pose.pose.pose.orientation.w,
                ]
            )[2],
            0.0,
        )
        if not self.controller_ready:
            self.controller.update_traj(state, self.waypoints)
            self.controller_ready = True
        return state


    def calculate_points(self):
        station_x = self.charging_station_pose.pose.pose.position.x
        station_y = self.charging_station_pose.pose.pose.position.y

        q = self.charging_station_pose.pose.pose.orientation
        quat = [q.x, q.y, q.z, q.w]

        _, _, station_yaw = euler_from_quaternion(quat)

        offset = 0.8

        pointOne = [
            station_x + offset/3 * np.cos(station_yaw),
            station_y + offset/3 * np.sin(station_yaw)
        ]

        pointTwo = [
            station_x + 2*offset * np.cos(station_yaw),
            station_y + 2*offset * np.sin(station_yaw)
        ]
        self.charging_station_identified_mocap = True
        return pointOne, pointTwo

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

    def publish_errors(self):
        _, _, _, vel = self.localizer.get_state()
        
        self.cross_track_error_pub.publish(Float32(data=self.controller.cross_track_error)) #m
        self.yaw_error_pub.publish(Float32(data=np.rad2deg(self.controller.yaw_error))) #degrees

        # Velocity error
        vel_err = self.controller.target_velocity - vel
        self.velocity_error_pub.publish(Float32(data=vel_err))



if __name__ == '__main__':
    stanley_control.main()
