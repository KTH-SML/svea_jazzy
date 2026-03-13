#! /usr/bin/env python3

import numpy as np
from geometry_msgs.msg import Point
from geometry_msgs.msg import PoseArray
from visualization_msgs.msg import Marker

from svea_core.interfaces import LocalizationInterface
from svea_core.controllers.pure_pursuit import PurePursuitController
from svea_core.interfaces import ActuationInterface, ShowMarker, ShowPath
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

class pure_pursuit(rx.Node):
    DELTA_TIME = 0.1
    TRAJ_LEN = 20

    endPoint = rx.Parameter('[1.885, 1.348]') #x= -1.360,y=  1.382, yaw = 90deg
    target_velocity = rx.Parameter(0.7)
    use_aruco_goal = rx.Parameter(False)
    aruco_goal_topic = rx.Parameter("aruco/poses")
    aruco_pose_is_car_in_marker_frame = rx.Parameter(True)
    aruco_goal_offset = rx.Parameter(0.0)  # stop short of marker center [m]
    aruco_distance_topic = rx.Parameter("aruco/distance_m") # distance to aruco marker, updated by subscriber
    # Interfaces
    actuation = ActuationInterface()
    localizer = LocalizationInterface()
    goal_tolerance = rx.Parameter(0.05) #m

    goal_marker = ShowMarker() # for goal visualization
    path = ShowPath() # for path visualization


    #Publishers
    goal_pub = rx.Publisher(Marker, 'goal_marker', qos_pubber)
    path_pub = rx.Publisher(Marker, 'path_marker', qos_pubber)
    traj_pub = rx.Publisher(Marker, 'traj_marker', qos_pubber)
    velocity_error_pub = rx.Publisher(Float32, 'velocity_error', qos_pubber)
    dist_to_goal = rx.Publisher(Float32, 'dist_to_goal', qos_pubber)


    @rx.Subscriber(Float32, aruco_distance_topic)
    def _aruco_distance_cb(self, msg: Float32):
        if not bool(self.use_aruco_goal):
            return
        self.aruco_distance = msg.data

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
        mid_x = 0.5 * (x + aruco_in_map[0])
        mid_y = 0.5 * (y + aruco_in_map[1])
        self.waypoints = [[x, y], [mid_x, mid_y], self.goal]
        self.reached_goal = False


    def on_startup(self):
        self.reached_goal = False
        self.counter = 0

        self.controller = PurePursuitController()
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
        self.publish_goal_marker(self.goal)

        #self.publish_waypoints_marker(self.waypoints)

        self.update_traj(x, y)
        self.create_timer(self.DELTA_TIME, self.loop)


    def loop(self):
        """
        Main loop of the Stanley controller. 
        """
        state = self.localizer.get_state()
        x, y, yaw, vel = state

        dist = self.distance_to_goal(state)
        if dist <= self.goal_tolerance:
            if not self.reached_goal:
                self.get_logger().info("Reached goal!")
                self.reached_goal = True

        #self.update_goal()
        self.update_traj(x, y)

        if not self.reached_goal:
            steering, velocity = self.controller.compute_control(state)
            self.get_logger().info(f"Steering: {steering}, Velocity: {velocity}")
        else:
            steering, velocity = 0.0, 0.0

        self.actuation.send_control(steering, velocity)
        

        if self.counter % 5 == 0: # publish markers every .5 seconds
            self.publish_goal_marker(self.goal)
            # self.publish_waypoints_marker(self.waypoints)
            # self.publish_trajectory_marker(self.controller.cx, self.controller.cy)
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
        # Velocity error
        vel_err = self.controller.target_velocity - vel
        self.velocity_error_pub.publish(Float32(data=vel_err))

    def update_traj(self, x, y):
        """
        Update the trajectory based on the current state and the goal. It
        generates a linear trajectory from the current position to the goal
        position, and updates the controller's trajectory points.
        The trajectory is visualized using the ShowPath interface.
        """
        xs = np.linspace(x, self.goal[0], self.TRAJ_LEN)
        ys = np.linspace(y, self.goal[1], self.TRAJ_LEN)
        self.controller.traj_x = xs
        self.controller.traj_y = ys
        self.path.publish_path(xs,ys)

if __name__ == '__main__':
    pure_pursuit.main()
