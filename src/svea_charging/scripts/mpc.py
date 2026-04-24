#! /usr/bin/env python3

import numpy as np
import math
import tf_transformations as tf
from svea_core.models.bicycle import Bicycle4DWithESC
from svea_core.interfaces import LocalizationInterface
try:
    from svea_mocap.mocap import MotionCaptureInterface
except ImportError:
    pass
from svea_core.controllers.mpc import MPC
from std_msgs.msg import Float32, String
from geometry_msgs.msg import PoseArray, PoseStamped, PoseWithCovarianceStamped, Pose, Point
from visualization_msgs.msg import Marker
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.clock import Clock
from tf_transformations import euler_from_quaternion

from svea_core import rosonic as rx

import time

qos_subber = QoSProfile(depth=10 # Size of the queue 
                        )

qos_pubber = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)

class mpc(rx.Node):
    is_sim = rx.Parameter(True)
    state = [-3.0, 0.0, 0.0, 0.0]  # x, y, yaw, velocity
    mpc_freq = rx.Parameter(10)  # Hz
    delta_s = rx.Parameter(0.4)  # m
    mpc_config_ns = rx.Parameter('/mpc')
    target_speed = rx.Parameter(0.2)  # m/s
    controller_name = rx.Parameter("mpc")
    active_controller = rx.Parameter("idle")
    steering_cmd_topic = rx.Parameter("mpc/cmd_steering_rad")
    velocity_cmd_topic = rx.Parameter("mpc/cmd_velocity_mps")
    svea_mocap_name = rx.Parameter("svea")
    prediction_horizon = rx.Parameter(5)
    final_state_weight_matrix = rx.Parameter(None)  # Weight matrix for the final state in MPC

    localizer = LocalizationInterface()

    ## MPC parameters 
    GOAL_REACHED_DIST = 0.02 # The distance threshold (in meters) within which the goal is considered reached.
    GOAL_REACHED_YAW = 0.1    # The yaw angle threshold (in radians) within which the goal orientation is considered reached.
    UPDATE_MPC_PARAM = True   # A flag indicating if the MPC parameters can be updated when the system is approaching the target.
    RESET_MPC_PARAM = False   # A flag indicating if the MPC parameters should be reset when the system is moving away from the target.
    predicted_state = None
    SLOWDOWN_DISTANCE = 0.2   # Distance from the goal where commanded speed starts ramping down in simulation.
    MIN_APPROACH_SPEED = 0.12 # Keep a small crawl speed in simulation without dropping too close to zero.

    ## Static Planner parameters
    APPROACH_TARGET_THR = 3.0   # The distance threshold (in meters) to define when the system is "approaching" the target.
    NEW_REFERENCE_THR = 1     # The distance threshold (in meters) to update the next intermediate reference point. 
    goal_pose = None
    static_path_plan = np.empty((3, 0))
    current_index_static_plan = 0
    is_last_point = False

    ## Other parameters
    steering = 0
    velocity = 0
    state = None

    mocap_state = [0.0, 0.0, 0.0]  # x, y, yaw from mocap

    dt =0.01

    aruco_goal_topic = rx.Parameter("aruco/poses")
  
    steering_cmd_pub = rx.Publisher(Float32, steering_cmd_topic, qos_profile=qos_pubber)
    velocity_cmd_pub = rx.Publisher(Float32, velocity_cmd_topic, qos_profile=qos_pubber)
    steering_pub = rx.Publisher(Float32, '/target_steering_angle', qos_profile=qos_pubber)
    velocity_pub = rx.Publisher(Float32, '/target_speed', qos_profile=qos_pubber)
    velocity_measured_pub = rx.Publisher(Float32, '/measured_speed', qos_profile=qos_pubber)
    predicted_trajectory_pub = rx.Publisher(PoseArray, '/predicted_path', qos_profile=qos_pubber)
    static_trajectory_pub = rx.Publisher(PoseArray, '/static_path', qos_profile=qos_pubber)
    path_pub = rx.Publisher(Marker, '/mpc_path', qos_profile=qos_pubber)

    @rx.Subscriber(String, "mission/active_controller", qos_profile=qos_pubber)
    def mission_active_callback(self, msg: String):
        self.active_controller = msg.data

    @rx.Subscriber(PoseStamped, '/mpc_target', qos_profile=qos_subber)
    def mpc_target_callback(self, msg):
        """
        Callback function that sets a new goal position and calculates a trajectory.
        :param msg: PoseStamped message containing the goal position.
        """
        # Set the goal position and log the new goal
        self.goal_pose = msg
        self.get_logger().info(f"New goal position received: ({self.goal_pose.pose.position.x}, {self.goal_pose.pose.position.y})")
        # Compute trajectory (straight line from current position to goal)
        self.compute_trajectory()

    @rx.Subscriber(PoseWithCovarianceStamped, '/mocap/svea/pose', qos_profile=qos_subber)
    def mocap_pose_callback(self, msg):
        """
        Callback function that updates the current state of the SVEA based on motion capture data.
        :param msg: PoseWithCovarianceStamped message containing the current pose of the SVEA.
        """
        self.mocap_state[0] = msg.pose.pose.position.x
        self.mocap_state[1] = msg.pose.pose.position.y
        orientation = msg.pose.pose.orientation
        quaternion = (orientation.x, orientation.y, orientation.z, orientation.w)
        # Use tf to convert quaternion to Euler angles
        euler = tf.euler_from_quaternion(quaternion)
        self.mocap_state[2] = euler[2]
        self.has_mocap_state = True

    @rx.Subscriber(Pose, 'aruco_to_map', qos_profile=qos_subber)
    def _aruco_goal_cb(self, msg: Pose):
        
        self.aruco_x = msg.position.x
        self.aruco_y = msg.position.y
        self.aruco_yaw = msg.orientation.z


   
    def on_startup(self):
        self.has_mocap_state = False
        self.state = None
        self.aruco_x = None
        self.aruco_y = None
        self.aruco_yaw = None

        ## Define the unitless steering biases for each SVEA.
        ## These values represent the measured steering actuations when the SVEA is not actually steering.
        self.unitless_steering_map = {
            "svea0": 28,
            "svea7": 7
        }

        self.controller = MPC(self)
        self.DELTA_TIME = 1.0/self.mpc_freq
        self.initial_horizon = self.prediction_horizon
        self.initial_Qf = self.final_state_weight_matrix

        if not self.is_sim:
            svea_name = self.svea_mocap_name.lower()  # Ensure case-insensitivity  
            unitless_steering = 29
            PERC_TO_LLI_COEFF = 1.27
            MAX_STEERING_ANGLE = 40 * math.pi / 180
            steer_percent = unitless_steering / PERC_TO_LLI_COEFF
            self.steering_bias = (steer_percent / 100.0) * MAX_STEERING_ANGLE
        else:
            self.steering_bias = 0
        
        self.create_timer(self.DELTA_TIME, self.loop)


    def _get_current_state(self):
        state = self.localizer.get_state()
        if state is None:
            return None

        state = list(state)
        # if self.aruco_x is not None:
            
        #     state[0] = self.aruco_x
        #     state[1] = self.aruco_y
        #     state[2] = self.aruco_yaw
            

        return state

   


    def loop(self):
        if self.active_controller != str(self.controller_name):
            return

        # Retrieve current state from SVEA localization and optionally override
        # position/yaw with the mapped ArUco pose used during docking.
        self.state = self._get_current_state()
        if self.state is None:
            return

        # if self.goal_pose is not None and self.static_path_plan.size == 0:
        #     self.compute_trajectory()

        # If a static path plan has been computed, run the mpc.
        if self.static_path_plan.size > 0 :
            # If enough time has passed, run the MPC computation
            current_time = Clock().now().to_msg()
            time_diff_sec = (current_time.sec - self.mpc_last_time.sec) + \
                (current_time.nanosec - self.mpc_last_time.nanosec) / 1e9
            measured_dt = time_diff_sec
                        # current_time - self.mpc_last_time
            if measured_dt >= self.DELTA_TIME:
                reference_trajectory, distance_to_goal = self.get_mpc_current_reference()
                if self.is_last_point and distance_to_goal <= self.APPROACH_TARGET_THR and self.UPDATE_MPC_PARAM:
                    # Update the prediction horizon and final state weight matrix only once when approaching target to achieve better parking.
                    # Increase x-weight relative to y-weight to improve x-coordinate precision at the goal.
                    new_Qf = np.array([140, 0, 0, 0,
                                        0, 70, 0, 0,
                                        0, 0, 30, 0,
                                        0, 0, 0, 0]).reshape((4, 4))
                    self.controller.set_new_weight_matrix('Qf', new_Qf)
                    self.UPDATE_MPC_PARAM = False
                    self.RESET_MPC_PARAM = True  # Allow resetting when moving away

                elif self.is_last_point and distance_to_goal > self.APPROACH_TARGET_THR and self.RESET_MPC_PARAM:
                    # Reset to initial values only once when moving away from target
                    self.current_horizon = self.initial_horizon
                    self.controller.set_new_prediction_horizon(self.initial_horizon)
                    self.controller.set_new_weight_matrix('Qf', self.initial_Qf)
                    self.UPDATE_MPC_PARAM = True  # Allow updating again when re-approaching
                    self.RESET_MPC_PARAM = False  # Prevent repeated resetting

                if  not self.is_goal_reached(distance_to_goal):
                    # Run the MPC to compute control
                    feedback_state = self.get_feedback_state()
                    steering_rate, acceleration = self.controller.compute_control(feedback_state, reference_trajectory)
                    self.steering += steering_rate * measured_dt
                    self.steering = float(
                        np.clip(
                            self.steering,
                            self.controller.min_steering,
                            self.controller.max_steering,
                        )
                    )
                    self.velocity += acceleration * measured_dt
                    speed_limit = self.compute_approach_speed_limit(distance_to_goal)
                    self.velocity = float(
                        np.clip(
                            self.velocity,
                            self.controller.min_velocity,
                            min(self.controller.max_velocity, speed_limit),
                        )
                    )
                    self.predicted_state = self.controller.get_optimal_states()
                else:
                    # Stop the vehicle if the goal is reached
                    self.steering, self.velocity = 0, 0

                # Update the last time the MPC was computed
                self.mpc_last_time = current_time
            # self.get_logger().info(f"Steering: {self.steering}, Velocity: {self.velocity}")
        
        # if self.velocity != 0 and self.velocity > 0.0:
        #     self.velocity = max(0.36, self.velocity)  # Ensure a minimum velocity of 0.1 m/s when moving forward
        # elif self.velocity != 0 and self.velocity < 0.0:
        #     self.velocity = min(-0.36, self.velocity)  # Ensure a minimum velocity of -0.1 m/s when moving backward
        # self.get_logger().info(f"Steering: {self.steering}, Velocity: {self.velocity}")
            
        # Publish the latest control target and the estimated speed( from mocap or indoors loc. or outdoors loc.).
        # self.publish_to_foxglove(self.steering, self.velocity, self.state[3])
        # Visualization data and send control

        steering_cmd = self.steering + self.steering_bias
        bounded_velocity = float(
            np.clip(
                self.velocity,
                self.controller.min_velocity,
                self.controller.max_velocity,
            )
        )
        self.velocity = bounded_velocity
        self.steering_cmd_pub.publish(Float32(data=float(steering_cmd)))
        self.velocity_cmd_pub.publish(Float32(data=bounded_velocity))
        # self.svea.visualize_data()
        

    def publish_to_foxglove(self,target_steering,target_speed,measured_speed):
        self.steering_pub.publish(target_steering)
        self.velocity_pub.publish(target_speed)
        self.velocity_measured_pub.publish(measured_speed)

    

    def get_mpc_current_reference(self):
        """
        Retrieves a forward-looking reference sequence for the MPC based on the
        closest remaining point on the static path.

        Returns:
            tuple: (x_ref, distance_to_goal), where x_ref is the reference state for
            the prediction horizon (shape: [4, N+1]) and distance_to_goal is the
            distance to the final point in the plan.
        """
        last_index = self.static_path_plan.shape[1] - 1
        self.current_index_static_plan = self.find_closest_remaining_point_index()
        self.is_last_point = self.current_index_static_plan >= max(0, last_index - 1)

        start_index = self.current_index_static_plan
        end_index = start_index + self.initial_horizon + 1
        x_ref = self.static_path_plan[:, start_index:min(end_index, last_index + 1)]

        while x_ref.shape[1] < self.initial_horizon + 1:
            x_ref = np.concatenate((x_ref, self.static_path_plan[:, -1:]), axis=1)

        target_speed_row = np.full((1, x_ref.shape[1]), self.target_speed)
        x_ref = np.concatenate((x_ref, target_speed_row), axis=0)

        distance_to_goal = self.compute_distance(self.state, self.static_path_plan[:, -1])
        return x_ref, distance_to_goal

    def find_closest_remaining_point_index(self):
        """
        Finds the closest point on the remaining plan while preventing the
        reference from jumping backwards along the path.
        """
        remaining_plan = self.static_path_plan[:2, self.current_index_static_plan:]
        if remaining_plan.size == 0:
            return self.static_path_plan.shape[1] - 1

        distances = np.linalg.norm(
            remaining_plan - np.array(self.state[:2], dtype=float)[:, None],
            axis=0,
        )
        return self.current_index_static_plan + int(np.argmin(distances))

    def compute_approach_speed_limit(self, distance_to_goal):
        """
        Linearly reduce the maximum forward speed as the vehicle approaches the
        final docking point. The final reference point uses zero speed so the
        optimizer has a true stopping target even when GOAL_REACHED_DIST is low.
        """
        max_forward_speed = min(float(self.target_speed), float(self.controller.max_velocity))
        if not self.is_last_point:
            return max_forward_speed

        if distance_to_goal <= self.GOAL_REACHED_DIST:
            return 0.0

        # On the real platform, low crawl speeds map to very small integer throttle
        # commands and the car may not move at all. Keep the full requested docking
        # speed until the stop tolerance is satisfied.
        if not self.is_sim:
            return max_forward_speed

        slowdown_ratio = np.clip(distance_to_goal / self.SLOWDOWN_DISTANCE, 0.0, 1.0)
        return max(self.MIN_APPROACH_SPEED, max_forward_speed * slowdown_ratio)

    def get_feedback_state(self):
        x, y, yaw, velocity = self.state
        
        # if self.aruco_x is not None:
     
        #     x = self.aruco_x
        #     y = self.aruco_y
        #     yaw = self.aruco_yaw
        return [x, y, yaw, velocity, self.steering]


    def publish_waypoints_marker(self, waypoints):
        # Draw straight segments between waypoints
        msg = Marker()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.ns = "mpc_waypoints"
        msg.id = 0
        msg.type = Marker.LINE_STRIP
        msg.action = Marker.ADD

        msg.scale.x = 0.02  # line width

        msg.color.r = 1.0
        msg.color.g = 1.0
        msg.color.b = 0.0
        msg.color.a = 1.0

        msg.points = []
        for wp in np.asarray(waypoints).T:
            p = Point()
            p.x = float(wp[0])
            p.y = float(wp[1])
            p.z = 0.05
            msg.points.append(p)

        self.path_pub.publish(msg)

    def compute_trajectory(self):
        """
        Compute a straight-line trajectory from the current position to the goal using delta_s,
        including the heading for each point, and publish the path.
        """
        if self.goal_pose is None:
            self.get_logger().warning("Missing goal for trajectory computation.")
            return

        self.state = self._get_current_state()
        if self.state is None:
            self.get_logger().warning("Missing goal or current state for trajectory computation.")
            return
        # Reset previous trajectory related variables
        self.current_index_static_plan = 0
        self.is_last_point = False
        self.static_path_plan = np.empty((3, 0))
        # reset control actions
        self.velocity = 0
        self.steering = 0
        # reset mpc parameters
        self.UPDATE_MPC_PARAM = True  
        self.RESET_MPC_PARAM = False
        self.mpc_last_time = Clock().now().to_msg()
        self.controller.reset_parameters()

        # Calculate the straight-line trajectory between current state and goal position
        start_x, start_y = self.state[0], self.state[1]
        goal_x, goal_y = self.goal_pose.pose.position.x, self.goal_pose.pose.position.y
        distance = self.compute_distance([start_x, start_y], [goal_x, goal_y])
        goal_yaw = self.get_yaw_from_pose(self.goal_pose)

        # Compute intermediate points at intervals of delta_s
        num_points = int(distance // self.delta_s)

        for i in range(num_points):
            ratio = ((i+1) * self.delta_s) / distance
            x = start_x + ratio * (goal_x - start_x)
            y = start_y + ratio * (goal_y - start_y)
            
            # Calculate the heading for this point
            heading = math.atan2(goal_y - start_y, goal_x - start_x)
            
            # Stack the computed point as a new column in the array
            new_point = np.array([[x], [y], [heading]])
            self.static_path_plan = np.hstack((self.static_path_plan, new_point))

        # Calculate distance between the last appended point and the goal point.
        if self.static_path_plan.size != 0:
            last_appended_x = self.static_path_plan[0, -1]
            last_appended_y = self.static_path_plan[1, -1]    
            distance = self.compute_distance([last_appended_x, last_appended_y], [goal_x, goal_y])

            # If the distance to the goal is too small, replace the last point with the goal directly.
            if distance < self.delta_s / 2:
                # Replace the last appended point with the goal point
                self.static_path_plan[:, -1] = np.array([goal_x, goal_y, goal_yaw])
            else:
                # Otherwise, append the last point as usual
                new_point = np.array([[goal_x], [goal_y], [goal_yaw]])
                self.static_path_plan = np.hstack((self.static_path_plan, new_point))
        else:
            # Otherwise, append the last point as usual
            new_point = np.array([[goal_x], [goal_y], [goal_yaw]])
            self.static_path_plan = np.hstack((self.static_path_plan, new_point))

        self.get_logger().warning("Complete trajectory computation.")
        self.publish_waypoints_marker(self.static_path_plan)
        
    def compute_distance(self,point1,point2):
        return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)
    
    def get_yaw_from_pose(self,pose_stamped):
        """
        Extracts the yaw from a PoseStamped message.
        """
        # Convert the quaternion to Euler angles
        orientation = pose_stamped.pose.orientation
        quaternion = (orientation.x, orientation.y, orientation.z, orientation.w)
        
        # Use tf to convert quaternion to Euler angles
        euler = tf.euler_from_quaternion(quaternion)
        
        # Return the yaw
        return euler[2]  
    
    def is_goal_reached(self,distance):
        if  not self.is_last_point:
            return False        
        elif distance < self.GOAL_REACHED_DIST:
            yaw_error = math.atan2(
                math.sin(self.state[2] - self.static_path_plan[2, -1]),
                math.cos(self.state[2] - self.static_path_plan[2, -1]),
            )
            if  abs(yaw_error) < self.GOAL_REACHED_YAW:
                return True
            else:
                return False
        else:
            return False
    

if __name__ == '__main__':
    mpc.main()
    
