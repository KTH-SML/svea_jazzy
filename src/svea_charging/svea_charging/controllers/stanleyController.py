"""

Stanley control based on the PythonRobotics library by Atsushi Sakai (@Atsushi_twi)

Reference:
    - [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics?tab=readme-ov-file#stanley-control)

"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import pathlib
from svea_core import rosonic as rx

from svea_core.interfaces import LocalizationInterface

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))
from svea_charging.third_party.PythonRobotics.PathPlanning.CubicSpline import cubic_spline_planner

# Parameters
k = 3.0 # control gain
Kp = 0.5  # speed proportional gain
dt = 0.05  # [s] time difference
L = 0.2  # [m] Wheel base of vehicle (TODO: check this value)
max_steer = np.radians(50.0)  # [rad] max steering angle (TODO: check this value)

Ki = .2
Kd = 0.01




class StanleyController:
    """
    Class representing the state of a vehicle.

    :param x: (float) x-coordinate
    :param y: (float) y-coordinate
    :param yaw: (float) yaw angle
    :param v: (float) speed
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0, node: rx.Node = None):
        """Instantiate the object."""
        super().__init__()
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        
        self.ax = []
        self.ay = []
        self.cx = []
        self.cy = []
        self.cyaw = []
        self.ck = []
        self.s = []
        self.target_idx = 0
        self.target_velocity = 0.0

        self.error_integral = 0.0
        self.error_derivative = 0.0
        self.error_prev = 0.0

        self.cross_track_error = 0.0
        self.yaw_error = 0.0
        self.node = node

    def update(self, state):
        """
        Update the state of the vehicle.
        """
        x, y, yaw, vel = state
        self.x = x
        self.y = y
        self.yaw = yaw
        self.yaw = self.normalize_angle(self.yaw)
        self.v = vel


    def pid_control(self):
        """
        Proportional control for the speed.

        :param target: (float)
        :param current: (float)
        :return: (float)
        """
        error = self.target_velocity - self.v

        error_i = (error + self.error_prev) / 2 * dt
        error_d = (error - self.error_prev) / dt
        self.error_derivative = error_d

        self.error_integral += error_i
        self.error_integral = np.clip(self.error_integral, -self.target_velocity*1.5, self.target_velocity*1.5)  # Anti-windup

        self.error_prev = error

        return Kp * error + Ki * self.error_integral + Kd * self.error_derivative


    def stanley_control(self, cx, cy, cyaw, last_target_idx):
        """
        Stanley steering control.

        :param state: (State object)
        :param cx: ([float])
        :param cy: ([float])
        :param cyaw: ([float])
        :param last_target_idx: (int)
        :return: (float, int)
        """
        if not cx or not cy or not cyaw:
            raise ValueError("Stanley path is empty; cannot compute steering.")

        current_target_idx, error_front_axle = self.calc_target_index(cx, cy)
        self.cross_track_error = error_front_axle # for providing output to external


        if last_target_idx >= current_target_idx:
            current_target_idx = last_target_idx

        # Path can be rebuilt every loop; keep stale indices within bounds.
        max_idx = min(len(cx), len(cy), len(cyaw)) - 1
        current_target_idx = int(np.clip(current_target_idx, 0, max_idx))

        # theta_e corrects the heading error
        theta_e = self.normalize_angle(cyaw[current_target_idx] - self.yaw)
        self.yaw_error = theta_e # for providing output to external
        # theta_d corrects the cross track error
        theta_d = np.arctan2(k * error_front_axle, max(self.v, 0.4))# added division to reduce steering angle for better stability at low speeds
        # self.node.get_logger().info(f"Theta_e: {theta_e}, Theta_d: {theta_d}")
        # Steering control
        delta = theta_e + theta_d
        # delta = theta_e

        return delta, current_target_idx


    def normalize_angle(self, angle):
        """
        Normalize an angle to [-pi, pi].

        :param angle: (float)
        :return: (float) Angle in radian in [-pi, pi]
        """
        #return angle_mod(angle) # this is from the original robotic library example
        return (angle + np.pi) % (2 * np.pi) - np.pi #reimplementation using numpy due to dependencies on scipy


    def calc_target_index(self, cx, cy):
        """
        Compute index in the trajectory list of the target.

        :param state: (State object)
        :param cx: [float]
        :param cy: [float]
        :return: (int, float)
        """
        if not cx or not cy:
            raise ValueError("Stanley path is empty; cannot compute target index.")

        # Calc front axle position
        fx = self.x + L * np.cos(self.yaw)
        fy = self.y + L * np.sin(self.yaw)

        # Search nearest point index
        dx = [fx - icx for icx in cx]
        dy = [fy - icy for icy in cy]
        d = np.hypot(dx, dy)        
        target_idx = np.argmin(d) #(TODO: fix fallback))

        # Project RMS error onto front axle vector
        front_axle_vec = [-np.cos(self.yaw + np.pi / 2),
                        -np.sin(self.yaw + np.pi / 2)]
        error_front_axle = np.dot([dx[target_idx], dy[target_idx]], front_axle_vec)

        return target_idx, error_front_axle


    def update_traj(self, state, waypoints):
        x, y, yaw, v = state
        #assuming waypoints list of (x, y) tuples
        '''
        #this takes the first two waypoints only
        first_wp, second_wp = waypoints[0], waypoints[1]
        first_wp_x, first_wp_y = first_wp
        second_wp_x, second_wp_y = second_wp

        self.ax = [x, first_wp_x, second_wp_x]
        self.ay = [y, first_wp_y, second_wp_y]
        '''
        '''
        for point in waypoints:
            self.ax.append(point[0])
            self.ay.append(point[1])'''
        #added to reduce time lag
        self.ax = [x] + [p[0] for p in waypoints]
        self.ay = [y] + [p[1] for p in waypoints]


        self.cx, self.cy, self.cyaw, self.ck, self.s = cubic_spline_planner.calc_spline_course(
            self.ax, self.ay, ds=0.2)

        # Keep target index valid when trajectory size changes.
        if self.cx:
            self.target_idx = int(np.clip(self.target_idx, 0, len(self.cx) - 1))
        else:
            self.target_idx = 0


    def compute_steering(self):
        cx = self.cx
        cy = self.cy
        cyaw = self.cyaw

        delta, self.target_idx = self.stanley_control(cx, cy, cyaw, self.target_idx)
        return delta


    def compute_velocity(self):
        return self.pid_control()


    def compute_control(self, state):
        self.update(state)
        steering = self.compute_steering()
        velocity = self.compute_velocity()
        

        return steering, velocity
