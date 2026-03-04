#! /usr/bin/env python3

import rclpy
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from mocap4r2_msgs.msg import RigidBodies
from tf2_ros import TransformListener, Buffer
from tf2_ros import TransformException
import tf2_geometry_msgs
import time

from svea_core import rosonic as rx


qos_normal = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
)

qos_subber = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,  # BEST_EFFORT
    history=QoSHistoryPolicy.KEEP_LAST,         # Keep the last N messages
    durability=QoSDurabilityPolicy.VOLATILE,    # Volatile
    depth=10,                                   # Size of the queue
)

class MocapToPose(rx.Node):

    ## Subscribers ##
    mocap_topic = rx.Parameter('/rigid_bodies')

    ## Publishers ##
    mocap_svea_pub = rx.Publisher(PoseWithCovarianceStamped, '/mocap/svea/pose', qos_normal)
    mocap_trailer_pub = rx.Publisher(PoseWithCovarianceStamped, '/mocap/trailer/pose', qos_normal)
    initialpose_pub = rx.Publisher(PoseWithCovarianceStamped, '/set_pose', qos_normal)    

    _svea_pose_msg = PoseWithCovarianceStamped()
    _trailer_pose_msg = PoseWithCovarianceStamped()
    _initialpose_msg = PoseWithCovarianceStamped()

    def on_startup(self):
        """Start the Mocap interface by subscribing to the pose topic."""
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.initial_pose = False
        time.sleep(3.0) # Wait for TF buffer to fill up with transforms
        self.get_logger().info("Starting Mocap interface Node...")
        self.get_logger().info("Mocap is ready.")

        self.create_timer(2.0, self._timer_callback)
        
    def _timer_callback(self):
        self.initial_pose = False

    @rx.Subscriber(RigidBodies, mocap_topic, qos_profile=qos_subber)
    def _rigid_bodies_pose_cb(self, msg: RigidBodies) -> None:
        for rigid_body in msg.rigidbodies:
            if rigid_body.rigid_body_name == "svea67":
                self._svea_pose_msg.header = msg.header
                self._svea_pose_msg.pose.pose = rigid_body.pose
                self.mocap_svea_pub.publish(self._svea_pose_msg)
                if not self.initial_pose and rigid_body.pose.position.x is not None:
                    try:
                        pose_stamped = PoseStamped()
                        pose_stamped.header = msg.header
                        pose_stamped.pose = rigid_body.pose
                        transform = self.tf_buffer.lookup_transform('map', 'mocap', rclpy.time.Time(),timeout=rclpy.duration.Duration(seconds=1.0))
                        pose_transformed = tf2_geometry_msgs.do_transform_pose_stamped(
                            pose_stamped,
                            transform
                        )
                        self._initialpose_msg.header = pose_transformed.header
                        self._initialpose_msg.pose.pose = pose_transformed.pose
                        self.initialpose_pub.publish(self._initialpose_msg)
                        self.get_logger().info("Mocap svea initial pose published: x={:.2f}, y={:.2f}".format(
                            pose_transformed.pose.position.x, pose_transformed.pose.position.y))
                        self.initial_pose = True
                    except TransformException as ex:
                        self.get_logger().warn(f'Could not transform: {ex}')
            elif rigid_body.rigid_body_name == "trailer":
                self._trailer_pose_msg.header = msg.header
                self._trailer_pose_msg.pose.pose = rigid_body.pose
                self.mocap_trailer_pub.publish(self._trailer_pose_msg)

if __name__ == '__main__':
    MocapToPose.main()

    
