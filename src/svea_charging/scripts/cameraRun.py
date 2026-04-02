#!/usr/bin/env python3

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class CameraBridgeForCalibration(Node):
    def __init__(self):
        super().__init__("camera_bridge_for_calibration")

        self.publisher_image = self.create_publisher(Image, "/camera/image_raw", 1)
        self.bridge = CvBridge()

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera index 0")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30.0)

        self.create_timer(1.0 / 30.0, self.loop)

    def loop(self):
        ok, frame = self.cap.read()
        if not ok:
            return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera"
        self.publisher_image.publish(msg)

    def on_shutdown(self):
        if self.cap is not None:
            self.cap.release()


def main():
    rclpy.init()
    try:
        node = CameraBridgeForCalibration()
    except Exception as exc:
        print(f"Failed to start camera bridge: {exc}")
        rclpy.shutdown()
        return
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
