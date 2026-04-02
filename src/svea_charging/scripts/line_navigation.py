#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from svea_core.interfaces import ActuationInterface


VEL = 0.4
Kp = 1
Ki = 0
Kd = 0

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()

    actuation = ActuationInterface()


    def listener_callback(self, data):
        # Convert ROS Image message to OpenCV image
        current_frame = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        
        # Convert BGR to HSV
        hsv_image = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)

        # Define range of blue color in HSV
        lower_blue = np.array([100, 50, 50])   # Lower bound of blue color
        upper_blue = np.array([130, 255, 255])  # Upper bound of blue color

        # Create a binary mask
        blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

        # Apply the mask to the original image
        blue_segmented_image = cv2.bitwise_and(current_frame, current_frame, mask=blue_mask)


        # Detect line and get its centroid
        line = self.get_contour_data(blue_mask)

        # Display the segmented image with line centroid
        if line:
            cv2.circle(blue_segmented_image, (line['x'], line['y']), 5, (0, 0, 255), 7)

        # Move depending on detection 
        _, width, _ = blue_segmented_image.shape
        if line:
            x = line['x']

            error = x - width//2

            self.actuation.send_control(steering, velocity)
            cv2.circle(blue_segmented_image, (line['x'], line['y']), 5, (0, 0, 255), 7)
        
        # Determine the speed to turn and get the line in the center of the camera.
        steering = float(error) * -Kp + Ki...
        print("Error: {} | Angular Z: {}, ".format(error, steering))

        # Send the command to execute
        self.publisher.publish(cmd)




        # Display the segmented image
        cv2.imshow("Blue Segmented Image", blue_segmented_image)
        cv2.waitKey(1)


        def get_contour_data(self, mask):
            """
            Return the centroid of the largest contour in the binary image 'mask' (the line) 
            """ 
            # Constants
            MIN_AREA_TRACK = 50  # Minimum area for track marks

            # get a list of contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            line = {}

            for contour in contours:
                M = cv2.moments(contour)

                if (M['m00'] > MIN_AREA_TRACK):
                    # Contour is part of the track
                    line['x'] = int(M["m10"]/M["m00"])
                    line['y'] = int(M["m01"]/M["m00"])

            return (line)


def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()