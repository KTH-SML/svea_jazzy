#!/bin/bash
# DEVICE=$(ls /dev/serial/by-id/*SVEA-LLI* | head -1)
# ros2 run micro_ros_agent micro_ros_agent serial --dev "$DEVICE" -b 115200

ros2 run micro_ros_agent micro_ros_agent serial --dev /dev/ttyACM0 -b 115200