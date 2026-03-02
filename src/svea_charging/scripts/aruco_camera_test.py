#!/usr/bin/env python3
"""ROS 2 ArUco camera test node for quick local webcam-based detection."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseArray
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Int32MultiArray, String, Float32

from svea_core import rosonic as rx


def get_aruco_module():
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "OpenCV ArUco module saknas. Installera 'opencv-contrib-python'."
        )
    return cv2.aruco


def get_dictionary(aruco, dictionary_name: str):
    if not hasattr(aruco, dictionary_name):
        available = [name for name in dir(aruco) if name.startswith("DICT_")]
        raise ValueError(
            f"Okänd dictionary '{dictionary_name}'. Exempel: {', '.join(available[:10])}"
        )

    dictionary_id = getattr(aruco, dictionary_name)
    if hasattr(aruco, "getPredefinedDictionary"):
        return aruco.getPredefinedDictionary(dictionary_id)
    return aruco.Dictionary_get(dictionary_id)


def generate_marker(
    aruco,
    dictionary,
    marker_id: int,
    marker_size_px: int,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(aruco, "generateImageMarker"):
        marker_img = aruco.generateImageMarker(dictionary, marker_id, marker_size_px)
    else:
        marker_img = aruco.drawMarker(dictionary, marker_id, marker_size_px)

    cv2.imwrite(str(output_path), marker_img)


def create_detector(aruco, dictionary, *, use_aruco_detector_api: bool = False):
    # Some OpenCV builds (especially in containers) segfault when constructing
    # DetectorParameters. Avoid touching it unless we explicitly use the newer
    # ArucoDetector API.
    if not use_aruco_detector_api:
        return None, None

    if hasattr(aruco, "DetectorParameters"):
        parameters = aruco.DetectorParameters()
    else:
        parameters = aruco.DetectorParameters_create()

    # Improve robustness for webcam images.
    if hasattr(parameters, "cornerRefinementMethod") and hasattr(aruco, "CORNER_REFINE_SUBPIX"):
        parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

    if use_aruco_detector_api and hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(dictionary, parameters)
        return detector, None

    return None, parameters


def detect_markers(aruco, detector, parameters, dictionary, frame):
    if detector is not None:
        return detector.detectMarkers(frame)
    if parameters is None:
        return aruco.detectMarkers(frame, dictionary)
    return aruco.detectMarkers(frame, dictionary, parameters=parameters)


def load_calibration(calibration_file: Path | None):
    if calibration_file is None:
        return None, None

    data = np.load(str(calibration_file))
    if "camera_matrix" not in data or "dist_coeffs" not in data:
        raise ValueError(
            "Kalibreringsfilen måste innehålla 'camera_matrix' och 'dist_coeffs' (.npz)."
        )

    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]
    return camera_matrix, dist_coeffs


def get_fallback_camera_matrix(frame_shape, focal_length_px: float | None):
    height, width = frame_shape[:2]
    fx = focal_length_px if focal_length_px is not None else 0.9 * width
    fy = focal_length_px if focal_length_px is not None else 0.9 * width
    cx = width / 2.0
    cy = height / 2.0

    camera_matrix = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)
    return camera_matrix, dist_coeffs


def rvec_to_euler_deg(rvec):
    rot_mat, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rot_mat[0, 0] ** 2 + rot_mat[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(rot_mat[2, 1], rot_mat[2, 2])
        pitch = np.arctan2(-rot_mat[2, 0], sy)
        yaw = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
    else:
        roll = np.arctan2(-rot_mat[1, 2], rot_mat[1, 1])
        pitch = np.arctan2(-rot_mat[2, 0], sy)
        yaw = 0.0

    return np.degrees([roll, pitch, yaw])


def rotation_matrix_to_quaternion(rot_mat):
    """Return quaternion as [x, y, z, w]."""
    trace = rot_mat[0, 0] + rot_mat[1, 1] + rot_mat[2, 2]
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rot_mat[2, 1] - rot_mat[1, 2]) * s
        y = (rot_mat[0, 2] - rot_mat[2, 0]) * s
        z = (rot_mat[1, 0] - rot_mat[0, 1]) * s
    elif rot_mat[0, 0] > rot_mat[1, 1] and rot_mat[0, 0] > rot_mat[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rot_mat[0, 0] - rot_mat[1, 1] - rot_mat[2, 2])
        w = (rot_mat[2, 1] - rot_mat[1, 2]) / s
        x = 0.25 * s
        y = (rot_mat[0, 1] + rot_mat[1, 0]) / s
        z = (rot_mat[0, 2] + rot_mat[2, 0]) / s
    elif rot_mat[1, 1] > rot_mat[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rot_mat[1, 1] - rot_mat[0, 0] - rot_mat[2, 2])
        w = (rot_mat[0, 2] - rot_mat[2, 0]) / s
        x = (rot_mat[0, 1] + rot_mat[1, 0]) / s
        y = 0.25 * s
        z = (rot_mat[1, 2] + rot_mat[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + rot_mat[2, 2] - rot_mat[0, 0] - rot_mat[1, 1])
        w = (rot_mat[1, 0] - rot_mat[0, 1]) / s
        x = (rot_mat[0, 2] + rot_mat[2, 0]) / s
        y = (rot_mat[1, 2] + rot_mat[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w], dtype=float)


def rvec_to_quaternion_xyzw(rvec):
    rot_mat, _ = cv2.Rodrigues(rvec)
    return rotation_matrix_to_quaternion(rot_mat)


def estimate_pose_for_markers(aruco, corners, marker_length_m, camera_matrix, dist_coeffs):
    if hasattr(aruco, "estimatePoseSingleMarkers"):
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
            corners, marker_length_m, camera_matrix, dist_coeffs
        )
        return rvecs, tvecs

    raise RuntimeError("Din OpenCV-version saknar estimatePoseSingleMarkers för ArUco.")


def draw_axes(frame, camera_matrix, dist_coeffs, rvec, tvec, axis_length_m):
    if hasattr(cv2, "drawFrameAxes"):
        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, axis_length_m)


class aruco_camera_test(rx.Node):
    dictionary = rx.Parameter("DICT_4X4_50")
    marker_id = rx.Parameter(0)
    marker_size_px = rx.Parameter(400)
    output = rx.Parameter("aruco_marker.png")
    generate_marker_on_startup = rx.Parameter(False)

    camera_index = rx.Parameter(0)
    marker_length_m = rx.Parameter(0.05)
    calibration_file = rx.Parameter("")
    focal_length_px = rx.Parameter(-1.0)
    display = rx.Parameter(False)
    loop_hz = rx.Parameter(30.0)
    frame_id = rx.Parameter("camera")
    camera_backend = rx.Parameter("ANY")  # ANY, V4L2, GSTREAMER, FFMPEG
    use_aruco_detector_api = rx.Parameter(False)
    publish_debug_image = rx.Parameter(True)
    jpeg_quality = rx.Parameter(80)

    detected_ids_pub = rx.Publisher(Int32MultiArray, "aruco/detected_ids")
    poses_pub = rx.Publisher(PoseArray, "aruco/poses")
    status_pub = rx.Publisher(String, "aruco/status")
    debug_image_pub = rx.Publisher(CompressedImage, "aruco/debug_image/compressed")
    distance_pub = rx.Publisher(Float32, "aruco/distance_m")

    def on_startup(self):
        self.cap = None
        self._warned_fallback_intrinsics = False

        try:
            self.get_logger().info("Initializing OpenCV ArUco module...")
            self.aruco = get_aruco_module()
            self.get_logger().info(f"Loading dictionary: {self.dictionary}")
            self.dictionary_obj = get_dictionary(self.aruco, str(self.dictionary))
        except Exception as exc:
            self.get_logger().error(f"ArUco setup failed: {exc}")
            return

        if self.generate_marker_on_startup:
            try:
                out_path = Path(str(self.output))
                generate_marker(
                    self.aruco,
                    self.dictionary_obj,
                    int(self.marker_id),
                    int(self.marker_size_px),
                    out_path,
                )
                self.get_logger().info(f"Generated marker image: {out_path.resolve()}")
            except Exception as exc:
                self.get_logger().error(f"Marker generation failed: {exc}")

        self.get_logger().info(
            f"Creating detector (use_aruco_detector_api={bool(self.use_aruco_detector_api)})..."
        )
        self.detector, self.detector_parameters = create_detector(
            self.aruco,
            self.dictionary_obj,
            use_aruco_detector_api=bool(self.use_aruco_detector_api),
        )

        marker_length = float(self.marker_length_m)
        self._marker_length_m = marker_length if marker_length > 0.0 else None

        focal_length = float(self.focal_length_px)
        self._focal_length_px = focal_length if focal_length > 0.0 else None

        calib_str = str(self.calibration_file).strip()
        calib_path = Path(calib_str) if calib_str else None
        try:
            if calib_path is not None:
                self.get_logger().info(f"Loading calibration file: {calib_path}")
            self.calibrated_camera_matrix, self.calibrated_dist_coeffs = load_calibration(
                calib_path
            )
        except Exception as exc:
            self.get_logger().error(f"Calibration load failed: {exc}")
            self.calibrated_camera_matrix, self.calibrated_dist_coeffs = None, None

        backend_name = str(self.camera_backend).upper()
        backend_map = {
            "ANY": getattr(cv2, "CAP_ANY", 0),
            "V4L2": getattr(cv2, "CAP_V4L2", getattr(cv2, "CAP_ANY", 0)),
            "GSTREAMER": getattr(cv2, "CAP_GSTREAMER", getattr(cv2, "CAP_ANY", 0)),
            "FFMPEG": getattr(cv2, "CAP_FFMPEG", getattr(cv2, "CAP_ANY", 0)),
        }
        backend = backend_map.get(backend_name, getattr(cv2, "CAP_ANY", 0))
        self.get_logger().info(
            f"Opening camera index {int(self.camera_index)} with backend={backend_name}..."
        )
        self.cap = cv2.VideoCapture(int(self.camera_index), backend)
        if not self.cap.isOpened():
            self.get_logger().error(
                f"Could not open camera index {int(self.camera_index)} with backend={backend_name}"
            )
            self.cap.release()
            self.cap = None
            return

        self.get_logger().info("Camera opened. Performing first frame read...")
        ok, _ = self.cap.read()
        if not ok:
            self.get_logger().error("Camera opened but first frame read failed.")
            self.cap.release()
            self.cap = None
            return

        self.get_logger().info(
            "ArUco camera node started "
            f"(camera_index={int(self.camera_index)}, backend={backend_name}, dictionary={self.dictionary})"
        )
        if self._marker_length_m is not None and self.calibrated_camera_matrix is None:
            self.get_logger().warning(
                "Pose estimation uses fallback intrinsics (no calibration file provided)."
            )

        period = 1.0 / max(float(self.loop_hz), 1.0)
        self.create_timer(period, self.loop)

    def on_shutdown(self):
        if getattr(self, "cap", None) is not None:
            self.cap.release()
            self.cap = None
        if bool(self.display):
            cv2.destroyAllWindows()

    def _publish_status(self, text: str):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)

    def _publish_ids(self, ids_list):
        msg = Int32MultiArray()
        msg.data = [int(i) for i in ids_list]
        self.detected_ids_pub.publish(msg)

    def _empty_pose_array(self):
        msg = PoseArray()
        msg.header.frame_id = str(self.frame_id)
        msg.header.stamp = self.get_clock().now().to_msg()
        return msg

    def _publish_debug_image(self, frame):
        if not bool(self.publish_debug_image):
            return
        quality = int(max(10, min(100, int(self.jpeg_quality))))
        ok, enc = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        )
        if not ok:
            return
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = str(self.frame_id)
        msg.format = "jpeg"
        msg.data = enc.tobytes()
        self.debug_image_pub.publish(msg)

    def loop(self):
        if self.cap is None:
            return

        ok, frame = self.cap.read()
        if not ok:
            self._publish_ids([])
            self.poses_pub.publish(self._empty_pose_array())
            self._publish_status("camera_read_failed")
            return

        corners, ids, rejected = detect_markers(
            self.aruco,
            self.detector,
            self.detector_parameters,
            self.dictionary_obj,
            frame,
        )

        ids_list = []
        pose_array = self._empty_pose_array()
        status_text = ""
        overlay_color = (0, 0, 255)

        if ids is not None and len(ids) > 0:
            ids_list = [int(i) for i in ids.flatten()]
            self.aruco.drawDetectedMarkers(frame, corners, ids)
            status_text = f"Detected IDs: {ids_list}"
            overlay_color = (0, 200, 0)

            if self._marker_length_m is not None:
                if self.calibrated_camera_matrix is None:
                    camera_matrix, dist_coeffs = get_fallback_camera_matrix(
                        frame.shape, self._focal_length_px
                    )
                    self._warned_fallback_intrinsics = True
                else:
                    camera_matrix, dist_coeffs = (
                        self.calibrated_camera_matrix,
                        self.calibrated_dist_coeffs,
                    )

                rvecs, tvecs = estimate_pose_for_markers(
                    self.aruco,
                    corners,
                    self._marker_length_m,
                    camera_matrix,
                    dist_coeffs,
                )

                for i, marker_id in enumerate(ids.flatten()):
                    rvec = rvecs[i]
                    tvec = tvecs[i]
                    txyz = np.asarray(tvec).reshape(-1)
                    qxyzw = rvec_to_quaternion_xyzw(rvec)
                    roll_deg, pitch_deg, yaw_deg = rvec_to_euler_deg(rvec)

                    pose = Pose()
                    pose.position.x = float(txyz[0])
                    pose.position.y = float(txyz[1])
                    pose.position.z = float(txyz[2])
                    pose.orientation.x = float(qxyzw[0])
                    pose.orientation.y = float(qxyzw[1])
                    pose.orientation.z = float(qxyzw[2])
                    pose.orientation.w = float(qxyzw[3])
                    pose_array.poses.append(pose)

                    draw_axes(
                        frame,
                        camera_matrix,
                        dist_coeffs,
                        rvec,
                        tvec,
                        axis_length_m=max(self._marker_length_m * 0.5, 0.02),
                    )

                    anchor = corners[i][0][0]
                    x_px, y_px = int(anchor[0]), int(anchor[1])
                    distance_m = float(np.linalg.norm(txyz))
                    lines = [
                        f"ID {int(marker_id)}: {distance_m:.2f} m",
                        f"R/P/Y: {roll_deg:.0f}/{pitch_deg:.0f}/{yaw_deg:.0f} deg",
                    ]
                    for line_idx, line in enumerate(lines):
                        cv2.putText(
                            frame,
                            line,
                            (x_px, y_px - 12 - line_idx * 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 220, 0),
                            2,
                            cv2.LINE_AA,
                        )
        else:
            rejected_count = len(rejected) if rejected is not None else 0
            status_text = f"No markers | rejected: {rejected_count}"

        self._publish_ids(ids_list)
        self.poses_pub.publish(pose_array)
        self._publish_status(status_text)
        self.distance_pub.publish(Float32(data=distance_m if ids_list else -1.0))

        if bool(self.display) or bool(self.publish_debug_image):
            cv2.putText(
                frame,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                overlay_color,
                2,
                cv2.LINE_AA,
            )

        self._publish_debug_image(frame)

        if bool(self.display):
            cv2.imshow("ArUco Camera Test", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self.get_logger().info("Shutdown requested from display window (q).")
                rclpy.shutdown()


if __name__ == "__main__":
    aruco_camera_test.main()
