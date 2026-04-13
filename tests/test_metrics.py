import unittest

from driver_drowsiness.metrics import (
    eye_aspect_ratio,
    mouth_aspect_ratio,
)

try:
    from driver_drowsiness.pose import estimate_head_pose
except ModuleNotFoundError:
    estimate_head_pose = None


class MetricTests(unittest.TestCase):
    def test_eye_aspect_ratio_for_known_eye_shape(self):
        eye = [
            (0.0, 0.0),
            (1.0, 2.0),
            (2.0, 2.0),
            (4.0, 0.0),
            (2.0, -2.0),
            (1.0, -2.0),
        ]
        self.assertAlmostEqual(eye_aspect_ratio(eye), 1.0, places=6)

    def test_mouth_aspect_ratio_increases_when_mouth_is_open(self):
        mouth = [
            (0.0, 0.0),
            (1.0, 2.0),
            (2.0, 3.0),
            (3.0, 2.0),
            (4.0, 0.0),
            (3.0, -2.0),
            (2.0, -3.0),
            (1.0, -2.0),
        ]
        self.assertGreater(mouth_aspect_ratio(mouth), 1.0)

    def test_estimate_head_pose_returns_float_angles(self):
        if estimate_head_pose is None:
            self.skipTest("OpenCV not available in the current Python environment")
        pose = estimate_head_pose(
            (480, 640, 3),
            nose_tip=(320.0, 240.0),
            chin=(320.0, 360.0),
            left_eye_corner=(250.0, 200.0),
            right_eye_corner=(390.0, 200.0),
            left_mouth_corner=(275.0, 300.0),
            right_mouth_corner=(365.0, 300.0),
        )
        self.assertIsInstance(pose.yaw, float)
        self.assertIsInstance(pose.pitch, float)
        self.assertIsInstance(pose.roll, float)


if __name__ == "__main__":
    unittest.main()
