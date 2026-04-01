import unittest

from driver_drowsiness.metrics import eye_aspect_ratio


class EyeAspectRatioTests(unittest.TestCase):
    def test_eye_aspect_ratio_for_known_eye_shape(self):
        eye = [
            (0.0, 0.0),
            (1.0, 2.0),
            (2.0, 2.0),
            (4.0, 0.0),
            (2.0, -2.0),
            (1.0, -2.0),
        ]

        ratio = eye_aspect_ratio(eye)

        self.assertAlmostEqual(ratio, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
