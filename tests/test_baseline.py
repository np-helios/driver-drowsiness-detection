import tempfile
import unittest
from pathlib import Path

from driver_drowsiness.baseline import BaselineCalibrator, BaselineStore, RollingBehavior
from driver_drowsiness.severity import classify_deviation
from driver_drowsiness.yawn import is_confirmed_yawn


class BaselineTests(unittest.TestCase):
    def test_baseline_store_round_trip(self):
        calibrator = BaselineCalibrator()
        for ear in (150.0, 152.0, 149.0):
            calibrator.add_open_ear(ear)
        for mar in (0.32, 0.31, 0.33):
            calibrator.add_mar(mar)
        calibrator.add_head_pose(yaw=0.0, roll=0.0, pitch=5.0)
        calibrator.add_head_pose(yaw=1.0, roll=1.0, pitch=6.0)
        calibrator.add_blink_metrics(0.3, 4.5)
        calibrator.add_blink_metrics(0.35, 4.2)
        calibrator.add_yawn_duration(1.0)
        profile = calibrator.build("alice")

        with tempfile.TemporaryDirectory() as tmpdir:
            store = BaselineStore(Path(tmpdir) / "alice.json")
            store.save(profile)
            loaded = store.load()

        self.assertEqual(loaded, profile)

    def test_personalized_deviation_escalates_for_large_shift(self):
        calibrator = BaselineCalibrator()
        for ear in (150.0, 151.0, 149.0, 152.0):
            calibrator.add_open_ear(ear)
        for mar in (0.31, 0.32, 0.33, 0.31):
            calibrator.add_mar(mar)
        calibrator.add_head_pose(yaw=0.0, roll=0.0, pitch=5.0)
        calibrator.add_head_pose(yaw=1.0, roll=1.0, pitch=6.0)
        calibrator.add_head_pose(yaw=-1.0, roll=-1.0, pitch=5.5)
        calibrator.add_blink_metrics(0.25, 5.0)
        calibrator.add_blink_metrics(0.3, 4.8)
        calibrator.add_blink_metrics(0.28, 5.2)
        calibrator.add_yawn_duration(1.0)
        calibrator.add_yawn_duration(0.9)
        profile = calibrator.build("alice")

        behavior = RollingBehavior()
        behavior.add_open_ear(120.0)
        behavior.add_mar(0.72)
        behavior.add_head_pose(yaw=10.0, roll=8.0, pitch=18.0)
        behavior.add_blink_metrics(blink_duration=1.4, open_interval=1.0)
        behavior.add_yawn_duration(2.2)

        result = classify_deviation(
            profile,
            behavior,
            current_ear=120.0,
            current_mar=0.72,
            current_head_yaw=10.0,
            current_head_roll=8.0,
            current_head_pitch=18.0,
        )

        self.assertEqual(result.level, 0)
        self.assertEqual(result.alarm_name, "power")

    def test_personalized_deviation_stays_low_near_baseline(self):
        calibrator = BaselineCalibrator()
        for ear in (150.0, 151.0, 149.0, 152.0):
            calibrator.add_open_ear(ear)
        for mar in (0.31, 0.32, 0.33, 0.31):
            calibrator.add_mar(mar)
        calibrator.add_head_pose(yaw=0.0, roll=0.0, pitch=5.0)
        calibrator.add_head_pose(yaw=1.0, roll=1.0, pitch=6.0)
        calibrator.add_head_pose(yaw=-1.0, roll=-1.0, pitch=5.5)
        calibrator.add_blink_metrics(0.25, 5.0)
        calibrator.add_blink_metrics(0.3, 4.8)
        calibrator.add_blink_metrics(0.28, 5.2)
        calibrator.add_yawn_duration(1.0)
        calibrator.add_yawn_duration(0.9)
        profile = calibrator.build("alice")

        behavior = RollingBehavior()
        behavior.add_open_ear(149.5)
        behavior.add_mar(0.32)
        behavior.add_head_pose(yaw=0.5, roll=0.5, pitch=5.7)
        behavior.add_blink_metrics(blink_duration=0.3, open_interval=4.9)
        behavior.add_yawn_duration(1.0)

        result = classify_deviation(
            profile,
            behavior,
            current_ear=149.5,
            current_mar=0.32,
            current_head_yaw=0.5,
            current_head_roll=0.5,
            current_head_pitch=5.7,
        )

        self.assertEqual(result.level, 2)

    def test_head_pose_wraparound_does_not_explode_score(self):
        calibrator = BaselineCalibrator()
        for ear in (150.0, 151.0, 149.0, 152.0):
            calibrator.add_open_ear(ear)
        for mar in (0.31, 0.32, 0.33, 0.31):
            calibrator.add_mar(mar)
        calibrator.add_head_pose(yaw=0.0, roll=0.0, pitch=179.0)
        calibrator.add_head_pose(yaw=1.0, roll=1.0, pitch=-179.0)
        calibrator.add_head_pose(yaw=-1.0, roll=-1.0, pitch=178.5)
        calibrator.add_blink_metrics(0.25, 5.0)
        calibrator.add_blink_metrics(0.3, 4.8)
        calibrator.add_blink_metrics(0.28, 5.2)
        profile = calibrator.build("alice")

        behavior = RollingBehavior()
        behavior.add_open_ear(149.5)
        behavior.add_mar(0.32)
        behavior.add_head_pose(yaw=0.4, roll=0.3, pitch=179.4)
        behavior.add_blink_metrics(blink_duration=0.3, open_interval=4.9)

        result = classify_deviation(
            profile,
            behavior,
            current_ear=149.5,
            current_mar=0.32,
            current_head_yaw=0.4,
            current_head_roll=0.3,
            current_head_pitch=179.4,
        )

        self.assertLess(result.score, 1.0)

    def test_short_mouth_opening_is_not_confirmed_as_yawn(self):
        calibrator = BaselineCalibrator()
        for mar in (0.010, 0.011, 0.012, 0.010):
            calibrator.add_mar(mar)
        profile = calibrator.build("alice")

        self.assertFalse(
            is_confirmed_yawn(
                duration_seconds=0.9,
                peak_mar=0.028,
                baseline_profile=profile,
                min_duration_seconds=1.4,
            )
        )

    def test_sustained_large_mouth_opening_is_confirmed_as_yawn(self):
        calibrator = BaselineCalibrator()
        for mar in (0.010, 0.011, 0.012, 0.010):
            calibrator.add_mar(mar)
        profile = calibrator.build("alice")

        self.assertTrue(
            is_confirmed_yawn(
                duration_seconds=1.8,
                peak_mar=0.045,
                baseline_profile=profile,
                min_duration_seconds=1.4,
            )
        )


if __name__ == "__main__":
    unittest.main()
