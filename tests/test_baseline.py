import tempfile
import unittest
from pathlib import Path

from driver_drowsiness.baseline import BaselineCalibrator, BaselineStore, RollingBehavior
from driver_drowsiness.severity import classify_deviation


class BaselineTests(unittest.TestCase):
    def test_baseline_store_round_trip(self):
        calibrator = BaselineCalibrator()
        for ear in (150.0, 152.0, 149.0):
            calibrator.add_open_ear(ear)
        calibrator.add_blink_metrics(0.3, 4.5)
        calibrator.add_blink_metrics(0.35, 4.2)
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
        calibrator.add_blink_metrics(0.25, 5.0)
        calibrator.add_blink_metrics(0.3, 4.8)
        calibrator.add_blink_metrics(0.28, 5.2)
        profile = calibrator.build("alice")

        behavior = RollingBehavior()
        behavior.add_open_ear(120.0)
        behavior.add_blink_metrics(blink_duration=1.4, open_interval=1.0)

        result = classify_deviation(profile, behavior, current_ear=120.0)

        self.assertEqual(result.level, 0)
        self.assertEqual(result.alarm_name, "power")

    def test_personalized_deviation_stays_low_near_baseline(self):
        calibrator = BaselineCalibrator()
        for ear in (150.0, 151.0, 149.0, 152.0):
            calibrator.add_open_ear(ear)
        calibrator.add_blink_metrics(0.25, 5.0)
        calibrator.add_blink_metrics(0.3, 4.8)
        calibrator.add_blink_metrics(0.28, 5.2)
        profile = calibrator.build("alice")

        behavior = RollingBehavior()
        behavior.add_open_ear(149.5)
        behavior.add_blink_metrics(blink_duration=0.3, open_interval=4.9)

        result = classify_deviation(profile, behavior, current_ear=149.5)

        self.assertEqual(result.level, 2)


if __name__ == "__main__":
    unittest.main()
