from __future__ import annotations

import argparse
import time
import timeit
from dataclasses import dataclass, field
from pathlib import Path
from threading import Thread

import cv2
import imutils
import numpy as np
from imutils.video import VideoStream

from .audio import sound_alarm
from .baseline import BaselineCalibrator, BaselineProfile, BaselineStore, RollingBehavior
from .config import DetectorConfig
from .landmarks import FaceMeshLandmarks
from .logging_utils import EventLogger
from .metrics import eye_aspect_ratio
from .preprocess import light_removing
from .severity import SeverityResult, classify_deviation


@dataclass
class RuntimeState:
    open_ear: float = 0.0
    ear_threshold: float = 0.0
    frame_counter: int = 0
    timer_flag: bool = False
    alarm_flag: bool = False
    alarm_count: int = 0
    running_time: float = 0.0
    previous_open_term: float = 0.0
    current_ear: float = 0.0
    start_closing: float = 0.0
    closed_eyes_time: list[float] = field(default_factory=list)
    test_data: list[list[float]] = field(default_factory=list)
    result_data: list[int] = field(default_factory=list)
    latest_face_detected: bool = False
    baseline_ready: bool = False
    baseline_started_at: float = 0.0


class DrowsinessDetectorApp:
    def __init__(self, config: DetectorConfig):
        self.config = config
        self.state = RuntimeState()
        self.landmarks = FaceMeshLandmarks(model_path=config.face_landmarker_task_path)
        self.logger = EventLogger(config.event_log_path)
        self.baseline_store = BaselineStore(config.baseline_path)
        self.baseline_calibrator = BaselineCalibrator()
        self.behavior = RollingBehavior()
        self.baseline_profile: BaselineProfile | None = self.baseline_store.load()
        self.video_stream = VideoStream(src=config.camera_source).start()
        self.state.baseline_ready = self.baseline_profile is not None
        if self.baseline_profile is not None:
            self.state.open_ear = self.baseline_profile.open_ear_mean
            self.state.ear_threshold = max(
                self.baseline_profile.open_ear_mean
                - max(self.baseline_profile.open_ear_std * 2.5, 8.0),
                0.0,
            )

    def start(self) -> None:
        print("loading MediaPipe face mesh...")
        print("starting video stream thread...")
        time.sleep(self.config.warmup_seconds)
        self.state.baseline_started_at = timeit.default_timer()
        if self.baseline_profile is None:
            print(
                "No saved driver baseline found. Starting personalized calibration "
                f"for {self.config.personalized_calibration_seconds:.0f} seconds."
            )
        else:
            print(f"Loaded saved baseline for driver '{self.baseline_profile.driver_id}'.")

        try:
            self._run_loop()
        finally:
            self._finalize_baseline_if_needed(force=True)
            cv2.destroyAllWindows()
            self.video_stream.stop()
            self.landmarks.close()

    def _run_loop(self) -> None:
        while True:
            frame = self.video_stream.read()
            if frame is None:
                continue
            frame = imutils.resize(frame, width=self.config.frame_width)
            _, _processed_frame = light_removing(frame, blur_kernel=self.config.light_blur_kernel)
            eye_landmarks = self.landmarks.detect(frame)
            self.state.latest_face_detected = eye_landmarks is not None

            if eye_landmarks is not None:
                left_ear = eye_aspect_ratio(eye_landmarks.left_eye)
                right_ear = eye_aspect_ratio(eye_landmarks.right_eye)
                self.state.current_ear = (left_ear + right_ear) * self.config.eye_scale_factor
                self.behavior.add_open_ear(self.state.current_ear)
                self._update_personal_baseline()

                cv2.drawContours(
                    frame,
                    [cv2.convexHull(_as_int_points(eye_landmarks.left_eye))],
                    -1,
                    (0, 255, 0),
                    1,
                )
                cv2.drawContours(
                    frame,
                    [cv2.convexHull(_as_int_points(eye_landmarks.right_eye))],
                    -1,
                    (0, 255, 0),
                    1,
                )

                if self._is_eye_closed():
                    self._handle_closed_eyes()
                else:
                    self._handle_open_eyes()

            cv2.putText(
                frame,
                f"EAR : {self.state.current_ear:.2f}",
                (250, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 30, 20),
                2,
            )
            if not self.state.baseline_ready:
                remaining = max(
                    0.0,
                    self.config.personalized_calibration_seconds
                    - (timeit.default_timer() - self.state.baseline_started_at),
                )
                cv2.putText(
                    frame,
                    f"Building baseline: {remaining:.0f}s",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    def _update_personal_baseline(self) -> None:
        if self.state.baseline_ready:
            return
        self.baseline_calibrator.add_open_ear(self.state.current_ear)
        self._finalize_baseline_if_needed()

    def _finalize_baseline_if_needed(self, force: bool = False) -> None:
        if self.state.baseline_ready:
            return
        elapsed = timeit.default_timer() - self.state.baseline_started_at
        if not force and elapsed < self.config.personalized_calibration_seconds:
            return
        profile = self.baseline_calibrator.build(self.config.driver_id)
        if profile.sample_count <= 0 and not force:
            return
        self.baseline_profile = profile
        self.baseline_store.save(profile)
        self.state.baseline_ready = True
        self.state.open_ear = profile.open_ear_mean
        self.state.ear_threshold = max(profile.open_ear_mean - max(profile.open_ear_std * 2.5, 8.0), 0.0)
        print(f"Saved personalized baseline for '{profile.driver_id}'.")
        print(f"Baseline open EAR mean={profile.open_ear_mean:.3f}, std={profile.open_ear_std:.3f}")
        self.logger.log(
            "baseline_ready",
            driver_id=profile.driver_id,
            baseline_profile=profile,
            ear_threshold=self.state.ear_threshold,
        )

    def _is_eye_closed(self) -> bool:
        threshold = self.state.ear_threshold
        if threshold <= 0:
            return False
        return self.state.current_ear < threshold

    def _handle_closed_eyes(self) -> None:
        if not self.state.timer_flag:
            self.state.start_closing = timeit.default_timer()
            self.state.timer_flag = True

        self.state.frame_counter += 1
        if self.state.frame_counter < self.config.ear_consecutive_frames:
            return

        mid_closing = timeit.default_timer()
        closing_time = round(mid_closing - self.state.start_closing, 3)
        if closing_time < self.state.running_time:
            return

        if self.state.running_time == 0:
            current_term = timeit.default_timer()
            opened_eyes_time = round(current_term - self.state.previous_open_term, 3)
            self.state.previous_open_term = current_term
            self.state.running_time = 1.75
        else:
            opened_eyes_time = round(timeit.default_timer() - self.state.previous_open_term, 3)

        self.state.running_time += 2
        self.state.alarm_flag = True
        self.state.alarm_count += 1

        print(f"{self.state.alarm_count}st ALARM")
        print(f"The time eyes are open before the alarm: {opened_eyes_time}")
        print(f"closing time: {closing_time}")

        self.state.test_data.append([opened_eyes_time, round(closing_time * 10, 3)])
        self.behavior.add_blink_metrics(closing_time, opened_eyes_time)
        if not self.state.baseline_ready or self.baseline_profile is None:
            severity = SeverityResult(
                level=1,
                alarm_name="normal",
                rationale="baseline_not_ready",
                score=0.0,
            )
        else:
            self.baseline_calibrator.add_blink_metrics(closing_time, opened_eyes_time)
            severity = classify_deviation(
                baseline=self.baseline_profile,
                behavior=self.behavior,
                current_ear=self.state.current_ear,
            )
        self.state.result_data.append(severity.level)
        self.logger.log(
            "alarm_triggered",
            alarm_count=self.state.alarm_count,
            opened_eyes_time=opened_eyes_time,
            closing_time=closing_time,
            severity=severity,
            ear_threshold=round(self.state.ear_threshold, 3),
            current_ear=round(self.state.current_ear, 3),
        )
        self._play_async_alarm(self._alarm_path_for_result(severity))

    def _handle_open_eyes(self) -> None:
        self.state.frame_counter = 0
        self.state.timer_flag = False
        self.state.running_time = 0

        if self.state.alarm_flag:
            end_closing = timeit.default_timer()
            closed_time = round(end_closing - self.state.start_closing, 3)
            self.state.closed_eyes_time.append(closed_time)
            print(f"The time eyes were closed: {self.state.closed_eyes_time}")

        self.state.alarm_flag = False

    def _alarm_path_for_result(self, result: int) -> Path:
        level = result.level if isinstance(result, SeverityResult) else result
        if level == 0:
            return self.config.power_alarm_path
        if level == 1:
            return self.config.normal_alarm_path
        return self.config.short_alarm_path

    def _play_async_alarm(self, path: Path) -> None:
        Thread(target=sound_alarm, args=(path,), daemon=True).start()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Driver drowsiness detection with OpenCV and MediaPipe.")
    parser.add_argument("--driver-id", default="default-driver", help="Unique identifier for the driver profile.")
    parser.add_argument("--camera-source", type=int, default=0, help="Camera index to open.")
    parser.add_argument("--frame-width", type=int, default=400, help="Display width for the camera frame.")
    parser.add_argument(
        "--event-log",
        type=Path,
        default=DetectorConfig.event_log_path,
        help="Path to a JSONL event log file. Use an empty value in code to disable logging.",
    )
    parser.add_argument(
        "--baseline-seconds",
        type=float,
        default=90.0,
        help="Duration of initial personalized baseline capture in seconds.",
    )
    parser.add_argument(
        "--baseline-path",
        type=Path,
        default=None,
        help="Optional path for the saved driver baseline JSON.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    baseline_path = args.baseline_path or (DetectorConfig.baseline_path.parent / f"{args.driver_id}.json")
    config = DetectorConfig(
        driver_id=args.driver_id,
        face_landmarker_task_path=DetectorConfig.face_landmarker_task_path,
        camera_source=args.camera_source,
        frame_width=args.frame_width,
        baseline_path=baseline_path,
        personalized_calibration_seconds=args.baseline_seconds,
        event_log_path=args.event_log,
    )
    app = DrowsinessDetectorApp(config)
    app.start()


def _as_int_points(points: list[tuple[float, float]]):
    return np.array([[int(x), int(y)] for x, y in points], dtype=np.int32)


if __name__ == "__main__":
    main()
