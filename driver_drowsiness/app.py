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

from .analytics import SessionAnalytics, save_session_summary
from .audio import sound_alarm, speak_alert
from .baseline import BaselineCalibrator, BaselineProfile, BaselineStore, RollingBehavior
from .config import DetectorConfig
from .landmarks import FaceMeshLandmarks
from .logging_utils import EventLogger
from .metrics import eye_aspect_ratio, mouth_aspect_ratio
from .pose import estimate_head_pose
from .preprocess import light_removing
from .severity import SeverityResult, classify_deviation
from .yawn import is_confirmed_yawn, mar_activation_threshold


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
    current_mar: float = 0.0
    current_head_yaw: float = 0.0
    current_head_roll: float = 0.0
    current_head_pitch: float = 0.0
    current_fatigue_score: float = 0.0
    start_closing: float = 0.0
    start_yawning: float = 0.0
    closed_eyes_time: list[float] = field(default_factory=list)
    yawn_durations: list[float] = field(default_factory=list)
    test_data: list[list[float]] = field(default_factory=list)
    result_data: list[int] = field(default_factory=list)
    latest_face_detected: bool = False
    baseline_ready: bool = False
    baseline_started_at: float = 0.0
    yawn_active: bool = False
    break_recommended: bool = False
    last_alarm_at: float = 0.0
    yawn_peak_mar: float = 0.0


class DrowsinessDetectorApp:
    def __init__(self, config: DetectorConfig):
        self.config = config
        self.state = RuntimeState()
        self.landmarks = FaceMeshLandmarks(model_path=config.face_landmarker_task_path)
        self.logger = EventLogger(config.event_log_path)
        self.baseline_store = BaselineStore(config.baseline_path)
        self.baseline_calibrator = BaselineCalibrator()
        self.behavior = RollingBehavior()
        self.analytics = SessionAnalytics()
        self.baseline_profile: BaselineProfile | None = None if config.force_recalibration else self.baseline_store.load()
        self.video_stream = VideoStream(src=config.camera_source).start()
        self.state.baseline_ready = self.baseline_profile is not None
        if self.baseline_profile is not None:
            self.state.open_ear = self.baseline_profile.open_ear_mean
            self.state.ear_threshold = _compute_ear_threshold(
                self.baseline_profile.open_ear_mean,
                self.baseline_profile.open_ear_std,
            )

    def start(self) -> None:
        print("loading MediaPipe face mesh...")
        print("starting video stream thread...")
        time.sleep(self.config.warmup_seconds)
        self.state.baseline_started_at = timeit.default_timer()
        if self.config.force_recalibration:
            print(
                "Forced recalibration enabled. Building a fresh personalized baseline "
                f"for {self.config.personalized_calibration_seconds:.0f} seconds."
            )
        elif self.baseline_profile is None:
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
            self._persist_session_summary()
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
                self.state.current_mar = mouth_aspect_ratio(eye_landmarks.mouth)
                head_pose = estimate_head_pose(
                    frame.shape,
                    nose_tip=eye_landmarks.nose_tip,
                    chin=eye_landmarks.chin,
                    left_eye_corner=eye_landmarks.left_eye_corner,
                    right_eye_corner=eye_landmarks.right_eye_corner,
                    left_mouth_corner=eye_landmarks.left_mouth_corner,
                    right_mouth_corner=eye_landmarks.right_mouth_corner,
                )
                self.state.current_head_yaw = head_pose.yaw
                self.state.current_head_roll = head_pose.roll
                self.state.current_head_pitch = head_pose.pitch
                self.state.current_ear = (left_ear + right_ear) * self.config.eye_scale_factor
                self.behavior.add_open_ear(self.state.current_ear)
                self.behavior.add_mar(self.state.current_mar)
                self.behavior.add_head_pose(
                    self.state.current_head_yaw,
                    self.state.current_head_roll,
                    self.state.current_head_pitch,
                )
                self._update_personal_baseline()
                self._update_yawn_state()
                self._update_live_fatigue_score()

                self._draw_detection_boxes(frame, eye_landmarks)

                if self._is_eye_closed():
                    self._handle_closed_eyes()
                else:
                    self._handle_open_eyes()
            else:
                self._handle_open_eyes()

            self.analytics.add_frame(
                ear=self.state.current_ear,
                mar=self.state.current_mar,
                head_yaw=self.state.current_head_yaw,
                head_pitch=self.state.current_head_pitch,
                face_detected=self.state.latest_face_detected,
            )

            self._draw_status_strip(frame)
            if not self.state.baseline_ready:
                remaining = max(
                    0.0,
                    self.config.personalized_calibration_seconds
                    - (timeit.default_timer() - self.state.baseline_started_at),
                )
                self._draw_mode_badge(frame, f"Calibrating {remaining:.0f}s", (0, 220, 255))
            else:
                self._draw_mode_badge(frame, "Monitoring", (90, 255, 170))

            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    def _update_personal_baseline(self) -> None:
        if self.state.baseline_ready:
            return
        self.baseline_calibrator.add_open_ear(self.state.current_ear)
        self.baseline_calibrator.add_mar(self.state.current_mar)
        self.baseline_calibrator.add_head_pose(
            self.state.current_head_yaw,
            self.state.current_head_roll,
            self.state.current_head_pitch,
        )
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
        self.state.ear_threshold = _compute_ear_threshold(profile.open_ear_mean, profile.open_ear_std)
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

    def _is_yawn_candidate(self) -> bool:
        if self.baseline_profile is None:
            return self.state.current_mar > 0.03
        return self.state.current_mar > mar_activation_threshold(
            self.baseline_profile.mar_mean,
            self.baseline_profile.mar_std,
        )

    def _update_yawn_state(self) -> None:
        if self._is_yawn_candidate():
            if not self.state.yawn_active:
                self.state.start_yawning = timeit.default_timer()
                self.state.yawn_active = True
                self.state.yawn_peak_mar = self.state.current_mar
            else:
                self.state.yawn_peak_mar = max(self.state.yawn_peak_mar, self.state.current_mar)
            return

        if not self.state.yawn_active:
            return
        yawn_duration = round(timeit.default_timer() - self.state.start_yawning, 3)
        self.state.yawn_active = False
        if not is_confirmed_yawn(
            duration_seconds=yawn_duration,
            peak_mar=self.state.yawn_peak_mar,
            baseline_profile=self.baseline_profile,
            min_duration_seconds=self.config.min_yawn_duration_seconds,
        ):
            self.state.yawn_peak_mar = 0.0
            return
        self.state.yawn_durations.append(yawn_duration)
        self.behavior.add_yawn_duration(yawn_duration)
        self.baseline_calibrator.add_yawn_duration(yawn_duration)
        self.analytics.yawn_events += 1
        self.logger.log(
            "yawn_detected",
            duration=yawn_duration,
            current_mar=round(self.state.current_mar, 3),
            peak_mar=round(self.state.yawn_peak_mar, 3),
        )
        self.state.yawn_peak_mar = 0.0

    def _handle_closed_eyes(self) -> None:
        if not self.state.timer_flag:
            self.state.start_closing = timeit.default_timer()
            self.state.timer_flag = True

        self.state.frame_counter += 1
        if self.state.frame_counter < self.config.ear_consecutive_frames:
            return

        if self.state.alarm_flag:
            return

        mid_closing = timeit.default_timer()
        closing_time = round(mid_closing - self.state.start_closing, 3)
        if closing_time < self.config.min_closed_duration_seconds:
            return

        if self.state.last_alarm_at and (mid_closing - self.state.last_alarm_at) < self.config.alarm_cooldown_seconds:
            return

        if self.state.previous_open_term == 0.0:
            self.state.previous_open_term = self.state.baseline_started_at
        opened_eyes_time = round(mid_closing - self.state.previous_open_term, 3)
        self.state.previous_open_term = mid_closing
        self.state.last_alarm_at = mid_closing
        self.state.alarm_flag = True
        self.state.alarm_count += 1

        print(f"{self.state.alarm_count}st ALARM")
        print(f"The time eyes are open before the alarm: {opened_eyes_time}")
        print(f"closing time: {closing_time}")

        self.state.test_data.append([opened_eyes_time, round(closing_time * 10, 3)])
        self.behavior.add_blink_metrics(closing_time, opened_eyes_time)
        self.analytics.blink_events += 1
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
                current_mar=self.state.current_mar,
                current_head_yaw=self.state.current_head_yaw,
                current_head_roll=self.state.current_head_roll,
                current_head_pitch=self.state.current_head_pitch,
            )
        self.state.result_data.append(severity.level)
        self.state.current_fatigue_score = severity.score
        self.state.break_recommended = severity.break_recommended
        self.analytics.alerts_triggered += 1
        self.analytics.record_score(severity.score)
        if severity.break_recommended:
            self.analytics.break_recommendations += 1
        self.logger.log(
            "alarm_triggered",
            alarm_count=self.state.alarm_count,
            opened_eyes_time=opened_eyes_time,
            closing_time=closing_time,
            severity=severity,
            ear_threshold=round(self.state.ear_threshold, 3),
            current_ear=round(self.state.current_ear, 3),
            current_mar=round(self.state.current_mar, 3),
            current_head_yaw=round(self.state.current_head_yaw, 3),
            current_head_roll=round(self.state.current_head_roll, 3),
            current_head_pitch=round(self.state.current_head_pitch, 3),
            break_recommended=severity.break_recommended,
        )
        self._play_async_alarm(self._alarm_path_for_result(severity))
        if severity.break_recommended:
            Thread(target=speak_alert, args=("Fatigue detected. Please take a short break.",), daemon=True).start()

    def _handle_open_eyes(self) -> None:
        self.state.frame_counter = 0
        self.state.timer_flag = False
        self.state.running_time = 0

        if self.state.alarm_flag:
            end_closing = timeit.default_timer()
            closed_time = round(end_closing - self.state.start_closing, 3)
            self.state.closed_eyes_time.append(closed_time)
            print(f"The time eyes were closed: {self.state.closed_eyes_time}")
            self.state.previous_open_term = end_closing

        self.state.alarm_flag = False

    def _update_live_fatigue_score(self) -> None:
        if not self.state.baseline_ready or self.baseline_profile is None:
            self.state.current_fatigue_score = 0.0
            self.state.break_recommended = False
            return
        severity = classify_deviation(
            baseline=self.baseline_profile,
            behavior=self.behavior,
            current_ear=self.state.current_ear,
            current_mar=self.state.current_mar,
            current_head_yaw=self.state.current_head_yaw,
            current_head_roll=self.state.current_head_roll,
            current_head_pitch=self.state.current_head_pitch,
        )
        self.state.current_fatigue_score = severity.score
        self.state.break_recommended = severity.break_recommended
        self.analytics.record_score(severity.score)

    def _draw_detection_boxes(self, frame, eye_landmarks) -> None:
        feature_points = (
            eye_landmarks.left_eye
            + eye_landmarks.right_eye
            + eye_landmarks.mouth
            + [
                eye_landmarks.nose_tip,
                eye_landmarks.chin,
                eye_landmarks.left_eye_corner,
                eye_landmarks.right_eye_corner,
                eye_landmarks.left_mouth_corner,
                eye_landmarks.right_mouth_corner,
            ]
        )
        face_box = _expanded_box(feature_points, frame.shape, pad_x=28, pad_y=36)
        left_eye_box = _expanded_box(eye_landmarks.left_eye, frame.shape, pad_x=5, pad_y=4)
        right_eye_box = _expanded_box(eye_landmarks.right_eye, frame.shape, pad_x=5, pad_y=4)
        mouth_box = _expanded_box(eye_landmarks.mouth, frame.shape, pad_x=6, pad_y=6)

        _draw_box(frame, face_box, (0, 232, 255), 1)
        _draw_box(frame, left_eye_box, (188, 80, 255), 1)
        _draw_box(frame, right_eye_box, (188, 80, 255), 1)
        mouth_color = (64, 170, 255) if self.state.yawn_active else (90, 110, 255)
        _draw_box(frame, mouth_box, mouth_color, 1)

    def _draw_status_strip(self, frame) -> None:
        frame_h, frame_w = frame.shape[:2]
        strip_height = 28
        strip_left = 10
        strip_right = frame_w - 10
        strip_top = frame_h - strip_height - 10
        strip_bottom = frame_h - 10

        overlay = frame.copy()
        cv2.rectangle(overlay, (strip_left, strip_top), (strip_right, strip_bottom), (18, 18, 18), -1)
        cv2.addWeighted(overlay, 0.42, frame, 0.58, 0, frame)

        font = cv2.FONT_HERSHEY_DUPLEX
        line = cv2.LINE_AA
        x = strip_left + 10
        y = strip_top + 18
        score_color = (0, 220, 0) if self.state.current_fatigue_score < 1.8 else (0, 190, 255) if self.state.current_fatigue_score < 3.2 else (0, 70, 255)
        segments = [
            (f"EAR {self.state.current_ear:.1f}", (245, 245, 245)),
            (f"MAR {self.state.current_mar:.3f}", (245, 245, 245)),
            (f"Fatigue {self.state.current_fatigue_score:.2f}", score_color),
            (f"Alerts {self.analytics.alerts_triggered}", (245, 245, 245)),
            (f"Yawns {self.analytics.yawn_events}", (245, 245, 245)),
        ]
        for text, color in segments:
            cv2.putText(frame, text, (x, y), font, 0.32, color, 1, line)
            text_width = cv2.getTextSize(text, font, 0.32, 1)[0][0]
            x += text_width + 16

        if self.state.break_recommended:
            warning = "Break recommended"
            warning_width = cv2.getTextSize(warning, font, 0.34, 1)[0][0]
            cv2.putText(
                frame,
                warning,
                (strip_right - warning_width - 10, y),
                font,
                0.34,
                (0, 70, 255),
                1,
                line,
            )

    def _draw_mode_badge(self, frame, label: str, color: tuple[int, int, int]) -> None:
        font = cv2.FONT_HERSHEY_DUPLEX
        line = cv2.LINE_AA
        text_size, _ = cv2.getTextSize(label, font, 0.38, 1)
        x = 12
        y = 14
        pad_x = 8
        pad_y = 6
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x, y),
            (x + text_size[0] + pad_x * 2, y + text_size[1] + pad_y * 2),
            (28, 28, 28),
            -1,
        )
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
        cv2.rectangle(
            frame,
            (x, y),
            (x + text_size[0] + pad_x * 2, y + text_size[1] + pad_y * 2),
            color,
            1,
            line,
        )
        cv2.putText(frame, label, (x + pad_x, y + text_size[1] + 1), font, 0.38, color, 1, line)

    def _persist_session_summary(self) -> None:
        summary = self.analytics.summary()
        self.logger.log("session_summary", **summary)
        save_session_summary(self.config.session_summary_path, summary)

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
    parser.add_argument(
        "--recalibrate",
        action="store_true",
        help="Ignore any saved profile for this driver and build a fresh baseline.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    baseline_path = args.baseline_path or (DetectorConfig.baseline_path.parent / f"{args.driver_id}.json")
    config = DetectorConfig(
        driver_id=args.driver_id,
        force_recalibration=args.recalibrate,
        face_landmarker_task_path=DetectorConfig.face_landmarker_task_path,
        camera_source=args.camera_source,
        frame_width=args.frame_width,
        baseline_path=baseline_path,
        personalized_calibration_seconds=args.baseline_seconds,
        event_log_path=args.event_log,
        session_summary_path=DetectorConfig.session_summary_path,
    )
    app = DrowsinessDetectorApp(config)
    app.start()


def _as_int_points(points: list[tuple[float, float]]):
    return np.array([[int(round(x)), int(round(y))] for x, y in points], dtype=np.int32)


def _compute_ear_threshold(open_ear_mean: float, open_ear_std: float) -> float:
    return max(open_ear_mean - max(open_ear_std * 2.0, open_ear_mean * 0.28), 0.0)


def _expanded_box(
    points: list[tuple[float, float]],
    frame_shape,
    *,
    pad_x: int,
    pad_y: int,
) -> tuple[int, int, int, int]:
    frame_h, frame_w = frame_shape[:2]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    left = max(0, int(round(min(xs) - pad_x)))
    top = max(0, int(round(min(ys) - pad_y)))
    right = min(frame_w - 1, int(round(max(xs) + pad_x)))
    bottom = min(frame_h - 1, int(round(max(ys) + pad_y)))
    return left, top, right, bottom


def _draw_box(
    frame,
    box: tuple[int, int, int, int],
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    left, top, right, bottom = box
    cv2.rectangle(frame, (left, top), (right, bottom), color, thickness, cv2.LINE_AA)


if __name__ == "__main__":
    main()
