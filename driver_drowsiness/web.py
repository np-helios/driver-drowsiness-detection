from __future__ import annotations

import timeit
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from .analytics import SessionAnalytics
from .baseline import BaselineCalibrator, BaselineProfile, BaselineStore, RollingBehavior
from .config import BASE_DIR, DetectorConfig
from .landmarks import FaceMeshLandmarks
from .metrics import eye_aspect_ratio, mouth_aspect_ratio
from .pose import estimate_head_pose
from .severity import SeverityResult, classify_deviation
from .yawn import is_confirmed_yawn, mar_activation_threshold


_LANDMARKS = FaceMeshLandmarks(model_path=DetectorConfig.face_landmarker_task_path)


@dataclass
class WebSessionState:
    driver_id: str
    baseline_seconds: float
    recalibrate: bool
    started_at: float = field(default_factory=timeit.default_timer)
    baseline_ready: bool = False
    baseline_profile: BaselineProfile | None = None
    calibrator: BaselineCalibrator = field(default_factory=BaselineCalibrator)
    behavior: RollingBehavior = field(default_factory=RollingBehavior)
    analytics: SessionAnalytics = field(default_factory=SessionAnalytics)
    current_ear: float = 0.0
    current_mar: float = 0.0
    current_head_yaw: float = 0.0
    current_head_roll: float = 0.0
    current_head_pitch: float = 0.0
    current_fatigue_score: float = 0.0
    ear_threshold: float = 0.0
    frame_counter: int = 0
    timer_flag: bool = False
    alarm_flag: bool = False
    start_closing: float = 0.0
    previous_open_term: float = 0.0
    last_alarm_at: float = 0.0
    start_yawning: float = 0.0
    yawn_active: bool = False
    yawn_peak_mar: float = 0.0
    break_recommended: bool = False
    alert_text: str = "No alert"
    alert_level: str = "Normal"


def create_session(driver_id: str, baseline_seconds: float, recalibrate: bool) -> tuple[WebSessionState, str]:
    driver_id = (driver_id or "web-demo").strip()
    store = BaselineStore(BASE_DIR / "profiles" / f"{driver_id}.json")
    profile = None if recalibrate else store.load()
    session = WebSessionState(
        driver_id=driver_id,
        baseline_seconds=baseline_seconds,
        recalibrate=recalibrate,
        baseline_ready=profile is not None,
        baseline_profile=profile,
    )
    if profile is not None:
        session.ear_threshold = _compute_ear_threshold(profile.open_ear_mean, profile.open_ear_std)
        status = f"Loaded saved baseline for `{driver_id}`. Live monitoring is ready."
    else:
        status = f"Started a new calibration session for `{driver_id}`."
    return session, _session_markdown(session, status)


def process_stream(frame_rgb: np.ndarray | None, session: WebSessionState | None):
    if frame_rgb is None:
        return None, _session_markdown(session) if session is not None else "Waiting for webcam frames...", _metrics_html(session), _alert_html(session), session
    if session is None:
        session, _ = create_session("web-demo", 20.0, True)

    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    landmarks = _LANDMARKS.detect(frame_bgr)
    now = timeit.default_timer()

    if landmarks is not None:
        left_ear = eye_aspect_ratio(landmarks.left_eye)
        right_ear = eye_aspect_ratio(landmarks.right_eye)
        session.current_ear = (left_ear + right_ear) * DetectorConfig.eye_scale_factor
        session.current_mar = mouth_aspect_ratio(landmarks.mouth)
        pose = estimate_head_pose(
            frame_bgr.shape,
            nose_tip=landmarks.nose_tip,
            chin=landmarks.chin,
            left_eye_corner=landmarks.left_eye_corner,
            right_eye_corner=landmarks.right_eye_corner,
            left_mouth_corner=landmarks.left_mouth_corner,
            right_mouth_corner=landmarks.right_mouth_corner,
        )
        session.current_head_yaw = pose.yaw
        session.current_head_roll = pose.roll
        session.current_head_pitch = pose.pitch
        session.behavior.add_open_ear(session.current_ear)
        session.behavior.add_mar(session.current_mar)
        session.behavior.add_head_pose(session.current_head_yaw, session.current_head_roll, session.current_head_pitch)

        if not session.baseline_ready:
            session.calibrator.add_open_ear(session.current_ear)
            session.calibrator.add_mar(session.current_mar)
            session.calibrator.add_head_pose(session.current_head_yaw, session.current_head_roll, session.current_head_pitch)
            elapsed = now - session.started_at
            if elapsed >= session.baseline_seconds:
                profile = session.calibrator.build(session.driver_id)
                session.baseline_profile = profile
                session.baseline_ready = True
                session.ear_threshold = _compute_ear_threshold(profile.open_ear_mean, profile.open_ear_std)
                BaselineStore(BASE_DIR / "profiles" / f"{session.driver_id}.json").save(profile)
        else:
            _update_yawn_state(session, now)
            _update_live_score(session)
            if session.current_ear < session.ear_threshold:
                _handle_closed_eyes(session, now)
            else:
                _handle_open_eyes(session, now)

        _draw_detection_boxes(frame_bgr, landmarks, session.yawn_active)

    _draw_mode_badge(
        frame_bgr,
        f"Calibrating {max(0.0, session.baseline_seconds - (now - session.started_at)):.0f}s" if not session.baseline_ready else "Monitoring",
        (0, 220, 255) if not session.baseline_ready else (90, 255, 170),
    )
    _draw_status_strip(frame_bgr, session)
    _draw_alert_banner(frame_bgr, session)
    annotated_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    status = _session_markdown(session)
    return annotated_rgb, status, _metrics_html(session), _alert_html(session), session


def build_demo() -> gr.Blocks:
    css = """
    .app-shell {max-width: 1320px; margin: 0 auto;}
    .metric-grid {display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:12px;}
    .metric-card {background:#101418; border:1px solid rgba(255,255,255,0.08); border-radius:14px; padding:14px;}
    .metric-label {font-size:12px; text-transform:uppercase; letter-spacing:0.08em; color:#9db0bd;}
    .metric-value {font-size:28px; font-weight:700; color:#f4f7fb; margin-top:6px;}
    .metric-sub {font-size:13px; color:#c6d1da; margin-top:4px;}
    .alert-panel {border-radius:14px; padding:14px 16px; margin-top:12px; border:1px solid rgba(255,255,255,0.08);}
    .alert-ok {background:#0f1c14; color:#9df0b8;}
    .alert-warn {background:#2a170c; color:#ffd08a;}
    .alert-danger {background:#2c1010; color:#ff9b9b;}
    .session-note {background:#11161b; border-radius:14px; padding:14px 16px; color:#d4dde4; border:1px solid rgba(255,255,255,0.08);}
    """
    with gr.Blocks(title="Driver Drowsiness Detector", css=css) as demo:
        gr.Markdown(
            """
            # Driver Drowsiness Detector
            Browser-deployed personalized monitoring with live face, eye, and mouth detection overlays.
            """
        )
        with gr.Row(elem_classes=["app-shell"]):
            with gr.Column(scale=3):
                processed = gr.Image(label="Live Detector Output", type="numpy", streaming=True, height=560)
                with gr.Accordion("Raw webcam input", open=False):
                    webcam = gr.Image(
                        label="Webcam Input",
                        sources="webcam",
                        type="numpy",
                        streaming=True,
                        webcam_options=gr.WebcamOptions(
                            mirror=True,
                            constraints={"width": {"ideal": 960}, "height": {"ideal": 540}, "facingMode": "user"},
                        ),
                    )
            with gr.Column(scale=2):
                driver_id = gr.Textbox(label="Driver ID", value="exam_demo")
                baseline_seconds = gr.Slider(label="Baseline seconds", minimum=10, maximum=60, value=20, step=5)
                recalibrate = gr.Checkbox(label="Force recalibration", value=True)
                start_btn = gr.Button("Start / Reset Session", variant="primary")
                status = gr.Markdown("Click **Start / Reset Session** and allow webcam access.", elem_classes=["session-note"])
                metrics = gr.HTML(_metrics_html(None))
                alert_box = gr.HTML(_alert_html(None))
        session_state = gr.State(None)

        start_btn.click(
            _create_session_ui,
            [driver_id, baseline_seconds, recalibrate],
            [session_state, status, metrics, alert_box],
        )
        webcam.stream(
            process_stream,
            [webcam, session_state],
            [processed, status, metrics, alert_box, session_state],
            time_limit=600,
            stream_every=0.1,
            concurrency_limit=1,
        )

    return demo


def _update_live_score(session: WebSessionState) -> None:
    if not session.baseline_ready or session.baseline_profile is None:
        session.current_fatigue_score = 0.0
        session.break_recommended = False
        return
    severity = classify_deviation(
        baseline=session.baseline_profile,
        behavior=session.behavior,
        current_ear=session.current_ear,
        current_mar=session.current_mar,
        current_head_yaw=session.current_head_yaw,
        current_head_roll=session.current_head_roll,
        current_head_pitch=session.current_head_pitch,
    )
    session.current_fatigue_score = severity.score
    session.break_recommended = severity.break_recommended


def _handle_closed_eyes(session: WebSessionState, now: float) -> None:
    if not session.timer_flag:
        session.start_closing = now
        session.timer_flag = True
    session.frame_counter += 1
    if session.frame_counter < DetectorConfig.ear_consecutive_frames or session.alarm_flag:
        return
    closing_time = now - session.start_closing
    if closing_time < DetectorConfig.min_closed_duration_seconds:
        return
    if session.last_alarm_at and (now - session.last_alarm_at) < DetectorConfig.alarm_cooldown_seconds:
        return
    if session.previous_open_term == 0.0:
        session.previous_open_term = session.started_at
    open_interval = now - session.previous_open_term
    session.previous_open_term = now
    session.last_alarm_at = now
    session.alarm_flag = True
    session.analytics.alerts_triggered += 1
    session.behavior.add_blink_metrics(closing_time, open_interval)
    if session.baseline_profile is not None:
        session.calibrator.add_blink_metrics(closing_time, open_interval)
        severity = classify_deviation(
            baseline=session.baseline_profile,
            behavior=session.behavior,
            current_ear=session.current_ear,
            current_mar=session.current_mar,
            current_head_yaw=session.current_head_yaw,
            current_head_roll=session.current_head_roll,
            current_head_pitch=session.current_head_pitch,
        )
    else:
        severity = SeverityResult(level=1, alarm_name="normal", rationale="baseline_not_ready", score=0.0)
    session.current_fatigue_score = severity.score
    session.break_recommended = severity.break_recommended
    session.alert_level = severity.alarm_name.title()
    session.alert_text = "Drowsiness alert"


def _handle_open_eyes(session: WebSessionState, now: float) -> None:
    session.frame_counter = 0
    session.timer_flag = False
    if session.alarm_flag:
        session.previous_open_term = now
    session.alarm_flag = False
    if not session.break_recommended:
        session.alert_text = "No alert"
        session.alert_level = "Normal"


def _update_yawn_state(session: WebSessionState, now: float) -> None:
    if _is_yawn_candidate(session):
        if not session.yawn_active:
            session.start_yawning = now
            session.yawn_active = True
            session.yawn_peak_mar = session.current_mar
        else:
            session.yawn_peak_mar = max(session.yawn_peak_mar, session.current_mar)
        return
    if not session.yawn_active:
        return
    duration = now - session.start_yawning
    session.yawn_active = False
    if not is_confirmed_yawn(
        duration_seconds=duration,
        peak_mar=session.yawn_peak_mar,
        baseline_profile=session.baseline_profile,
        min_duration_seconds=DetectorConfig.min_yawn_duration_seconds,
    ):
        session.yawn_peak_mar = 0.0
        return
    session.analytics.yawn_events += 1
    session.behavior.add_yawn_duration(duration)
    session.calibrator.add_yawn_duration(duration)
    session.yawn_peak_mar = 0.0


def _is_yawn_candidate(session: WebSessionState) -> bool:
    if session.baseline_profile is None:
        return session.current_mar > 0.03
    return session.current_mar > mar_activation_threshold(
        session.baseline_profile.mar_mean,
        session.baseline_profile.mar_std,
    )


def _draw_detection_boxes(frame, landmarks, yawn_active: bool) -> None:
    feature_points = (
        landmarks.left_eye
        + landmarks.right_eye
        + landmarks.mouth
        + [
            landmarks.nose_tip,
            landmarks.chin,
            landmarks.left_eye_corner,
            landmarks.right_eye_corner,
            landmarks.left_mouth_corner,
            landmarks.right_mouth_corner,
        ]
    )
    face_box = _expanded_box(feature_points, frame.shape, pad_x=28, pad_y=36)
    left_eye_box = _expanded_box(landmarks.left_eye, frame.shape, pad_x=5, pad_y=4)
    right_eye_box = _expanded_box(landmarks.right_eye, frame.shape, pad_x=5, pad_y=4)
    mouth_box = _expanded_box(landmarks.mouth, frame.shape, pad_x=6, pad_y=6)
    _draw_box(frame, face_box, (0, 232, 255))
    _draw_box(frame, left_eye_box, (188, 80, 255))
    _draw_box(frame, right_eye_box, (188, 80, 255))
    _draw_box(frame, mouth_box, (64, 170, 255) if yawn_active else (90, 110, 255))


def _draw_status_strip(frame, session: WebSessionState) -> None:
    frame_h, frame_w = frame.shape[:2]
    strip_height = 28
    left = 10
    right = frame_w - 10
    top = frame_h - strip_height - 10
    bottom = frame_h - 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (left, top), (right, bottom), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.42, frame, 0.58, 0, frame)
    font = cv2.FONT_HERSHEY_DUPLEX
    line = cv2.LINE_AA
    score_color = (0, 220, 0) if session.current_fatigue_score < 1.8 else (0, 190, 255) if session.current_fatigue_score < 3.2 else (0, 70, 255)
    segments = [
        (f"EAR {session.current_ear:.1f}", (245, 245, 245)),
        (f"MAR {session.current_mar:.3f}", (245, 245, 245)),
        (f"Fatigue {session.current_fatigue_score:.2f}", score_color),
        (f"Alerts {session.analytics.alerts_triggered}", (245, 245, 245)),
        (f"Yawns {session.analytics.yawn_events}", (245, 245, 245)),
        (session.alert_text, (0, 70, 255) if session.alert_text != "No alert" else (190, 255, 190)),
    ]
    x = left + 10
    y = top + 18
    for text, color in segments:
        cv2.putText(frame, text, (x, y), font, 0.31, color, 1, line)
        x += cv2.getTextSize(text, font, 0.31, 1)[0][0] + 14


def _draw_alert_banner(frame, session: WebSessionState) -> None:
    if session.alert_text == "No alert" and not session.break_recommended:
        return
    color = (0, 190, 255) if session.alert_level == "Normal" else (0, 140, 255) if session.alert_level == "Short" else (0, 70, 255)
    font = cv2.FONT_HERSHEY_DUPLEX
    line = cv2.LINE_AA
    label = "Break recommended" if session.break_recommended else session.alert_text
    text_size, _ = cv2.getTextSize(label, font, 0.48, 1)
    x = 14
    y = 52
    pad_x = 10
    pad_y = 7
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x, y),
        (x + text_size[0] + pad_x * 2, y + text_size[1] + pad_y * 2),
        (25, 25, 25),
        -1,
    )
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    cv2.rectangle(
        frame,
        (x, y),
        (x + text_size[0] + pad_x * 2, y + text_size[1] + pad_y * 2),
        color,
        1,
        line,
    )
    cv2.putText(frame, label, (x + pad_x, y + text_size[1] + 1), font, 0.48, color, 1, line)


def _draw_mode_badge(frame, label: str, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_DUPLEX
    line = cv2.LINE_AA
    text_size, _ = cv2.getTextSize(label, font, 0.38, 1)
    x = 12
    y = 14
    pad_x = 8
    pad_y = 6
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + text_size[0] + pad_x * 2, y + text_size[1] + pad_y * 2), (28, 28, 28), -1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
    cv2.rectangle(frame, (x, y), (x + text_size[0] + pad_x * 2, y + text_size[1] + pad_y * 2), color, 1, line)
    cv2.putText(frame, label, (x + pad_x, y + text_size[1] + 1), font, 0.38, color, 1, line)


def _expanded_box(points: list[tuple[float, float]], frame_shape, *, pad_x: int, pad_y: int) -> tuple[int, int, int, int]:
    frame_h, frame_w = frame_shape[:2]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    left = max(0, int(round(min(xs) - pad_x)))
    top = max(0, int(round(min(ys) - pad_y)))
    right = min(frame_w - 1, int(round(max(xs) + pad_x)))
    bottom = min(frame_h - 1, int(round(max(ys) + pad_y)))
    return left, top, right, bottom


def _draw_box(frame, box: tuple[int, int, int, int], color: tuple[int, int, int], thickness: int = 1) -> None:
    left, top, right, bottom = box
    cv2.rectangle(frame, (left, top), (right, bottom), color, thickness, cv2.LINE_AA)


def _compute_ear_threshold(open_ear_mean: float, open_ear_std: float) -> float:
    return max(open_ear_mean - max(open_ear_std * 2.0, open_ear_mean * 0.28), 0.0)


def _session_markdown(session: WebSessionState, prefix: str | None = None) -> str:
    if session is None:
        return "Click **Start / Reset Session** and allow webcam access."
    mode = "Monitoring" if session.baseline_ready else "Calibrating"
    lines = []
    if prefix:
        lines.append(prefix)
    lines.extend(
        [
            f"**Mode:** {mode}",
            f"**Driver ID:** `{session.driver_id}`",
            f"**EAR:** `{session.current_ear:.1f}`",
            f"**MAR:** `{session.current_mar:.3f}`",
            f"**Fatigue Score:** `{session.current_fatigue_score:.2f}`",
            f"**Alerts:** `{session.analytics.alerts_triggered}`",
            f"**Yawns:** `{session.analytics.yawn_events}`",
            f"**Alert State:** {session.alert_text}",
        ]
    )
    return "\n\n".join(lines)


def _metrics_html(session: WebSessionState | None) -> str:
    if session is None:
        return """
        <div class="metric-grid">
          <div class="metric-card"><div class="metric-label">EAR</div><div class="metric-value">--</div><div class="metric-sub">Eye Aspect Ratio</div></div>
          <div class="metric-card"><div class="metric-label">MAR</div><div class="metric-value">--</div><div class="metric-sub">Mouth Aspect Ratio</div></div>
          <div class="metric-card"><div class="metric-label">Fatigue</div><div class="metric-value">--</div><div class="metric-sub">Live deviation score</div></div>
          <div class="metric-card"><div class="metric-label">Events</div><div class="metric-value">--</div><div class="metric-sub">Alerts and yawns</div></div>
        </div>
        """
    return f"""
    <div class="metric-grid">
      <div class="metric-card">
        <div class="metric-label">EAR</div>
        <div class="metric-value">{session.current_ear:.1f}</div>
        <div class="metric-sub">Eye closure signal</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">MAR</div>
        <div class="metric-value">{session.current_mar:.3f}</div>
        <div class="metric-sub">Mouth opening signal</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Fatigue</div>
        <div class="metric-value">{session.current_fatigue_score:.2f}</div>
        <div class="metric-sub">Personalized deviation score</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Events</div>
        <div class="metric-value">{session.analytics.alerts_triggered} / {session.analytics.yawn_events}</div>
        <div class="metric-sub">Alerts / Yawns</div>
      </div>
    </div>
    """


def _alert_html(session: WebSessionState | None) -> str:
    if session is None or (session.alert_text == "No alert" and not session.break_recommended):
        return '<div class="alert-panel alert-ok"><strong>Status:</strong> Monitoring normally. No active drowsiness alert.</div>'
    if session.break_recommended:
        return '<div class="alert-panel alert-danger"><strong>Action needed:</strong> Sustained fatigue detected. Please take a short break.</div>'
    if session.alert_level in {"Power", "Normal"}:
        return '<div class="alert-panel alert-danger"><strong>Drowsiness alert:</strong> Sustained eye closure detected.</div>'
    return '<div class="alert-panel alert-warn"><strong>Warning:</strong> Mild fatigue deviation detected.</div>'


def _create_session_ui(driver_id: str, baseline_seconds: float, recalibrate: bool):
    session, status = create_session(driver_id, baseline_seconds, recalibrate)
    return session, status, _metrics_html(session), _alert_html(session)
