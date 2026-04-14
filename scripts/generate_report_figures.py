from __future__ import annotations

from math import cos, radians, sin
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "report_figures"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figure_1_system_architecture()
    figure_2_landmark_regions()
    figure_3_ear_diagram()
    figure_4_mar_diagram()
    figure_5_baseline_workflow()
    figure_6_deviation_scoring()
    figure_7_deployment_comparison()
    figure_8_alert_flow()
    figure_9_ui_mockup()
    figure_10_contribution_summary()
    print(f"Saved figures to {OUT_DIR}")


def setup_canvas(figsize=(14, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT_DIR / name, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def box(ax, x, y, w, h, text, fc="#eef4ff", ec="#2b5fab", fontsize=12, lw=1.8, roundness=0.08, color="#132238"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.02,rounding_size={roundness * min(w, h)}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=color, weight="bold", wrap=True)
    return patch


def arrow(ax, start, end, color="#334155", lw=1.8, style="-|>"):
    arr = FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=14, linewidth=lw, color=color)
    ax.add_patch(arr)


def title(ax, text):
    ax.text(50, 96, text, ha="center", va="center", fontsize=20, weight="bold", color="#0f172a")


def subtitle(ax, text):
    ax.text(50, 91.5, text, ha="center", va="center", fontsize=10.5, color="#475569")


def figure_1_system_architecture():
    fig, ax = setup_canvas((15, 8))
    title(ax, "Figure 1. Overall System Architecture")
    subtitle(ax, "End-to-end pipeline for adaptive driver drowsiness detection")
    xs = [3, 19, 35, 53, 71, 87]
    labels = [
        "Webcam\nInput",
        "Face Landmark\nDetection\n(MediaPipe)",
        "Feature Extraction\nEAR, MAR,\n3D Head Pose",
        "Personalized\nBaseline\nModeling",
        "Deviation-Based\nFatigue\nScoring",
        "Alerts,\nLogging &\nSession Summary",
    ]
    colors = ["#edf7ff", "#eef8f0", "#fff7e8", "#eef4ff", "#fff3f0", "#f3f0ff"]
    edges = ["#268bd2", "#2f855a", "#d69e2e", "#4c51bf", "#dd6b20", "#805ad5"]
    for i, x in enumerate(xs):
        box(ax, x, 52, 12, 22, labels[i], fc=colors[i], ec=edges[i], fontsize=11)
        if i < len(xs) - 1:
            arrow(ax, (x + 12, 63), (xs[i + 1], 63))
    box(ax, 28, 14, 18, 16, "Driver Profile\n(JSON baseline)", fc="#f0fdf4", ec="#16a34a", fontsize=11)
    box(ax, 54, 14, 18, 16, "Runtime Event Log\n(JSONL)", fc="#fefce8", ec="#ca8a04", fontsize=11)
    arrow(ax, (59, 52), (37, 30), color="#4c51bf")
    arrow(ax, (93, 52), (63, 30), color="#7c3aed")
    save(fig, "figure_01_system_architecture.png")


def figure_2_landmark_regions():
    fig, ax = setup_canvas((10, 12))
    title(ax, "Figure 2. Facial Landmark Regions Used")
    subtitle(ax, "Eye, mouth, and pose anchor regions employed by the detector")
    face = Circle((50, 52), 28, facecolor="#f7dcc8", edgecolor="#caa58a", linewidth=2)
    ax.add_patch(face)
    hair = Polygon([(24, 66), (34, 84), (50, 88), (66, 84), (76, 66), (75, 62), (25, 62)], closed=True, facecolor="#1f2937")
    ax.add_patch(hair)
    left_eye = Rectangle((37, 56), 8, 3.5, linewidth=2, edgecolor="#a855f7", facecolor="none")
    right_eye = Rectangle((55, 56), 8, 3.5, linewidth=2, edgecolor="#a855f7", facecolor="none")
    mouth = Rectangle((43, 38), 14, 6, linewidth=2, edgecolor="#fb7185", facecolor="none")
    face_box = Rectangle((28, 28), 44, 44, linewidth=2, edgecolor="#eab308", facecolor="none")
    ax.add_patch(face_box)
    ax.add_patch(left_eye)
    ax.add_patch(right_eye)
    ax.add_patch(mouth)
    points = {
        "Nose tip": (50, 48),
        "Chin": (50, 24),
        "Left eye corner": (36, 58),
        "Right eye corner": (64, 58),
        "Left mouth corner": (43, 41),
        "Right mouth corner": (57, 41),
    }
    for label, (x, y) in points.items():
        ax.add_patch(Circle((x, y), 0.8, color="#0f172a"))
    callouts = [
        ("Left eye region", (20, 71), (37, 58), "#a855f7"),
        ("Right eye region", (78, 71), (63, 58), "#a855f7"),
        ("Mouth region", (78, 34), (57, 41), "#fb7185"),
        ("Face box", (18, 22), (28, 28), "#eab308"),
        ("Pose anchors", (18, 46), (50, 48), "#0f172a"),
    ]
    for text, src, dst, color in callouts:
        ax.text(src[0], src[1], text, fontsize=10.5, color=color, weight="bold")
        arrow(ax, src, dst, color=color, lw=1.4)
    save(fig, "figure_02_landmark_regions.png")


def figure_3_ear_diagram():
    fig, ax = setup_canvas((11, 7))
    title(ax, "Figure 3. Eye Aspect Ratio (EAR) Computation")
    subtitle(ax, "Six eye landmarks define horizontal and vertical distances")
    pts = [(18, 40), (30, 58), (50, 60), (82, 40), (50, 20), (30, 22)]
    labels = ["p1", "p2", "p3", "p4", "p5", "p6"]
    ax.plot([p[0] for p in pts + [pts[0]]], [p[1] for p in pts + [pts[0]]], color="#2563eb", linewidth=2)
    for (x, y), label in zip(pts, labels):
        ax.add_patch(Circle((x, y), 1.2, facecolor="#ffffff", edgecolor="#0f172a", linewidth=1.2))
        ax.text(x + 1.5, y + 2, label, fontsize=10, weight="bold")
    ax.plot([30, 30], [22, 58], linestyle="--", color="#f97316", linewidth=1.6)
    ax.plot([50, 50], [20, 60], linestyle="--", color="#f97316", linewidth=1.6)
    ax.plot([18, 82], [40, 40], linestyle="--", color="#16a34a", linewidth=1.8)
    ax.text(33, 42, "A", color="#f97316", fontsize=12, weight="bold")
    ax.text(53, 42, "B", color="#f97316", fontsize=12, weight="bold")
    ax.text(49, 33, "C", color="#16a34a", fontsize=12, weight="bold")
    ax.text(56, 76, "EAR = (A + B) / (2C)", fontsize=16, weight="bold", color="#0f172a")
    ax.text(56, 67, "Lower EAR indicates stronger eye closure", fontsize=10.5, color="#475569")
    save(fig, "figure_03_ear_computation.png")


def figure_4_mar_diagram():
    fig, ax = setup_canvas((11, 7))
    title(ax, "Figure 4. Mouth Aspect Ratio (MAR) Computation")
    subtitle(ax, "Mouth opening intensity is estimated from vertical-to-horizontal geometry")
    pts = [(18, 35), (30, 52), (44, 58), (58, 52), (82, 35), (58, 18), (44, 12), (30, 18)]
    ax.plot([p[0] for p in pts + [pts[0]]], [p[1] for p in pts + [pts[0]]], color="#ef4444", linewidth=2)
    for i, (x, y) in enumerate(pts, start=1):
        ax.add_patch(Circle((x, y), 1.2, facecolor="#ffffff", edgecolor="#0f172a", linewidth=1.2))
        ax.text(x + 1.2, y + 1.5, f"m{i}", fontsize=9)
    for x in [30, 44, 58]:
        ax.plot([x, x], [18, 52] if x != 44 else [12, 58], linestyle="--", color="#f59e0b", linewidth=1.4)
    ax.plot([18, 82], [35, 35], linestyle="--", color="#10b981", linewidth=1.8)
    ax.text(64, 76, "MAR = mean(vertical openings) / mouth width", fontsize=14, weight="bold", color="#0f172a")
    ax.text(64, 67, "Used for yawn candidate and confirmed yawn validation", fontsize=10.5, color="#475569")
    save(fig, "figure_04_mar_computation.png")


def figure_5_baseline_workflow():
    fig, ax = setup_canvas((12, 10))
    title(ax, "Figure 5. Personalized Baseline Workflow")
    subtitle(ax, "Driver-specific calibration and reuse pipeline")
    nodes = [
        (35, 78, 30, 10, "Start Session"),
        (35, 63, 30, 10, "Capture Initial Driver Behavior"),
        (35, 48, 30, 10, "Compute EAR, MAR,\nHead Pose"),
        (35, 33, 30, 10, "Build Personalized\nBaseline Profile"),
        (12, 18, 30, 10, "Save Profile\nper Driver ID"),
        (58, 18, 30, 10, "Live Monitoring\nUsing Saved Profile"),
    ]
    for x, y, w, h, text in nodes:
        box(ax, x, y, w, h, text, fc="#f8fbff", ec="#3b82f6", fontsize=11)
    for i in range(4):
        arrow(ax, (50, nodes[i][1]), (50, nodes[i + 1][1] + 10))
    arrow(ax, (42, 33), (27, 28))
    arrow(ax, (58, 33), (73, 28))
    ax.text(50, 8, "Deviation from the stored baseline drives fatigue scoring", ha="center", fontsize=12, color="#0f172a", weight="bold")
    save(fig, "figure_05_baseline_workflow.png")


def figure_6_deviation_scoring():
    fig, ax = setup_canvas((14, 8))
    title(ax, "Figure 6. Deviation-Based Fatigue Scoring")
    subtitle(ax, "Personalized normal behavior is compared against current live state")
    box(ax, 7, 25, 36, 48, "Normal Driver Baseline\n\nEAR: stable open-eye mean\nMAR: low resting mouth opening\nHead pose: normal orientation\nBlink duration: short\nOpen intervals: regular", fc="#effcf4", ec="#16a34a", fontsize=12)
    box(ax, 57, 25, 36, 48, "Detected Deviation\n\nEAR: drops below personal norm\nMAR: sustained high opening\nHead pose: abnormal tilt / yaw\nBlink duration: prolonged\nOpen intervals: irregular", fc="#fff4f2", ec="#dc2626", fontsize=12)
    arrow(ax, (43, 49), (57, 49), color="#334155", lw=2.0)
    ax.text(50, 53, "Deviation analysis", ha="center", fontsize=11, color="#334155", weight="bold")
    box(ax, 33, 8, 34, 10, "Weighted fatigue score -> Mild / Moderate / Strong alert", fc="#eef2ff", ec="#4f46e5", fontsize=11)
    arrow(ax, (50, 25), (50, 18), color="#4f46e5")
    save(fig, "figure_06_deviation_scoring.png")


def figure_7_deployment_comparison():
    fig, ax = setup_canvas((15, 8))
    title(ax, "Figure 7. Local Edge vs Hosted Web Deployment")
    subtitle(ax, "Comparison of deployment modes used in the project")
    box(ax, 8, 58, 34, 26, "Local Edge Deployment\n\nCamera -> Local inference ->\nLive overlay -> Alarm", fc="#effcf4", ec="#16a34a", fontsize=13)
    box(ax, 58, 58, 34, 26, "Hosted Web Deployment\n\nBrowser webcam -> Server inference ->\nProcessed output -> Browser", fc="#fff7ed", ec="#ea580c", fontsize=13)
    box(ax, 10, 18, 30, 24, "Advantages\n\nLow latency\nDirect hardware access\nCloser to in-car edge systems", fc="#f8fafc", ec="#16a34a", fontsize=11)
    box(ax, 60, 18, 30, 24, "Advantages / Trade-off\n\nPublic URL\nEasy sharing\nHigher latency due to browser-server loop", fc="#f8fafc", ec="#ea580c", fontsize=11)
    arrow(ax, (25, 58), (25, 42), color="#16a34a")
    arrow(ax, (75, 58), (75, 42), color="#ea580c")
    ax.text(50, 8, "Real automotive systems typically follow the local edge deployment model", ha="center", fontsize=11, color="#0f172a", weight="bold")
    save(fig, "figure_07_deployment_comparison.png")


def diamond(ax, cx, cy, w, h, text, fc="#fff7e8", ec="#d97706"):
    pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
    patch = Polygon(pts, closed=True, facecolor=fc, edgecolor=ec, linewidth=1.8)
    ax.add_patch(patch)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=10.5, weight="bold", color="#132238", wrap=True)


def figure_8_alert_flow():
    fig, ax = setup_canvas((13, 10))
    title(ax, "Figure 8. Alert Decision Flow")
    subtitle(ax, "How the system decides whether to trigger a drowsiness alert")
    box(ax, 34, 82, 32, 8, "Detect face and compute EAR / MAR / Head Pose", fontsize=11)
    diamond(ax, 50, 68, 30, 12, "Eyes closed below personalized threshold?")
    diamond(ax, 50, 50, 30, 12, "Sustained closure long enough?")
    diamond(ax, 50, 32, 30, 12, "Score / yawn / pose indicate fatigue?")
    box(ax, 8, 62, 22, 8, "Continue monitoring", fc="#effcf4", ec="#16a34a", fontsize=10.5)
    box(ax, 8, 44, 22, 8, "Ignore normal blink", fc="#effcf4", ec="#16a34a", fontsize=10.5)
    box(ax, 63, 12, 28, 10, "Trigger alert\nand recommend break\nif severity is high", fc="#fff4f2", ec="#dc2626", fontsize=11)
    arrow(ax, (50, 82), (50, 74))
    arrow(ax, (35, 68), (30, 66))
    arrow(ax, (50, 62), (50, 56))
    arrow(ax, (35, 50), (30, 48))
    arrow(ax, (50, 44), (50, 38))
    arrow(ax, (50, 26), (63, 18))
    save(fig, "figure_08_alert_flow.png")


def figure_9_ui_mockup():
    fig, ax = setup_canvas((14, 9))
    title(ax, "Figure 9. Real-Time Output Interface Mockup")
    subtitle(ax, "Styled representation of the final detector UI")
    screen = FancyBboxPatch((12, 12), 76, 68, boxstyle="round,pad=0.02,rounding_size=2.5", facecolor="#d8c7a2", edgecolor="#0f172a", linewidth=1.8)
    ax.add_patch(screen)
    box(ax, 16, 72, 18, 7, "Monitoring", fc="#1b2c1d", ec="#8df3a9", fontsize=13, color="#9df7b1")
    ax.add_patch(Rectangle((40, 30), 25, 30, fill=False, edgecolor="#f6d32d", linewidth=2))
    ax.add_patch(Rectangle((44.5, 46), 7, 4, fill=False, edgecolor="#e879f9", linewidth=2))
    ax.add_patch(Rectangle((54.5, 46), 7, 4, fill=False, edgecolor="#e879f9", linewidth=2))
    ax.add_patch(Rectangle((47, 35), 12, 5, fill=False, edgecolor="#fb7185", linewidth=2))
    bar = Rectangle((14, 14), 72, 8, facecolor="#111827", alpha=0.78, edgecolor="none")
    ax.add_patch(bar)
    ax.text(18, 18, "EAR 245.0    MAR 0.029    Fatigue 2.06    Alerts 3    Yawns 1", fontsize=12, color="white", va="center", weight="bold")
    ax.text(50, 7, "Local runtime overlay used for live demonstration", ha="center", fontsize=11, color="#334155")
    save(fig, "figure_09_ui_mockup.png")


def figure_10_contribution_summary():
    fig, ax = setup_canvas((14, 8))
    title(ax, "Figure 10. Research Contribution Summary")
    subtitle(ax, "Key contributions achieved in the project")
    panels = [
        (8, 48, "Personalized\nBaseline Modeling", "#eef4ff", "#2563eb", "Learns driver-specific\nnormal behavior"),
        (54, 48, "Real-Time\nEAR / MAR Monitoring", "#effcf4", "#16a34a", "Tracks eye and mouth\nfatigue cues"),
        (8, 15, "3D Head Pose\nEstimation", "#fff7ed", "#ea580c", "Uses yaw, pitch,\nand roll as cues"),
        (54, 15, "Deployable\nArchitecture", "#f5f3ff", "#7c3aed", "Supports local edge\nand hosted demo paths"),
    ]
    for x, y, heading, fc, ec, sub in panels:
        box(ax, x, y, 38, 22, heading, fc=fc, ec=ec, fontsize=14)
        ax.text(x + 19, y + 5.5, sub, ha="center", va="center", fontsize=11, color="#475569")
    save(fig, "figure_10_contribution_summary.png")


if __name__ == "__main__":
    main()
