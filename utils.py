"""
utils.py — Driver Monitoring System Utilities
============================================================
Contains:
  • EAR  – Eye Aspect Ratio  (drowsiness)
  • MAR  – Mouth Aspect Ratio (yawning)
  • HeadPoseEstimator         (gaze direction)
  • AlertManager              (sound + on-screen warnings)
  • draw_dashboard            (HUD overlay renderer)
  • draw_landmarks            (debug face-mesh overlay)
"""

import time
import threading
import platform
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import cv2
import numpy as np

# ── optional: playsound for cross-platform audio ──────────────────────────────
try:
    import playsound
    _PLAYSOUND_AVAILABLE = True
except ImportError:
    _PLAYSOUND_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DriverState:
    """Snapshot of the driver's current monitoring state."""
    ear:             float  = 1.0    # current Eye Aspect Ratio
    mar:             float  = 0.0    # current Mouth Aspect Ratio
    yaw:             float  = 0.0    # head yaw  in degrees
    pitch:           float  = 0.0    # head pitch in degrees
    roll:            float  = 0.0    # head roll  in degrees
    gaze_x:          float  = 0.0   # iris gaze horizontal offset (-1 left … +1 right)
    gaze_y:          float  = 0.0   # iris gaze vertical offset   (-1 up   … +1 down)
    drowsy:          bool   = False  # drowsiness flag
    yawning:         bool   = False  # yawning flag
    head_distracted: bool   = False  # head turned away flag
    phone_detected:  bool   = False  # phone-in-hand flag
    fps:             float  = 0.0    # current processing FPS
    status:          str    = "ALERT"  # overall label shown on HUD

    @property
    def any_alert(self) -> bool:
        return self.drowsy or self.yawning or self.head_distracted or self.phone_detected


# ══════════════════════════════════════════════════════════════════════════════
#  EAR  —  Eye Aspect Ratio
# ══════════════════════════════════════════════════════════════════════════════

# MediaPipe face-mesh landmark indices for each eye
# Left eye  (from driver's perspective)
LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
# Right eye
RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]

def _eye_aspect_ratio(landmarks, eye_indices: List[int], w: int, h: int) -> float:
    """
    Compute the Eye Aspect Ratio for one eye.

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Args:
        landmarks : MediaPipe NormalizedLandmarkList
        eye_indices: 6 landmark indices [p1..p6] in Soukupová order
        w, h       : frame dimensions for de-normalisation

    Returns:
        float – EAR value (lower → more closed)
    """
    def _pt(idx):
        lm = landmarks[idx]
        return np.array([lm.x * w, lm.y * h])

    p1, p2, p3, p4, p5, p6 = [_pt(i) for i in eye_indices]
    A = np.linalg.norm(p2 - p6)
    B = np.linalg.norm(p3 - p5)
    C = np.linalg.norm(p1 - p4)
    return (A + B) / (2.0 * C + 1e-6)


def compute_ear(landmarks, w: int, h: int) -> float:
    """Average EAR across both eyes."""
    left  = _eye_aspect_ratio(landmarks, LEFT_EYE_IDX,  w, h)
    right = _eye_aspect_ratio(landmarks, RIGHT_EYE_IDX, w, h)
    return (left + right) / 2.0


# ══════════════════════════════════════════════════════════════════════════════
#  MAR  —  Mouth Aspect Ratio (yawn detection)
# ══════════════════════════════════════════════════════════════════════════════

# Outer lip landmarks from MediaPipe 468-point mesh
MOUTH_IDX = [61, 291, 39, 181, 0, 17, 269, 405]

def compute_mar(landmarks, w: int, h: int) -> float:
    """
    Mouth Aspect Ratio – analogous to EAR for the mouth.
    Uses 4 vertical pairs and 1 horizontal pair.

    Returns:
        float – MAR value (higher → wider open)
    """
    def _pt(idx):
        lm = landmarks[idx]
        return np.array([lm.x * w, lm.y * h])

    p1, p2, p3, p4, p5, p6, p7, p8 = [_pt(i) for i in MOUTH_IDX]
    # Vertical distances
    A = np.linalg.norm(p3 - p7)
    B = np.linalg.norm(p4 - p8)
    C = np.linalg.norm(p5 - p6)  # center vertical
    # Horizontal distance
    D = np.linalg.norm(p1 - p2)
    return (A + B + C) / (2.0 * D + 1e-6)


# ══════════════════════════════════════════════════════════════════════════════
#  HEAD POSE ESTIMATOR
# ══════════════════════════════════════════════════════════════════════════════

# 3-D model points of canonical face (nose-tip centred, cm scale)
_MODEL_POINTS = np.array([
    ( 0.0,    0.0,    0.0  ),   # Nose tip            (1)
    ( 0.0,   -330.0, -65.0),   # Chin                (8)
    (-225.0,  170.0, -135.0),  # Left eye corner     (36)
    ( 225.0,  170.0, -135.0),  # Right eye corner    (45)
    (-150.0, -150.0, -125.0),  # Left mouth corner   (48)
    ( 150.0, -150.0, -125.0),  # Right mouth corner  (54)
], dtype=np.float64)

# Corresponding MediaPipe face-mesh indices
_MP_POSE_IDX = [1, 152, 263, 33, 287, 57]


def estimate_head_pose(landmarks, w: int, h: int) -> Tuple[float, float, float]:
    """
    Estimate yaw, pitch, roll (degrees) using solvePnP.

    Returns:
        (yaw, pitch, roll) – positive yaw = head turned right
    """
    image_points = np.array([
        (landmarks[i].x * w, landmarks[i].y * h)
        for i in _MP_POSE_IDX
    ], dtype=np.float64)

    focal  = w
    center = (w / 2.0, h / 2.0)
    camera_matrix = np.array([
        [focal,  0,      center[0]],
        [0,      focal,  center[1]],
        [0,      0,      1       ],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    ok, rvec, _ = cv2.solvePnP(
        _MODEL_POINTS, image_points,
        camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return 0.0, 0.0, 0.0

    rot_mat, _ = cv2.Rodrigues(rvec)
    proj_matrix = np.hstack([rot_mat, np.zeros((3, 1))])
    _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj_matrix)

    # euler is shape (3,1) — flatten to get plain scalars
    euler = euler.flatten()
    pitch = float(euler[0])
    yaw   = float(euler[1])
    roll  = float(euler[2])
    return yaw, pitch, roll


# ══════════════════════════════════════════════════════════════════════════════
#  ROLLING COUNTER  –  consecutive-frame threshold helper
# ══════════════════════════════════════════════════════════════════════════════

class RollingCounter:
    """
    Counts consecutive frames where a condition is True,
    resets to zero when condition becomes False.
    """
    def __init__(self, threshold: int):
        self.threshold = threshold
        self.count     = 0

    def update(self, condition: bool) -> bool:
        """Update counter; returns True when threshold is exceeded."""
        if condition:
            self.count += 1
        else:
            self.count = 0
        return self.count >= self.threshold

    def reset(self):
        self.count = 0


# ══════════════════════════════════════════════════════════════════════════════
#  ALERT MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class AlertManager:
    """
    Manages audio alerts with a cooldown to avoid spamming.
    Beep runs in a background thread so it doesn't block video.
    """
    def __init__(self, cooldown_sec: float = 3.0,
                 freq: int = 1000, duration_ms: int = 500):
        self._cooldown   = cooldown_sec
        self._freq       = freq
        self._dur        = duration_ms
        self._last_alert = 0.0
        self._os         = platform.system()

    def trigger(self, reason: str = ""):
        """Fire an alert if the cooldown has elapsed."""
        now = time.time()
        if now - self._last_alert >= self._cooldown:
            self._last_alert = now
            label = f"[ALERT] {reason}" if reason else "[ALERT]"
            print(label)
            t = threading.Thread(target=self._beep, daemon=True)
            t.start()

    def _beep(self):
        try:
            if self._os == "Windows":
                import winsound
                winsound.Beep(self._freq, self._dur)
            elif self._os == "Darwin":           # macOS
                import subprocess
                subprocess.call(["afplay", "/System/Library/Sounds/Funk.aiff"])
            else:                                # Linux
                # Try ALSA speaker-test, fall back silently
                import subprocess
                subprocess.call(
                    ["speaker-test", "-t", "sine", "-f", str(self._freq),
                     "-l", "1"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        except Exception:
            pass  # audio failure is non-critical


# ══════════════════════════════════════════════════════════════════════════════
#  DRAWING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

# Colour palette  (BGR)
_COL = {
    "green":  (0,   220,  80),
    "yellow": (0,   210, 255),
    "red":    (30,   30, 230),
    "orange": (10,  130, 255),
    "white":  (240, 240, 240),
    "black":  (10,   10,  10),
    "bg":     (18,   18,  24),
    "card":   (28,   28,  38),
    "accent": (60,  200, 255),
}

def _text(img, txt, pos, scale=0.55, color=_COL["white"], thickness=1):
    cv2.putText(img, txt, pos, cv2.FONT_HERSHEY_DUPLEX,
                scale, color, thickness, cv2.LINE_AA)


def draw_landmarks(frame, face_landmarks, w: int, h: int,
                   draw_eyes=True, draw_mouth=True):
    """Draw eye and mouth landmark dots for debugging."""
    if draw_eyes:
        for idx in LEFT_EYE_IDX + RIGHT_EYE_IDX:
            lm = face_landmarks[idx]
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 2, _COL["accent"], -1)
    if draw_mouth:
        for idx in MOUTH_IDX:
            lm = face_landmarks[idx]
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 2, _COL["yellow"], -1)


def draw_head_axis(frame, face_landmarks, w: int, h: int,
                   yaw: float, pitch: float):
    """Draw a direction indicator near the nose tip."""
    nose = face_landmarks[1]
    nx, ny = int(nose.x * w), int(nose.y * h)
    length = 60
    yaw_r   = math.radians(yaw)
    pitch_r = math.radians(pitch)
    ex = nx + int(length * math.sin(yaw_r))
    ey = ny - int(length * math.sin(pitch_r))
    cv2.arrowedLine(frame, (nx, ny), (ex, ey), _COL["accent"], 2,
                    tipLength=0.3)


def draw_dashboard(frame, state: DriverState, dashboard_h: int):
    """
    Render a semi-transparent HUD dashboard at the bottom of the frame.

    Layout (left → right):
        [STATUS]  |  EAR  MAR  YAW  PITCH  |  FLAGS  |  FPS
    """
    fh, fw = frame.shape[:2]
    y0 = fh - dashboard_h

    # ── dark background strip ────────────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y0), (fw, fh), _COL["bg"], -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    # top border line
    border_col = _COL["red"] if state.any_alert else _COL["green"]
    cv2.line(frame, (0, y0), (fw, y0), border_col, 2)

    # ── STATUS badge ─────────────────────────────────────────────────────────
    badge_w = 200
    status_color = _COL["green"]
    if state.drowsy or state.yawning:
        status_color = _COL["red"]
        status_label = "DROWSY"
    elif state.phone_detected:
        status_color = _COL["orange"]
        status_label = "PHONE DETECTED"
    elif state.head_distracted:
        status_color = _COL["yellow"]
        status_label = "DISTRACTED"
    else:
        status_label = "ALERT"

    cv2.rectangle(frame, (10, y0 + 10), (badge_w, y0 + 60), status_color, -1)
    cv2.rectangle(frame, (10, y0 + 10), (badge_w, y0 + 60), _COL["white"], 1)
    tw, th = cv2.getTextSize(status_label, cv2.FONT_HERSHEY_DUPLEX,
                             0.65, 2)[0]
    tx = 10 + (badge_w - 10 - tw) // 2
    ty = y0 + 10 + (50 + th) // 2
    cv2.putText(frame, status_label, (tx, ty),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, _COL["black"], 2, cv2.LINE_AA)

    # ── Metrics ──────────────────────────────────────────────────────────────
    mx = badge_w + 30
    metrics = [
        ("EAR",   f"{state.ear:.3f}",
         _COL["red"] if state.ear < 0.22 else _COL["white"]),
        ("MAR",   f"{state.mar:.3f}",
         _COL["orange"] if state.mar > 0.6 else _COL["white"]),
        ("YAW",   f"{state.yaw:+.1f}°",
         _COL["yellow"] if abs(state.yaw) > 30 else _COL["white"]),
        ("PITCH", f"{state.pitch:+.1f}°",
         _COL["yellow"] if abs(state.pitch) > 25 else _COL["white"]),
    ]
    for label, val, col in metrics:
        _text(frame, label, (mx, y0 + 30), 0.42, _COL["accent"])
        _text(frame, val,   (mx, y0 + 55), 0.60, col, 1)
        mx += 110

    # ── Active-flag pills ────────────────────────────────────────────────────
    flags = []
    if state.drowsy:          flags.append(("EYES CLOSED",  _COL["red"]))
    if state.yawning:         flags.append(("YAWNING",      _COL["orange"]))
    if state.head_distracted: flags.append(("HEAD AWAY",    _COL["yellow"]))
    if state.phone_detected:  flags.append(("PHONE",        _COL["orange"]))

    pill_x = mx + 20
    for txt, col in flags:
        tw2 = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 0.44, 1)[0][0]
        cv2.rectangle(frame,
                      (pill_x - 4,      y0 + 18),
                      (pill_x + tw2 + 4, y0 + 48), col, -1)
        _text(frame, txt, (pill_x, y0 + 40), 0.44, _COL["black"], 1)
        pill_x += tw2 + 20

    # ── FPS counter ──────────────────────────────────────────────────────────
    fps_txt = f"{state.fps:.1f} fps"
    tw3 = cv2.getTextSize(fps_txt, cv2.FONT_HERSHEY_DUPLEX, 0.50, 1)[0][0]
    _text(frame, fps_txt, (fw - tw3 - 12, y0 + 55), 0.50, _COL["accent"])

    # ── second row: consecutive-frame progress bars ───────────────────────────
    bar_y = y0 + 75
    bar_labels = [
        ("Drowsy", state.drowsy),
        ("Yawning", state.yawning),
        ("Head",   state.head_distracted),
        ("Phone",  state.phone_detected),
    ]
    bx = 15
    for lbl, active in bar_labels:
        col2 = _COL["red"] if active else _COL["card"]
        cv2.rectangle(frame, (bx, bar_y), (bx + 90, bar_y + 20), col2, -1)
        cv2.rectangle(frame, (bx, bar_y), (bx + 90, bar_y + 20),
                      _COL["accent"], 1)
        _text(frame, lbl, (bx + 4, bar_y + 14), 0.38,
              _COL["black"] if active else _COL["accent"])
        bx += 105

# ══════════════════════════════════════════════════════════════════════════════
#  IRIS / GAZE TRACKING
# ══════════════════════════════════════════════════════════════════════════════
#
#  MediaPipe 478-landmark model includes 10 iris landmarks:
#    Left  iris : 468 (centre), 469 (right), 470 (bottom), 471 (left), 472 (top)
#    Right iris : 473 (centre), 474 (right), 475 (bottom), 476 (left), 477 (top)
#
#  Gaze is estimated by comparing the iris centre position against the eye
#  corner landmarks to produce a normalised offset in [-1, +1] for both axes.
# ──────────────────────────────────────────────────────────────────────────────

# Iris centre landmark indices
_L_IRIS_C = 468
_R_IRIS_C = 473

# Iris edge landmarks (right, bottom, left, top) for radius estimation
_L_IRIS_EDGE = [469, 470, 471, 472]
_R_IRIS_EDGE = [474, 475, 476, 477]

# Eye-corner landmarks used as reference for gaze offset calculation
_L_EYE_LEFT_CORNER  = 263   # outer (temporal) corner — left eye
_L_EYE_RIGHT_CORNER = 362   # inner (nasal)    corner — left eye
_R_EYE_LEFT_CORNER  = 33    # inner (nasal)    corner — right eye
_R_EYE_RIGHT_CORNER = 133   # outer (temporal) corner — right eye
_L_EYE_TOP          = 386   # upper lid centre — left eye
_L_EYE_BOT          = 374   # lower lid centre — left eye
_R_EYE_TOP          = 159   # upper lid centre — right eye
_R_EYE_BOT          = 145   # lower lid centre — right eye


def _lm_pt(landmarks, idx: int, w: int, h: int) -> np.ndarray:
    """Return pixel-space (x, y) for a landmark index."""
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)


def _iris_radius(landmarks, edge_indices: list, centre: np.ndarray,
                 w: int, h: int) -> float:
    """Estimate iris radius as mean distance from centre to 4 edge points."""
    dists = [np.linalg.norm(_lm_pt(landmarks, i, w, h) - centre)
             for i in edge_indices]
    return float(np.mean(dists))


def compute_gaze(landmarks, w: int, h: int) -> Tuple[float, float,
                                                       np.ndarray, float,
                                                       np.ndarray, float]:
    """
    Compute iris gaze offset for both eyes.

    Returns:
        gaze_x  : horizontal gaze in [-1, +1]  (+1 = looking right)
        gaze_y  : vertical   gaze in [-1, +1]  (+1 = looking down)
        l_centre: left  iris centre pixel (x, y)
        l_radius: left  iris radius in pixels
        r_centre: right iris centre pixel (x, y)
        r_radius: right iris radius in pixels
    """
    # ── iris centres ──────────────────────────────────────────────────────────
    lc = _lm_pt(landmarks, _L_IRIS_C, w, h)
    rc = _lm_pt(landmarks, _R_IRIS_C, w, h)

    l_radius = _iris_radius(landmarks, _L_IRIS_EDGE, lc, w, h)
    r_radius = _iris_radius(landmarks, _R_IRIS_EDGE, rc, w, h)

    # ── per-eye gaze offset (normalised by eye width / height) ───────────────
    def _eye_gaze(iris_c, c_left, c_right, c_top, c_bot):
        eye_left  = _lm_pt(landmarks, c_left,  w, h)
        eye_right = _lm_pt(landmarks, c_right, w, h)
        eye_top   = _lm_pt(landmarks, c_top,   w, h)
        eye_bot   = _lm_pt(landmarks, c_bot,   w, h)

        eye_centre = (eye_left + eye_right) / 2.0
        eye_w = np.linalg.norm(eye_right - eye_left) + 1e-6
        eye_h = np.linalg.norm(eye_bot   - eye_top)  + 1e-6

        dx = float((iris_c[0] - eye_centre[0]) / (eye_w / 2.0))
        dy = float((iris_c[1] - eye_centre[1]) / (eye_h / 2.0))
        return np.clip(dx, -1.0, 1.0), np.clip(dy, -1.0, 1.0)

    lx, ly = _eye_gaze(lc, _L_EYE_LEFT_CORNER, _L_EYE_RIGHT_CORNER,
                        _L_EYE_TOP, _L_EYE_BOT)
    rx, ry = _eye_gaze(rc, _R_EYE_LEFT_CORNER, _R_EYE_RIGHT_CORNER,
                        _R_EYE_TOP, _R_EYE_BOT)

    gaze_x = float((lx + rx) / 2.0)
    gaze_y = float((ly + ry) / 2.0)

    return gaze_x, gaze_y, lc, l_radius, rc, r_radius


# ──────────────────────────────────────────────────────────────────────────────
#  GAZE VISUALISATION
# ──────────────────────────────────────────────────────────────────────────────

def draw_eye_tracking(frame,
                      landmarks,
                      w: int, h: int,
                      gaze_x: float, gaze_y: float,
                      l_centre: np.ndarray, l_radius: float,
                      r_centre: np.ndarray, r_radius: float):
    """
    Draw the full webcam-style eye-tracking overlay:
      • Coloured iris circles on each eye
      • Gaze-direction arrow from each iris
      • Semi-transparent eye bounding ellipse
      • Mini gaze-map panel (top-right corner)
      • Gaze metrics text panel
    """
    # ── colours ──────────────────────────────────────────────────────────────
    C_IRIS      = (255, 220,  50)   # gold iris ring
    C_PUPIL     = ( 20,  20,  20)   # dark pupil
    C_GAZE      = ( 50, 255, 120)   # green gaze arrow
    C_ELLIPSE   = ( 80, 180, 255)   # blue eye ellipse
    C_PANEL_BG  = ( 15,  15,  20)
    C_ACCENT    = ( 60, 200, 255)
    C_WHITE     = (230, 230, 230)
    C_RED       = ( 40,  40, 220)
    C_YELLOW    = ( 30, 210, 255)

    ARROW_LEN   = int(l_radius * 3.5)

    def _draw_one_eye(centre, radius, edge_indices):
        cx, cy = int(centre[0]), int(centre[1])
        r      = max(2, int(radius))

        # Semi-transparent iris fill
        overlay = frame.copy()
        cv2.circle(overlay, (cx, cy), r, C_IRIS, -1)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        # Iris ring
        cv2.circle(frame, (cx, cy), r,     C_IRIS,  1, cv2.LINE_AA)
        # Pupil dot
        cv2.circle(frame, (cx, cy), max(1, r // 3), C_PUPIL, -1, cv2.LINE_AA)
        # Outer highlight ring
        cv2.circle(frame, (cx, cy), r + 2, C_ELLIPSE, 1, cv2.LINE_AA)

    _draw_one_eye(l_centre, l_radius, _L_IRIS_EDGE)
    _draw_one_eye(r_centre, r_radius, _R_IRIS_EDGE)

    # ── gaze arrows from each iris ────────────────────────────────────────────
    def _gaze_arrow(centre, gx, gy, length):
        cx, cy = int(centre[0]), int(centre[1])
        ex = cx + int(gx * length)
        ey = cy + int(gy * length)
        cv2.arrowedLine(frame, (cx, cy), (ex, ey), C_GAZE, 2,
                        tipLength=0.35, line_type=cv2.LINE_AA)

    _gaze_arrow(l_centre, gaze_x, gaze_y, ARROW_LEN)
    _gaze_arrow(r_centre, gaze_x, gaze_y, ARROW_LEN)

    # ── eye bounding ellipses ─────────────────────────────────────────────────
    def _eye_ellipse(c_left_idx, c_right_idx, c_top_idx, c_bot_idx):
        p_l = _lm_pt(landmarks, c_left_idx,  w, h)
        p_r = _lm_pt(landmarks, c_right_idx, w, h)
        p_t = _lm_pt(landmarks, c_top_idx,   w, h)
        p_b = _lm_pt(landmarks, c_bot_idx,   w, h)
        centre_e = ((p_l + p_r) / 2).astype(int)
        ew = int(np.linalg.norm(p_r - p_l) / 2) + 4
        eh = int(np.linalg.norm(p_b - p_t) / 2) + 4
        if ew > 2 and eh > 2:
            cv2.ellipse(frame, tuple(centre_e), (ew, eh), 0, 0, 360,
                        C_ELLIPSE, 1, cv2.LINE_AA)

    _eye_ellipse(_L_EYE_LEFT_CORNER, _L_EYE_RIGHT_CORNER,
                 _L_EYE_TOP, _L_EYE_BOT)
    _eye_ellipse(_R_EYE_LEFT_CORNER, _R_EYE_RIGHT_CORNER,
                 _R_EYE_TOP, _R_EYE_BOT)

    # ══════════════════════════════════════════════════════════════════════════
    #  GAZE MAP PANEL  (top-right corner)
    #  A small square showing where the driver is looking, like a crosshair map
    # ══════════════════════════════════════════════════════════════════════════
    MAP_SIZE = 110
    MAP_PAD  = 10
    mx0 = w - MAP_SIZE - MAP_PAD
    my0 = MAP_PAD

    # Background
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (mx0, my0), (mx0 + MAP_SIZE, my0 + MAP_SIZE),
                  C_PANEL_BG, -1)
    cv2.addWeighted(overlay2, 0.75, frame, 0.25, 0, frame)

    # Border
    cv2.rectangle(frame, (mx0, my0), (mx0 + MAP_SIZE, my0 + MAP_SIZE),
                  C_ACCENT, 1)

    # Grid lines
    mid_x = mx0 + MAP_SIZE // 2
    mid_y = my0 + MAP_SIZE // 2
    cv2.line(frame, (mid_x, my0), (mid_x, my0 + MAP_SIZE), (40, 40, 50), 1)
    cv2.line(frame, (mx0, mid_y), (mx0 + MAP_SIZE, mid_y), (40, 40, 50), 1)

    # Outer ellipse — screen boundary metaphor
    cv2.ellipse(frame, (mid_x, mid_y),
                (MAP_SIZE // 2 - 8, MAP_SIZE // 2 - 8),
                0, 0, 360, (50, 60, 70), 1)

    # Gaze dot position
    dot_x = mid_x + int(gaze_x * (MAP_SIZE // 2 - 12))
    dot_y = mid_y + int(gaze_y * (MAP_SIZE // 2 - 12))
    dot_x = int(np.clip(dot_x, mx0 + 5, mx0 + MAP_SIZE - 5))
    dot_y = int(np.clip(dot_y, my0 + 5, my0 + MAP_SIZE - 5))

    # Gaze dot with glow effect
    cv2.circle(frame, (dot_x, dot_y), 8, C_GAZE, -1, cv2.LINE_AA)
    cv2.circle(frame, (dot_x, dot_y), 11, C_GAZE, 1,  cv2.LINE_AA)

    # Label
    cv2.putText(frame, "GAZE MAP", (mx0 + 4, my0 + MAP_SIZE + 14),
                cv2.FONT_HERSHEY_DUPLEX, 0.38, C_ACCENT, 1, cv2.LINE_AA)

    # ══════════════════════════════════════════════════════════════════════════
    #  3-D PERSPECTIVE CONE  (gaze projection — top-left corner)
    # ══════════════════════════════════════════════════════════════════════════
    CONE_X  = 12
    CONE_Y  = 12
    CONE_W  = 160
    CONE_H  = 110

    # Background panel
    ovl3 = frame.copy()
    cv2.rectangle(ovl3, (CONE_X, CONE_Y),
                  (CONE_X + CONE_W, CONE_Y + CONE_H), C_PANEL_BG, -1)
    cv2.addWeighted(ovl3, 0.72, frame, 0.28, 0, frame)
    cv2.rectangle(frame, (CONE_X, CONE_Y),
                  (CONE_X + CONE_W, CONE_Y + CONE_H), C_ACCENT, 1)

    # Eye position (left side of panel)
    eye_px = CONE_X + 28
    eye_py = CONE_Y + CONE_H // 2
    cv2.circle(frame, (eye_px, eye_py), 7, C_IRIS, -1, cv2.LINE_AA)
    cv2.circle(frame, (eye_px, eye_py), 3, C_PUPIL, -1, cv2.LINE_AA)

    # Screen rectangle (right side of panel)
    scr_x1 = CONE_X + CONE_W - 42
    scr_y1 = CONE_Y + 14
    scr_x2 = CONE_X + CONE_W - 12
    scr_y2 = CONE_Y + CONE_H - 14
    cv2.rectangle(frame, (scr_x1, scr_y1), (scr_x2, scr_y2), C_ELLIPSE, 1)

    # Gaze point on the screen rectangle
    scr_mid_x = (scr_x1 + scr_x2) // 2
    scr_mid_y = (scr_y1 + scr_y2) // 2
    gpt_x = scr_mid_x + int(gaze_x * (scr_x2 - scr_x1) // 2)
    gpt_y = scr_mid_y + int(gaze_y * (scr_y2 - scr_y1) // 2)
    gpt_x = int(np.clip(gpt_x, scr_x1 + 2, scr_x2 - 2))
    gpt_y = int(np.clip(gpt_y, scr_y1 + 2, scr_y2 - 2))

    # Cone lines from eye to screen corners
    for corner in [(scr_x1, scr_y1), (scr_x1, scr_y2),
                   (scr_x2, scr_y1), (scr_x2, scr_y2)]:
        cv2.line(frame, (eye_px, eye_py), corner, (40, 60, 80), 1, cv2.LINE_AA)

    # Main gaze ray
    cv2.line(frame, (eye_px, eye_py), (gpt_x, gpt_y), C_GAZE, 2, cv2.LINE_AA)
    cv2.circle(frame, (gpt_x, gpt_y), 4, C_GAZE, -1, cv2.LINE_AA)

    # Labels
    cv2.putText(frame, "EYE", (eye_px - 10, eye_py + 18),
                cv2.FONT_HERSHEY_DUPLEX, 0.32, C_IRIS, 1, cv2.LINE_AA)
    cv2.putText(frame, "SCR", (scr_x1, scr_y2 + 12),
                cv2.FONT_HERSHEY_DUPLEX, 0.32, C_ELLIPSE, 1, cv2.LINE_AA)

    # ══════════════════════════════════════════════════════════════════════════
    #  GAZE METRICS TEXT  (below cone panel)
    # ══════════════════════════════════════════════════════════════════════════
    ty = CONE_Y + CONE_H + 18
    metrics = [
        (f"gaze X  {gaze_x:+.2f}",
         C_RED    if abs(gaze_x) > 0.4 else C_WHITE),
        (f"gaze Y  {gaze_y:+.2f}",
         C_YELLOW if abs(gaze_y) > 0.4 else C_WHITE),
        (f"L iris r {l_radius:.1f}px", C_WHITE),
        (f"R iris r {r_radius:.1f}px", C_WHITE),
    ]
    for txt, col in metrics:
        cv2.putText(frame, txt, (CONE_X + 4, ty),
                    cv2.FONT_HERSHEY_DUPLEX, 0.38, col, 1, cv2.LINE_AA)
        ty += 16
