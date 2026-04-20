# 🚗 Advanced Driver Monitoring System

A real-time driver safety system built with Python, OpenCV, MediaPipe, and YOLOv8.

---

## ✨ Features

| Feature | Method | Trigger |
|---|---|---|
| **Drowsiness detection** | Eye Aspect Ratio (EAR) | Eyes closed for ≥ 20 frames |
| **Yawn detection** | Mouth Aspect Ratio (MAR) | Mouth open wide for ≥ 15 frames |
| **Head-pose tracking** | solvePnP (6-DoF) | Yaw > 30° or Pitch > 25° for ≥ 15 frames |
| **Phone detection** | YOLOv8 nano (COCO) | Phone visible for ≥ 10 frames |
| **Audio alert** | OS beep / afplay | Fires on any alert, 3 s cooldown |
| **HUD dashboard** | OpenCV overlay | Live EAR / MAR / yaw / pitch + status badge |

---

## 📁 Project Structure

```
driver_monitor/
├── main.py          ← entry point, main loop, camera, YOLO, HUD
├── utils.py         ← EAR, MAR, head-pose, AlertManager, drawing helpers
├── config.py        ← all tunable thresholds and constants
├── requirements.txt ← Python dependencies
└── README.md        ← this file
```

---

## 🛠️ Installation

### 1. Clone / copy the project

```bash
# If from git:
git clone <repo-url>
cd driver_monitor

# Or just ensure all four .py files are in the same folder
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `ultralytics` will automatically download the YOLOv8 nano weights  
> (`yolov8n.pt`, ~6 MB) on the **first run**.

### 4. (Optional) Verify your camera

```bash
python -c "import cv2; cap=cv2.VideoCapture(0); print('OK' if cap.isOpened() else 'FAIL'); cap.release()"
```

---

## ▶️ Running the System

### Default (webcam index 0)

```bash
python main.py
```

### Use a different camera

```bash
python main.py --source 1
```

### Run on a recorded video file

```bash
python main.py --source path/to/video.mp4
```

### Enable debug overlay (landmark dots + head-axis arrow)

```bash
python main.py --debug
```

### Combine flags

```bash
python main.py --source 1 --debug
```

**Press `Q` or `ESC` to quit.**

---

## ⚙️ Tuning Thresholds

Edit `config.py` to adjust sensitivity:

```python
# Drowsiness — lower EAR_THRESHOLD catches earlier signs
EAR_THRESHOLD     = 0.22   # 0.18–0.25 typical range
EAR_CONSEC_FRAMES = 20     # reduce for faster alerts

# Yawning
MAR_THRESHOLD     = 0.6    # 0.5–0.7
MAR_CONSEC_FRAMES = 15

# Head pose
HEAD_YAW_THRESHOLD   = 30  # degrees
HEAD_PITCH_THRESHOLD = 25  # degrees

# Phone detection confidence
PHONE_CONF_THRESHOLD = 0.45  # 0.3–0.6
```

---

## 🖥️ HUD Legend

```
┌─────────────────────────────────────────────────────────┐
│                      VIDEO FEED                         │
├──────────┬──────────────────────────┬───────────────────┤
│ [ALERT]  │ EAR  MAR  YAW   PITCH   │ FLAGS       n fps │
│          │                          │                   │
│ [STATUS] │  progress bar indicators │                   │
└──────────┴──────────────────────────┴───────────────────┘
```

**Status colours:**
- 🟢 `ALERT` — driver is attentive
- 🔴 `DROWSY` — eyes closed too long or yawning
- 🟡 `DISTRACTED` — head turned away from road
- 🟠 `PHONE DETECTED` — mobile phone visible in frame

---

## 🔊 Audio Alerts

| OS | Method |
|---|---|
| Windows | `winsound.Beep()` |
| macOS | `afplay /System/Library/Sounds/Funk.aiff` |
| Linux | `speaker-test` (ALSA) |

If audio fails silently it is non-critical — visual alerts still work.

---

## 📦 Dependencies

| Package | Purpose | Version |
|---|---|---|
| `opencv-python` | Frame capture & rendering | ≥ 4.8 |
| `mediapipe` | 468-point face mesh + head pose | ≥ 0.10 |
| `ultralytics` | YOLOv8 phone detection | ≥ 8.0 |
| `numpy` | Numerical operations | ≥ 1.24 |

---

## 🔬 How It Works

### Eye Aspect Ratio (EAR)
```
EAR = (||p2-p6|| + ||p3-p5||) / (2 × ||p1-p4||)
```
Six MediaPipe landmarks per eye form the Soukupová ratio.  
EAR ≈ 0.3 open · 0.0 closed. Alert fires below **0.22** for 20 frames.

### Mouth Aspect Ratio (MAR)
Same principle applied to 8 lip landmarks.  
MAR > **0.6** for 15 frames = yawn.

### Head Pose (solvePnP)
Six 3-D face model points are matched to 2-D MediaPipe landmarks via  
`cv2.solvePnP` → rotation matrix → Euler angles (yaw / pitch / roll).

### Phone Detection (YOLOv8)
COCO class **67** (`cell phone`) is extracted from YOLOv8 nano inference.  
Only detections with ≥ 45 % confidence for ≥ 10 consecutive frames trigger an alert.

---

## ⚠️ Troubleshooting

| Problem | Fix |
|---|---|
| Black screen / camera not found | Try `--source 1` or `--source 2` |
| `ModuleNotFoundError: mediapipe` | `pip install mediapipe` |
| YOLO weights not downloading | Check internet; set `YOLO_MODEL_NAME` to a local `.pt` path |
| No audio on Linux | Install ALSA: `sudo apt install alsa-utils` |
| Face not detected | Ensure good lighting; sit 40–80 cm from camera |

---

## 📄 License

MIT — free to use, modify, and distribute.
