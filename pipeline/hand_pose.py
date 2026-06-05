"""
hand_pose.py
────────────
Accurate wrist tracking using MediaPipe HandLandmarker (Tasks API).

MediaPipe's hand model works well from any camera angle — including
top-down GoPro footage — because it was trained specifically on hands,
not full-body poses. It detects up to 2 hands and returns the wrist
position (landmark 0) reliably.

For each frame:
  - Detect up to 2 hands
  - Get the WRIST landmark (index 0) for each hand
  - Draw a clean circle at each wrist
  - Label "L" (red) for left hand, "R" (blue) for right hand
  - Save annotated PNG + keypoint JSON

Usage:
  From pre-extracted frames:
    python pipeline/hand_pose.py --frames "assets/processed/frames/Cutting Banana"

  From video:
    python pipeline/hand_pose.py assets/videos/WashingCup.mp4

  Quick test (saves 5 sample frames):
    python pipeline/hand_pose.py --test
"""

# ── Imports ───────────────────────────────────────────────────────────────────

import sys
import json
import time
from pathlib import Path

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision


# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH = Path("assets/models/hand_landmarker.task")   # 7.5 MB, already downloaded

ANNOTATED_ROOT = Path("assets/processed/annotated")
HAND_POSE_ROOT = Path("assets/processed/hand_pose")

# Test mode
TEST_FRAMES_DIR = Path("assets/processed/frames/Cutting Banana")
TEST_VIDEO      = Path("assets/videos/Cutting Banana.mp4")
TEST_OUTPUT_DIR = Path("assets/processed/holistic_test")
TEST_MAX_FRAMES = 100
TEST_SAVE_AT    = [1, 25, 50, 75, 100]

# Drawing sizes (scaled to 4K footage)
DOT_RADIUS      = 28     # outer circle radius
DOT_INNER       = 14     # inner white center radius
LINE_THICKNESS  = 6      # label outline thickness

# Colors (BGR)
COLOR_LEFT  = (50,  50,  230)   # red   — left  hand
COLOR_RIGHT = (230, 80,  50 )   # blue  — right hand
COLOR_WHITE = (255, 255, 255)   # white center dot


# ── MediaPipe setup ───────────────────────────────────────────────────────────

def load_detector():
    """Create and return the MediaPipe HandLandmarker detector."""
    if not MODEL_PATH.exists():
        print(f"[ERROR] Hand landmarker model not found: {MODEL_PATH}")
        print("  Download with:")
        print("  curl -L -o assets/models/hand_landmarker.task \\")
        print("    https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
              "hand_landmarker/float16/latest/hand_landmarker.task")
        sys.exit(1)

    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        num_hands=2,                           # detect up to 2 hands
        min_hand_detection_confidence=0.5,     # initial detection threshold
        min_hand_presence_confidence=0.5,      # tracking threshold
        min_tracking_confidence=0.5,           # frame-to-frame tracking
        running_mode=mp_vision.RunningMode.IMAGE,   # one frame at a time
    )
    return mp_vision.HandLandmarker.create_from_options(options)


# ── Core: detect wrists and draw ─────────────────────────────────────────────

def process_frame(detector, frame: np.ndarray, width: int, height: int) -> tuple:
    """
    Detect hands in one frame and draw a dot at each wrist.

    Returns:
      annotated  — BGR image with wrist dots drawn
      wrist_data — list of dicts, one per detected hand:
                   {"label": "Left"|"Right", "confidence": float,
                    "px": int, "py": int}
    """

    annotated  = frame.copy()
    wrist_data = []

    # Convert BGR → RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # Run detection
    result = detector.detect(mp_image)

    # Pair each hand's landmarks with its handedness classification
    for landmarks, handedness in zip(
        result.hand_landmarks,    # list of 21-landmark lists
        result.handedness,        # list of classification results
    ):
        # Handedness: "Left" or "Right" (from MediaPipe's perspective,
        # which is mirrored vs. real world for front-facing cameras —
        # but for GoPro top-down this is less of an issue)
        label      = handedness[0].category_name   # "Left" or "Right"
        confidence = round(handedness[0].score, 4)

        # WRIST is landmark index 0 — the base of the palm
        wrist_lm = landmarks[0]
        px = int(wrist_lm.x * width)    # convert normalised (0-1) → pixels
        py = int(wrist_lm.y * height)

        # Choose color based on hand label
        color = COLOR_LEFT if label == "Left" else COLOR_RIGHT
        side  = "L" if label == "Left" else "R"

        # ── Draw wrist dot ────────────────────────────────────────────────────
        cv2.circle(annotated, (px, py), DOT_RADIUS,     color,       -1, cv2.LINE_AA)
        cv2.circle(annotated, (px, py), DOT_INNER,      COLOR_WHITE, -1, cv2.LINE_AA)
        cv2.circle(annotated, (px, py), DOT_RADIUS + 3, (0,0,0),      3, cv2.LINE_AA)  # black ring

        # ── Draw label "L" or "R" next to the dot ────────────────────────────
        lx = px + DOT_RADIUS + 10   # place label to the right of the dot
        ly = py + 14                # vertically centered

        # Black outline
        for dx, dy in [(-3,-3),(-3,3),(3,-3),(3,3),(0,-3),(0,3),(-3,0),(3,0)]:
            cv2.putText(annotated, side, (lx+dx, ly+dy),
                        cv2.FONT_HERSHEY_DUPLEX, 1.8, (0,0,0), 5, cv2.LINE_AA)
        # Colored label on top
        cv2.putText(annotated, side, (lx, ly),
                    cv2.FONT_HERSHEY_DUPLEX, 1.8, color, 2, cv2.LINE_AA)

        wrist_data.append({
            "label":      label,
            "confidence": confidence,
            "px":         px,
            "py":         py,
            "x_norm":     round(wrist_lm.x, 6),   # normalised x  (0-1)
            "y_norm":     round(wrist_lm.y, 6),   # normalised y  (0-1)
        })

    return annotated, wrist_data


# ── Test mode ─────────────────────────────────────────────────────────────────

def run_test():
    """Process 100 frames and save 5 sample PNGs to check quality."""

    print(f"\n{'=' * 60}")
    print(f"  Wrist Tracking  —  TEST MODE  (MediaPipe HandLandmarker)")
    print(f"  Frames : first {TEST_MAX_FRAMES}")
    print(f"  Saving : frames {TEST_SAVE_AT}")
    print(f"{'=' * 60}\n")

    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detector = load_detector()

    # Use pre-extracted frames if available, otherwise open video
    if TEST_FRAMES_DIR.exists():
        frame_files = sorted(
            list(TEST_FRAMES_DIR.glob("frame_*.jpg")) +
            list(TEST_FRAMES_DIR.glob("frame_*.png"))
        )[:TEST_MAX_FRAMES]
        first   = cv2.imread(str(frame_files[0]))
        height, width = first.shape[:2]
        source_type = "frames"
    else:
        frame_files = None
        source_type = "video"

    start = time.time()

    def frames():
        if source_type == "frames":
            for i, p in enumerate(frame_files):
                img = cv2.imread(str(p))
                if img is not None:
                    yield i, img
        else:
            cap = cv2.VideoCapture(str(TEST_VIDEO))
            h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            nonlocal height, width
            height, width = h, w
            fid = 0
            while fid < TEST_MAX_FRAMES:
                ret, img = cap.read()
                if not ret: break
                yield fid, img
                fid += 1
            cap.release()

    height = width = 0

    for frame_id, frame in frames():
        if height == 0:
            height, width = frame.shape[:2]

        display = frame_id + 1
        annotated, wrist_data = process_frame(detector, frame, width, height)

        if display in TEST_SAVE_AT:
            # Frame number overlay
            for dx, dy in [(-2,-2),(-2,2),(2,-2),(2,2)]:
                cv2.putText(annotated, f"Frame {display}", (30+dx, 70+dy),
                            cv2.FONT_HERSHEY_DUPLEX, 2.0, (0,0,0), 6, cv2.LINE_AA)
            cv2.putText(annotated, f"Frame {display}", (30, 70),
                        cv2.FONT_HERSHEY_DUPLEX, 2.0, (255,255,255), 3, cv2.LINE_AA)

            out = TEST_OUTPUT_DIR / f"frame_{str(display).zfill(3)}.png"
            cv2.imwrite(str(out), annotated)

            yes_no = lambda b: "✅ Yes" if b else "❌ No"
            left  = any(w["label"] == "Left"  for w in wrist_data)
            right = any(w["label"] == "Right" for w in wrist_data)
            print(f"  Frame {display:>3}  →  {out.name}")
            print(f"          Hands found  : {len(wrist_data)}")
            print(f"          Left  wrist  : {yes_no(left)}")
            print(f"          Right wrist  : {yes_no(right)}\n")

    detector.close()
    print(f"  TEST COMPLETE  ({time.time()-start:.1f}s)")
    print(f"  Output → {TEST_OUTPUT_DIR}/\n")


# ── Frames folder mode ────────────────────────────────────────────────────────

def process_frames(frames_dir: str, fps: float = 29.97):
    """Read pre-extracted frames, detect wrists, save annotated PNGs + JSON."""

    frames_dir  = Path(frames_dir)
    clip_name   = frames_dir.name
    frame_files = sorted(
        list(frames_dir.glob("frame_*.jpg")) +
        list(frames_dir.glob("frame_*.png"))
    )

    if not frame_files:
        print(f"[ERROR] No frames in {frames_dir}")
        sys.exit(1)

    first         = cv2.imread(str(frame_files[0]))
    height, width = first.shape[:2]
    total_frames  = len(frame_files)

    print(f"\n{'=' * 60}")
    print(f"  Wrist Tracking  (MediaPipe HandLandmarker)")
    print(f"  Clip   : {clip_name}")
    print(f"  Source : {frames_dir}")
    print(f"  Frames : {total_frames}  |  {width}x{height}")
    print(f"{'=' * 60}\n")

    annotated_dir = ANNOTATED_ROOT / clip_name
    annotated_dir.mkdir(parents=True, exist_ok=True)
    HAND_POSE_ROOT.mkdir(parents=True, exist_ok=True)

    detector    = load_detector()
    frames_data = []
    frame_id    = 0
    left_count  = 0
    right_count = 0
    start_time  = time.time()

    for frame_file in frame_files:

        frame = cv2.imread(str(frame_file))
        if frame is None:
            frame_id += 1
            continue

        annotated, wrist_data = process_frame(detector, frame, width, height)

        cv2.imwrite(str(annotated_dir / f"frame_{str(frame_id).zfill(6)}.png"), annotated)

        has_left  = any(w["label"] == "Left"  for w in wrist_data)
        has_right = any(w["label"] == "Right" for w in wrist_data)
        if has_left:  left_count  += 1
        if has_right: right_count += 1

        frames_data.append({
            "frame_id":      frame_id,
            "timestamp_sec": round(frame_id / fps, 4),
            "hands_found":   len(wrist_data),
            "wrists":        wrist_data,
        })

        if frame_id % 100 == 0:
            elapsed = time.time() - start_time
            pct     = frame_id / total_frames * 100
            print(
                f"  Frame {frame_id:>6} / {total_frames}"
                f"  ({pct:5.1f}%)"
                f"  |  L:{left_count}  R:{right_count}"
                f"  |  {elapsed:.1f}s"
            )

        frame_id += 1

    detector.close()

    json_path = HAND_POSE_ROOT / f"{clip_name}.json"
    with open(json_path, "w") as f:
        json.dump({
            "clip_name":    clip_name,
            "total_frames": frame_id,
            "fps":          fps,
            "resolution":   {"width": width, "height": height},
            "frames":       frames_data,
        }, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\n{'─' * 60}")
    print(f"  DONE  —  {clip_name}")
    print(f"{'─' * 60}")
    print(f"  Total frames  : {frame_id:,}")
    print(f"  Left  wrist   : {left_count:,}  ({left_count/frame_id*100:.1f}%)")
    print(f"  Right wrist   : {right_count:,}  ({right_count/frame_id*100:.1f}%)")
    print(f"  Time          : {elapsed/60:.1f} min")
    print(f"\n  Frames  →  {annotated_dir}/")
    print(f"  JSON    →  {json_path}\n")


# ── Video file mode ───────────────────────────────────────────────────────────

def process_video(video_path: str):
    """Read directly from video file and process every frame."""

    video_path = Path(video_path)
    if not video_path.exists():
        print(f"[ERROR] Not found: {video_path}")
        sys.exit(1)

    clip_name    = video_path.stem
    cap          = cv2.VideoCapture(str(video_path))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n{'=' * 60}")
    print(f"  Wrist Tracking  (from video)")
    print(f"  {clip_name}  |  {width}x{height}  |  {total_frames} frames")
    print(f"{'=' * 60}\n")

    annotated_dir = ANNOTATED_ROOT / clip_name
    annotated_dir.mkdir(parents=True, exist_ok=True)
    HAND_POSE_ROOT.mkdir(parents=True, exist_ok=True)

    detector    = load_detector()
    frames_data = []
    frame_id    = 0
    left_count  = 0
    right_count = 0
    start_time  = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, wrist_data = process_frame(detector, frame, width, height)
        cv2.imwrite(str(annotated_dir / f"frame_{str(frame_id).zfill(6)}.png"), annotated)

        if any(w["label"] == "Left"  for w in wrist_data): left_count  += 1
        if any(w["label"] == "Right" for w in wrist_data): right_count += 1

        frames_data.append({
            "frame_id":      frame_id,
            "timestamp_sec": round(frame_id / fps, 4),
            "hands_found":   len(wrist_data),
            "wrists":        wrist_data,
        })

        if frame_id % 100 == 0:
            elapsed = time.time() - start_time
            pct     = frame_id / total_frames * 100
            print(f"  Frame {frame_id:>6} / {total_frames}  ({pct:5.1f}%)  "
                  f"|  L:{left_count}  R:{right_count}  |  {elapsed:.1f}s")

        frame_id += 1

    cap.release()
    detector.close()

    json_path = HAND_POSE_ROOT / f"{clip_name}.json"
    with open(json_path, "w") as f:
        json.dump({
            "clip_name":    clip_name,
            "total_frames": frame_id,
            "fps":          fps,
            "resolution":   {"width": width, "height": height},
            "frames":       frames_data,
        }, f, indent=2)

    print(f"\n  DONE  —  {frame_id:,} frames  in  {(time.time()-start_time)/60:.1f} min")
    print(f"  Frames → {annotated_dir}/  |  JSON → {json_path}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    if "--test" in sys.argv:
        run_test()

    elif "--frames" in sys.argv:
        idx = sys.argv.index("--frames")
        if idx + 1 >= len(sys.argv):
            print("Usage: python pipeline/hand_pose.py --frames <folder>")
            sys.exit(1)
        process_frames(sys.argv[idx + 1])

    elif len(sys.argv) == 2:
        process_video(sys.argv[1])

    else:
        print("Usage:")
        print("  Test   : python pipeline/hand_pose.py --test")
        print("  Frames : python pipeline/hand_pose.py --frames <folder>")
        print("  Video  : python pipeline/hand_pose.py <video_path>")
        sys.exit(1)
