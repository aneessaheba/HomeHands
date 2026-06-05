"""
hand_pose.py
────────────
Full arm skeleton tracking using YOLOv8-Pose.

Detects 17 COCO body keypoints per person per frame, then draws
the arm skeleton — shoulder → elbow → wrist — on both sides.

Why YOLOv8-Pose instead of MediaPipe?
  MediaPipe Holistic was trained on front-facing cameras and places
  shoulder/elbow landmarks on the chest/torso in top-down GoPro footage.
  YOLOv8-Pose generalises better to unusual camera angles.

COCO keypoints we use:
  5 = left_shoulder    6 = right_shoulder
  7 = left_elbow       8 = right_elbow
  9 = left_wrist      10 = right_wrist

Usage:
  Quick test (first 100 frames of WashingCup):
    python pipeline/hand_pose.py --test

  From pre-extracted frames folder:
    python pipeline/hand_pose.py --frames "assets/processed/frames/Cutting Banana"

  From video file:
    python pipeline/hand_pose.py assets/videos/WashingCup.mp4
"""

# ── Imports ───────────────────────────────────────────────────────────────────

import sys           # command-line arguments
import json          # write keypoint JSON
import time          # measure elapsed time
from pathlib import Path

import cv2           # read frames, draw shapes, save PNGs
import numpy as np   # array operations

from ultralytics import YOLO   # YOLOv8-Pose — replaces MediaPipe


# ── Config ────────────────────────────────────────────────────────────────────

# Model: yolov8s-pose.pt = "small" — good balance of speed + accuracy.
# Downloads automatically (~23 MB) on first run to ~/.cache/ultralytics/
YOLO_MODEL = "yolov8s-pose.pt"

# Only draw a keypoint if YOLO's confidence for that joint is above this.
# Raise it (e.g. 0.5) to draw fewer but more reliable joints.
KP_CONF_THRESHOLD = 0.35

# Output folders
ANNOTATED_ROOT = Path("assets/processed/annotated")
HAND_POSE_ROOT = Path("assets/processed/hand_pose")

# Test mode settings
TEST_VIDEO      = Path("assets/videos/Cutting Banana.mp4")
TEST_FRAMES_DIR = Path("assets/processed/frames/Cutting Banana")
TEST_OUTPUT_DIR = Path("assets/processed/holistic_test")
TEST_MAX_FRAMES = 100
TEST_SAVE_AT    = [1, 25, 50, 75, 100]


# ── COCO keypoint indices for arm joints ──────────────────────────────────────

# YOLOv8-Pose returns 17 keypoints in COCO order.
# We only care about the 6 arm joints below.
L_SHOULDER = 5    # left  shoulder
R_SHOULDER = 6    # right shoulder
L_ELBOW    = 7    # left  elbow
R_ELBOW    = 8    # right elbow
L_WRIST    = 9    # left  wrist
R_WRIST    = 10   # right wrist

# Which joints to connect with lines to form the arm skeleton.
# Each tuple: (joint_index_A, joint_index_B)
ARM_CONNECTIONS = [
    (L_SHOULDER, L_ELBOW),    # left  upper arm
    (L_ELBOW,    L_WRIST),    # left  forearm
    (R_SHOULDER, R_ELBOW),    # right upper arm
    (R_ELBOW,    R_WRIST),    # right forearm
]

# Colors (BGR — OpenCV uses Blue, Green, Red order)
COLOR_LEFT  = (60,  60,  230)   # red-ish   — left  arm
COLOR_RIGHT = (230, 60,  60 )   # blue-ish  — right arm
COLOR_JOINT = (255, 255, 255)   # white     — joint dots


# ── Detection + drawing ───────────────────────────────────────────────────────

def detect_and_draw(model, frame: np.ndarray) -> tuple:
    """
    Run YOLOv8-Pose on one frame, draw the arm skeleton, return keypoint data.

    model   — loaded YOLO model
    frame   — BGR image (NumPy array from cv2.imread / cap.read)

    Returns:
      annotated  — BGR image with skeleton drawn on top
      kp_data    — list of dicts, one per detected person, with arm keypoints
    """

    annotated = frame.copy()   # don't modify the original

    # Run YOLO — verbose=False suppresses the per-frame console output
    results = model(frame, verbose=False)

    kp_data = []   # will collect one dict per person detected

    # results[0] = detection result for this frame
    # .keypoints may be None if nothing was detected
    if results[0].keypoints is None:
        return annotated, kp_data

    # .xy  → pixel coords,  shape: (num_persons, 17, 2)
    # .conf → confidence,   shape: (num_persons, 17)
    kps  = results[0].keypoints.xy.cpu().numpy()
    conf = results[0].keypoints.conf.cpu().numpy()

    # Loop over every person detected in this frame
    for person_idx in range(len(kps)):

        person_kps  = kps[person_idx]    # shape (17, 2) — (x,y) per joint
        person_conf = conf[person_idx]   # shape (17,)   — confidence per joint

        # Helper: get pixel coords for one joint if confidence is high enough
        def pt(idx):
            """Return (int_x, int_y) if confident, else None."""
            if person_conf[idx] < KP_CONF_THRESHOLD:
                return None   # too uncertain — skip this joint
            x, y = person_kps[idx]
            return (int(x), int(y))

        # Extract the 6 arm joints
        pts = {
            "left_shoulder":  pt(L_SHOULDER),
            "left_elbow":     pt(L_ELBOW),
            "left_wrist":     pt(L_WRIST),
            "right_shoulder": pt(R_SHOULDER),
            "right_elbow":    pt(R_ELBOW),
            "right_wrist":    pt(R_WRIST),
        }

        # ── Draw arm bone lines ───────────────────────────────────────────────
        for a_idx, b_idx in ARM_CONNECTIONS:
            p1 = pt(a_idx)   # start joint
            p2 = pt(b_idx)   # end joint
            if p1 and p2:    # only draw if both joints are confident
                # Choose color based on which arm
                color = COLOR_LEFT if a_idx in (L_SHOULDER, L_ELBOW) else COLOR_RIGHT
                cv2.line(annotated, p1, p2, color, 5, cv2.LINE_AA)   # thick bone line

        # ── Draw joint dots ───────────────────────────────────────────────────
        for joint_name, p in pts.items():
            if p:
                color = COLOR_LEFT if "left" in joint_name else COLOR_RIGHT
                cv2.circle(annotated, p, 12, color,       -1, cv2.LINE_AA)  # colored outer
                cv2.circle(annotated, p,  6, COLOR_JOINT, -1, cv2.LINE_AA)  # white center

        # ── Store keypoint data for JSON ──────────────────────────────────────
        # Only store joints that were actually detected (confidence above threshold)
        kp_entry = {}
        for name, joint_idx in {
            "left_shoulder":  L_SHOULDER,
            "left_elbow":     L_ELBOW,
            "left_wrist":     L_WRIST,
            "right_shoulder": R_SHOULDER,
            "right_elbow":    R_ELBOW,
            "right_wrist":    R_WRIST,
        }.items():
            if person_conf[joint_idx] >= KP_CONF_THRESHOLD:
                x, y = person_kps[joint_idx]
                kp_entry[name] = {
                    "px":   int(x),
                    "py":   int(y),
                    "conf": round(float(person_conf[joint_idx]), 4),
                }

        kp_data.append({
            "person_id":  person_idx,
            "keypoints":  kp_entry,
            "arms_detected": {
                "left":  pt(L_SHOULDER) is not None or pt(L_WRIST) is not None,
                "right": pt(R_SHOULDER) is not None or pt(R_WRIST) is not None,
            },
        })

    return annotated, kp_data


# ── Test mode ─────────────────────────────────────────────────────────────────

def run_test():
    """
    Process only the first 100 frames (from pre-extracted folder or video)
    and save 5 sample PNGs to assets/processed/holistic_test/.
    """

    print(f"\n{'=' * 60}")
    print(f"  YOLOv8-Pose  —  TEST MODE")
    print(f"  Frames : first {TEST_MAX_FRAMES}")
    print(f"  Saving : {TEST_SAVE_AT}")
    print(f"{'=' * 60}\n")

    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"  Loading YOLOv8-Pose model ({YOLO_MODEL}) ...")
    model = YOLO(YOLO_MODEL)
    print(f"  Model ready.\n")

    # Prefer pre-extracted frames; fall back to video
    if TEST_FRAMES_DIR.exists():
        frame_files = sorted(
            list(TEST_FRAMES_DIR.glob("frame_*.jpg")) +
            list(TEST_FRAMES_DIR.glob("frame_*.png"))
        )[:TEST_MAX_FRAMES]
        source = f"frames folder ({TEST_FRAMES_DIR.name})"
    elif TEST_VIDEO.exists():
        frame_files = None
        source = f"video ({TEST_VIDEO.name})"
    else:
        print(f"[ERROR] Neither frames folder nor video found.")
        sys.exit(1)

    print(f"  Source: {source}\n")

    cap      = None
    frame_id = 0
    start    = time.time()
    saved    = 0

    # Generator that yields (frame_id, frame_bgr)
    def frame_source():
        nonlocal cap
        if frame_files:
            for fid, path in enumerate(frame_files):
                img = cv2.imread(str(path))
                if img is not None:
                    yield fid, img
        else:
            cap = cv2.VideoCapture(str(TEST_VIDEO))
            fid = 0
            while fid < TEST_MAX_FRAMES:
                ret, img = cap.read()
                if not ret:
                    break
                yield fid, img
                fid += 1
            cap.release()

    for frame_id, frame in frame_source():
        display_num = frame_id + 1   # 1-indexed for display

        annotated, kp_data = detect_and_draw(model, frame)

        if display_num in TEST_SAVE_AT:
            # Add frame number label
            for dx, dy in [(-2,-2),(-2,2),(2,-2),(2,2)]:
                cv2.putText(annotated, f"Frame {display_num}", (20+dx, 60+dy),
                            cv2.FONT_HERSHEY_DUPLEX, 1.5, (0,0,0), 4, cv2.LINE_AA)
            cv2.putText(annotated, f"Frame {display_num}", (20, 60),
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)

            out = TEST_OUTPUT_DIR / f"frame_{str(display_num).zfill(3)}.png"
            cv2.imwrite(str(out), annotated)
            saved += 1

            yes_no = lambda b: "✅ Yes" if b else "❌ No"
            persons = len(kp_data)
            left  = any(p["arms_detected"]["left"]  for p in kp_data)
            right = any(p["arms_detected"]["right"] for p in kp_data)
            print(f"  Frame {display_num:>3}  →  {out.name}")
            print(f"          Persons detected : {persons}")
            print(f"          Left  arm        : {yes_no(left)}")
            print(f"          Right arm        : {yes_no(right)}\n")

    elapsed = time.time() - start
    print(f"  {'─' * 50}")
    print(f"  TEST COMPLETE  —  {saved} frames saved  ({elapsed:.1f}s)")
    print(f"  Output → {TEST_OUTPUT_DIR}/\n")


# ── Frames folder mode ────────────────────────────────────────────────────────

def process_frames(frames_dir: str, fps: float = 29.97):
    """
    Read pre-extracted frames from a folder, detect + draw arm skeleton,
    save annotated PNGs and keypoint JSON.
    """

    frames_dir = Path(frames_dir)
    if not frames_dir.exists():
        print(f"[ERROR] Frames folder not found: {frames_dir}")
        sys.exit(1)

    clip_name   = frames_dir.name
    frame_files = sorted(
        list(frames_dir.glob("frame_*.jpg")) +
        list(frames_dir.glob("frame_*.png"))
    )
    total_frames = len(frame_files)

    if not frame_files:
        print(f"[ERROR] No frame files in {frames_dir}")
        sys.exit(1)

    first  = cv2.imread(str(frame_files[0]))
    height, width = first.shape[:2]

    print(f"\n{'=' * 60}")
    print(f"  YOLOv8-Pose Arm Tracking")
    print(f"  Clip       : {clip_name}")
    print(f"  Source     : {frames_dir}")
    print(f"{'=' * 60}")
    print(f"  Resolution : {width} x {height}")
    print(f"  Frames     : {total_frames}")
    print(f"  Model      : {YOLO_MODEL}\n")

    annotated_dir = ANNOTATED_ROOT / clip_name
    annotated_dir.mkdir(parents=True, exist_ok=True)
    HAND_POSE_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"  Loading YOLOv8-Pose model ...")
    model = YOLO(YOLO_MODEL)
    print(f"  Model ready.\n")

    frames_data = []
    frame_id    = 0
    arms_left   = 0
    arms_right  = 0
    start_time  = time.time()

    for frame_file in frame_files:

        frame = cv2.imread(str(frame_file))
        if frame is None:
            frame_id += 1
            continue

        annotated, kp_data = detect_and_draw(model, frame)

        # Save annotated frame
        cv2.imwrite(str(annotated_dir / f"frame_{str(frame_id).zfill(6)}.png"), annotated)

        # Count detections
        has_left  = any(p["arms_detected"]["left"]  for p in kp_data)
        has_right = any(p["arms_detected"]["right"] for p in kp_data)
        if has_left:  arms_left  += 1
        if has_right: arms_right += 1

        frames_data.append({
            "frame_id":      frame_id,
            "timestamp_sec": round(frame_id / fps, 4),
            "persons":       kp_data,
        })

        if frame_id % 100 == 0:
            elapsed = time.time() - start_time
            pct     = (frame_id / total_frames) * 100
            print(
                f"  Frame {frame_id:>6} / {total_frames}"
                f"  ({pct:5.1f}%)"
                f"  |  L:{arms_left}  R:{arms_right}"
                f"  |  {elapsed:.1f}s"
            )

        frame_id += 1

    # Save JSON
    json_path = HAND_POSE_ROOT / f"{clip_name}.json"
    with open(json_path, "w") as f:
        json.dump({
            "clip_name":    clip_name,
            "model":        YOLO_MODEL,
            "total_frames": frame_id,
            "fps":          fps,
            "resolution":   {"width": width, "height": height},
            "frames":       frames_data,
        }, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\n{'─' * 60}")
    print(f"  SUMMARY  —  {clip_name}")
    print(f"{'─' * 60}")
    print(f"  Total frames  : {frame_id:,}")
    print(f"  Left arm      : {arms_left:,}  ({arms_left/frame_id*100:.1f}%)")
    print(f"  Right arm     : {arms_right:,}  ({arms_right/frame_id*100:.1f}%)")
    print(f"  Time taken    : {elapsed/60:.1f} min")
    print(f"\n  Frames  →  {annotated_dir}/")
    print(f"  JSON    →  {json_path}\n")


# ── Full video mode ───────────────────────────────────────────────────────────

def process_video(video_path: str):
    """Read directly from a video file and process every frame."""

    video_path = Path(video_path)
    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}")
        sys.exit(1)

    clip_name = video_path.stem

    cap          = cv2.VideoCapture(str(video_path))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n{'=' * 60}")
    print(f"  YOLOv8-Pose Arm Tracking  (from video)")
    print(f"  Clip : {clip_name}  |  {width}x{height}  |  {fps:.2f}fps  |  {total_frames} frames")
    print(f"{'=' * 60}\n")

    annotated_dir = ANNOTATED_ROOT / clip_name
    annotated_dir.mkdir(parents=True, exist_ok=True)
    HAND_POSE_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"  Loading YOLOv8-Pose model ...")
    model = YOLO(YOLO_MODEL)
    print(f"  Model ready.\n")

    frames_data = []
    frame_id    = 0
    arms_left   = 0
    arms_right  = 0
    start_time  = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, kp_data = detect_and_draw(model, frame)
        cv2.imwrite(str(annotated_dir / f"frame_{str(frame_id).zfill(6)}.png"), annotated)

        has_left  = any(p["arms_detected"]["left"]  for p in kp_data)
        has_right = any(p["arms_detected"]["right"] for p in kp_data)
        if has_left:  arms_left  += 1
        if has_right: arms_right += 1

        frames_data.append({
            "frame_id":      frame_id,
            "timestamp_sec": round(frame_id / fps, 4),
            "persons":       kp_data,
        })

        if frame_id % 100 == 0:
            elapsed = time.time() - start_time
            pct     = (frame_id / total_frames) * 100
            print(f"  Frame {frame_id:>6} / {total_frames}  ({pct:5.1f}%)  "
                  f"|  L:{arms_left}  R:{arms_right}  |  {elapsed:.1f}s")

        frame_id += 1

    cap.release()

    json_path = HAND_POSE_ROOT / f"{clip_name}.json"
    with open(json_path, "w") as f:
        json.dump({
            "clip_name":    clip_name,
            "model":        YOLO_MODEL,
            "total_frames": frame_id,
            "fps":          fps,
            "resolution":   {"width": width, "height": height},
            "frames":       frames_data,
        }, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\n  DONE  —  {frame_id:,} frames  |  {elapsed/60:.1f} min")
    print(f"  Frames  →  {annotated_dir}/")
    print(f"  JSON    →  {json_path}\n")


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
        print("  Quick test   : python pipeline/hand_pose.py --test")
        print("  From frames  : python pipeline/hand_pose.py --frames <folder>")
        print("  From video   : python pipeline/hand_pose.py <video_path>")
        sys.exit(1)
