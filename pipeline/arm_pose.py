"""
arm_pose.py
───────────
Combined hand + arm tracking for egocentric (head-mounted) video.

  • MediaPipe HandLandmarker  — 21-point hand skeleton (red = left, blue = right)
  • YOLOv8 pose              — wrist → elbow arm segment (green)

YOLO keypoints used (COCO format):
  7  = left  elbow
  8  = right elbow
  9  = left  wrist
  10 = right wrist

Only the highest-confidence YOLO detection is used per frame.
Keypoints at (0, 0) are treated as invisible and skipped.

Usage:
  Test (saves 5 sample frames):
    python pipeline/arm_pose.py --test

  From pre-extracted frames:
    python pipeline/arm_pose.py --frames "assets/processed/frames/Cutting Banana"

  From video:
    python pipeline/arm_pose.py assets/videos/WashingCup.mp4

  From video + save output video:
    python pipeline/arm_pose.py assets/videos/WashingCup.mp4 --output-video
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
from ultralytics import YOLO


# ── Config ────────────────────────────────────────────────────────────────────

MEDIAPIPE_MODEL = Path("assets/models/hand_landmarker.task")
YOLO_MODEL      = "yolov8n-pose.pt"   # downloaded automatically if missing

ANNOTATED_ROOT  = Path("assets/processed/annotated")
ARM_POSE_ROOT   = Path("assets/processed/arm_pose")

# Test mode
TEST_FRAMES_DIR = Path("assets/processed/frames/Cutting Banana")
TEST_VIDEO      = Path("assets/videos/Cutting Banana.mp4")
TEST_OUTPUT_DIR = Path("assets/processed/arm_pose_test")
TEST_MAX_FRAMES = 100
TEST_SAVE_AT    = [1, 25, 50, 75, 100]

# Drawing — matched to hand_pose.py scale (4K footage)
DOT_RADIUS  = 18
DOT_INNER   = 8
BONE_WIDTH  = 5

# Colors (BGR)
COLOR_LEFT   = (50,  50,  230)    # red   — MediaPipe left hand
COLOR_RIGHT  = (230, 80,  50 )    # blue  — MediaPipe right hand
COLOR_ARM_LEFT  = (230, 80,  50 )    # blue — YOLO left arm (mirrored)
COLOR_ARM_RIGHT = (50,  50,  230)    # red  — YOLO right arm (mirrored)
COLOR_WHITE  = (255, 255, 255)    # white dot centers

# MediaPipe hand connections (21 landmarks)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

LANDMARK_NAMES = [
    "WRIST","THUMB_CMC","THUMB_MCP","THUMB_IP","THUMB_TIP",
    "INDEX_MCP","INDEX_PIP","INDEX_DIP","INDEX_TIP",
    "MIDDLE_MCP","MIDDLE_PIP","MIDDLE_DIP","MIDDLE_TIP",
    "RING_MCP","RING_PIP","RING_DIP","RING_TIP",
    "PINKY_MCP","PINKY_PIP","PINKY_DIP","PINKY_TIP",
]

# YOLO COCO keypoint indices we care about
YOLO_LEFT_ELBOW  = 7
YOLO_RIGHT_ELBOW = 8
YOLO_LEFT_WRIST  = 9
YOLO_RIGHT_WRIST = 10


# ── Model loaders ─────────────────────────────────────────────────────────────

def load_mediapipe():
    """Load MediaPipe HandLandmarker."""
    if not MEDIAPIPE_MODEL.exists():
        print(f"[ERROR] MediaPipe model not found: {MEDIAPIPE_MODEL}")
        print("  Download with:")
        print("  curl -L -o assets/models/hand_landmarker.task \\")
        print("    https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
              "hand_landmarker/float16/latest/hand_landmarker.task")
        sys.exit(1)

    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(MEDIAPIPE_MODEL)),
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def load_yolo():
    """Load YOLOv8 pose model."""
    return YOLO(YOLO_MODEL)


# ── YOLO arm helpers ──────────────────────────────────────────────────────────

def is_valid_kp(pt):
    """Return True if keypoint is not a zero/missing point."""
    return not (pt[0] < 1.0 and pt[1] < 1.0)


def draw_arm_dot(img, pt, color):
    """Draw a filled dot with white center at pt."""
    cv2.circle(img, pt, DOT_RADIUS, color,       -1, cv2.LINE_AA)
    cv2.circle(img, pt, DOT_INNER,  COLOR_WHITE, -1, cv2.LINE_AA)


def extract_arm_keypoints(yolo_results, width, height):
    """
    From YOLO results pick the highest-confidence detection and extract
    wrist + elbow pixel coordinates.

    Returns dict:
      {
        "left_wrist":  (x, y) or None,
        "left_elbow":  (x, y) or None,
        "right_wrist": (x, y) or None,
        "right_elbow": (x, y) or None,
      }
    """
    arm_kps = {
        "left_wrist":  None,
        "left_elbow":  None,
        "right_wrist": None,
        "right_elbow": None,
    }

    best_result = yolo_results[0]
    if best_result.keypoints is None or len(best_result.keypoints.xy) == 0:
        return arm_kps

    # Pick the detection with the highest box confidence
    boxes = best_result.boxes
    if boxes is None or len(boxes.conf) == 0:
        return arm_kps

    best_idx = int(boxes.conf.argmax())
    kps = best_result.keypoints.xy[best_idx]   # shape [17, 2]

    def to_px(idx):
        pt = (float(kps[idx][0]), float(kps[idx][1]))
        if is_valid_kp(pt):
            return (int(pt[0]), int(pt[1]))
        return None

    arm_kps["left_wrist"]  = to_px(YOLO_LEFT_WRIST)
    arm_kps["left_elbow"]  = to_px(YOLO_LEFT_ELBOW)
    arm_kps["right_wrist"] = to_px(YOLO_RIGHT_WRIST)
    arm_kps["right_elbow"] = to_px(YOLO_RIGHT_ELBOW)

    return arm_kps


def draw_arm_segments(img, arm_kps):
    """
    Draw wrist→elbow lines and dots for both arms.
    Only draws a segment if both endpoints are valid.
    """
    pairs = [
        ("left_wrist",  "left_elbow",  COLOR_ARM_LEFT),
        ("right_wrist", "right_elbow", COLOR_ARM_RIGHT),
    ]

    for wrist_key, elbow_key, color in pairs:
        w = arm_kps[wrist_key]
        e = arm_kps[elbow_key]

        if w is not None and e is not None:
            cv2.line(img, w, e, color, BONE_WIDTH, cv2.LINE_AA)
        if w is not None:
            draw_arm_dot(img, w, color)
        if e is not None:
            draw_arm_dot(img, e, color)


# ── MediaPipe hand helpers ────────────────────────────────────────────────────

def draw_hand_skeleton(img, result, width, height):
    """
    Draw 21-point hand skeletons from MediaPipe result.
    Returns list of wrist data dicts for JSON.
    """
    wrist_data = []

    for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        label      = handedness[0].category_name
        confidence = round(handedness[0].score, 4)
        color      = COLOR_LEFT if label == "Left" else COLOR_RIGHT
        side       = "L" if label == "Left" else "R"

        pts = [
            (int(lm.x * width), int(lm.y * height))
            for lm in landmarks
        ]

        # Bone lines
        for a, b in HAND_CONNECTIONS:
            cv2.line(img, pts[a], pts[b], color, BONE_WIDTH, cv2.LINE_AA)

        # Joint dots
        for px, py in pts:
            cv2.circle(img, (px, py), DOT_RADIUS, color,       -1, cv2.LINE_AA)
            cv2.circle(img, (px, py), DOT_INNER,  COLOR_WHITE, -1, cv2.LINE_AA)

        # L / R label at wrist
        wx, wy = pts[0]
        lx, ly = wx + DOT_RADIUS + 10, wy + 14
        for dx, dy in [(-3,-3),(-3,3),(3,-3),(3,3)]:
            cv2.putText(img, side, (lx+dx, ly+dy),
                        cv2.FONT_HERSHEY_DUPLEX, 1.8, (0,0,0), 5, cv2.LINE_AA)
        cv2.putText(img, side, (lx, ly),
                    cv2.FONT_HERSHEY_DUPLEX, 1.8, color, 2, cv2.LINE_AA)

        # Collect keypoints for JSON
        keypoints = {}
        for i, (lm, (px, py)) in enumerate(zip(landmarks, pts)):
            keypoints[LANDMARK_NAMES[i]] = {
                "px": px, "py": py,
                "x":  round(lm.x, 6),
                "y":  round(lm.y, 6),
                "z":  round(lm.z, 6),
            }

        wrist_data.append({
            "label":      label,
            "confidence": confidence,
            "px":         pts[0][0],
            "py":         pts[0][1],
            "keypoints":  keypoints,
        })

    return wrist_data


# ── Core: process one frame ───────────────────────────────────────────────────

def process_frame(mp_detector, yolo_model, frame: np.ndarray, width: int, height: int) -> tuple:
    """
    Run both MediaPipe and YOLO on one frame.

    Returns:
      annotated  — BGR image with hand skeleton + arm segments drawn
      frame_data — dict with wrist_data (MediaPipe) and arm_keypoints (YOLO)
    """
    annotated = frame.copy()

    # ── MediaPipe hands ────────────────────────────────────────────────────
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    mp_result = mp_detector.detect(mp_image)
    wrist_data = draw_hand_skeleton(annotated, mp_result, width, height)

    # ── YOLO arms ──────────────────────────────────────────────────────────
    yolo_results = yolo_model(frame, verbose=False)
    arm_kps      = extract_arm_keypoints(yolo_results, width, height)
    draw_arm_segments(annotated, arm_kps)

    # Serialise arm keypoints for JSON (None → null)
    def pt_to_dict(pt):
        return {"px": pt[0], "py": pt[1]} if pt is not None else None

    frame_data = {
        "hands_found":  len(wrist_data),
        "wrists":       wrist_data,
        "arm_keypoints": {
            "left_wrist":  pt_to_dict(arm_kps["left_wrist"]),
            "left_elbow":  pt_to_dict(arm_kps["left_elbow"]),
            "right_wrist": pt_to_dict(arm_kps["right_wrist"]),
            "right_elbow": pt_to_dict(arm_kps["right_elbow"]),
        },
    }

    return annotated, frame_data


# ── Test mode ─────────────────────────────────────────────────────────────────

def run_test():
    print(f"\n{'=' * 60}")
    print(f"  Arm + Hand Tracking  —  TEST MODE")
    print(f"  Frames : first {TEST_MAX_FRAMES}  |  Saving : {TEST_SAVE_AT}")
    print(f"{'=' * 60}\n")

    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mp_detector = load_mediapipe()
    yolo_model  = load_yolo()

    if TEST_FRAMES_DIR.exists():
        frame_files = sorted(
            list(TEST_FRAMES_DIR.glob("frame_*.jpg")) +
            list(TEST_FRAMES_DIR.glob("frame_*.png"))
        )[:TEST_MAX_FRAMES]
        first = cv2.imread(str(frame_files[0]))
        height, width = first.shape[:2]
        source_type = "frames"
    else:
        frame_files = None
        source_type = "video"
        height = width = 0

    start = time.time()

    def frames():
        nonlocal height, width
        if source_type == "frames":
            for i, p in enumerate(frame_files):
                img = cv2.imread(str(p))
                if img is not None:
                    yield i, img
        else:
            cap = cv2.VideoCapture(str(TEST_VIDEO))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            fid = 0
            while fid < TEST_MAX_FRAMES:
                ret, img = cap.read()
                if not ret: break
                yield fid, img
                fid += 1
            cap.release()

    for frame_id, frame in frames():
        if height == 0:
            height, width = frame.shape[:2]

        display  = frame_id + 1
        annotated, fdata = process_frame(mp_detector, yolo_model, frame, width, height)

        if display in TEST_SAVE_AT:
            for dx, dy in [(-2,-2),(-2,2),(2,-2),(2,2)]:
                cv2.putText(annotated, f"Frame {display}", (30+dx, 70+dy),
                            cv2.FONT_HERSHEY_DUPLEX, 2.0, (0,0,0), 6, cv2.LINE_AA)
            cv2.putText(annotated, f"Frame {display}", (30, 70),
                        cv2.FONT_HERSHEY_DUPLEX, 2.0, (255,255,255), 3, cv2.LINE_AA)

            out = TEST_OUTPUT_DIR / f"frame_{str(display).zfill(3)}.png"
            cv2.imwrite(str(out), annotated)

            ak = fdata["arm_keypoints"]
            print(f"  Frame {display:>3}  →  {out.name}")
            print(f"          Hands (MediaPipe) : {fdata['hands_found']}")
            print(f"          Left  elbow (YOLO): {'✅' if ak['left_elbow']  else '❌'}")
            print(f"          Right elbow (YOLO): {'✅' if ak['right_elbow'] else '❌'}\n")

    mp_detector.close()
    print(f"  TEST COMPLETE  ({time.time()-start:.1f}s)")
    print(f"  Output → {TEST_OUTPUT_DIR}/\n")


# ── Frames folder mode ────────────────────────────────────────────────────────

def process_frames_folder(frames_dir: str, fps: float = 29.97):
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
    print(f"  Arm + Hand Tracking  (from frames)")
    print(f"  Clip   : {clip_name}  |  {width}x{height}  |  {total_frames} frames")
    print(f"{'=' * 60}\n")

    annotated_dir = ANNOTATED_ROOT / clip_name
    annotated_dir.mkdir(parents=True, exist_ok=True)
    ARM_POSE_ROOT.mkdir(parents=True, exist_ok=True)

    mp_detector = load_mediapipe()
    yolo_model  = load_yolo()

    all_frames  = []
    frame_id    = 0
    start_time  = time.time()

    for frame_file in frame_files:
        frame = cv2.imread(str(frame_file))
        if frame is None:
            frame_id += 1
            continue

        annotated, fdata = process_frame(mp_detector, yolo_model, frame, width, height)
        cv2.imwrite(str(annotated_dir / f"frame_{str(frame_id).zfill(6)}.png"), annotated)

        all_frames.append({
            "frame_id":      frame_id,
            "timestamp_sec": round(frame_id / fps, 4),
            **fdata,
        })

        if frame_id % 100 == 0:
            elapsed = time.time() - start_time
            pct     = frame_id / total_frames * 100
            print(f"  Frame {frame_id:>6} / {total_frames}  ({pct:5.1f}%)  |  {elapsed:.1f}s")

        frame_id += 1

    mp_detector.close()
    _save_json(clip_name, frame_id, fps, width, height, all_frames)
    _print_summary(clip_name, frame_id, time.time() - start_time, annotated_dir)


# ── Video file mode ───────────────────────────────────────────────────────────

def process_video(video_path: str, output_video: bool = False):
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
    print(f"  Arm + Hand Tracking  (from video)")
    print(f"  {clip_name}  |  {width}x{height}  |  {total_frames} frames")
    print(f"{'=' * 60}\n")

    annotated_dir = ANNOTATED_ROOT / clip_name
    annotated_dir.mkdir(parents=True, exist_ok=True)
    ARM_POSE_ROOT.mkdir(parents=True, exist_ok=True)

    mp_detector = load_mediapipe()
    yolo_model  = load_yolo()

    all_frames = []
    frame_id   = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, fdata = process_frame(mp_detector, yolo_model, frame, width, height)
        cv2.imwrite(str(annotated_dir / f"frame_{str(frame_id).zfill(6)}.png"), annotated)

        all_frames.append({
            "frame_id":      frame_id,
            "timestamp_sec": round(frame_id / fps, 4),
            **fdata,
        })

        if frame_id % 100 == 0:
            elapsed = time.time() - start_time
            pct     = frame_id / total_frames * 100
            print(f"  Frame {frame_id:>6} / {total_frames}  ({pct:5.1f}%)  |  {elapsed:.1f}s")

        frame_id += 1

    cap.release()
    mp_detector.close()

    _save_json(clip_name, frame_id, fps, width, height, all_frames)
    _print_summary(clip_name, frame_id, time.time() - start_time, annotated_dir)

    if output_video:
        _stitch_video(annotated_dir, clip_name, fps, width, height)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save_json(clip_name, total_frames, fps, width, height, frames_data):
    json_path = ARM_POSE_ROOT / f"{clip_name}.json"
    with open(json_path, "w") as f:
        json.dump({
            "clip_name":    clip_name,
            "total_frames": total_frames,
            "fps":          fps,
            "resolution":   {"width": width, "height": height},
            "frames":       frames_data,
        }, f, indent=2)
    print(f"  JSON → {json_path}")


def _print_summary(clip_name, total_frames, elapsed, annotated_dir):
    print(f"\n{'─' * 60}")
    print(f"  DONE  —  {clip_name}")
    print(f"{'─' * 60}")
    print(f"  Total frames : {total_frames:,}")
    print(f"  Time         : {elapsed/60:.1f} min")
    print(f"  Frames → {annotated_dir}/\n")


def _stitch_video(annotated_dir, clip_name, fps, width, height):
    png_files = sorted(annotated_dir.glob("frame_*.png"))
    if not png_files:
        print("  [warn] No frames to stitch.")
        return
    out_path = Path("assets/processed") / f"{clip_name}_arm_pose.mp4"
    writer   = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    print("  Stitching frames into video...")
    for p in png_files:
        writer.write(cv2.imread(str(p)))
    writer.release()
    print(f"  Video → {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    output_video = "--output-video" in sys.argv
    clean_argv   = [a for a in sys.argv if a != "--output-video"]

    if "--test" in clean_argv:
        run_test()

    elif "--frames" in clean_argv:
        idx = clean_argv.index("--frames")
        if idx + 1 >= len(clean_argv):
            print("Usage: python pipeline/arm_pose.py --frames <folder>")
            sys.exit(1)
        process_frames_folder(clean_argv[idx + 1])

    elif len(clean_argv) == 2:
        process_video(clean_argv[1], output_video=output_video)

    else:
        print("Usage:")
        print("  Test   : python pipeline/arm_pose.py --test")
        print("  Frames : python pipeline/arm_pose.py --frames <folder>")
        print("  Video  : python pipeline/arm_pose.py <video_path> [--output-video]")
        sys.exit(1)