"""
hand_pose.py
────────────
Detects full arm + hand keypoints using MediaPipe HolisticLandmarker.

Extracts per frame:
  • Arm points  (from body pose): shoulder, elbow, wrist — left and right
  • Hand points (from hand model): all 21 finger keypoints — left and right

Two modes:
  Normal mode  — processes every frame of a video
  Test mode    — processes only the first 100 frames, saves 5 sample PNGs
                 and a small JSON. Great for checking everything works quickly.

Usage:
  Normal : python pipeline/hand_pose.py assets/videos/WashingCup.mp4
  Test   : python pipeline/hand_pose.py --test
"""

# ── Imports ───────────────────────────────────────────────────────────────────

import sys           # read command-line arguments
import json          # write Python dicts as .json files
import time          # measure elapsed time
from pathlib import Path   # cross-platform file path handling

import cv2           # OpenCV — open videos, read frames, save PNG images
import numpy as np   # NumPy — image array operations

import mediapipe as mp   # MediaPipe — detection framework

# Tasks API — the modern MediaPipe 0.10+ interface
# (mp.solutions.holistic is not available in MediaPipe 0.10+;
#  HolisticLandmarker via the Tasks API provides the same features)
from mediapipe.tasks import python as mp_tasks          # BaseOptions etc.
from mediapipe.tasks.python import vision as mp_vision  # HolisticLandmarker


# ── Config ────────────────────────────────────────────────────────────────────

# Path to the holistic model file (downloaded to assets/models/)
MODEL_PATH = Path("assets/models/holistic_landmarker.task")

# Default video used when running in test mode
TEST_VIDEO = Path("assets/videos/WashingCup.mp4")

# Output folder for test mode results
TEST_OUTPUT_DIR = Path("assets/processed/holistic_test")

# In test mode: only read this many frames from the video
TEST_MAX_FRAMES = 100

# In test mode: save a PNG snapshot at each of these frame numbers
TEST_SAVE_FRAMES = [1, 25, 50, 75, 100]

# Normal mode output folders
ANNOTATED_ROOT = Path("assets/processed/annotated")
HAND_POSE_ROOT = Path("assets/processed/hand_pose")


# ── Pose landmark indices ─────────────────────────────────────────────────────

# MediaPipe's pose model labels 33 body joints 0–32.
# We extract only the 6 upper-body joints needed for arm tracking.
POSE_LEFT_SHOULDER  = 11   # left shoulder
POSE_RIGHT_SHOULDER = 12   # right shoulder
POSE_LEFT_ELBOW     = 13   # left elbow
POSE_RIGHT_ELBOW    = 14   # right elbow
POSE_LEFT_WRIST     = 15   # left wrist  (where arm ends, hand begins)
POSE_RIGHT_WRIST    = 16   # right wrist


# ── Hand landmark names (MediaPipe order 0–20) ────────────────────────────────

LANDMARK_NAMES = [
    "WRIST",             # 0
    "THUMB_CMC",         # 1  — thumb base
    "THUMB_MCP",         # 2
    "THUMB_IP",          # 3
    "THUMB_TIP",         # 4  — thumb tip
    "INDEX_FINGER_MCP",  # 5  — index base
    "INDEX_FINGER_PIP",  # 6
    "INDEX_FINGER_DIP",  # 7
    "INDEX_FINGER_TIP",  # 8  — index tip
    "MIDDLE_FINGER_MCP", # 9
    "MIDDLE_FINGER_PIP", # 10
    "MIDDLE_FINGER_DIP", # 11
    "MIDDLE_FINGER_TIP", # 12
    "RING_FINGER_MCP",   # 13
    "RING_FINGER_PIP",   # 14
    "RING_FINGER_DIP",   # 15
    "RING_FINGER_TIP",   # 16
    "PINKY_MCP",         # 17
    "PINKY_PIP",         # 18
    "PINKY_DIP",         # 19
    "PINKY_TIP",         # 20
]

# Bone connections: each tuple (a, b) draws a line from landmark a to b
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),         # thumb
    (0,5),(5,6),(6,7),(7,8),         # index
    (0,9),(9,10),(10,11),(11,12),    # middle
    (0,13),(13,14),(14,15),(15,16),  # ring
    (0,17),(17,18),(18,19),(19,20),  # pinky
    (5,9),(9,13),(13,17),            # palm crossbars
]


# ── Colors (BGR — OpenCV uses Blue, Green, Red order) ─────────────────────────

# Left side  → RED tones
COLOR_LEFT_ARM   = (0,   60,  220)   # dark red  — arm line
COLOR_LEFT_BONE  = (80,  80,  255)   # light red — hand bones
COLOR_LEFT_DOT   = (0,   0,   255)   # pure red  — joint dots

# Right side → BLUE tones
COLOR_RIGHT_ARM  = (220, 60,  0  )   # dark blue — arm line
COLOR_RIGHT_BONE = (255, 120, 80 )   # light blue — hand bones
COLOR_RIGHT_DOT  = (255, 0,   0  )   # pure blue — joint dots


# ── Helper: normalised landmark → pixel coords ────────────────────────────────

def lm_to_px(lm, width: int, height: int) -> tuple:
    """
    MediaPipe gives x, y as fractions of image size (0.0 to 1.0).
    Multiplying by frame dimensions converts them to actual pixel positions.
    Returns (pixel_x, pixel_y) as a tuple of ints.
    """
    return (int(lm.x * width), int(lm.y * height))


# ── Helper: draw the full arm + hand skeleton ─────────────────────────────────

def draw_skeleton(frame, arm, hand_pts, arm_color, bone_color, dot_color):
    """
    Draw a continuous skeleton from shoulder tip to fingertips.

    arm       — dict with keys "shoulder", "elbow", "wrist" → (px, py) or None
    hand_pts  — list of 21 (px, py) tuples for finger landmarks
    *_color   — BGR color tuples for arm line, hand bones, and joint dots
    """

    shoulder = arm.get("shoulder")   # (px, py) or None if not detected
    elbow    = arm.get("elbow")
    wrist    = arm.get("wrist")      # pose wrist (arm endpoint)

    # ── 1. Draw arm line: shoulder → elbow → wrist ───────────────────────────
    if shoulder and elbow:
        cv2.line(frame, shoulder, elbow, arm_color, 5, cv2.LINE_AA)   # thick line

    if elbow and wrist:
        cv2.line(frame, elbow, wrist, arm_color, 5, cv2.LINE_AA)

    # Draw large circles at each arm joint so they stand out
    for pt in [shoulder, elbow, wrist]:
        if pt:
            cv2.circle(frame, pt, 10, arm_color, -1, cv2.LINE_AA)     # filled circle
            cv2.circle(frame, pt,  5, (255, 255, 255), -1, cv2.LINE_AA)  # white center

    # ── 2–4 only run if we actually have hand landmark data ──────────────────
    # hand_pts is [] when the hand wasn't detected — skip hand drawing in that case
    if not hand_pts:
        return   # arm drawn above is enough; no hand data to add

    # ── 2. Connect arm wrist to the hand skeleton ─────────────────────────────
    if wrist and hand_pts[0]:
        cv2.line(frame, wrist, hand_pts[0], arm_color, 3, cv2.LINE_AA)

    # ── 3. Draw hand bone lines ───────────────────────────────────────────────
    for a_idx, b_idx in HAND_CONNECTIONS:
        p1 = hand_pts[a_idx]   # start point of this bone
        p2 = hand_pts[b_idx]   # end point of this bone
        if p1 and p2:
            cv2.line(frame, p1, p2, bone_color, 2, cv2.LINE_AA)

    # ── 4. Draw a dot at every hand keypoint ─────────────────────────────────
    for pt in hand_pts:
        if pt:
            cv2.circle(frame, pt, 6, bone_color, -1, cv2.LINE_AA)   # outer ring
            cv2.circle(frame, pt, 3, dot_color,  -1, cv2.LINE_AA)   # bright center


# ── Core detection logic (shared by normal and test mode) ─────────────────────

def detect_frame(result, width, height):
    """
    Extract arm + hand data from one HolisticLandmarker result.

    result  — the object returned by detector.detect(mp_image)
    width   — frame pixel width  (needed to convert normalised → pixels)
    height  — frame pixel height

    Returns a dict ready to be saved in the JSON:
    {
      "left_arm_detected":   True/False,
      "right_arm_detected":  True/False,
      "left_hand_detected":  True/False,
      "right_hand_detected": True/False,
      "hands": [
        {
          "label": "Left",
          "arm":   {"shoulder": {"px":…,"py":…}, "elbow":…, "wrist":…},
          "keypoints": {"WRIST": {"x":…,"y":…,"px":…,"py":…}, … 21 points}
        },
        …
      ]
    }
    """

    # ── Pose (arm) landmarks ──────────────────────────────────────────────────
    # result.pose_landmarks is a list of 33 NormalizedLandmark objects,
    # or an empty list [] if no body was detected in this frame.
    pose = result.pose_landmarks   # shorthand

    def pose_pt(idx):
        """Get pixel coords for pose landmark at index idx, or None."""
        if not pose or idx >= len(pose):
            return None   # no pose detected
        return lm_to_px(pose[idx], width, height)

    # Extract 3 joints per side
    left_arm  = {
        "shoulder": pose_pt(POSE_LEFT_SHOULDER),
        "elbow":    pose_pt(POSE_LEFT_ELBOW),
        "wrist":    pose_pt(POSE_LEFT_WRIST),
    }
    right_arm = {
        "shoulder": pose_pt(POSE_RIGHT_SHOULDER),
        "elbow":    pose_pt(POSE_RIGHT_ELBOW),
        "wrist":    pose_pt(POSE_RIGHT_WRIST),
    }

    # Arm is "detected" if at least the shoulder point was found
    left_arm_detected  = left_arm["shoulder"]  is not None
    right_arm_detected = right_arm["shoulder"] is not None

    # ── Hand landmarks ────────────────────────────────────────────────────────
    # HolisticLandmarker gives left and right hands separately.
    # Each is a list of 21 NormalizedLandmark objects, or [] if not detected.
    left_lms  = result.left_hand_landmarks    # 21 landmarks or []
    right_lms = result.right_hand_landmarks   # 21 landmarks or []

    left_hand_detected  = len(left_lms)  == 21   # True only if all 21 found
    right_hand_detected = len(right_lms) == 21

    def to_pts(lms):
        """Convert list of 21 NormalizedLandmarks to list of (px,py) tuples."""
        if not lms:
            return []
        return [lm_to_px(lm, width, height) for lm in lms]

    left_pts  = to_pts(left_lms)    # list of 21 (px,py) or []
    right_pts = to_pts(right_lms)   # list of 21 (px,py) or []

    def build_kp(pts, lms):
        """Build keypoints dict: landmark name → {x, y, z, px, py}."""
        kp = {}
        for i, (pt, lm) in enumerate(zip(pts, lms)):
            kp[LANDMARK_NAMES[i]] = {
                "x":  round(lm.x, 6),   # normalised 0–1
                "y":  round(lm.y, 6),
                "z":  round(lm.z, 6),   # relative depth
                "px": pt[0],             # pixel x
                "py": pt[1],             # pixel y
            }
        return kp

    def build_arm_json(arm_dict):
        """Convert arm dict of (px,py) to JSON-serialisable form."""
        out = {}
        for key, pt in arm_dict.items():
            out[key] = {"px": pt[0], "py": pt[1]} if pt else None
        return out

    # Assemble the hands list (left first, then right)
    hands = []
    if left_hand_detected:
        hands.append({
            "label":      "Left",
            "arm":        build_arm_json(left_arm),
            "keypoints":  build_kp(left_pts, left_lms),
        })
    if right_hand_detected:
        hands.append({
            "label":      "Right",
            "arm":        build_arm_json(right_arm),
            "keypoints":  build_kp(right_pts, right_lms),
        })

    return {
        "left_arm_detected":   left_arm_detected,
        "right_arm_detected":  right_arm_detected,
        "left_hand_detected":  left_hand_detected,
        "right_hand_detected": right_hand_detected,
        "left_arm":   build_arm_json(left_arm),
        "right_arm":  build_arm_json(right_arm),
        "left_pts":   left_pts,    # pixel tuples — used for drawing, not saved to JSON
        "right_pts":  right_pts,
        "hands":      hands,
    }


def arm_json_to_pts(arm_json: dict) -> dict:
    """
    Convert arm JSON dict {"shoulder": {"px":…,"py":…}, …}
    to the tuple dict {"shoulder": (px,py), …} that draw_skeleton expects.
    Returns None for any joint that wasn't detected.
    """
    result = {}
    for key, val in arm_json.items():
        # val is either {"px": int, "py": int} or None
        if val and "px" in val:
            result[key] = (val["px"], val["py"])   # convert to tuple
        else:
            result[key] = None                     # joint not detected
    return result


def annotate_frame(frame, data):
    """
    Draw arm + hand skeleton on frame using data returned by detect_frame().
    Modifies frame in-place.
    """
    # Convert arm JSON dicts → tuple dicts for drawing
    left_arm_pts  = arm_json_to_pts(data["left_arm"])
    right_arm_pts = arm_json_to_pts(data["right_arm"])

    # Draw left arm + hand (red)
    if data["left_arm_detected"] or data["left_hand_detected"]:
        draw_skeleton(
            frame,
            arm        = left_arm_pts,
            hand_pts   = data["left_pts"],
            arm_color  = COLOR_LEFT_ARM,
            bone_color = COLOR_LEFT_BONE,
            dot_color  = COLOR_LEFT_DOT,
        )

    # Draw right arm + hand (blue)
    if data["right_arm_detected"] or data["right_hand_detected"]:
        draw_skeleton(
            frame,
            arm        = right_arm_pts,
            hand_pts   = data["right_pts"],
            arm_color  = COLOR_RIGHT_ARM,
            bone_color = COLOR_RIGHT_BONE,
            dot_color  = COLOR_RIGHT_DOT,
        )


# ── Test mode ─────────────────────────────────────────────────────────────────

def run_test():
    """
    Quick test: process only the first 100 frames of WashingCup.mp4.
    Save 5 sample annotated PNGs and a small keypoints JSON.
    Print a detection report for each saved frame.
    """

    print(f"\n{'=' * 60}")
    print(f"  Holistic Landmarker — TEST MODE")
    print(f"  Video  : {TEST_VIDEO}")
    print(f"  Frames : first {TEST_MAX_FRAMES} only")
    print(f"  Saving : frames {TEST_SAVE_FRAMES}")
    print(f"{'=' * 60}\n")

    # Check the video file exists
    if not TEST_VIDEO.exists():
        print(f"[ERROR] Video not found: {TEST_VIDEO}")
        sys.exit(1)

    # Check the model file exists
    if not MODEL_PATH.exists():
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        sys.exit(1)

    # Create the output folder (holistic_test/)
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Output folder: {TEST_OUTPUT_DIR}\n")

    # ── Open video ────────────────────────────────────────────
    cap    = cv2.VideoCapture(str(TEST_VIDEO))   # open the video file
    fps    = cap.get(cv2.CAP_PROP_FPS)            # frames per second
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  Resolution : {width} x {height}  |  FPS: {fps}\n")

    # ── Set up MediaPipe HolisticLandmarker ───────────────────
    print("  Loading HolisticLandmarker model ...")
    options = mp_vision.HolisticLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp_vision.RunningMode.IMAGE,   # one frame at a time
        min_pose_detection_confidence=0.5,
        min_pose_landmarks_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
    )
    detector = mp_vision.HolisticLandmarker.create_from_options(options)
    print("  Model ready.\n")

    # ── Process first 100 frames ──────────────────────────────
    saved_frames  = {}    # {frame_id: detection_data} — only for TEST_SAVE_FRAMES
    frame_id      = 0     # current frame counter (0-indexed)
    start         = time.time()

    print(f"  Processing frames 0 – {TEST_MAX_FRAMES - 1} ...")
    print(f"  {'─' * 50}")

    while frame_id < TEST_MAX_FRAMES:   # stop after TEST_MAX_FRAMES frames

        ret, frame = cap.read()   # read next frame
        if not ret:
            break   # video ended early

        # Convert BGR → RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Run detection
        result = detector.detect(mp_image)

        # Extract all landmarks into a clean dict
        data = detect_frame(result, width, height)

        # ── Save this frame if it's one of our 5 sample frames ───────────────
        # frame_id is 0-indexed, but TEST_SAVE_FRAMES uses 1-indexed numbers
        # so frame_id 0 = "frame 1", frame_id 24 = "frame 25", etc.
        display_num = frame_id + 1   # human-friendly 1-based number

        if display_num in TEST_SAVE_FRAMES:

            # Draw skeleton on a copy of the frame
            annotated = frame.copy()
            annotate_frame(annotated, data)

            # Add frame number label in top-left corner (white text, black outline)
            label = f"Frame {display_num}"
            for dx, dy in [(-1,-1),(-1,1),(1,-1),(1,1)]:   # draw black outline first
                cv2.putText(annotated, label, (20+dx, 50+dy),
                            cv2.FONT_HERSHEY_DUPLEX, 1.2, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(annotated, label, (20, 50),   # then white fill on top
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (255,255,255), 2, cv2.LINE_AA)

            # Save the annotated PNG
            out_path = TEST_OUTPUT_DIR / f"frame_{str(display_num).zfill(3)}.png"
            cv2.imwrite(str(out_path), annotated)

            # Store detection data for JSON (remove drawing-only fields)
            json_data = {
                "frame_number":        display_num,
                "timestamp_sec":       round(frame_id / fps, 4),
                "left_arm_detected":   data["left_arm_detected"],
                "right_arm_detected":  data["right_arm_detected"],
                "left_hand_detected":  data["left_hand_detected"],
                "right_hand_detected": data["right_hand_detected"],
                "hands":               data["hands"],
            }
            saved_frames[display_num] = json_data

            # ── Print detection report for this frame ─────────────────────────
            yes_no = lambda b: "✅ Yes" if b else "❌ No"   # helper for clean output
            print(f"\n  Frame {display_num:>3}  saved → {out_path.name}")
            print(f"          Left  arm  detected : {yes_no(data['left_arm_detected'])}")
            print(f"          Right arm  detected : {yes_no(data['right_arm_detected'])}")
            print(f"          Left  hand detected : {yes_no(data['left_hand_detected'])}")
            print(f"          Right hand detected : {yes_no(data['right_hand_detected'])}")

        frame_id += 1   # advance to next frame

    cap.release()      # close video file
    detector.close()   # free MediaPipe model memory

    # ── Save test keypoints JSON ──────────────────────────────
    json_path = TEST_OUTPUT_DIR / "test_keypoints.json"
    with open(json_path, "w") as f:
        json.dump({
            "video":           str(TEST_VIDEO),
            "frames_tested":   TEST_MAX_FRAMES,
            "frames_saved":    TEST_SAVE_FRAMES,
            "fps":             fps,
            "resolution":      {"width": width, "height": height},
            "results":         list(saved_frames.values()),
        }, f, indent=2)

    # ── Final summary ─────────────────────────────────────────
    elapsed = time.time() - start
    print(f"\n  {'─' * 50}")
    print(f"  TEST COMPLETE")
    print(f"  {'─' * 50}")
    print(f"  Frames processed : {frame_id}")
    print(f"  PNGs saved       : {len(saved_frames)}")
    print(f"  Time taken       : {elapsed:.1f}s")
    print(f"\n  Sample frames → {TEST_OUTPUT_DIR}/")
    print(f"  Keypoints JSON → {json_path}")
    print()


# ── Frames folder mode ───────────────────────────────────────────────────────

def process_frames(frames_dir: str, fps: float = 29.97):
    """
    Read already-extracted frames from a folder, draw the arm + hand skeleton
    on each one, and save annotated PNGs.

    frames_dir  — path to a folder of JPEG/PNG frames named frame_000000.jpg etc.
    fps         — original video FPS (used only for timestamp calculation in JSON)

    Why read from frames instead of video?
      We already extracted raw frames in a separate step. Reading PNGs/JPEGs
      is faster than re-decoding a compressed video, and it avoids doing the
      same work twice.
    """

    frames_dir = Path(frames_dir)

    if not frames_dir.exists():
        print(f"[ERROR] Frames folder not found: {frames_dir}")
        sys.exit(1)

    if not MODEL_PATH.exists():
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        sys.exit(1)

    # Clip name comes from the folder name, e.g. "Cutting Banana"
    clip_name = frames_dir.name

    # Collect all frame files, sorted so they process in the correct order
    # Accept both .jpg and .png frames
    frame_files = sorted(
        list(frames_dir.glob("frame_*.jpg")) +
        list(frames_dir.glob("frame_*.png"))
    )

    if not frame_files:
        print(f"[ERROR] No frame files found in {frames_dir}")
        sys.exit(1)

    total_frames = len(frame_files)   # total number of frames to process

    # Read the first frame to get resolution
    first = cv2.imread(str(frame_files[0]))
    height, width = first.shape[:2]   # shape is (H, W, channels)

    print(f"\n{'=' * 60}")
    print(f"  Hand Pose  —  from extracted frames")
    print(f"  Clip       : {clip_name}")
    print(f"  Frames dir : {frames_dir}")
    print(f"{'=' * 60}")
    print(f"  Resolution : {width} x {height}")
    print(f"  FPS        : {fps}")
    print(f"  Frames     : {total_frames}\n")

    # Create output folders
    annotated_dir = ANNOTATED_ROOT / clip_name
    annotated_dir.mkdir(parents=True, exist_ok=True)
    HAND_POSE_ROOT.mkdir(parents=True, exist_ok=True)

    # Load MediaPipe HolisticLandmarker
    options = mp_vision.HolisticLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp_vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.5,
        min_pose_landmarks_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
    )
    detector = mp_vision.HolisticLandmarker.create_from_options(options)

    frames_data = []    # per-frame JSON entries
    frame_id    = 0     # current frame index (0-based)
    count_2     = 0
    count_1     = 0
    count_0     = 0
    start_time  = time.time()

    for frame_file in frame_files:   # loop over every frame file in sorted order

        # Read the raw frame from disk
        frame = cv2.imread(str(frame_file))

        if frame is None:
            print(f"  [WARN] Could not read {frame_file.name} — skipping")
            frame_id += 1
            continue

        # Convert BGR → RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Run detection
        result = detector.detect(mp_image)
        data   = detect_frame(result, width, height)

        # Draw skeleton on a copy and save
        annotated = frame.copy()
        annotate_frame(annotated, data)
        cv2.imwrite(str(annotated_dir / f"frame_{str(frame_id).zfill(6)}.png"), annotated)

        # Count hands
        num_hands = int(data["left_hand_detected"]) + int(data["right_hand_detected"])
        if num_hands == 2: count_2 += 1
        elif num_hands == 1: count_1 += 1
        else: count_0 += 1

        frames_data.append({
            "frame_id":       frame_id,
            "timestamp_sec":  round(frame_id / fps, 4),
            "hands_detected": num_hands,
            "hands":          data["hands"],
        })

        if frame_id % 100 == 0:
            elapsed = time.time() - start_time
            pct     = (frame_id / total_frames) * 100
            print(
                f"  Frame {frame_id:>6} / {total_frames}"
                f"  ({pct:5.1f}%)"
                f"  |  hands: 2={count_2}  1={count_1}  0={count_0}"
                f"  |  {elapsed:.1f}s"
            )

        frame_id += 1

    detector.close()

    # Save JSON
    json_path = HAND_POSE_ROOT / f"{clip_name}.json"
    with open(json_path, "w") as f:
        json.dump({
            "clip_name":    clip_name,
            "total_frames": frame_id,
            "fps":          fps,
            "resolution":   {"width": width, "height": height},
            "frames":       frames_data,
        }, f, indent=2)

    total_time    = time.time() - start_time
    frames_hands  = count_1 + count_2
    detection_rate = (frames_hands / frame_id) * 100 if frame_id > 0 else 0

    print(f"\n{'─' * 60}")
    print(f"  SUMMARY  —  {clip_name}")
    print(f"{'─' * 60}")
    print(f"  Total frames   : {frame_id}")
    print(f"  With hands     : {frames_hands}  ({detection_rate:.1f}%)")
    print(f"  2 hands        : {count_2}")
    print(f"  1 hand         : {count_1}")
    print(f"  0 hands        : {count_0}")
    print(f"  Time taken     : {total_time:.1f}s")
    print(f"\n  Annotated frames → {annotated_dir}/")
    print(f"  Keypoints JSON   → {json_path}\n")


# ── Full video mode ───────────────────────────────────────────────────────────

def process_video(video_path: str):
    """
    Full pipeline: detect arm + hand keypoints for every frame of a video.
    Saves annotated PNGs and a complete keypoints JSON.
    """

    video_path = Path(video_path)

    if not video_path.exists():
        print(f"[ERROR] File not found: {video_path}")
        sys.exit(1)

    if not MODEL_PATH.exists():
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        sys.exit(1)

    clip_name = video_path.stem   # e.g. "WashingCup"

    print(f"\n{'=' * 60}")
    print(f"  Hand Pose Pipeline  (MediaPipe Holistic)")
    print(f"  Clip : {clip_name}")
    print(f"{'=' * 60}\n")

    cap          = cv2.VideoCapture(str(video_path))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  Resolution : {width} x {height}  |  FPS: {fps}  |  Frames: {total_frames}\n")

    annotated_dir = ANNOTATED_ROOT / clip_name
    annotated_dir.mkdir(parents=True, exist_ok=True)
    HAND_POSE_ROOT.mkdir(parents=True, exist_ok=True)

    # Load model
    options = mp_vision.HolisticLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp_vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.5,
        min_pose_landmarks_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
    )
    detector = mp_vision.HolisticLandmarker.create_from_options(options)

    frames_data   = []   # one dict per frame for the JSON
    frame_id      = 0
    count_2       = 0    # frames with 2 hands
    count_1       = 0    # frames with 1 hand
    count_0       = 0    # frames with 0 hands
    start_time    = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result    = detector.detect(mp_image)
        data      = detect_frame(result, width, height)

        # Draw and save annotated frame
        annotated = frame.copy()
        annotate_frame(annotated, data)
        cv2.imwrite(str(annotated_dir / f"frame_{str(frame_id).zfill(6)}.png"), annotated)

        # Count hands for summary
        num_hands = int(data["left_hand_detected"]) + int(data["right_hand_detected"])
        if num_hands == 2: count_2 += 1
        elif num_hands == 1: count_1 += 1
        else: count_0 += 1

        # Build JSON entry (no drawing-only fields)
        frames_data.append({
            "frame_id":       frame_id,
            "timestamp_sec":  round(frame_id / fps, 4),
            "hands_detected": num_hands,
            "hands":          data["hands"],
        })

        if frame_id % 100 == 0:
            elapsed  = time.time() - start_time
            pct      = (frame_id / total_frames) * 100
            print(
                f"  Frame {frame_id:>6} / {total_frames}"
                f"  ({pct:5.1f}%)"
                f"  |  hands: 2={count_2}  1={count_1}  0={count_0}"
                f"  |  {elapsed:.1f}s"
            )

        frame_id += 1

    cap.release()
    detector.close()

    # Save JSON
    json_path = HAND_POSE_ROOT / f"{clip_name}.json"
    with open(json_path, "w") as f:
        json.dump({
            "clip_name":    clip_name,
            "total_frames": frame_id,
            "fps":          fps,
            "resolution":   {"width": width, "height": height},
            "frames":       frames_data,
        }, f, indent=2)

    total_time    = time.time() - start_time
    frames_hands  = count_1 + count_2
    detection_rate = (frames_hands / frame_id) * 100

    print(f"\n{'─' * 60}")
    print(f"  SUMMARY  —  {clip_name}")
    print(f"{'─' * 60}")
    print(f"  Total frames   : {frame_id}")
    print(f"  With hands     : {frames_hands}  ({detection_rate:.1f}%)")
    print(f"  2 hands        : {count_2}")
    print(f"  1 hand         : {count_1}")
    print(f"  0 hands        : {count_0}")
    print(f"  Time taken     : {total_time:.1f}s")
    print(f"\n  Frames → {annotated_dir}/")
    print(f"  JSON   → {json_path}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    if "--test" in sys.argv:
        run_test()   # quick 100-frame test on first 100 frames of WashingCup

    elif "--frames" in sys.argv:
        # Read from an already-extracted frames folder
        # Usage: python pipeline/hand_pose.py --frames "assets/processed/frames/Cutting Banana"
        idx = sys.argv.index("--frames")
        if idx + 1 >= len(sys.argv):
            print("Usage: python pipeline/hand_pose.py --frames <frames_folder>")
            sys.exit(1)
        process_frames(sys.argv[idx + 1])

    elif len(sys.argv) == 2:
        process_video(sys.argv[1])   # full video file

    else:
        print("Usage:")
        print("  From frames folder : python pipeline/hand_pose.py --frames <folder>")
        print("  From video file    : python pipeline/hand_pose.py <video_path>")
        print("  Quick test         : python pipeline/hand_pose.py --test")
        sys.exit(1)
