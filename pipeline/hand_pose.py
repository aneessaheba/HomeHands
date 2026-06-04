"""
hand_pose.py
────────────
Reads every frame of a video, detects hands using MediaPipe Tasks API
(compatible with MediaPipe 0.10+), draws the 21-keypoint skeleton on
each frame, and saves:
  • annotated frame PNGs  →  assets/processed/annotated/<clip_name>/
  • keypoint JSON         →  assets/processed/hand_pose/<clip_name>.json

Usage:
  python pipeline/hand_pose.py assets/videos/WashingCup.mp4
"""

# ── Imports ───────────────────────────────────────────────────────────────────

import sys                    # read the command-line argument (video path)
import json                   # write Python dicts as .json files
import time                   # measure how long the script takes
from pathlib import Path      # cross-platform file path handling

import cv2                    # OpenCV — opens videos, reads frames, saves images
import numpy as np            # NumPy — needed for drawing connections manually
import mediapipe as mp        # MediaPipe — hand detection

# MediaPipe Tasks API — the modern interface for MediaPipe 0.10+
from mediapipe.tasks import python as mp_tasks               # base options etc.
from mediapipe.tasks.python import vision as mp_vision       # HandLandmarker lives here
from mediapipe.tasks.python.components.containers import \
    landmark as mp_landmark                                  # landmark types


# ── Model path ────────────────────────────────────────────────────────────────

# Path to the downloaded hand_landmarker.task model file
# (run:  curl -L -o assets/models/hand_landmarker.task \
#   https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task)
MODEL_PATH = Path("assets/models/hand_landmarker.task")


# ── Output folders ────────────────────────────────────────────────────────────

ANNOTATED_ROOT = Path("assets/processed/annotated")   # annotated frame PNGs
HAND_POSE_ROOT = Path("assets/processed/hand_pose")   # keypoint JSON files


# ── Landmark name list ────────────────────────────────────────────────────────

# MediaPipe returns 21 landmarks numbered 0–20.
# This list maps each index to its human-readable name.
LANDMARK_NAMES = [
    "WRIST",             # 0
    "THUMB_CMC",         # 1
    "THUMB_MCP",         # 2
    "THUMB_IP",          # 3
    "THUMB_TIP",         # 4
    "INDEX_FINGER_MCP",  # 5
    "INDEX_FINGER_PIP",  # 6
    "INDEX_FINGER_DIP",  # 7
    "INDEX_FINGER_TIP",  # 8
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

# Which landmark indices are connected by lines to form the skeleton
# Each tuple is a pair of indices that should be drawn as a bone
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),         # thumb
    (0,5),(5,6),(6,7),(7,8),         # index finger
    (0,9),(9,10),(10,11),(11,12),    # middle finger
    (0,13),(13,14),(14,15),(15,16),  # ring finger
    (0,17),(17,18),(18,19),(19,20),  # pinky
    (5,9),(9,13),(13,17),            # palm crossbars
]


# ── Drawing helper ────────────────────────────────────────────────────────────

def draw_hand_skeleton(frame: np.ndarray, landmarks: list, width: int, height: int):
    """
    Draw the 21-keypoint hand skeleton directly on a frame (in place).

    landmarks — list of NormalizedLandmark objects (x, y are 0.0–1.0)
    width, height — pixel dimensions of the frame (for converting normalised → px)
    """
    # Convert all normalised coordinates to pixel positions
    pts = [
        (int(lm.x * width), int(lm.y * height))   # (px_x, px_y) for each landmark
        for lm in landmarks
    ]

    # Draw bones (lines between connected landmark pairs)
    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(
            frame,
            pts[start_idx],    # start point (x, y) in pixels
            pts[end_idx],      # end point   (x, y) in pixels
            (0, 200, 0),       # colour: green in BGR
            2,                 # line thickness in pixels
        )

    # Draw a filled circle at each of the 21 keypoints
    for px, py in pts:
        cv2.circle(frame, (px, py), 5, (0, 0, 255), -1)   # red filled dot, radius 5


# ── Main function ─────────────────────────────────────────────────────────────

def process_video(video_path: str):
    """
    Full pipeline for one video:
      1. Open video with OpenCV
      2. Set up MediaPipe HandLandmarker (Tasks API)
      3. Process every frame
      4. Draw skeleton + save annotated PNG
      5. Collect all keypoints
      6. Write JSON
      7. Print summary
    """

    video_path = Path(video_path)          # convert string to Path object

    if not video_path.exists():
        print(f"[ERROR] File not found: {video_path}")
        sys.exit(1)

    # Verify the model file is present — it must be downloaded first
    if not MODEL_PATH.exists():
        print(f"[ERROR] Model file not found: {MODEL_PATH}")
        print("  Run this to download it:")
        print("  curl -L -o assets/models/hand_landmarker.task \\")
        print("    https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
        sys.exit(1)

    clip_name = video_path.stem            # e.g. "WashingCup"

    print(f"\n{'=' * 60}")
    print(f"  Hand Pose Pipeline")
    print(f"  Clip : {clip_name}")
    print(f"{'=' * 60}\n")

    # ── Open video ────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))   # open the video file

    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        sys.exit(1)

    fps          = cap.get(cv2.CAP_PROP_FPS)                   # frames per second
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))      # pixel width
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))     # pixel height
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))      # total frame count

    print(f"  Resolution : {width} x {height}")
    print(f"  FPS        : {fps}")
    print(f"  Frames     : {total_frames}\n")

    # ── Create output folders ─────────────────────────────────
    annotated_dir = ANNOTATED_ROOT / clip_name           # e.g. .../annotated/WashingCup/
    annotated_dir.mkdir(parents=True, exist_ok=True)     # create if missing
    HAND_POSE_ROOT.mkdir(parents=True, exist_ok=True)    # create if missing

    # ── Set up MediaPipe HandLandmarker (Tasks API) ───────────

    # BaseOptions tells MediaPipe where the model file is
    base_opts = mp_tasks.BaseOptions(model_asset_path=str(MODEL_PATH))

    # HandLandmarkerOptions configures the detector
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_opts,
        num_hands=2,                    # detect up to 2 hands per frame
        min_hand_detection_confidence=0.5,   # minimum confidence to detect a hand
        min_hand_presence_confidence=0.5,    # minimum confidence to keep tracking
        min_tracking_confidence=0.5,         # minimum confidence for frame-to-frame tracking
        running_mode=mp_vision.RunningMode.IMAGE,  # IMAGE mode = process one frame at a time
    )

    # Create the HandLandmarker detector from these options
    detector = mp_vision.HandLandmarker.create_from_options(options)

    # ── Counters ──────────────────────────────────────────────
    frames_data   = []       # one dict per frame, collected for the JSON
    frame_id      = 0        # current frame index (starts at 0)
    count_2_hands = 0        # frames with 2 hands detected
    count_1_hand  = 0        # frames with 1 hand detected
    count_0_hands = 0        # frames with no hands detected
    start_time    = time.time()

    # ── Frame loop ────────────────────────────────────────────
    while True:
        ret, frame = cap.read()    # read next frame; ret=False at end of video
        if not ret:
            break                  # no more frames

        # Convert BGR (OpenCV) → RGB (MediaPipe)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Wrap the NumPy array in a MediaPipe Image object
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,   # standard RGB colour space
            data=frame_rgb,                      # the pixel data
        )

        # Run hand detection on this frame
        detection_result = detector.detect(mp_image)

        # detection_result.hand_landmarks  — list of 21-landmark lists (one per hand)
        # detection_result.handedness      — list of handedness classifications
        num_hands = len(detection_result.hand_landmarks)   # 0, 1, or 2

        # Update counters
        if num_hands == 2:
            count_2_hands += 1
        elif num_hands == 1:
            count_1_hand += 1
        else:
            count_0_hands += 1

        # ── Draw skeleton on a copy of the frame ─────────────
        annotated_frame = frame.copy()    # don't modify the original
        for hand_landmarks in detection_result.hand_landmarks:
            draw_hand_skeleton(annotated_frame, hand_landmarks, width, height)

        # ── Save annotated frame as PNG ───────────────────────
        fname = f"frame_{str(frame_id).zfill(6)}.png"   # zero-padded filename
        cv2.imwrite(str(annotated_dir / fname), annotated_frame)

        # ── Build keypoint data for JSON ──────────────────────
        hands_list = []

        for hand_lms, handedness in zip(
            detection_result.hand_landmarks,    # 21 landmarks per hand
            detection_result.handedness,        # classification per hand
        ):
            # handedness[0] contains the best classification result
            hand_label      = handedness[0].category_name   # "Left" or "Right"
            hand_confidence = round(handedness[0].score, 4) # confidence score

            # Build a dict of all 21 named keypoints
            keypoints_dict = {}
            for idx, lm in enumerate(hand_lms):
                name = LANDMARK_NAMES[idx]
                keypoints_dict[name] = {
                    "x":  round(lm.x, 6),           # normalised x (0–1)
                    "y":  round(lm.y, 6),           # normalised y (0–1)
                    "z":  round(lm.z, 6),           # relative depth
                    "px": int(lm.x * width),        # pixel x
                    "py": int(lm.y * height),       # pixel y
                }

            hands_list.append({
                "label":      hand_label,
                "confidence": hand_confidence,
                "keypoints":  keypoints_dict,
            })

        # ── Append frame entry ────────────────────────────────
        frames_data.append({
            "frame_id":       frame_id,
            "timestamp_sec":  round(frame_id / fps, 4),
            "hands_detected": num_hands,
            "hands":          hands_list,
        })

        # ── Progress print every 100 frames ──────────────────
        if frame_id % 100 == 0:
            elapsed  = time.time() - start_time
            pct_done = (frame_id / total_frames) * 100
            print(
                f"  Frame {frame_id:>6} / {total_frames}"
                f"  ({pct_done:5.1f}%)"
                f"  |  hands: 2={count_2_hands}  1={count_1_hand}  0={count_0_hands}"
                f"  |  {elapsed:.1f}s"
            )

        frame_id += 1   # advance frame counter

    # ── Release resources ─────────────────────────────────────
    cap.release()      # close video file
    detector.close()   # release MediaPipe model resources

    # ── Write JSON ────────────────────────────────────────────
    output_json = {
        "clip_name":    clip_name,
        "total_frames": frame_id,
        "fps":          fps,
        "resolution":   {"width": width, "height": height},
        "frames":       frames_data,
    }

    json_path = HAND_POSE_ROOT / f"{clip_name}.json"
    with open(json_path, "w") as f:
        json.dump(output_json, f, indent=2)

    # ── Summary ───────────────────────────────────────────────
    total_time        = time.time() - start_time
    frames_with_hands = count_1_hand + count_2_hands
    detection_rate    = (frames_with_hands / frame_id) * 100

    print(f"\n{'─' * 60}")
    print(f"  SUMMARY  —  {clip_name}")
    print(f"{'─' * 60}")
    print(f"  Total frames processed : {frame_id}")
    print(f"  Frames with hands      : {frames_with_hands}")
    print(f"  Frames with 2 hands    : {count_2_hands}")
    print(f"  Frames with 1 hand     : {count_1_hand}")
    print(f"  Frames with 0 hands    : {count_0_hands}")
    print(f"  Detection rate         : {detection_rate:.1f}%")
    print(f"  Time taken             : {total_time:.1f}s")
    print(f"\n  Annotated frames  ->  {annotated_dir}/")
    print(f"  Keypoint JSON     ->  {json_path}")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage  : python pipeline/hand_pose.py <path_to_video>")
        print("Example: python pipeline/hand_pose.py assets/videos/WashingCup.mp4")
        sys.exit(1)
    process_video(sys.argv[1])
