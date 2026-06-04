"""
annotate_video.py
─────────────────
Full pipeline for one video — hand pose, SAM2 segmentation, then
stitches everything into a single annotated output video showing:
  • Original footage as background (inside the segmented frames)
  • Red mask over the left hand
  • Purple mask over the right hand
  • 21-point hand skeleton (dots + connecting lines) drawn on top
  • Frame number and hand count shown in the top-left corner

Usage:
  python pipeline/annotate_video.py assets/videos/WashingCup.mp4

Output:
  assets/processed/WashingCup_annotated.mp4
"""

# ── Imports ───────────────────────────────────────────────────────────────────

import sys          # read the command-line argument (video path)
import json         # load the hand pose keypoint JSON
import time         # measure elapsed time
from pathlib import Path   # clean file path handling

import cv2          # OpenCV — read PNG frames, draw shapes, write video
import numpy as np  # NumPy — needed for image array operations

# Add the pipeline/ folder to Python's module search path so we can
# import hand_pose.py and segmentation.py as modules
sys.path.insert(0, str(Path(__file__).parent))

import hand_pose     # our hand detection script
import segmentation  # our SAM2 segmentation script


# ── Paths ─────────────────────────────────────────────────────────────────────

# Where segmentation.py saves its colored mask frame PNGs
SEGMENTED_ROOT = Path("assets/processed/segmented")

# Where hand_pose.py saves its keypoint JSON
HAND_POSE_ROOT = Path("assets/processed/hand_pose")

# Where we will save the final annotated video
OUTPUT_ROOT = Path("assets/processed")


# ── Hand skeleton definition ──────────────────────────────────────────────────

# MediaPipe landmark names — index in this list = landmark ID (0-20)
LANDMARK_NAMES = [
    "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]

# Each tuple (a, b) means "draw a line from landmark a to landmark b"
# These connections form the hand skeleton when drawn
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),         # thumb
    (0,5),(5,6),(6,7),(7,8),         # index finger
    (0,9),(9,10),(10,11),(11,12),    # middle finger
    (0,13),(13,14),(14,15),(15,16),  # ring finger
    (0,17),(17,18),(18,19),(19,20),  # pinky
    (5,9),(9,13),(13,17),            # palm crossbars
]

# Colors for skeleton drawing — OpenCV uses BGR (Blue, Green, Red) order
COLOR_BONE_RIGHT   = (200, 180, 255)   # light purple for right hand bones
COLOR_DOT_RIGHT    = (255, 0,   255)   # bright magenta for right hand dots
COLOR_BONE_LEFT    = (100, 100, 255)   # light red for left hand bones
COLOR_DOT_LEFT     = (0,   0,   255)   # bright red for left hand dots


# ── Helper: run a pipeline step ───────────────────────────────────────────────

def run_pipeline_step(step_name: str, func, video_path: Path):
    """
    Call one of our pipeline functions (hand_pose or segmentation).
    Print a status line showing success or failure.

    step_name  — label shown in the terminal, e.g. "Hand Pose"
    func       — the process_video function to call
    video_path — path to the video file
    """
    print(f"\n  Running {step_name} ...")
    start = time.time()

    try:
        func(str(video_path))    # call the module's main function
        elapsed = time.time() - start
        print(f"  ✅ {step_name} complete  ({elapsed:.1f}s)")
        return True
    except Exception as e:
        print(f"  ❌ {step_name} FAILED: {e}")
        return False


# ── Helper: draw skeleton on a frame ─────────────────────────────────────────

def draw_skeleton(frame: np.ndarray, hand_data: dict, width: int, height: int):
    """
    Draw the 21-point hand skeleton (bones as lines, joints as dots)
    directly on the frame image.

    frame      — the image to draw on (modified in-place)
    hand_data  — one hand dict from the JSON: {"label": ..., "keypoints": {...}}
    width      — frame pixel width (used to know which colors to use)
    height     — frame pixel height
    """
    label     = hand_data.get("label", "")       # "Left" or "Right"
    keypoints = hand_data.get("keypoints", {})   # dict of landmark name → {px, py, ...}

    # Choose bone and dot colors based on which hand this is
    if label == "Right":
        bone_color = COLOR_BONE_RIGHT
        dot_color  = COLOR_DOT_RIGHT
    else:
        bone_color = COLOR_BONE_LEFT
        dot_color  = COLOR_DOT_LEFT

    # Build a list of (px, py) pixel positions, one per landmark index
    # We index by LANDMARK_NAMES so the connection tuples work correctly
    pts = []
    for name in LANDMARK_NAMES:
        kp = keypoints.get(name)
        if kp:
            pts.append((int(kp["px"]), int(kp["py"])))   # pixel coords from JSON
        else:
            pts.append(None)   # landmark missing — skip connections involving it

    # Draw the bone lines first (so dots appear on top)
    for start_idx, end_idx in HAND_CONNECTIONS:
        p1 = pts[start_idx]
        p2 = pts[end_idx]
        if p1 and p2:                        # only draw if both endpoints exist
            cv2.line(frame, p1, p2, bone_color, 2, cv2.LINE_AA)   # anti-aliased line

    # Draw a filled circle at each keypoint (joint)
    for pt in pts:
        if pt:
            cv2.circle(frame, pt, 6, bone_color, -1, cv2.LINE_AA)  # filled circle
            cv2.circle(frame, pt, 3, dot_color,  -1, cv2.LINE_AA)  # smaller dot on top


# ── Helper: draw text with a black outline ────────────────────────────────────

def draw_text_outlined(frame: np.ndarray, text: str, x: int, y: int,
                       scale: float = 0.6, thickness: int = 1):
    """
    Draw white text with a black outline so it stays readable
    over any background color (dark or light).

    We draw the text 8 times in black at small offsets (the outline),
    then once in white at the exact position (the fill).
    """
    font = cv2.FONT_HERSHEY_DUPLEX

    # Draw black outline — shifted one pixel in every diagonal direction
    for dx, dy in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
        cv2.putText(frame, text, (x+dx, y+dy), font, scale, (0,0,0), thickness+1, cv2.LINE_AA)

    # Draw white fill on top
    cv2.putText(frame, text, (x, y), font, scale, (255,255,255), thickness, cv2.LINE_AA)


# ── Step 3: stitch frames into annotated video ────────────────────────────────

def stitch_video(clip_name: str, fps: float, output_path: Path):
    """
    Load every segmented frame PNG, draw the skeleton on top using keypoints
    from the JSON, add a text overlay, then write all frames to an .mp4 video.

    clip_name   — e.g. "WashingCup"
    fps         — the video's frame rate (from the JSON metadata)
    output_path — where to save the final annotated video
    """

    # ── Load the hand pose JSON ───────────────────────────────
    json_path = HAND_POSE_ROOT / f"{clip_name}.json"
    print(f"\n  Loading keypoint JSON: {json_path}")

    with open(json_path, "r") as f:
        pose_data = json.load(f)

    # Build a frame_id → frame_dict lookup for fast access
    # Without this we'd have to search the whole list for every frame
    frame_lookup = {frame["frame_id"]: frame for frame in pose_data["frames"]}
    total_frames = len(frame_lookup)

    print(f"  {total_frames} frames loaded from JSON")

    # ── Collect segmented frame files ─────────────────────────
    segmented_dir = SEGMENTED_ROOT / clip_name
    # glob returns all matching files; sorted() ensures they are in frame order
    frame_files = sorted(segmented_dir.glob("frame_*.png"))

    if not frame_files:
        print(f"  [ERROR] No segmented frames found in {segmented_dir}")
        print(f"          Run segmentation.py first.")
        return False

    print(f"  {len(frame_files)} segmented frames found in {segmented_dir}")

    # ── Read the first frame to get image dimensions ──────────
    # We need width and height to set up the VideoWriter
    first_frame = cv2.imread(str(frame_files[0]))
    if first_frame is None:
        print(f"  [ERROR] Could not read first frame: {frame_files[0]}")
        return False

    height, width = first_frame.shape[:2]   # shape is (H, W, channels)
    print(f"  Frame size: {width} x {height}  |  FPS: {fps}")

    # ── Set up the VideoWriter ────────────────────────────────
    # mp4v is the codec for .mp4 files — widely supported
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        print(f"  [ERROR] Could not open VideoWriter for: {output_path}")
        return False

    print(f"\n  Stitching frames → {output_path}")
    print(f"  {'─' * 50}")

    # ── Process every frame ───────────────────────────────────
    stitch_start = time.time()

    for i, frame_file in enumerate(frame_files):

        # ── Load the segmented frame image ────────────────────
        # This PNG already has the colored SAM2 masks blended in
        frame = cv2.imread(str(frame_file))

        if frame is None:
            # If a frame failed to load, write a black frame as a placeholder
            frame = np.zeros((height, width, 3), dtype=np.uint8)

        # ── Get keypoint data for this frame ──────────────────
        # The frame filename is "frame_000042.png" — extract the number
        frame_id = int(frame_file.stem.split("_")[1])  # "000042" → 42

        # Look up this frame in the JSON (returns {} if not found)
        frame_data   = frame_lookup.get(frame_id, {})
        hands        = frame_data.get("hands", [])           # list of hand dicts
        num_hands    = frame_data.get("hands_detected", 0)   # 0, 1, or 2

        # ── Draw the hand skeleton on top ─────────────────────
        # The segmented frame has the colored masks; we add the skeleton on top
        for hand in hands:
            draw_skeleton(frame, hand, width, height)

        # ── Draw text overlay in top-left corner ──────────────
        # Line 1: frame counter
        draw_text_outlined(frame, f"Frame: {frame_id}", x=20, y=40, scale=0.75)

        # Line 2: hands detected count
        hand_str = f"Hands: {num_hands}"
        draw_text_outlined(frame, hand_str, x=20, y=75, scale=0.75)

        # ── Write this frame to the output video ──────────────
        writer.write(frame)

        # ── Print progress every 100 frames ───────────────────
        if i % 100 == 0:
            elapsed  = time.time() - stitch_start
            pct_done = (i / len(frame_files)) * 100
            print(
                f"  Frame {i:>6} / {len(frame_files)}"
                f"  ({pct_done:5.1f}%)"
                f"  |  {elapsed:.1f}s elapsed"
            )

    # ── Finish writing ────────────────────────────────────────
    writer.release()   # flush any remaining frames and close the file

    total_stitch_time = time.time() - stitch_start
    print(f"\n  ✅ Video saved → {output_path}")
    print(f"     {len(frame_files)} frames  |  {fps} fps  |  {total_stitch_time:.1f}s")
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    """
    Orchestrate all three steps for one video:
      Step 1 — hand_pose.py   (detect hands, save keypoints)
      Step 2 — segmentation.py (SAM2 colored masks)
      Step 3 — stitch frames into final annotated video
    """

    # ── Read the video path from the command line ─────────────
    if len(sys.argv) != 2:
        print("Usage  : python pipeline/annotate_video.py <path_to_video>")
        print("Example: python pipeline/annotate_video.py assets/videos/WashingCup.mp4")
        sys.exit(1)

    video_path = Path(sys.argv[1])

    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}")
        sys.exit(1)

    clip_name   = video_path.stem   # e.g. "WashingCup"
    output_path = OUTPUT_ROOT / f"{clip_name}_annotated.mp4"

    print(f"\n{'=' * 60}")
    print(f"  Full Annotation Pipeline")
    print(f"  Clip   : {clip_name}")
    print(f"  Output : {output_path}")
    print(f"{'=' * 60}")

    overall_start = time.time()

    # ── Step 1: Hand Pose ─────────────────────────────────────
    # Detect the 21 hand landmarks in every frame and save them to JSON.
    # This gives us precise pixel coordinates for each keypoint.
    print(f"\n{'─' * 60}")
    print("  STEP 1 of 3 — Hand Pose Detection")
    print("  MediaPipe finds 21 landmarks per hand per frame.")
    print(f"{'─' * 60}")

    pose_ok = run_pipeline_step("Hand Pose", hand_pose.process_video, video_path)
    if not pose_ok:
        print("  Cannot continue without hand pose data. Exiting.")
        sys.exit(1)

    # ── Step 2: SAM2 Segmentation ─────────────────────────────
    # Use the wrist coordinates from Step 1 as prompts for SAM2.
    # SAM2 segments the hand region and saves colored mask frames.
    print(f"\n{'─' * 60}")
    print("  STEP 2 of 3 — SAM2 Segmentation")
    print("  SAM2 uses wrist coordinates as prompts to segment hand regions.")
    print("  Left hand → red mask.  Right hand → purple mask.")
    print(f"{'─' * 60}")

    seg_ok = run_pipeline_step("Segmentation", segmentation.process_video, video_path)
    if not seg_ok:
        print("  Cannot stitch video without segmented frames. Exiting.")
        sys.exit(1)

    # ── Step 3: Stitch Annotated Video ────────────────────────
    # Load all the colored mask PNGs, draw skeleton on top, add text,
    # and combine into a single .mp4 file.
    print(f"\n{'─' * 60}")
    print("  STEP 3 of 3 — Stitching Annotated Video")
    print("  Loading segmented frames, drawing skeleton, adding text overlay.")
    print(f"{'─' * 60}")

    # Read FPS from the hand pose JSON so the output video matches the original
    json_path = HAND_POSE_ROOT / f"{clip_name}.json"
    with open(json_path, "r") as f:
        pose_meta = json.load(f)

    fps = pose_meta.get("fps", 30.0)   # default to 30 if not in JSON

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)   # create output folder

    stitch_ok = stitch_video(clip_name, fps, output_path)

    # ── Final summary ─────────────────────────────────────────
    total_time = time.time() - overall_start
    print(f"\n{'=' * 60}")
    if stitch_ok:
        print(f"  Pipeline complete!")
        print(f"  Output video : {output_path}")
    else:
        print(f"  Pipeline finished with errors.")
    print(f"  Total time   : {total_time:.1f}s")
    print(f"{'=' * 60}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

# This block only runs when you execute the file directly:
#   python pipeline/annotate_video.py assets/videos/WashingCup.mp4
if __name__ == "__main__":
    main()
