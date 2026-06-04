"""
segmentation.py
───────────────
Reads every frame of a video, loads wrist coordinates from the
existing hand pose JSON, and uses SAM2 to segment the hand regions.

Outputs:
  • colored segmentation frame PNGs → assets/processed/segmented/<clip_name>/
  • updated hand pose JSON          → assets/processed/hand_pose/<clip_name>.json
    (segmentation data is added into each frame entry)

Usage:
  python pipeline/segmentation.py assets/videos/WashingCup.mp4

NOTE: Run hand_pose.py on this video first — segmentation.py reads
      the wrist pixel coordinates that hand_pose.py produces.
"""

# ── Imports ───────────────────────────────────────────────────────────────────

import sys                         # read command-line arguments
import json                        # read and write JSON files
import time                        # measure how long the script takes
import urllib.request              # download the SAM2 checkpoint file
from pathlib import Path           # clean cross-platform file path handling

import cv2                         # OpenCV — open video, read frames, save images
import numpy as np                 # NumPy — arrays for mask math and pixel operations
import torch                       # PyTorch — required internally by SAM2

# SAM2 model builder and the per-image predictor interface
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# ── Folder paths ──────────────────────────────────────────────────────────────

# Where hand_pose.py wrote the keypoint JSON files
HAND_POSE_ROOT = Path("assets/processed/hand_pose")

# Where we will save segmented frame PNG images
SEGMENTED_ROOT = Path("assets/processed/segmented")

# Where we will store the downloaded SAM2 model weights
MODELS_DIR = Path("assets/models")


# ── SAM2 model settings ───────────────────────────────────────────────────────

# Official download URL for the SAM2 tiny checkpoint (~155 MB) from Meta
SAM2_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt"
)

# Local file path where we save the downloaded checkpoint
SAM2_CHECKPOINT_PATH = MODELS_DIR / "sam2_hiera_tiny.pt"

# Config filename bundled inside the sam2 pip package (tiny variant)
SAM2_CONFIG = "sam2_hiera_t.yaml"


# ── Mask color settings ───────────────────────────────────────────────────────

# Colors in RGB format (used in JSON output so they are human-readable)
COLOR_RIGHT_HAND_RGB = (128, 0, 128)   # purple
COLOR_LEFT_HAND_RGB  = (255, 0, 0)     # red

# OpenCV needs BGR (Blue, Green, Red) — the channel order is reversed vs RGB
# Purple (128, 0, 128) RGB → (128, 0, 128) BGR  — same because R==B here
# Red    (255, 0,   0) RGB → (  0, 0, 255) BGR  — channels are swapped
COLOR_RIGHT_HAND_BGR = (128, 0, 128)   # purple in BGR
COLOR_LEFT_HAND_BGR  = (0,   0, 255)   # red in BGR

# Mask transparency: 0.0 = invisible overlay, 1.0 = fully solid overlay
# 0.45 means 45% mask color + 55% original frame pixels
MASK_ALPHA = 0.45


# ── Device selection ──────────────────────────────────────────────────────────

def get_device() -> str:
    """
    Return the best compute device available on this machine.
    SAM2 runs fastest on a GPU; falls back to CPU if none is found.
      cuda — NVIDIA GPU
      mps  — Apple Silicon (M1/M2/M3) GPU via Metal
      cpu  — always available, but slow for large models
    """
    if torch.cuda.is_available():              # check for NVIDIA GPU
        return "cuda"
    if torch.backends.mps.is_available():      # check for Apple Silicon GPU
        return "mps"
    return "cpu"                               # plain CPU fallback


# ── Checkpoint download ───────────────────────────────────────────────────────

def download_checkpoint():
    """
    Download the SAM2 tiny model weights from Meta's servers if they are
    not already on disk. The file is around 155 MB and only downloads once.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)    # create assets/models/ if missing

    if SAM2_CHECKPOINT_PATH.exists():                # already downloaded — nothing to do
        print(f"  [model] Checkpoint found: {SAM2_CHECKPOINT_PATH}")
        return

    print(f"  [model] Downloading SAM2 tiny checkpoint (~155 MB) ...")
    print(f"          From : {SAM2_CHECKPOINT_URL}")
    print(f"          To   : {SAM2_CHECKPOINT_PATH}")

    # urlretrieve downloads the URL and saves it to the given local path
    urllib.request.urlretrieve(SAM2_CHECKPOINT_URL, SAM2_CHECKPOINT_PATH)

    print(f"  [model] Download complete.\n")


# ── SAM2 model loader ─────────────────────────────────────────────────────────

def load_sam2(device: str) -> SAM2ImagePredictor:
    """
    Build and return a SAM2ImagePredictor ready to process images.

    How SAM2ImagePredictor works:
      1. predictor.set_image(frame_rgb)
             → SAM2's image encoder runs on this frame once
      2. predictor.predict(point_coords=..., point_labels=...)
             → SAM2 generates a segmentation mask from the given points
    """
    print(f"  [model] Loading SAM2 tiny  (device: {device}) ...")

    # build_sam2 reads the YAML config and loads the neural network weights
    sam2_model = build_sam2(
        SAM2_CONFIG,                      # config YAML bundled in the sam2 package
        str(SAM2_CHECKPOINT_PATH),        # path to downloaded weights
        device=device,                    # where to run inference
    )

    # Wrap the raw model in the SAM2ImagePredictor interface
    predictor = SAM2ImagePredictor(sam2_model)

    print("  [model] SAM2 ready.\n")
    return predictor


# ── Per-frame segmentation ────────────────────────────────────────────────────

def segment_frame(
    frame_bgr:  np.ndarray,           # raw video frame in OpenCV BGR format
    hands:      list,                 # list of hand dicts from the hand pose JSON
    predictor:  SAM2ImagePredictor,   # loaded SAM2 model
    frame_h:    int,                  # frame height in pixels
    frame_w:    int,                  # frame width in pixels
) -> tuple:
    """
    Run SAM2 on one frame.

    For each hand found in this frame:
      - use the wrist (px, py) as a SAM2 foreground point prompt
      - get the segmentation mask back from SAM2
      - blend the colored mask onto the frame

    Returns:
      annotated_frame  — BGR image with hand masks overlaid
      masks_info       — list of dicts with mask statistics (for JSON)
    """

    annotated_frame = frame_bgr.copy()   # copy so we don't modify the original frame
    masks_info = []                      # will collect one dict per hand mask

    # If no hands were detected in this frame, return the unchanged frame
    if not hands:
        return annotated_frame, masks_info

    # ── Convert frame to RGB for SAM2 ────────────────────────
    # OpenCV stores pixels as BGR; SAM2 expects RGB — we swap the channel order
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # ── Encode the frame once — expensive step done only once per frame ───
    # SAM2's image encoder (a Vision Transformer) converts the raw pixels
    # into a rich feature map. This runs once per frame regardless of how
    # many hands we then segment from it.
    with torch.no_grad():              # no_grad: don't store gradient history (saves RAM)
        predictor.set_image(frame_rgb)

    # ── Segment each detected hand ────────────────────────────
    for hand in hands:

        # Determine mask color based on which hand ("Left" or "Right")
        hand_label = hand.get("label", "")

        if hand_label == "Right":
            color_bgr  = COLOR_RIGHT_HAND_BGR   # purple — for drawing on frame
            color_rgb  = COLOR_RIGHT_HAND_RGB   # purple — for JSON storage
            mask_label = "right_hand"
        else:
            color_bgr  = COLOR_LEFT_HAND_BGR    # red — for drawing on frame
            color_rgb  = COLOR_LEFT_HAND_RGB    # red — for JSON storage
            mask_label = "left_hand"

        # ── Get wrist pixel coordinates ───────────────────────
        # The wrist is the base of the hand — a stable, reliable prompt point
        keypoints = hand.get("keypoints", {})   # dict of all 21 landmark positions
        wrist     = keypoints.get("WRIST")      # the WRIST entry: {"px": ..., "py": ...}

        if wrist is None:
            continue   # this hand has no wrist data — skip it

        px = wrist["px"]   # wrist x position in pixels (0 = left edge of frame)
        py = wrist["py"]   # wrist y position in pixels (0 = top edge of frame)

        # ── Build the SAM2 point prompt arrays ───────────────
        # SAM2 accepts a list of (x, y) pixel coordinates.
        # shape of point_coords must be (N, 2) where N = number of points
        # We provide just one point — the wrist.
        point_coords = np.array([[px, py]], dtype=np.float32)   # shape: (1, 2)

        # point_labels tells SAM2 whether each point is foreground (1) or background (0)
        # We mark the wrist as foreground (1) so SAM2 segments toward the hand
        point_labels = np.array([1], dtype=np.int32)            # shape: (1,)

        # ── Run SAM2 mask prediction ──────────────────────────
        # predictor.predict() returns three things:
        #   masks  — boolean array of shape (N_masks, H, W)
        #            True  = this pixel belongs to the segmented object
        #            False = background
        #   scores — confidence score for each mask (higher is better)
        #   _      — logits (raw model output before thresholding — we ignore these)
        with torch.no_grad():
            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False,    # False = return single best mask
            )

        # masks[0] is the first (and only) mask when multimask_output=False
        # SAM2 on MPS/CPU may return uint8 or float — force to bool so NumPy
        # array indexing (overlay[mask] = color) works correctly on all devices
        mask = masks[0].astype(bool)

        # ── Blend the colored mask onto the frame ────────────
        # Strategy: paint the mask color onto a copy (overlay), then
        # mix the overlay and the original with addWeighted.
        # This keeps the underlying texture partially visible.
        overlay        = annotated_frame.copy()   # copy of current annotated frame
        overlay[mask]  = color_bgr                # paint every hand pixel with the color

        # addWeighted formula per pixel:
        #   result = overlay * MASK_ALPHA + annotated_frame * (1 - MASK_ALPHA)
        annotated_frame = cv2.addWeighted(
            overlay,         MASK_ALPHA,          # mask color layer
            annotated_frame, 1.0 - MASK_ALPHA,    # original frame layer
            0,                                    # gamma brightness offset (0 = no shift)
        )

        # ── Compute mask statistics for the JSON ─────────────
        pixel_count  = int(mask.sum())                                         # count True pixels
        coverage_pct = round((pixel_count / (frame_h * frame_w)) * 100, 4)    # % of frame covered

        # Compute bounding box — convert bool mask to uint8 for findContours
        mask_uint8 = mask.astype(np.uint8) * 255       # True→255, False→0
        contours, _ = cv2.findContours(
            mask_uint8,
            cv2.RETR_EXTERNAL,          # only outer contours (no nested holes)
            cv2.CHAIN_APPROX_SIMPLE,    # compress straight lines to endpoints
        )

        if contours:
            # Pick the largest contour (most likely the actual hand)
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)   # bounding rectangle
            bbox = [int(x), int(y), int(w), int(h)]  # convert to plain ints for JSON
        else:
            bbox = [0, 0, 0, 0]    # empty fallback

        # Append this hand's mask info to the list
        masks_info.append({
            "label":        mask_label,           # "right_hand" or "left_hand"
            "color":        list(color_rgb),      # RGB e.g. [128, 0, 128]
            "pixel_count":  pixel_count,          # raw pixel count inside the mask
            "coverage_pct": coverage_pct,         # percentage of the full frame
            "bbox":         bbox,                 # [x, y, width, height]
        })

    return annotated_frame, masks_info    # return the drawn frame and mask data


# ── Main pipeline ─────────────────────────────────────────────────────────────

def process_video(video_path: str):
    """
    Runs the full segmentation pipeline for one video file.
    """

    video_path = Path(video_path)    # convert string to Path object

    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}")
        sys.exit(1)

    clip_name = video_path.stem      # e.g. "WashingCup" (filename without extension)

    print(f"\n{'=' * 60}")
    print(f"  Segmentation Pipeline  (SAM2)")
    print(f"  Clip : {clip_name}")
    print(f"{'=' * 60}\n")

    # ── Step 1: load the hand pose JSON ──────────────────────
    json_path = HAND_POSE_ROOT / f"{clip_name}.json"

    if not json_path.exists():
        print(f"[ERROR] Hand pose JSON not found: {json_path}")
        print(f"        Please run hand_pose.py first:")
        print(f"          python pipeline/hand_pose.py {video_path}")
        sys.exit(1)

    print(f"  [1] Loading hand pose JSON ...")
    with open(json_path, "r") as f:
        pose_data = json.load(f)      # load the whole JSON into a Python dict

    # Build a frame_id → frame_dict lookup for fast access during the video loop
    # Without this we would have to scan the whole list for each frame (slow)
    frame_lookup = {frame["frame_id"]: frame for frame in pose_data["frames"]}

    print(f"       {len(pose_data['frames'])} frames loaded\n")

    # ── Step 2: download + load SAM2 ─────────────────────────
    print("  [2] Setting up SAM2 model ...")
    download_checkpoint()              # downloads only if not already on disk
    device    = get_device()           # pick best available device
    predictor = load_sam2(device)      # load model weights

    # ── Step 3: open video ────────────────────────────────────
    print(f"  [3] Opening video ...")
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"[ERROR] Could not open: {video_path}")
        sys.exit(1)

    fps          = cap.get(cv2.CAP_PROP_FPS)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"       {width} x {height}  |  {fps} fps  |  {total_frames} frames\n")

    # ── Step 4: create output folder ─────────────────────────
    segmented_dir = SEGMENTED_ROOT / clip_name
    segmented_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 5: loop over every frame ────────────────────────

    frame_id           = 0      # current frame counter
    count_segmented    = 0      # frames where SAM2 produced at least one mask
    total_coverage_sum = 0.0    # running sum of coverage % for the average calculation
    total_masks_count  = 0      # total number of individual masks across all frames
    start_time         = time.time()

    while True:

        ret, frame = cap.read()    # read the next frame from the video file
        if not ret:
            break                  # end of video — exit the loop

        # Look up the pose data for this frame (returns {} if not found)
        frame_pose = frame_lookup.get(frame_id, {})
        hands      = frame_pose.get("hands", [])   # list of hand dicts (may be empty)

        # ── Run SAM2 segmentation on this frame ──────────────
        annotated_frame, masks_info = segment_frame(
            frame_bgr=frame,
            hands=hands,
            predictor=predictor,
            frame_h=height,
            frame_w=width,
        )

        # ── Save annotated frame as PNG ───────────────────────
        fname       = f"frame_{str(frame_id).zfill(6)}.png"
        save_path   = segmented_dir / fname
        cv2.imwrite(str(save_path), annotated_frame)

        # ── Update statistics ─────────────────────────────────
        if masks_info:
            count_segmented += 1
            for m in masks_info:
                total_coverage_sum += m["coverage_pct"]
                total_masks_count  += 1

        # ── Attach segmentation result to the JSON frame entry
        # frame_lookup[frame_id] is the same dict that lives inside pose_data["frames"],
        # so updating it here automatically updates pose_data too (Python passes by reference)
        if frame_id in frame_lookup:
            frame_lookup[frame_id]["segmentation"] = {
                "method": "SAM2",       # record which algorithm produced this
                "masks":  masks_info,   # list of mask dicts (empty if no hands)
            }

        # ── Print progress every 100 frames ──────────────────
        if frame_id % 100 == 0:
            elapsed  = time.time() - start_time
            pct_done = (frame_id / total_frames) * 100
            print(
                f"  Frame {frame_id:>6} / {total_frames}"
                f"  ({pct_done:5.1f}%)"
                f"  |  segmented: {count_segmented}"
                f"  |  {elapsed:.1f}s"
            )

        frame_id += 1    # advance frame counter

    # ── Step 6: release video ─────────────────────────────────
    cap.release()    # close the video file handle

    # ── Step 7: write the updated JSON ───────────────────────
    # pose_data["frames"] now contains the new "segmentation" key on each frame
    print(f"\n  [6] Saving updated JSON → {json_path}")
    with open(json_path, "w") as f:
        json.dump(pose_data, f, indent=2)
    print(f"       Saved.\n")

    # ── Step 8: final summary ─────────────────────────────────

    total_time   = time.time() - start_time
    avg_coverage = (total_coverage_sum / total_masks_count) if total_masks_count > 0 else 0.0

    print(f"{'─' * 60}")
    print(f"  SUMMARY  —  {clip_name}")
    print(f"{'─' * 60}")
    print(f"  Total frames processed      : {frame_id}")
    print(f"  Frames with segmentation    : {count_segmented}")
    print(f"  Frames without segmentation : {frame_id - count_segmented}")
    print(f"  Total hand masks created    : {total_masks_count}")
    print(f"  Average hand coverage       : {avg_coverage:.2f}%")
    print(f"  Device used                 : {device}")
    print(f"  Time taken                  : {total_time:.1f}s")
    print(f"\n  Segmented frames  ->  {segmented_dir}/")
    print(f"  Updated JSON      ->  {json_path}")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

# This block only executes when the file is run directly from the terminal.
# It is skipped when another script does  "import segmentation".
if __name__ == "__main__":

    # sys.argv is the list of tokens typed on the command line:
    #   sys.argv[0] = "pipeline/segmentation.py"  (script name, always present)
    #   sys.argv[1] = the video path the user typed (what we need)
    if len(sys.argv) != 2:
        print("Usage  : python pipeline/segmentation.py <path_to_video>")
        print("Example: python pipeline/segmentation.py assets/videos/WashingCup.mp4")
        print()
        print("IMPORTANT: Run hand_pose.py on the same video first.")
        sys.exit(1)

    process_video(sys.argv[1])    # call the main function with the user's video path
