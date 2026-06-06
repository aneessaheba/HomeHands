"""
segmentation.py
───────────────
SAM3-based arm segmentation pipeline for egocentric household activity videos.

Processes every 3rd frame with SAM3; interpolates (copies) masks for the 2
frames in between. Left arm is painted RED, right arm GREEN. Left arm is
always drawn on top so it wins when the two regions overlap.

Outputs:
  • colored PNG per frame   →  assets/processed/segmented/<clip_name>/
  • updated hand pose JSON  →  assets/processed/hand_pose/<clip_name>.json
    (sam3_segmentation block added to every frame entry)

Usage:
  python pipeline/segmentation.py assets/videos/WashingCup.mp4
"""

import sys
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from mlx_vlm.utils import load_model, get_model_path
from mlx_vlm.models.sam3.generate import Sam3Predictor
from mlx_vlm.models.sam3.processing_sam3 import Sam3Processor


# ── Paths ─────────────────────────────────────────────────────────────────────

HAND_POSE_ROOT = Path("assets/processed/hand_pose")
SEGMENTED_ROOT = Path("assets/processed/segmented")

# ── Colors ────────────────────────────────────────────────────────────────────

COLOR_LEFT_RGB  = (255,   0,   0)   # red   — left arm
COLOR_RIGHT_RGB = (  0, 255,   0)   # green — right arm

# ── Sampling ──────────────────────────────────────────────────────────────────

PROCESS_EVERY_N = 9   # run SAM3 on every 9th frame; copy result to frames in between


# ── Model loader ──────────────────────────────────────────────────────────────

def load_sam3() -> Sam3Predictor:
    print("  [model] Loading SAM3 8-bit (mlx-community/sam3-8bit) ...")
    model_path = get_model_path("mlx-community/sam3-8bit")
    model      = load_model(model_path)
    processor  = Sam3Processor.from_pretrained(str(model_path))
    predictor  = Sam3Predictor(model, processor, score_threshold=0.3)
    print("  [model] SAM3 ready.\n")
    return predictor


# ── SAM3 inference ────────────────────────────────────────────────────────────

def run_prompt(predictor: Sam3Predictor, pil_image: Image.Image, text_prompt: str):
    """
    Run SAM3 with a single text prompt.
    All returned masks are combined (OR) into one boolean H×W array.
    Returns the combined mask, or None if nothing was found or an error occurred.
    """
    try:
        result = predictor.predict(pil_image, text_prompt=text_prompt)

        if result is None:
            return None

        # Normalise: result may be a dict {'masks': ...} or a bare array/list
        if isinstance(result, dict):
            masks = result.get("masks", None)
        elif hasattr(result, "masks"):
            masks = result.masks
        else:
            masks = result

        if masks is None:
            return None

        arr = np.array(masks)

        if arr.size == 0:
            return None

        # Shape can be (H, W), (N, H, W), or (1, N, H, W) depending on version
        if arr.ndim == 2:
            combined = arr.astype(bool)
        elif arr.ndim == 3:
            combined = arr.any(axis=0).astype(bool)
        elif arr.ndim == 4:
            combined = arr.any(axis=(0, 1)).astype(bool)
        else:
            return None

        return combined if combined.any() else None

    except Exception:
        return None


# ── Frame painter ─────────────────────────────────────────────────────────────

def paint_solid(frame_bgr: np.ndarray, mask: np.ndarray, color_rgb: tuple) -> np.ndarray:
    """Return a copy of frame_bgr with mask pixels set to color_rgb (solid)."""
    out = frame_bgr.copy()
    out[mask] = (color_rgb[2], color_rgb[1], color_rgb[0])   # RGB → BGR
    return out


# ── Stats helper ──────────────────────────────────────────────────────────────

def arm_info(mask, frame_h: int, frame_w: int) -> dict:
    if mask is None:
        return {"found": False, "pixel_count": 0, "coverage_pct": 0.0}
    px  = int(mask.sum())
    cov = round((px / (frame_h * frame_w)) * 100, 4)
    return {"found": True, "pixel_count": px, "coverage_pct": cov}


# ── Per-frame segmentation ────────────────────────────────────────────────────

def segment_frame(predictor, frame_bgr, frame_h, frame_w):
    """
    Run SAM3 on one BGR frame.
    Returns (annotated_bgr, left_info_dict, right_info_dict).
    Right arm is painted first, left arm on top → left always wins on overlap.
    """
    frame_bgr = cv2.resize(frame_bgr, (1920, 1080))
    pil_img    = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    right_mask = run_prompt(predictor, pil_img, "right arm")
    left_mask  = run_prompt(predictor, pil_img, "left arm")

    original_h, original_w = frame_h, frame_w
    if left_mask is not None:
        left_mask = cv2.resize(left_mask.astype(np.uint8),
                    (original_w, original_h)).astype(bool)
    if right_mask is not None:
        right_mask = cv2.resize(right_mask.astype(np.uint8),
                     (original_w, original_h)).astype(bool)

    annotated = cv2.resize(frame_bgr, (original_w, original_h))
    if right_mask is not None:
        annotated = paint_solid(annotated, right_mask, COLOR_RIGHT_RGB)
    if left_mask is not None:
        annotated = paint_solid(annotated, left_mask,  COLOR_LEFT_RGB)

    return (
        annotated,
        arm_info(left_mask,  frame_h, frame_w),
        arm_info(right_mask, frame_h, frame_w),
    )


# ── Main pipeline ─────────────────────────────────────────────────────────────

def process_video(video_path_str: str):
    video_path = Path(video_path_str)
    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}")
        sys.exit(1)

    clip_name = video_path.stem

    print(f"\n{'=' * 60}")
    print(f"  Segmentation Pipeline  (SAM3)")
    print(f"  Clip : {clip_name}")
    print(f"{'=' * 60}\n")

    # ── Load hand pose JSON ───────────────────────────────────
    json_path = HAND_POSE_ROOT / f"{clip_name}.json"
    if not json_path.exists():
        print(f"[ERROR] Hand pose JSON not found: {json_path}")
        print(f"        Run hand_pose.py first.")
        sys.exit(1)

    print("  [1] Loading hand pose JSON ...")
    with open(json_path) as f:
        pose_data = json.load(f)
    frame_lookup = {fr["frame_id"]: fr for fr in pose_data["frames"]}
    print(f"       {len(pose_data['frames'])} frames loaded\n")

    # ── Load SAM3 model once ──────────────────────────────────
    print("  [2] Loading SAM3 model ...")
    predictor = load_sam3()

    # ── Open video ────────────────────────────────────────────
    print("  [3] Opening video ...")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Could not open: {video_path}")
        sys.exit(1)

    fps          = cap.get(cv2.CAP_PROP_FPS)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"       {width}x{height} | {fps:.1f} fps | {total_frames} frames\n")

    # ── Create output folder ──────────────────────────────────
    segmented_dir = SEGMENTED_ROOT / clip_name
    segmented_dir.mkdir(parents=True, exist_ok=True)

    # ── Frame loop ────────────────────────────────────────────
    frame_id   = 0
    start_time = time.time()

    # Running stats (counted over every frame including interpolated ones)
    left_found  = 0
    right_found = 0
    left_cov_sum  = 0.0
    right_cov_sum = 0.0

    # Carry-forward state for interpolated frames
    prev_annotated  = None
    prev_left_info  = {"found": False, "pixel_count": 0, "coverage_pct": 0.0}
    prev_right_info = {"found": False, "pixel_count": 0, "coverage_pct": 0.0}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % PROCESS_EVERY_N == 0:
            # SAM3 inference on this frame
            annotated, left_info, right_info = segment_frame(
                predictor, frame, height, width
            )
            prev_annotated  = annotated
            prev_left_info  = left_info
            prev_right_info = right_info
        else:
            # Interpolate: copy previous frame's result
            annotated  = prev_annotated if prev_annotated is not None else frame
            left_info  = prev_left_info
            right_info = prev_right_info

        # Save PNG
        cv2.imwrite(str(segmented_dir / f"frame_{str(frame_id).zfill(6)}.png"), annotated)

        # Attach to JSON
        if frame_id in frame_lookup:
            frame_lookup[frame_id]["sam3_segmentation"] = {
                "method":    "SAM3-8bit-MLX",
                "left_arm":  left_info,
                "right_arm": right_info,
            }

        # Accumulate stats
        if left_info["found"]:
            left_found   += 1
            left_cov_sum += left_info["coverage_pct"]
        if right_info["found"]:
            right_found   += 1
            right_cov_sum += right_info["coverage_pct"]

        # Progress print every 50 frames
        if frame_id > 0 and frame_id % 50 == 0:
            elapsed = time.time() - start_time
            lpx = left_info["pixel_count"]
            rpx = right_info["pixel_count"]
            print(
                f"  Frame {frame_id}/{total_frames}"
                f" | L:{lpx}px | R:{rpx}px"
                f" | {elapsed:.1f}s elapsed"
            )

        frame_id += 1

    cap.release()

    # ── Save updated JSON ─────────────────────────────────────
    print(f"\n  Saving updated JSON → {json_path}")
    with open(json_path, "w") as f:
        json.dump(pose_data, f, indent=2)

    # ── Final summary ─────────────────────────────────────────
    total_time    = time.time() - start_time
    avg_left_cov  = (left_cov_sum  / left_found)  if left_found  > 0 else 0.0
    avg_right_cov = (right_cov_sum / right_found) if right_found > 0 else 0.0
    left_pct      = (left_found  / frame_id * 100) if frame_id > 0 else 0.0
    right_pct     = (right_found / frame_id * 100) if frame_id > 0 else 0.0

    print(f"\n{'─' * 60}")
    print(f"  Frames processed     : {frame_id}")
    print(f"  Frames with left arm : {left_found} ({left_pct:.1f}%)")
    print(f"  Frames with right arm: {right_found} ({right_pct:.1f}%)")
    print(f"  Avg left coverage    : {avg_left_cov:.1f}%")
    print(f"  Avg right coverage   : {avg_right_cov:.1f}%")
    print(f"  Output saved to      : {segmented_dir}/")
    print(f"{'─' * 60}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage  : python pipeline/segmentation.py <path_to_video>")
        print("Example: python pipeline/segmentation.py assets/videos/WashingCup.mp4")
        sys.exit(1)

    process_video(sys.argv[1])
