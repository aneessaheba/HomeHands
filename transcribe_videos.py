#!/usr/bin/env python3
"""
transcribe_videos.py
────────────────────
Batch-processes every clip in VIDEOS_DIR:
  1. Extracts audio with ffmpeg
  2. Transcribes narration with OpenAI Whisper
  3. Writes  <clip>.srt   — subtitle file
             <clip>.txt   — plain transcript
             <clip>.json  — narration annotation for HomeHands dataset
  4. Burns subtitles into a new video:  <clip>_subtitled.mp4

Usage
-----
  python transcribe_videos.py

Requirements
------------
  brew install ffmpeg          # or apt-get install ffmpeg on Linux
  pip install openai-whisper   # installed automatically if missing
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


# ── CONFIG ────────────────────────────────────────────────────────────────────
VIDEOS_DIR    = Path("assets/videos")           # folder with source .mp4 clips
OUTPUT_DIR    = Path("assets/videos/subtitled") # where subtitled videos are saved
ANNOT_DIR     = Path("assets/data/narrations")  # SRT / TXT / JSON per clip
WHISPER_MODEL = "base"    # tiny | base | small | medium | large
                          # larger = more accurate, slower; "base" is a good start
# ──────────────────────────────────────────────────────────────────────────────


# ── DEPENDENCY CHECKS ─────────────────────────────────────────────────────────

def check_ffmpeg():
    """Abort early if ffmpeg is not on PATH."""
    if shutil.which("ffmpeg") is None:
        sys.exit(
            "\n[ERROR] ffmpeg not found.\n"
            "  macOS : brew install ffmpeg\n"
            "  Ubuntu: sudo apt-get install ffmpeg\n"
        )


def ensure_whisper():
    """Import whisper, installing it automatically if it is missing."""
    try:
        import whisper
        return whisper
    except ImportError:
        print("[setup] openai-whisper not found — installing …")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "openai-whisper"],
            check=True,
        )
        import whisper
        return whisper


# ── AUDIO EXTRACTION ──────────────────────────────────────────────────────────

def extract_audio(video_path: Path, audio_path: Path):
    """
    Pull audio from the video and save as 16 kHz mono WAV.
    Whisper was trained on 16 kHz audio, so this format gives best accuracy.
    """
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-ac", "1",        # mono — Whisper doesn't need stereo
            "-ar", "16000",    # 16 kHz sample rate
            "-vn",             # drop video stream, audio only
            str(audio_path),
        ],
        check=True,
        capture_output=True,   # suppress ffmpeg's verbose output
    )


# ── TRANSCRIPTION ─────────────────────────────────────────────────────────────

def transcribe(audio_path: Path, model) -> list[dict]:
    """
    Run Whisper on the WAV file.
    Returns a list of timed segments:
      [{"start": 0.0, "end": 2.3, "text": "I am picking up the cup"}, ...]
    """
    result = model.transcribe(
        str(audio_path),
        language="en",  # set explicitly so Whisper skips language detection
        fp16=False,     # fp16 only works on CUDA GPUs; CPU requires fp32
    )

    segments = []
    for seg in result["segments"]:
        segments.append({
            "start": round(float(seg["start"]), 3),
            "end":   round(float(seg["end"]),   3),
            "text":  seg["text"].strip(),
        })
    return segments


# ── OUTPUT FILE WRITERS ───────────────────────────────────────────────────────

def _srt_timestamp(seconds: float) -> str:
    """Convert float seconds → SRT timestamp  HH:MM:SS,mmm."""
    h  = int(seconds // 3600)
    m  = int((seconds % 3600) // 60)
    s  = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segments: list[dict], srt_path: Path):
    """
    Standard SubRip (.srt) format:

        1
        00:00:00,000 --> 00:00:02,300
        I am picking up the cup

        2
        ...
    """
    lines = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(
            f"{_srt_timestamp(seg['start'])} --> {_srt_timestamp(seg['end'])}"
        )
        lines.append(seg["text"])
        lines.append("")   # blank line required between SRT entries
    srt_path.write_text("\n".join(lines), encoding="utf-8")


def write_txt(segments: list[dict], txt_path: Path):
    """Plain transcript — one line per Whisper segment, no timestamps."""
    txt_path.write_text(
        "\n".join(seg["text"] for seg in segments),
        encoding="utf-8",
    )


def write_json(clip_name: str, segments: list[dict], json_path: Path):
    """
    HomeHands narration annotation format:
    {
      "clip": "WashingCup.mp4",
      "narrations": [
        {"start": 0.0, "end": 2.3, "text": "I am picking up the cup"},
        ...
      ]
    }
    """
    payload = {
        "clip": clip_name,
        "narrations": [
            {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
            for seg in segments
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ── SUBTITLE BURNING ──────────────────────────────────────────────────────────

def burn_subtitles(video_path: Path, srt_path: Path, out_path: Path):
    """
    Hardcode (burn) subtitles into a new video file using ffmpeg.

    The SRT file is copied to a temp location with no spaces in the name
    because ffmpeg's subtitles filter doesn't handle spaces in paths well.

    Style choices:
      - White text, black outline, no shadow
      - Bottom-center (Alignment=2 in SSA convention)
      - Arial 20pt — clean and readable on typical egocentric footage
    """
    subtitle_style = (
        "FontName=Arial,"
        "FontSize=20,"
        "PrimaryColour=&H00FFFFFF,"    # white text
        "OutlineColour=&H00000000,"    # black outline
        "Outline=2,"                   # 2px outline thickness
        "Shadow=0,"                    # no drop shadow
        "Alignment=2,"                 # bottom-center
        "MarginV=25"                   # pixels from bottom edge
    )

    # Copy SRT to a temp file with a space-free name to avoid ffmpeg path issues
    with tempfile.NamedTemporaryFile(suffix=".srt", delete=False) as tmp:
        tmp_srt = Path(tmp.name)
    shutil.copy(srt_path, tmp_srt)

    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-vf", f"subtitles={tmp_srt}:force_style='{subtitle_style}'",
                "-c:a", "copy",        # copy audio stream unchanged — no re-encode
                str(out_path),
            ],
            check=True,
        )
    finally:
        tmp_srt.unlink(missing_ok=True)   # clean up temp SRT regardless of outcome


# ── PER-CLIP PIPELINE ─────────────────────────────────────────────────────────

def process_clip(video_path: Path, model) -> dict:
    """
    Full pipeline for one clip.
    Returns a summary dict {"clip", "status", "segments" | "error"}.
    """
    stem = video_path.stem

    # Derive output paths
    srt_path  = ANNOT_DIR  / f"{stem}.srt"
    txt_path  = ANNOT_DIR  / f"{stem}.txt"
    json_path = ANNOT_DIR  / f"{stem}.json"
    out_video = OUTPUT_DIR / f"{stem}_subtitled.mp4"

    print(f"\n{'─' * 60}")
    print(f"  Clip: {video_path.name}")
    print(f"{'─' * 60}")

    # ── Step 1: extract audio to a temp WAV ──────────────────
    print("  [1/4] Extracting audio …")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_path = Path(tmp.name)
    try:
        extract_audio(video_path, audio_path)
    except subprocess.CalledProcessError as e:
        audio_path.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg audio extraction failed: {e}") from e

    # ── Step 2: transcribe ───────────────────────────────────
    print("  [2/4] Transcribing with Whisper …")
    try:
        segments = transcribe(audio_path, model)
    finally:
        audio_path.unlink(missing_ok=True)   # temp WAV no longer needed

    print(f"        → {len(segments)} segment(s) detected")
    for seg in segments:
        print(f"          [{seg['start']:6.2f}s – {seg['end']:6.2f}s]  {seg['text']}")

    # ── Step 3: write annotation files ───────────────────────
    print("  [3/4] Writing .srt / .txt / .json …")
    write_srt(segments,  srt_path)
    write_txt(segments,  txt_path)
    write_json(video_path.name, segments, json_path)

    # ── Step 4: burn subtitles ───────────────────────────────
    print("  [4/4] Burning subtitles into video …")
    burn_subtitles(video_path, srt_path, out_video)

    print(f"  ✓  Saved → {out_video.name}")
    return {"clip": video_path.name, "status": "ok", "segments": len(segments)}


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    check_ffmpeg()
    whisper = ensure_whisper()

    # Create output directories (no-op if they already exist)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ANNOT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect clips — skip anything that is already a subtitled output
    clips = sorted(
        p for p in VIDEOS_DIR.iterdir()
        if p.suffix.lower() in {".mp4", ".mov", ".avi"}
        and "_subtitled" not in p.stem
        and p.parent == VIDEOS_DIR   # don't recurse into subtitled/ subfolder
    )

    if not clips:
        sys.exit(f"\n[ERROR] No video files found in {VIDEOS_DIR}\n")

    print(f"\nFound {len(clips)} clip(s) to process.")
    print(f"Loading Whisper model '{WHISPER_MODEL}' …  (first run downloads the model)\n")
    model = whisper.load_model(WHISPER_MODEL)
    print("Model ready.\n")

    summary = []
    for clip in clips:
        try:
            result = process_clip(clip, model)
        except Exception as exc:
            print(f"  ✗  Error: {exc}")
            result = {"clip": clip.name, "status": "error", "error": str(exc)}
        summary.append(result)

    # ── Final summary ────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print("  SUMMARY")
    print(f"{'═' * 60}")
    ok  = [s for s in summary if s["status"] == "ok"]
    err = [s for s in summary if s["status"] != "ok"]
    for s in ok:
        print(f"  ✓  {s['clip']}  —  {s['segments']} segment(s)")
    for s in err:
        print(f"  ✗  {s['clip']}  —  {s.get('error', 'unknown error')}")
    print(f"\n  {len(ok)} succeeded  ·  {len(err)} failed")
    print(f"\n  Subtitled videos : {OUTPUT_DIR}/")
    print(f"  Annotations      : {ANNOT_DIR}/")
    print()


if __name__ == "__main__":
    main()
