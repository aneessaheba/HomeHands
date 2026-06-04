"""
transcribe.py
─────────────
Extracts audio from a video, transcribes it with local OpenAI Whisper
(no internet / no API key required), burns subtitles into a new video,
and saves everything as .srt, .json, and a subtitled .mp4.

Outputs:
  assets/processed/narrations/[clip_name].srt
  assets/processed/narrations/[clip_name].json
  assets/processed/subtitled/[clip_name]_subtitled.mp4

Usage:
  python pipeline/transcribe.py assets/videos/WashingCup.mp4
"""

# ── Imports ───────────────────────────────────────────────────────────────────

import sys          # read command-line arguments (the video path the user types)
import json         # write Python dicts to .json files
import shutil       # check whether ffmpeg is installed (shutil.which)
import subprocess   # run ffmpeg commands as external shell processes
import tempfile     # create a temporary .wav file that is auto-deleted when done
import time         # measure how long the script takes to run
from pathlib import Path   # clean, cross-platform file path handling

import whisper      # OpenAI Whisper — local speech-to-text, runs entirely offline


# ── Output folder paths ───────────────────────────────────────────────────────

# Where .srt and .json narration files will be saved
NARRATIONS_DIR = Path("assets/processed/narrations")

# Where the subtitled video copies will be saved
SUBTITLED_DIR  = Path("assets/processed/subtitled")


# ── Whisper model setting ─────────────────────────────────────────────────────

# "base" is the smallest useful model (~145 MB).
# Larger options: "small", "medium", "large" — more accurate but slower.
WHISPER_MODEL = "base"


# ── Subtitle style (for ffmpeg subtitle burning) ──────────────────────────────

# These settings are passed to ffmpeg's subtitle filter.
# They follow the SSA/ASS subtitle format (SubStation Alpha).
SUBTITLE_STYLE = (
    "FontName=Arial,"          # font family — Arial is clean and widely available
    "FontSize=24,"             # font size in points
    "PrimaryColour=&H00FFFFFF,"  # text fill color: white  (&H00 = fully opaque in SSA)
    "OutlineColour=&H00000000,"  # outline color: black
    "Outline=2,"               # outline thickness in pixels
    "Shadow=0,"                # no drop shadow
    "Alignment=2,"             # 2 = bottom-center in SSA alignment numbering
    "MarginV=25"               # vertical margin from the bottom edge in pixels
)


# ── Helper: check ffmpeg ───────────────────────────────────────────────────────

def check_ffmpeg():
    """
    Make sure ffmpeg is installed and reachable.
    If not found, print installation instructions and exit.
    """
    if shutil.which("ffmpeg") is None:    # shutil.which searches PATH like 'which' in a shell
        print("[ERROR] ffmpeg is not installed or not on your PATH.")
        print("        macOS  : brew install ffmpeg")
        print("        Ubuntu : sudo apt-get install ffmpeg")
        sys.exit(1)    # exit with error code so the shell knows something went wrong


# ── Step 1: extract audio ─────────────────────────────────────────────────────

def extract_audio(video_path: Path, audio_path: Path):
    """
    Pull only the audio stream out of the video and save it as a WAV file.

    Why WAV at 16 kHz mono?
      Whisper was trained on audio at 16 000 samples per second (16 kHz), mono.
      Giving it exactly that format gives the best transcription accuracy.
    """
    subprocess.run(
        [
            "ffmpeg",
            "-y",            # overwrite output file without asking
            "-i", str(video_path),   # input: the video file
            "-ac", "1",      # audio channels: 1 = mono (Whisper needs mono)
            "-ar", "16000",  # audio sample rate: 16 000 Hz = 16 kHz
            "-vn",           # -vn = "no video" — drop the video stream entirely
            str(audio_path), # output: the temporary .wav file
        ],
        check=True,           # raise an exception if ffmpeg exits with an error code
        capture_output=True,  # hide ffmpeg's verbose output from the terminal
    )


# ── Step 2: transcribe with Whisper ──────────────────────────────────────────

def transcribe_audio(audio_path: Path, model) -> list:
    """
    Run Whisper on the WAV file and return a list of timed segments.

    Each segment is a dict like:
      {"id": 1, "start": 0.0, "end": 2.3, "text": "I am picking up the cup"}

    Whisper breaks the audio into sentence-like chunks automatically.
    word_timestamps=True asks Whisper to also track timing at the word level,
    which makes the segment boundaries more precise.
    """
    print("  [2] Transcribing with Whisper (this may take a moment) ...")

    # model.transcribe() is the main Whisper call — it processes the entire file
    result = model.transcribe(
        str(audio_path),      # path to the .wav file
        language="en",        # set language explicitly — skips Whisper's auto-detect step
        fp16=False,           # fp16 (half-precision) only works on NVIDIA CUDA GPUs
                              # setting False forces full-precision (works everywhere)
        word_timestamps=True, # return per-word timing inside each segment
    )

    # result["segments"] is a list of dicts — one per spoken phrase/sentence
    # Each dict contains keys: "start", "end", "text", and more
    segments = []

    for i, seg in enumerate(result["segments"]):
        # Build a clean, simplified segment dict for our own use
        segment = {
            "id":    i + 1,                          # human-friendly 1-based index
            "start": round(float(seg["start"]), 3),  # start time in seconds (3 decimal places)
            "end":   round(float(seg["end"]),   3),  # end time in seconds
            "text":  seg["text"].strip(),             # transcribed text with whitespace removed
        }
        segments.append(segment)

        # Print each segment as soon as it is ready (Step 6 requirement)
        print(f"    [{segment['start']:.1f}s -> {segment['end']:.1f}s]  {segment['text']}")

    return segments   # return the full list of segment dicts


# ── Step 3: write .srt subtitle file ─────────────────────────────────────────

def write_srt(segments: list, srt_path: Path):
    """
    Write segments to a .srt (SubRip Text) subtitle file.

    The SRT format looks like this:

        1
        00:00:00,000 --> 00:00:02,300
        I am picking up the cup

        2
        00:00:02,300 --> 00:00:05,100
        I am turning on the tap

    The blank line between entries is required by the SRT standard.
    """

    def to_srt_timestamp(seconds: float) -> str:
        """Convert a float number of seconds into SRT timestamp format HH:MM:SS,mmm."""
        h  = int(seconds // 3600)          # hours  (integer division by 3600)
        m  = int((seconds % 3600) // 60)   # minutes (remainder after hours, ÷60)
        s  = int(seconds % 60)             # seconds (remainder after minutes)
        ms = int(round((seconds - int(seconds)) * 1000))  # milliseconds (fractional part × 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"   # format: 00:00:00,000

    lines = []    # we'll build the file content as a list of strings, then join them

    for seg in segments:
        lines.append(str(seg["id"]))                                          # subtitle number
        lines.append(
            f"{to_srt_timestamp(seg['start'])} --> {to_srt_timestamp(seg['end'])}"   # time range
        )
        lines.append(seg["text"])    # the transcribed words
        lines.append("")             # blank line — required SRT separator between entries

    # Write all lines to the file, joined by newlines
    srt_path.write_text("\n".join(lines), encoding="utf-8")


# ── Step 4: burn subtitles into a video copy ─────────────────────────────────

def burn_subtitles(video_path: Path, srt_path: Path, output_path: Path):
    """
    Use ffmpeg to hardcode (burn) the subtitle text directly into the video pixels.

    "Hardcoded" means the text is baked into each video frame — viewers cannot
    turn it off like soft subtitles. This is useful for a dataset demo.

    The SRT file path is copied to a temp location first because ffmpeg's subtitle
    filter does not handle spaces in file paths well on macOS/Linux.
    """
    import shutil as _shutil    # local import to avoid shadowing the top-level shutil
    import os                   # needed to create a temp file without auto-deleting it

    # Create a temp SRT file with a simple name (no spaces) to avoid ffmpeg issues
    tmp_srt_fd, tmp_srt_str = tempfile.mkstemp(suffix=".srt")   # mkstemp returns (fd, path)
    os.close(tmp_srt_fd)                    # close the file descriptor — we only need the path
    tmp_srt_path = Path(tmp_srt_str)        # convert string path to Path object
    _shutil.copy(srt_path, tmp_srt_path)    # copy the real SRT to the temp location

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",                                 # overwrite output without asking
                "-i", str(video_path),                # input: the original video
                "-vf",                                # -vf = "video filter"
                f"subtitles={tmp_srt_path}:force_style={SUBTITLE_STYLE}",
                                                      # NOTE: no extra single-quotes here —
                                                      # subprocess list args don't need shell quoting
                "-c:a", "copy",                       # copy audio stream unchanged (no re-encode)
                str(output_path),                     # output: the new subtitled video file
            ],
            check=True,    # raise an error if ffmpeg fails
        )
    finally:
        tmp_srt_path.unlink(missing_ok=True)   # always delete the temp SRT, even if ffmpeg fails


# ── Step 5: write narration JSON ─────────────────────────────────────────────

def write_json(clip_name: str, segments: list, total_duration: float, json_path: Path):
    """
    Save the transcription as a structured JSON file.

    Format:
    {
      "clip_name": "WashingCup",
      "total_segments": 12,
      "total_duration_sec": 76.1,
      "narrations": [ {"id": 1, "start": 0.0, "end": 2.3, "text": "..."}, ... ]
    }
    """
    payload = {
        "clip_name":         clip_name,             # e.g. "WashingCup"
        "total_segments":    len(segments),          # how many speech segments were found
        "total_duration_sec": round(total_duration, 3),  # how many seconds of audio were covered
        "narrations":        segments,               # the full list of timed text segments
    }

    # Write the dict to disk as a nicely indented JSON file
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)   # indent=2 adds two spaces per nesting level


# ── Main pipeline ─────────────────────────────────────────────────────────────

def process_video(video_path: str):
    """
    Runs all transcription steps for one video file end-to-end.
    """

    # Convert string argument to a Path object for easy manipulation
    video_path = Path(video_path)

    # Verify the video file actually exists before doing any work
    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}")
        sys.exit(1)

    clip_name = video_path.stem    # e.g. "WashingCup" (filename without extension)

    print(f"\n{'=' * 60}")
    print(f"  Transcription Pipeline  (local Whisper — no API key)")
    print(f"  Clip : {clip_name}")
    print(f"{'=' * 60}\n")

    # Make sure ffmpeg is available before we do anything else
    check_ffmpeg()

    # Create the output directories if they don't already exist
    # exist_ok=True means: do not raise an error if the folder is already there
    NARRATIONS_DIR.mkdir(parents=True, exist_ok=True)
    SUBTITLED_DIR.mkdir(parents=True, exist_ok=True)

    # Define all output file paths up front so they are easy to reference later
    srt_path       = NARRATIONS_DIR / f"{clip_name}.srt"           # subtitle file
    json_path      = NARRATIONS_DIR / f"{clip_name}.json"          # narration JSON
    subtitled_path = SUBTITLED_DIR  / f"{clip_name}_subtitled.mp4" # output video

    start_time = time.time()    # record when we started (for the final summary)

    # ── Step 1: extract audio to a temporary .wav file ───────

    print("  [1] Extracting audio from video ...")

    # tempfile.NamedTemporaryFile creates a temp file and gives us a path to it.
    # delete=False means the file is NOT automatically deleted when we close it —
    # we handle deletion ourselves in the 'finally' block below.
    # suffix=".wav" ensures the file has the right extension for ffmpeg.
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_wav_path = Path(tmp_wav.name)   # grab the path before closing the handle
    tmp_wav.close()                     # close the file handle so ffmpeg can write to it

    try:
        # Run ffmpeg to extract 16 kHz mono audio into the temp .wav
        extract_audio(video_path, tmp_wav_path)
        print(f"       Audio saved to temp file: {tmp_wav_path.name}")

        # ── Step 2: load Whisper and transcribe ──────────────

        print(f"\n  [2] Loading Whisper '{WHISPER_MODEL}' model ...")
        print(f"       (First run downloads ~145 MB model to ~/.cache/whisper)\n")

        # whisper.load_model downloads the model weights on the first call,
        # then caches them in ~/.cache/whisper for all future runs.
        model    = whisper.load_model(WHISPER_MODEL)

        print(f"\n  Model loaded. Starting transcription ...\n")

        # Run transcription — this is the main Whisper call
        # Each segment is printed to the terminal as it is returned (Step 6)
        segments = transcribe_audio(tmp_wav_path, model)

        # If Whisper found nothing (silent video, music only, etc.), warn and stop
        if not segments:
            print("\n  [WARN] No speech detected in this video.")
            print("         The video may be silent or contain only background noise.")
            return   # exit the function early — nothing to save

        print(f"\n  Transcription complete: {len(segments)} segment(s) found.\n")

        # Calculate total duration covered — from the start of segment 1 to end of last
        total_duration = segments[-1]["end"] - segments[0]["start"]  # last end minus first start

        # ── Step 3: write .srt subtitle file ─────────────────

        print(f"  [3] Writing SRT file  → {srt_path}")
        write_srt(segments, srt_path)
        print(f"       Done.\n")

        # ── Step 4: burn subtitles into the video ─────────────

        print(f"  [4] Burning subtitles into video ...")
        print(f"       Output → {subtitled_path}")
        burn_subtitles(video_path, srt_path, subtitled_path)
        print(f"       Done.\n")

        # ── Step 5: write narration JSON ──────────────────────

        print(f"  [5] Writing narration JSON → {json_path}")
        write_json(clip_name, segments, total_duration, json_path)
        print(f"       Done.\n")

    finally:
        # ── Step 8: clean up the temporary .wav file ──────────
        # The 'finally' block always runs, even if an error occurred above.
        # This guarantees the temp audio file is deleted no matter what.
        if tmp_wav_path.exists():
            tmp_wav_path.unlink()    # .unlink() deletes a file (like 'rm' in the shell)
            print(f"  [cleanup] Temp audio deleted: {tmp_wav_path.name}\n")

    # ── Step 7: print final summary ───────────────────────────

    total_time = time.time() - start_time    # seconds from start to now

    print(f"{'─' * 60}")
    print(f"  SUMMARY  —  {clip_name}")
    print(f"{'─' * 60}")
    print(f"  Total segments transcribed : {len(segments)}")
    print(f"  Total duration covered     : {total_duration:.1f}s")
    print(f"  Time taken                 : {total_time:.1f}s")
    print(f"\n  Output files saved to:")
    print(f"    SRT      →  {srt_path}")
    print(f"    JSON     →  {json_path}")
    print(f"    Video    →  {subtitled_path}")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

# This block only runs when you execute this file directly:
#   python pipeline/transcribe.py assets/videos/WashingCup.mp4
#
# It does NOT run when another script does "import transcribe".
if __name__ == "__main__":

    # sys.argv is the list of words you typed on the command line.
    # sys.argv[0] = "pipeline/transcribe.py"  ← always the script name
    # sys.argv[1] = "assets/videos/WashingCup.mp4"  ← what we need
    if len(sys.argv) != 2:
        # Wrong number of arguments — show the user what to type
        print("Usage  : python pipeline/transcribe.py <path_to_video>")
        print("Example: python pipeline/transcribe.py assets/videos/WashingCup.mp4")
        sys.exit(1)    # exit with error code 1 to signal something went wrong

    # Hand the video path to the main function
    process_video(sys.argv[1])
