#!/usr/bin/env python3
"""
Render each capture in a PUSCH dataset as one video frame and encode an MP4.

Usage:
    python generate_capture_video.py
    python generate_capture_video.py path/to/pusch_dataset.bin
    python generate_capture_video.py path/to/pusch_dataset.bin --output capture_video.mp4
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from plot_capture import plot_capture
from read_dataset import PUSCHDataset

DEFAULT_DATASET = "plugins/nr_pusch_capture/data/pusch_dataset.bin"
DEFAULT_FPS = 2.0
FRAME_PATTERN = "frame_%06d.png"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate an MP4 video using one frame per PUSCH capture."
    )
    parser.add_argument(
        "dataset_path",
        nargs="?",
        default=DEFAULT_DATASET,
        help=f"Path to pusch_dataset.bin (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output MP4 path (default: dataset path with .mp4 suffix)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_FPS,
        help="Frames per second (default: 2.0 = 0.5 seconds per frame)",
    )
    return parser.parse_args(argv)


def default_output_path(dataset_path):
    return dataset_path.with_suffix(".mp4")


def find_ffmpeg():
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg is required to encode the video but was not found in PATH."
        )
    return ffmpeg


def build_ffmpeg_command(ffmpeg, frame_pattern, fps, output_path):
    return [
        ffmpeg,
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frame_pattern),
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]


def render_frames(dataset_path, frame_dir):
    dataset = PUSCHDataset(dataset_path)
    total = len(dataset)
    if total == 0:
        raise ValueError(f"Dataset has no captures: {dataset_path}")

    print(f"Rendering {total} frames from {dataset_path}")
    for idx, capture in enumerate(dataset):
        frame_path = frame_dir / (FRAME_PATTERN % idx)
        plot_capture(capture, frame_path, verbose=False)
        if total <= 10 or idx == total - 1 or (idx + 1) % 10 == 0:
            print(f"Rendered frame {idx + 1}/{total}")

    return total


def encode_video(frame_dir, output_path, fps):
    ffmpeg = find_ffmpeg()
    frame_pattern = frame_dir / FRAME_PATTERN
    cmd = build_ffmpeg_command(ffmpeg, frame_pattern, fps, output_path)
    subprocess.run(cmd, check=True)


def main(argv=None):
    find_ffmpeg()
    args = parse_args(argv)

    if args.fps <= 0:
        raise ValueError(f"FPS must be positive, got {args.fps}")

    dataset_path = Path(args.dataset_path).resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    output_path = (
        Path(args.output).resolve()
        if args.output
        else default_output_path(dataset_path)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="capture_video_") as temp_dir:
        frame_dir = Path(temp_dir)
        total = render_frames(dataset_path, frame_dir)
        encode_video(frame_dir, output_path, args.fps)

    print(f"Encoded {total} frames to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
