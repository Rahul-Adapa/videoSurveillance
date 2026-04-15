#!/usr/bin/env python3
"""Download sample video clips from public surveillance datasets.

Downloads a small set of clips from MOT17 for testing the pipeline.
These are publicly available benchmark sequences.
"""

import hashlib
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

SAMPLES_DIR = Path(__file__).parent.parent / "sample_videos"

MOT17_URL = "https://motchallenge.net/data/MOT17.zip"

ALTERNATIVE_SOURCES = {
    "mot17_09": "https://motchallenge.net/data/MOT17.zip",
}


def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """Download a file with progress reporting."""
    if dest.exists():
        print(f"  Already exists: {dest}")
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {desc or url}...")

    try:
        urllib.request.urlretrieve(url, str(dest))
        print(f"  Saved to: {dest}")
        return True
    except Exception as e:
        print(f"  Failed: {e}")
        return False


def setup_sample_videos():
    """Set up sample videos for testing."""
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Sample Video Setup")
    print("=" * 60)
    print()
    print("For testing, please download clips from one of these datasets:")
    print()
    print("1. MOT17 (Recommended for tracking evaluation):")
    print("   https://motchallenge.net/data/MOT17/")
    print("   - Download MOT17 dataset")
    print("   - Extract video sequences from train/ or test/ folders")
    print()
    print("2. UCF-Crime (Real-world CCTV footage):")
    print("   https://www.crcv.ucf.edu/projects/real-world/")
    print("   - Contains 1,900 untrimmed surveillance videos")
    print()
    print("3. VIRAT (Ground surveillance):")
    print("   https://viratdata.org/")
    print("   - Multi-resolution surveillance clips")
    print()
    print("4. VisDrone (Drone footage):")
    print("   https://github.com/VisDrone/VisDrone-Dataset")
    print()
    print(f"Place downloaded video files in: {SAMPLES_DIR.absolute()}")
    print()

    create_test_video()


def create_test_video():
    """Create a synthetic test video with moving rectangles for basic testing."""
    import numpy as np

    try:
        import cv2
    except ImportError:
        print("OpenCV not installed. Skipping synthetic test video.")
        return

    print("Creating synthetic test video for pipeline validation...")

    width, height, fps, duration = 1280, 720, 30, 10
    total_frames = fps * duration
    out_path = SAMPLES_DIR / "synthetic_test.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    persons = [
        {"start": (100, 300), "end": (600, 400), "size": (50, 120)},
        {"start": (800, 200), "end": (400, 500), "size": (45, 110)},
        {"start": (300, 100), "end": (300, 600), "size": (55, 130)},
    ]

    for frame_i in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (60, 60, 60)

        t = frame_i / total_frames
        for p in persons:
            x = int(p["start"][0] + (p["end"][0] - p["start"][0]) * t)
            y = int(p["start"][1] + (p["end"][1] - p["start"][1]) * t)
            w, h = p["size"]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (180, 180, 220), -1)
            cv2.circle(frame, (x + w // 2, y - 15), 15, (180, 180, 220), -1)

        writer.write(frame)

    writer.release()
    print(f"  Synthetic test video saved: {out_path}")
    print(f"  {width}x{height}, {fps} FPS, {duration}s, {total_frames} frames")

    zones_path = SAMPLES_DIR / "synthetic_zones.json"
    import json
    zones = {
        "zones": [
            {
                "zone_id": "restricted_zone",
                "name": "Restricted Area",
                "type": "restricted",
                "polygon": [[200, 200], [500, 200], [500, 500], [200, 500]],
                "color": [0, 0, 255],
                "loiter_threshold_sec": 3.0,
            },
            {
                "zone_id": "monitored_zone",
                "name": "Monitored Zone",
                "type": "monitored",
                "polygon": [[600, 300], [900, 300], [900, 550], [600, 550]],
                "color": [0, 165, 255],
                "loiter_threshold_sec": 5.0,
            },
        ]
    }
    with open(zones_path, "w") as f:
        json.dump(zones, f, indent=2)
    print(f"  Zone config saved: {zones_path}")


if __name__ == "__main__":
    setup_sample_videos()
