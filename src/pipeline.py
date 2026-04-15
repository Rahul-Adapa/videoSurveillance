"""Main pipeline orchestrator.

Coordinates detection → tracking → event detection → visualization → output
in a clean, modular pipeline. Handles video I/O, frame sampling, and
progress reporting.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from loguru import logger
from tqdm import tqdm

from .event_detector import Event, EventDetector
from .tracker import MultiObjectTracker, Track
from .visualizer import Visualizer
from .zone_manager import ZoneManager


@dataclass
class PipelineConfig:
    """Configuration for the surveillance pipeline."""
    model_name: str = "yolov8n.pt"
    tracker_type: str = "bytetrack"
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45
    device: str = ""
    img_size: int = 640

    # Event detection
    loiter_displacement_threshold: float = 50.0
    zone_check_method: str = "bottom_center"
    cooldown_frames: int = 150
    min_track_age_frames: int = 5

    # Tracking
    track_buffer: int = 60

    # Visualization
    show_trajectories: bool = True
    trajectory_length: int = 30
    show_confidence: bool = True
    zone_opacity: float = 0.25

    # Output
    output_video: bool = True
    output_codec: str = "mp4v"
    output_fps: Optional[float] = None
    frame_skip: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> PipelineConfig:
        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_file(cls, path: str | Path) -> PipelineConfig:
        with open(path) as f:
            return cls.from_dict(json.load(f))


@dataclass
class PipelineStats:
    """Runtime statistics for the pipeline."""
    total_frames: int = 0
    processed_frames: int = 0
    total_detections: int = 0
    total_tracks: int = 0
    unique_track_ids: set = field(default_factory=set)
    total_events: int = 0
    processing_time_sec: float = 0.0
    avg_inference_ms: float = 0.0

    @property
    def avg_fps(self) -> float:
        if self.processing_time_sec == 0:
            return 0.0
        return self.processed_frames / self.processing_time_sec

    def to_dict(self) -> dict:
        return {
            "total_frames": self.total_frames,
            "processed_frames": self.processed_frames,
            "total_detections": self.total_detections,
            "total_tracks": self.total_tracks,
            "unique_persons": len(self.unique_track_ids),
            "total_events": self.total_events,
            "processing_time_sec": round(self.processing_time_sec, 2),
            "avg_fps": round(self.avg_fps, 2),
        }


class SurveillancePipeline:
    """End-to-end video surveillance pipeline.

    Pipeline stages:
    1. Video capture → frame extraction
    2. Person detection (YOLOv8)
    3. Multi-object tracking (ByteTrack)
    4. Zone-based event detection
    5. Frame annotation & visualization
    6. Output generation (video + event logs)
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.zone_manager = ZoneManager()
        self.stats = PipelineStats()

        self.tracker = MultiObjectTracker(
            model_name=self.config.model_name,
            tracker_type=self.config.tracker_type,
            confidence_threshold=self.config.confidence_threshold,
            iou_threshold=self.config.iou_threshold,
            device=self.config.device,
            img_size=self.config.img_size,
            track_buffer=self.config.track_buffer,
        )

        self.event_detector: Optional[EventDetector] = None
        self.visualizer: Optional[Visualizer] = None

    def load_zones(self, zones_path: str | Path) -> None:
        """Load zone definitions from a JSON config file."""
        self.zone_manager.load_from_file(zones_path)
        logger.info(f"Loaded {len(self.zone_manager.zones)} zones")

    def _init_event_detector(self, fps: float) -> None:
        self.event_detector = EventDetector(
            zone_manager=self.zone_manager,
            fps=fps,
            loiter_displacement_threshold=self.config.loiter_displacement_threshold,
            zone_check_method=self.config.zone_check_method,
            cooldown_frames=self.config.cooldown_frames,
            min_track_age_frames=self.config.min_track_age_frames,
        )

    def _init_visualizer(self) -> None:
        self.visualizer = Visualizer(
            zone_manager=self.zone_manager,
            show_trajectories=self.config.show_trajectories,
            trajectory_length=self.config.trajectory_length,
            show_confidence=self.config.show_confidence,
            zone_opacity=self.config.zone_opacity,
        )

    def process_video(
        self,
        video_path: str | Path,
        output_dir: str | Path,
        zones_path: Optional[str | Path] = None,
    ) -> dict:
        """Process a video file end-to-end.

        Args:
            video_path: Path to input video file
            output_dir: Directory for output files
            zones_path: Path to zone config JSON (optional)

        Returns:
            Summary dict with stats and event counts
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(
            f"Video: {video_path.name} | {width}x{height} | "
            f"{fps:.1f} FPS | {total_frames} frames | "
            f"{total_frames / fps:.1f}s"
        )

        self.stats = PipelineStats(total_frames=total_frames)

        if zones_path:
            self.load_zones(zones_path)
        elif len(self.zone_manager.zones) == 0:
            logger.warning("No zones defined — creating default zones")
            self.zone_manager.create_default_zones(width, height)

        self._init_event_detector(fps)
        self._init_visualizer()

        writer = None
        if self.config.output_video:
            out_video_path = output_dir / f"{video_path.stem}_annotated.mp4"
            out_fps = self.config.output_fps or fps
            fourcc = cv2.VideoWriter_fourcc(*self.config.output_codec)
            writer = cv2.VideoWriter(str(out_video_path), fourcc, out_fps, (width, height))
            logger.info(f"Output video: {out_video_path}")

        all_events: List[dict] = []
        start_time = time.time()
        frame_idx = 0

        pbar = tqdm(total=total_frames, desc="Processing", unit="frame")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if self.config.frame_skip > 0 and frame_idx % (self.config.frame_skip + 1) != 0:
                    frame_idx += 1
                    pbar.update(1)
                    continue

                frame_tracks = self.tracker.update(frame, frame_idx)

                self.stats.processed_frames += 1
                self.stats.total_tracks += len(frame_tracks.tracks)
                for t in frame_tracks.tracks:
                    self.stats.unique_track_ids.add(t.track_id)

                frame_events = self.event_detector.process_frame(
                    tracks=frame_tracks.tracks,
                    frame_idx=frame_idx,
                    track_histories=self.tracker.get_all_histories(),
                )

                for event in frame_events:
                    all_events.append(event.to_dict())
                    self.stats.total_events += 1

                elapsed = time.time() - start_time
                current_fps = self.stats.processed_frames / max(elapsed, 0.001)

                if writer or True:
                    annotated = self.visualizer.draw_frame(
                        frame=frame,
                        tracks=frame_tracks.tracks,
                        events=frame_events,
                        frame_idx=frame_idx,
                        fps=current_fps,
                    )
                    if writer:
                        writer.write(annotated)

                frame_idx += 1
                pbar.update(1)

        finally:
            pbar.close()
            cap.release()
            if writer:
                writer.release()

        self.stats.processing_time_sec = time.time() - start_time

        self._save_event_log(all_events, output_dir, video_path.stem)
        self._save_stats(output_dir, video_path.stem)

        summary = {
            "video": str(video_path),
            "stats": self.stats.to_dict(),
            "events_summary": self.event_detector.get_events_summary(),
        }

        summary_path = output_dir / f"{video_path.stem}_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(
            f"Done! Processed {self.stats.processed_frames} frames in "
            f"{self.stats.processing_time_sec:.1f}s "
            f"({self.stats.avg_fps:.1f} FPS avg)"
        )
        logger.info(
            f"Unique persons: {len(self.stats.unique_track_ids)} | "
            f"Events: {self.stats.total_events}"
        )

        return summary

    def _save_event_log(
        self, events: List[dict], output_dir: Path, stem: str
    ) -> None:
        """Save event log as JSON and CSV."""
        json_path = output_dir / f"{stem}_events.json"
        with open(json_path, "w") as f:
            json.dump(events, f, indent=2)
        logger.info(f"Event log saved: {json_path} ({len(events)} events)")

        csv_path = output_dir / f"{stem}_events.csv"
        if events:
            import csv
            fieldnames = [
                "event_id", "event_type", "severity", "track_id",
                "zone_id", "zone_name", "frame_idx", "timestamp_sec",
                "bbox", "confidence",
            ]
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for e in events:
                    row = {k: e.get(k, "") for k in fieldnames}
                    row["bbox"] = str(row["bbox"])
                    writer.writerow(row)
            logger.info(f"Event CSV saved: {csv_path}")

    def _save_stats(self, output_dir: Path, stem: str) -> None:
        """Save pipeline statistics."""
        stats_path = output_dir / f"{stem}_stats.json"
        stats_data = self.stats.to_dict()
        stats_data["track_histories"] = {}
        for tid, hist in self.tracker.get_all_histories().items():
            stats_data["track_histories"][str(tid)] = {
                "first_seen": hist.first_seen,
                "last_seen": hist.last_seen,
                "total_frames": len(hist.frame_indices),
                "age_frames": hist.age_frames,
            }
        with open(stats_path, "w") as f:
            json.dump(stats_data, f, indent=2)
