"""Multi-object tracking module.

Tracker Selection Rationale:
- ByteTrack chosen as the primary tracker for its robustness:
    - Associates both high and low-confidence detections (two-stage matching)
    - Handles occlusion better than single-threshold approaches
    - Pure motion-based — no appearance model needed, so it's fast
- BoT-SORT available as an alternative (adds Re-ID appearance features)
- Alternatives considered:
    - DeepSORT: Popular but slower due to appearance feature extraction CNN
    - StrongSORT: Improved DeepSORT, but heavier compute for marginal gains
    - OC-SORT: Good for non-linear motion but ByteTrack is more battle-tested

We leverage ultralytics' built-in tracker which wraps BoT-SORT/ByteTrack
with configurable parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from loguru import logger
from ultralytics import YOLO

from .detector import Detection


@dataclass
class Track:
    """A tracked person with a unique ID."""
    track_id: int
    bbox: np.ndarray               # [x1, y1, x2, y2]
    confidence: float
    frame_idx: int
    is_confirmed: bool = True

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2,
        )

    @property
    def bottom_center(self) -> tuple[float, float]:
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            self.bbox[3],
        )

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "bbox": self.bbox.tolist(),
            "confidence": round(float(self.confidence), 4),
        }


@dataclass
class FrameTracks:
    """All tracks in a single frame."""
    frame_idx: int
    tracks: List[Track] = field(default_factory=list)


@dataclass
class TrackHistory:
    """Historical information for a single track across frames."""
    track_id: int
    positions: List[tuple[float, float]] = field(default_factory=list)
    bboxes: List[np.ndarray] = field(default_factory=list)
    frame_indices: List[int] = field(default_factory=list)
    last_seen: int = 0
    first_seen: int = 0

    @property
    def age_frames(self) -> int:
        return self.last_seen - self.first_seen

    def displacement(self, window: int = 30) -> float:
        """Pixel displacement over last `window` positions."""
        recent = self.positions[-window:]
        if len(recent) < 2:
            return 0.0
        start = np.array(recent[0])
        end = np.array(recent[-1])
        return float(np.linalg.norm(end - start))


class MultiObjectTracker:
    """Wraps ultralytics' built-in tracker for integrated detect+track.

    Uses the YOLO model's .track() method which handles:
    - Detection → tracking association (ByteTrack or BoT-SORT)
    - Kalman filter state prediction
    - ID management and re-identification
    """

    PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        tracker_type: str = "bytetrack",
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        device: str = "",
        img_size: int = 640,
        track_buffer: int = 60,
    ):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.tracker_type = tracker_type

        logger.info(f"Loading model for tracking: {model_name}")
        self.model = YOLO(model_name)

        if device:
            self.device = device
        else:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tracker_config = self._get_tracker_config(tracker_type, track_buffer)
        self.track_histories: Dict[int, TrackHistory] = {}

        logger.info(
            f"Tracker initialized: {tracker_type} on {self.device}, "
            f"buffer={track_buffer} frames"
        )

    def _get_tracker_config(self, tracker_type: str, track_buffer: int) -> str:
        """Return the tracker config YAML name."""
        if tracker_type == "botsort":
            return "botsort.yaml"
        return "bytetrack.yaml"

    def update(self, frame: np.ndarray, frame_idx: int) -> FrameTracks:
        """Run detection + tracking on a single frame."""
        results = self.model.track(
            frame,
            classes=[self.PERSON_CLASS_ID],
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            device=self.device,
            tracker=self.tracker_config,
            persist=True,
            verbose=False,
        )

        result = results[0]
        tracks = []

        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy().astype(int)

            for bbox, conf, tid in zip(boxes, confs, track_ids):
                track = Track(
                    track_id=int(tid),
                    bbox=bbox.astype(np.float32),
                    confidence=float(conf),
                    frame_idx=frame_idx,
                )
                tracks.append(track)
                self._update_history(track)

        return FrameTracks(frame_idx=frame_idx, tracks=tracks)

    def _update_history(self, track: Track) -> None:
        """Maintain per-track position history for event analysis."""
        tid = track.track_id
        if tid not in self.track_histories:
            self.track_histories[tid] = TrackHistory(
                track_id=tid,
                first_seen=track.frame_idx,
            )

        hist = self.track_histories[tid]
        hist.positions.append(track.bottom_center)
        hist.bboxes.append(track.bbox.copy())
        hist.frame_indices.append(track.frame_idx)
        hist.last_seen = track.frame_idx

    def get_history(self, track_id: int) -> Optional[TrackHistory]:
        return self.track_histories.get(track_id)

    def get_all_histories(self) -> Dict[int, TrackHistory]:
        return self.track_histories

    def reset(self) -> None:
        """Reset tracker state (for processing a new video)."""
        self.model = YOLO(self.model.model_name if hasattr(self.model, 'model_name') else "yolov8n.pt")
        self.track_histories.clear()
        logger.info("Tracker state reset")
