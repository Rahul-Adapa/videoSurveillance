"""Visualization module for annotating video frames.

Draws bounding boxes, track IDs, zone overlays, and event indicators
onto video frames for output generation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .event_detector import Event, EventSeverity, EventType
from .tracker import Track
from .zone_manager import Zone, ZoneManager


SEVERITY_COLORS = {
    EventSeverity.LOW: (0, 200, 0),
    EventSeverity.MEDIUM: (0, 200, 255),
    EventSeverity.HIGH: (0, 100, 255),
    EventSeverity.CRITICAL: (0, 0, 255),
}

TRACK_COLORS = [
    (230, 159, 0),    # orange
    (86, 180, 233),   # sky blue
    (0, 158, 115),    # green
    (240, 228, 66),   # yellow
    (0, 114, 178),    # blue
    (213, 94, 0),     # vermilion
    (204, 121, 167),  # pink
    (120, 94, 240),   # purple
]


def get_track_color(track_id: int) -> Tuple[int, int, int]:
    return TRACK_COLORS[track_id % len(TRACK_COLORS)]


class Visualizer:
    """Annotates video frames with detection, tracking, and event info."""

    def __init__(
        self,
        zone_manager: Optional[ZoneManager] = None,
        show_trajectories: bool = True,
        trajectory_length: int = 30,
        show_confidence: bool = True,
        zone_opacity: float = 0.25,
        font_scale: float = 0.5,
        thickness: int = 2,
    ):
        self.zone_manager = zone_manager
        self.show_trajectories = show_trajectories
        self.trajectory_length = trajectory_length
        self.show_confidence = show_confidence
        self.zone_opacity = zone_opacity
        self.font_scale = font_scale
        self.thickness = thickness

        self._trajectory_points: Dict[int, list] = {}

    def draw_frame(
        self,
        frame: np.ndarray,
        tracks: List[Track],
        events: Optional[List[Event]] = None,
        frame_idx: int = 0,
        fps: float = 0.0,
    ) -> np.ndarray:
        """Draw all annotations onto a frame."""
        annotated = frame.copy()

        if self.zone_manager:
            annotated = self._draw_zones(annotated)

        for track in tracks:
            annotated = self._draw_track(annotated, track)

        if self.show_trajectories:
            annotated = self._draw_trajectories(annotated, tracks)

        if events:
            annotated = self._draw_events(annotated, events)

        annotated = self._draw_info_bar(annotated, frame_idx, len(tracks), fps, events)

        return annotated

    def _draw_zones(self, frame: np.ndarray) -> np.ndarray:
        """Draw semi-transparent zone overlays with labels."""
        overlay = frame.copy()

        for zone in self.zone_manager.get_enabled_zones():
            pts = zone.np_coords.reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], zone.color)

            cv2.polylines(frame, [pts], True, zone.color, 2)

            centroid = np.mean(zone.np_coords, axis=0).astype(int)
            label = f"{zone.name} [{zone.zone_type}]"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                frame,
                (centroid[0] - tw // 2 - 4, centroid[1] - th - 6),
                (centroid[0] + tw // 2 + 4, centroid[1] + 4),
                zone.color,
                -1,
            )
            cv2.putText(
                frame, label,
                (centroid[0] - tw // 2, centroid[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )

        return cv2.addWeighted(overlay, self.zone_opacity, frame, 1 - self.zone_opacity, 0)

    def _draw_track(self, frame: np.ndarray, track: Track) -> np.ndarray:
        """Draw bounding box and label for a single track."""
        x1, y1, x2, y2 = track.bbox.astype(int)
        color = get_track_color(track.track_id)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.thickness)

        label = f"ID:{track.track_id}"
        if self.show_confidence:
            label += f" {track.confidence:.0%}"

        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1
        )
        cv2.rectangle(
            frame,
            (x1, y1 - th - 10),
            (x1 + tw + 8, y1),
            color,
            -1,
        )
        cv2.putText(
            frame, label,
            (x1 + 4, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (255, 255, 255), 1,
        )

        bc = track.bottom_center
        cv2.circle(frame, (int(bc[0]), int(bc[1])), 4, color, -1)

        return frame

    def _draw_trajectories(self, frame: np.ndarray, tracks: List[Track]) -> np.ndarray:
        """Draw motion trails for tracked persons."""
        for track in tracks:
            tid = track.track_id
            if tid not in self._trajectory_points:
                self._trajectory_points[tid] = []

            self._trajectory_points[tid].append(track.bottom_center)
            if len(self._trajectory_points[tid]) > self.trajectory_length:
                self._trajectory_points[tid] = self._trajectory_points[tid][-self.trajectory_length:]

            points = self._trajectory_points[tid]
            color = get_track_color(tid)

            for i in range(1, len(points)):
                alpha = i / len(points)
                thickness = max(1, int(alpha * 3))
                pt1 = (int(points[i - 1][0]), int(points[i - 1][1]))
                pt2 = (int(points[i][0]), int(points[i][1]))
                faded_color = tuple(int(c * alpha) for c in color)
                cv2.line(frame, pt1, pt2, faded_color, thickness)

        return frame

    def _draw_events(self, frame: np.ndarray, events: List[Event]) -> np.ndarray:
        """Draw event indicators on the frame."""
        for event in events:
            if event.event_type == EventType.ZONE_EXIT:
                continue

            color = SEVERITY_COLORS.get(event.severity, (255, 255, 255))
            x1, y1, x2, y2 = event.bbox.astype(int)

            cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), color, 3)

            if event.event_type == EventType.ZONE_INTRUSION:
                label = "! INTRUSION"
            elif event.event_type == EventType.LOITERING:
                duration = event.metadata.get("duration_sec", 0)
                label = f"! LOITER {duration:.0f}s"
            else:
                label = f"! {event.event_type.value.upper()}"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                frame,
                (x1, y2 + 2),
                (x1 + tw + 8, y2 + th + 12),
                color,
                -1,
            )
            cv2.putText(
                frame, label,
                (x1 + 4, y2 + th + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
            )

        return frame

    def _draw_info_bar(
        self,
        frame: np.ndarray,
        frame_idx: int,
        num_tracks: int,
        fps: float,
        events: Optional[List[Event]],
    ) -> np.ndarray:
        """Draw a status bar at the top of the frame."""
        h, w = frame.shape[:2]
        bar_height = 32
        cv2.rectangle(frame, (0, 0), (w, bar_height), (40, 40, 40), -1)

        timestamp = frame_idx / max(fps, 1.0)
        minutes = int(timestamp // 60)
        seconds = timestamp % 60

        info_parts = [
            f"Frame: {frame_idx}",
            f"Time: {minutes:02d}:{seconds:05.2f}",
            f"Tracks: {num_tracks}",
        ]
        if fps > 0:
            info_parts.append(f"FPS: {fps:.1f}")

        if events:
            active_events = [e for e in events if e.event_type != EventType.ZONE_EXIT]
            if active_events:
                info_parts.append(f"EVENTS: {len(active_events)}")

        info_text = "  |  ".join(info_parts)
        cv2.putText(
            frame, info_text,
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1,
        )

        return frame

    def reset(self) -> None:
        self._trajectory_points.clear()
