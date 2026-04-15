"""Event detection logic for zone intrusion and loitering.

Implements temporal reasoning over tracked person positions to detect:
1. Zone Intrusion — person enters a restricted area
2. Loitering — person remains (near-)stationary in a zone beyond a threshold

State machine per (track_id, zone_id) pair to avoid duplicate alerts and
manage event lifecycle (enter → active → exit).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from loguru import logger

from .tracker import Track, TrackHistory
from .zone_manager import Zone, ZoneManager


class EventType(str, Enum):
    ZONE_INTRUSION = "zone_intrusion"
    LOITERING = "loitering"
    ZONE_EXIT = "zone_exit"


class EventSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Event:
    """A detected surveillance event."""
    event_id: str
    event_type: EventType
    severity: EventSeverity
    track_id: int
    zone_id: str
    zone_name: str
    frame_idx: int
    timestamp_sec: float
    bbox: np.ndarray
    confidence: float
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "track_id": self.track_id,
            "zone_id": self.zone_id,
            "zone_name": self.zone_name,
            "frame_idx": self.frame_idx,
            "timestamp_sec": round(self.timestamp_sec, 3),
            "bbox": self.bbox.tolist(),
            "confidence": round(self.confidence, 4),
            "metadata": self.metadata,
        }


@dataclass
class _TrackZoneState:
    """Internal state for a (track, zone) pair."""
    entered_frame: int = -1
    entered_time: float = 0.0
    last_seen_frame: int = -1
    is_inside: bool = False
    loitering_alerted: bool = False
    intrusion_alerted: bool = False
    positions_in_zone: List[tuple[float, float]] = field(default_factory=list)
    consecutive_still_frames: int = 0


class EventDetector:
    """Detects zone intrusions and loitering from tracked person data."""

    def __init__(
        self,
        zone_manager: ZoneManager,
        fps: float = 30.0,
        loiter_displacement_threshold: float = 50.0,
        zone_check_method: str = "bottom_center",
        cooldown_frames: int = 150,
        min_track_age_frames: int = 5,
    ):
        self.zone_manager = zone_manager
        self.fps = fps
        self.loiter_displacement_px = loiter_displacement_threshold
        self.zone_check_method = zone_check_method
        self.cooldown_frames = cooldown_frames
        self.min_track_age = min_track_age_frames

        self._states: Dict[Tuple[int, int], _TrackZoneState] = {}
        self._event_counter = 0
        self._events: List[Event] = []
        self._recent_events: Dict[str, int] = {}

    def _state_key(self, track_id: int, zone_id: str) -> Tuple[int, str]:
        return (track_id, zone_id)

    def _get_state(self, track_id: int, zone_id: str) -> _TrackZoneState:
        key = self._state_key(track_id, zone_id)
        if key not in self._states:
            self._states[key] = _TrackZoneState()
        return self._states[key]

    def _next_event_id(self) -> str:
        self._event_counter += 1
        return f"EVT-{self._event_counter:06d}"

    def _is_on_cooldown(self, event_type: str, track_id: int, zone_id: str, frame_idx: int) -> bool:
        key = f"{event_type}:{track_id}:{zone_id}"
        last_frame = self._recent_events.get(key, -999999)
        return (frame_idx - last_frame) < self.cooldown_frames

    def _record_cooldown(self, event_type: str, track_id: int, zone_id: str, frame_idx: int) -> None:
        key = f"{event_type}:{track_id}:{zone_id}"
        self._recent_events[key] = frame_idx

    def process_frame(
        self,
        tracks: List[Track],
        frame_idx: int,
        track_histories: Optional[Dict[int, TrackHistory]] = None,
    ) -> List[Event]:
        """Process a frame's tracks and return any new events."""
        frame_events: List[Event] = []
        timestamp = frame_idx / self.fps

        active_keys: Set[Tuple[int, str]] = set()

        for track in tracks:
            matched_zones = self.zone_manager.check_bbox(
                track.bbox, method=self.zone_check_method
            )

            for zone in matched_zones:
                key = self._state_key(track.track_id, zone.zone_id)
                active_keys.add(key)
                state = self._get_state(track.track_id, zone.zone_id)

                if not state.is_inside:
                    state.is_inside = True
                    state.entered_frame = frame_idx
                    state.entered_time = timestamp
                    state.positions_in_zone = [track.bottom_center]
                    state.consecutive_still_frames = 0
                    state.loitering_alerted = False

                    if not self._is_on_cooldown(
                        EventType.ZONE_INTRUSION, track.track_id, zone.zone_id, frame_idx
                    ):
                        event = Event(
                            event_id=self._next_event_id(),
                            event_type=EventType.ZONE_INTRUSION,
                            severity=self._intrusion_severity(zone),
                            track_id=track.track_id,
                            zone_id=zone.zone_id,
                            zone_name=zone.name,
                            frame_idx=frame_idx,
                            timestamp_sec=timestamp,
                            bbox=track.bbox.copy(),
                            confidence=track.confidence,
                            metadata={"zone_type": zone.zone_type},
                        )
                        frame_events.append(event)
                        state.intrusion_alerted = True
                        self._record_cooldown(
                            EventType.ZONE_INTRUSION, track.track_id, zone.zone_id, frame_idx
                        )
                else:
                    state.last_seen_frame = frame_idx
                    state.positions_in_zone.append(track.bottom_center)

                    loiter_events = self._check_loitering(
                        track, zone, state, frame_idx, timestamp
                    )
                    frame_events.extend(loiter_events)

            for zone in self.zone_manager.get_enabled_zones():
                if zone not in matched_zones:
                    key = self._state_key(track.track_id, zone.zone_id)
                    state = self._get_state(track.track_id, zone.zone_id)
                    if state.is_inside:
                        state.is_inside = False
                        duration = timestamp - state.entered_time
                        if duration > 1.0:
                            exit_event = Event(
                                event_id=self._next_event_id(),
                                event_type=EventType.ZONE_EXIT,
                                severity=EventSeverity.LOW,
                                track_id=track.track_id,
                                zone_id=zone.zone_id,
                                zone_name=zone.name,
                                frame_idx=frame_idx,
                                timestamp_sec=timestamp,
                                bbox=track.bbox.copy(),
                                confidence=track.confidence,
                                metadata={
                                    "duration_sec": round(duration, 2),
                                    "zone_type": zone.zone_type,
                                },
                            )
                            frame_events.append(exit_event)

        self._events.extend(frame_events)
        return frame_events

    def _check_loitering(
        self,
        track: Track,
        zone: Zone,
        state: _TrackZoneState,
        frame_idx: int,
        timestamp: float,
    ) -> List[Event]:
        """Check if a person has been loitering in a zone."""
        events = []

        duration_sec = timestamp - state.entered_time
        if duration_sec < zone.loiter_threshold_sec:
            return events

        if state.loitering_alerted:
            return events

        recent_positions = state.positions_in_zone[-int(self.fps * 5):]
        if len(recent_positions) >= 2:
            displacement = np.linalg.norm(
                np.array(recent_positions[-1]) - np.array(recent_positions[0])
            )
        else:
            displacement = 0.0

        window_size = min(len(state.positions_in_zone), int(self.fps * zone.loiter_threshold_sec))
        if window_size >= 2:
            positions_array = np.array(state.positions_in_zone[-window_size:])
            total_displacement = np.linalg.norm(positions_array[-1] - positions_array[0])
            max_spread = np.max(np.ptp(positions_array, axis=0))
        else:
            total_displacement = 0.0
            max_spread = 0.0

        is_loitering = (
            total_displacement < self.loiter_displacement_px * 3
            and max_spread < self.loiter_displacement_px * 5
        )

        if is_loitering and not self._is_on_cooldown(
            EventType.LOITERING, track.track_id, zone.zone_id, frame_idx
        ):
            event = Event(
                event_id=self._next_event_id(),
                event_type=EventType.LOITERING,
                severity=self._loiter_severity(duration_sec, zone),
                track_id=track.track_id,
                zone_id=zone.zone_id,
                zone_name=zone.name,
                frame_idx=frame_idx,
                timestamp_sec=timestamp,
                bbox=track.bbox.copy(),
                confidence=track.confidence,
                metadata={
                    "duration_sec": round(duration_sec, 2),
                    "displacement_px": round(float(total_displacement), 1),
                    "max_spread_px": round(float(max_spread), 1),
                    "zone_type": zone.zone_type,
                },
            )
            events.append(event)
            state.loitering_alerted = True
            self._record_cooldown(
                EventType.LOITERING, track.track_id, zone.zone_id, frame_idx
            )

        return events

    def _intrusion_severity(self, zone: Zone) -> EventSeverity:
        if zone.zone_type == "restricted":
            return EventSeverity.HIGH
        elif zone.zone_type == "monitored":
            return EventSeverity.MEDIUM
        return EventSeverity.LOW

    def _loiter_severity(self, duration_sec: float, zone: Zone) -> EventSeverity:
        if zone.zone_type == "restricted":
            if duration_sec > zone.loiter_threshold_sec * 2:
                return EventSeverity.CRITICAL
            return EventSeverity.HIGH
        if duration_sec > zone.loiter_threshold_sec * 3:
            return EventSeverity.HIGH
        if duration_sec > zone.loiter_threshold_sec * 1.5:
            return EventSeverity.MEDIUM
        return EventSeverity.LOW

    def get_all_events(self) -> List[Event]:
        return self._events

    def get_events_summary(self) -> dict:
        summary = {
            "total_events": len(self._events),
            "by_type": {},
            "by_severity": {},
            "by_zone": {},
        }
        for event in self._events:
            t = event.event_type.value
            summary["by_type"][t] = summary["by_type"].get(t, 0) + 1
            s = event.severity.value
            summary["by_severity"][s] = summary["by_severity"].get(s, 0) + 1
            z = event.zone_id
            summary["by_zone"][z] = summary["by_zone"].get(z, 0) + 1
        return summary

    def reset(self) -> None:
        self._states.clear()
        self._events.clear()
        self._recent_events.clear()
        self._event_counter = 0
