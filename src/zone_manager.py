"""Zone management for defining and querying regions of interest.

Handles loading zone polygons from JSON config, point-in-polygon testing,
and bounding-box–polygon intersection checks using Shapely for robust
computational geometry.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from shapely.geometry import Point, Polygon, box


@dataclass
class Zone:
    """A named polygonal region of interest."""
    zone_id: str
    name: str
    polygon: Polygon
    zone_type: str = "restricted"     # restricted | monitored | entry | exit
    color: Tuple[int, int, int] = (0, 0, 255)
    loiter_threshold_sec: float = 30.0
    enabled: bool = True

    @property
    def coords(self) -> list[list[float]]:
        """Return polygon coordinates as list of [x, y] pairs."""
        return [list(c) for c in self.polygon.exterior.coords[:-1]]

    @property
    def np_coords(self) -> np.ndarray:
        """Return polygon coordinates as numpy array for OpenCV drawing."""
        return np.array(self.polygon.exterior.coords[:-1], dtype=np.int32)

    def contains_point(self, x: float, y: float) -> bool:
        return self.polygon.contains(Point(x, y))

    def intersects_bbox(self, bbox: np.ndarray) -> bool:
        """Check if a bounding box [x1, y1, x2, y2] intersects the zone."""
        bbox_poly = box(bbox[0], bbox[1], bbox[2], bbox[3])
        return self.polygon.intersects(bbox_poly)

    def bbox_overlap_ratio(self, bbox: np.ndarray) -> float:
        """Fraction of the bounding box area that overlaps the zone."""
        bbox_poly = box(bbox[0], bbox[1], bbox[2], bbox[3])
        if bbox_poly.area == 0:
            return 0.0
        intersection = self.polygon.intersection(bbox_poly)
        return intersection.area / bbox_poly.area

    def to_dict(self) -> dict:
        return {
            "zone_id": self.zone_id,
            "name": self.name,
            "type": self.zone_type,
            "polygon": self.coords,
            "loiter_threshold_sec": self.loiter_threshold_sec,
            "color": list(self.color),
            "enabled": self.enabled,
        }


class ZoneManager:
    """Loads, stores, and queries zones of interest."""

    def __init__(self):
        self.zones: Dict[str, Zone] = {}

    def load_from_file(self, config_path: str | Path) -> None:
        """Load zones from a JSON config file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Zone config not found: {config_path}")

        with open(config_path) as f:
            data = json.load(f)

        zones_data = data if isinstance(data, list) else data.get("zones", [])

        for z in zones_data:
            self.add_zone_from_dict(z)

        logger.info(f"Loaded {len(self.zones)} zones from {config_path}")

    def add_zone_from_dict(self, zone_dict: dict) -> Zone:
        """Create and register a zone from a dict specification."""
        coords = zone_dict["polygon"]
        if len(coords) < 3:
            raise ValueError(
                f"Zone '{zone_dict.get('zone_id', '?')}' needs at least 3 points"
            )

        color = tuple(zone_dict.get("color", [0, 0, 255]))

        zone = Zone(
            zone_id=zone_dict["zone_id"],
            name=zone_dict.get("name", zone_dict["zone_id"]),
            polygon=Polygon(coords),
            zone_type=zone_dict.get("type", "restricted"),
            color=color,
            loiter_threshold_sec=zone_dict.get("loiter_threshold_sec", 30.0),
            enabled=zone_dict.get("enabled", True),
        )

        if not zone.polygon.is_valid:
            logger.warning(f"Zone '{zone.zone_id}' polygon is invalid, attempting fix")
            zone.polygon = zone.polygon.buffer(0)

        self.zones[zone.zone_id] = zone
        return zone

    def check_point(
        self, x: float, y: float
    ) -> List[Zone]:
        """Return all enabled zones containing the given point."""
        return [
            z for z in self.zones.values()
            if z.enabled and z.contains_point(x, y)
        ]

    def check_bbox(
        self, bbox: np.ndarray, method: str = "bottom_center"
    ) -> List[Zone]:
        """Check which zones a bounding box intersects with.

        Methods:
        - 'bottom_center': Check if bottom-center point is in zone (best for
          ground-plane zones, reduces false positives from tall bounding boxes)
        - 'center': Check if center point is in zone
        - 'intersect': Check geometric intersection of bbox with zone polygon
        - 'overlap': Check if >50% of bbox overlaps with zone
        """
        if method == "bottom_center":
            bx = (bbox[0] + bbox[2]) / 2
            by = bbox[3]
            return self.check_point(bx, by)
        elif method == "center":
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            return self.check_point(cx, cy)
        elif method == "intersect":
            return [
                z for z in self.zones.values()
                if z.enabled and z.intersects_bbox(bbox)
            ]
        elif method == "overlap":
            return [
                z for z in self.zones.values()
                if z.enabled and z.bbox_overlap_ratio(bbox) > 0.5
            ]
        else:
            raise ValueError(f"Unknown check method: {method}")

    def get_zone(self, zone_id: str) -> Optional[Zone]:
        return self.zones.get(zone_id)

    def get_all_zones(self) -> List[Zone]:
        return list(self.zones.values())

    def get_enabled_zones(self) -> List[Zone]:
        return [z for z in self.zones.values() if z.enabled]

    def create_default_zones(self, frame_width: int, frame_height: int) -> None:
        """Create default demo zones based on frame dimensions."""
        margin_x = frame_width * 0.1
        margin_y = frame_height * 0.1

        self.add_zone_from_dict({
            "zone_id": "restricted_1",
            "name": "Restricted Area",
            "type": "restricted",
            "polygon": [
                [margin_x, margin_y],
                [frame_width * 0.4, margin_y],
                [frame_width * 0.4, frame_height * 0.5],
                [margin_x, frame_height * 0.5],
            ],
            "color": [0, 0, 255],
            "loiter_threshold_sec": 30.0,
        })

        self.add_zone_from_dict({
            "zone_id": "monitored_1",
            "name": "Monitored Entrance",
            "type": "monitored",
            "polygon": [
                [frame_width * 0.6, frame_height * 0.5],
                [frame_width * 0.9, frame_height * 0.5],
                [frame_width * 0.9, frame_height * 0.9],
                [frame_width * 0.6, frame_height * 0.9],
            ],
            "color": [0, 165, 255],
            "loiter_threshold_sec": 60.0,
        })

        logger.info(f"Created {len(self.zones)} default zones for {frame_width}x{frame_height}")
