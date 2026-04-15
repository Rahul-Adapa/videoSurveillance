#!/usr/bin/env python3
"""Interactive zone creation tool.

Opens a video frame and allows the user to draw zone polygons
by clicking points. Saves the result as a zones JSON config.

Usage:
    python scripts/create_zones.py --video input.mp4 --output zones.json
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


class ZoneCreator:
    """Interactive polygon drawing tool for defining surveillance zones."""

    def __init__(self, frame: np.ndarray):
        self.frame = frame.copy()
        self.display = frame.copy()
        self.zones = []
        self.current_points = []
        self.zone_counter = 0
        self.colors = [
            (0, 0, 255), (0, 165, 255), (255, 165, 0),
            (0, 255, 0), (255, 0, 255), (255, 255, 0),
        ]

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_points.append([x, y])
            self._redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self.current_points) >= 3:
                self._finish_zone()

    def _redraw(self):
        self.display = self.frame.copy()

        for zone in self.zones:
            pts = np.array(zone["polygon"], dtype=np.int32)
            overlay = self.display.copy()
            cv2.fillPoly(overlay, [pts], zone["color"])
            self.display = cv2.addWeighted(overlay, 0.3, self.display, 0.7, 0)
            cv2.polylines(self.display, [pts], True, zone["color"], 2)

            centroid = np.mean(pts, axis=0).astype(int)
            cv2.putText(
                self.display, zone["name"],
                (centroid[0], centroid[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
            )

        if self.current_points:
            color = self.colors[self.zone_counter % len(self.colors)]
            for pt in self.current_points:
                cv2.circle(self.display, tuple(pt), 5, color, -1)
            if len(self.current_points) > 1:
                pts = np.array(self.current_points, dtype=np.int32)
                cv2.polylines(self.display, [pts], False, color, 2)

        instructions = [
            "Left-click: Add point | Right-click: Finish zone",
            "Press 'u': Undo last point | 'r': Reset current",
            "Press 's': Save & quit  | 'q': Quit without saving",
        ]
        for i, text in enumerate(instructions):
            cv2.putText(
                self.display, text,
                (10, 25 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )

    def _finish_zone(self):
        color = self.colors[self.zone_counter % len(self.colors)]
        self.zone_counter += 1

        zone = {
            "zone_id": f"zone_{self.zone_counter}",
            "name": f"Zone {self.zone_counter}",
            "type": "restricted",
            "polygon": self.current_points.copy(),
            "color": list(color),
            "loiter_threshold_sec": 30.0,
            "enabled": True,
        }
        self.zones.append(zone)
        self.current_points = []
        self._redraw()
        print(f"Zone {self.zone_counter} created with {len(zone['polygon'])} points")

    def run(self) -> list:
        cv2.namedWindow("Zone Creator", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Zone Creator", self.mouse_callback)
        self._redraw()

        while True:
            cv2.imshow("Zone Creator", self.display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("u") and self.current_points:
                self.current_points.pop()
                self._redraw()
            elif key == ord("r"):
                self.current_points = []
                self._redraw()
            elif key == ord("s"):
                break
            elif key == ord("q"):
                self.zones = []
                break

        cv2.destroyAllWindows()
        return self.zones


def main():
    parser = argparse.ArgumentParser(description="Interactive zone polygon creator")
    parser.add_argument("--video", required=True, help="Video file to extract frame from")
    parser.add_argument("--output", default="zones.json", help="Output zones JSON file")
    parser.add_argument("--frame", type=int, default=0, help="Frame number to display")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Cannot open video {args.video}")
        return

    if args.frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Cannot read frame")
        return

    creator = ZoneCreator(frame)
    zones = creator.run()

    if zones:
        output = {"zones": zones}
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved {len(zones)} zones to {args.output}")
    else:
        print("\nNo zones created.")


if __name__ == "__main__":
    main()
