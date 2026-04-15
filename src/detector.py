"""Person detection module using YOLOv8.

Model Selection Rationale:
- YOLOv8 (ultralytics) chosen for its excellent speed/accuracy trade-off
- Pre-trained on COCO dataset which includes 'person' class (class ID 0)
- Supports multiple model sizes (nano → xlarge) for different hardware
- Native ONNX/TensorRT export for production deployment
- Alternatives considered:
    - Faster R-CNN: Higher accuracy on small objects, but 3-5x slower inference
    - YOLOv5: Mature but superseded by v8 in both speed and accuracy
    - DETR: Transformer-based, better at occlusion but too slow for real-time
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
from loguru import logger
from ultralytics import YOLO


@dataclass
class Detection:
    """A single person detection in a frame."""
    bbox: np.ndarray          # [x1, y1, x2, y2] absolute pixel coords
    confidence: float
    class_id: int = 0         # COCO person class

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2,
        )

    @property
    def bottom_center(self) -> tuple[float, float]:
        """Foot position — better for zone checks on ground plane."""
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            self.bbox[3],
        )

    @property
    def area(self) -> float:
        return max(0, self.bbox[2] - self.bbox[0]) * max(0, self.bbox[3] - self.bbox[1])

    def to_dict(self) -> dict:
        return {
            "bbox": self.bbox.tolist(),
            "confidence": round(float(self.confidence), 4),
        }


@dataclass
class FrameDetections:
    """All detections in a single frame."""
    frame_idx: int
    detections: List[Detection] = field(default_factory=list)
    inference_ms: float = 0.0


class PersonDetector:
    """YOLOv8-based person detector.

    Filters detections to only the COCO 'person' class (ID 0) and applies
    confidence thresholding + optional NMS tuning.
    """

    PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        device: str = "",
        img_size: int = 640,
    ):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size

        logger.info(f"Loading detection model: {model_name}")
        self.model = YOLO(model_name)

        if device:
            self.device = device
        else:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Detector initialized on device: {self.device}")

    def detect(self, frame: np.ndarray, frame_idx: int = 0) -> FrameDetections:
        """Run person detection on a single frame."""
        results = self.model.predict(
            frame,
            classes=[self.PERSON_CLASS_ID],
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            device=self.device,
            verbose=False,
        )

        result = results[0]
        inference_ms = (result.speed["preprocess"] +
                        result.speed["inference"] +
                        result.speed["postprocess"])

        detections = []
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()

            for bbox, conf in zip(boxes, confs):
                detections.append(Detection(
                    bbox=bbox.astype(np.float32),
                    confidence=float(conf),
                ))

        return FrameDetections(
            frame_idx=frame_idx,
            detections=detections,
            inference_ms=inference_ms,
        )

    def detect_batch(
        self, frames: list[np.ndarray], start_idx: int = 0
    ) -> list[FrameDetections]:
        """Run detection on a batch of frames for better GPU utilization."""
        results = self.model.predict(
            frames,
            classes=[self.PERSON_CLASS_ID],
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            device=self.device,
            verbose=False,
        )

        frame_detections = []
        for i, result in enumerate(results):
            inference_ms = (result.speed["preprocess"] +
                            result.speed["inference"] +
                            result.speed["postprocess"])
            detections = []
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                for bbox, conf in zip(boxes, confs):
                    detections.append(Detection(
                        bbox=bbox.astype(np.float32),
                        confidence=float(conf),
                    ))

            frame_detections.append(FrameDetections(
                frame_idx=start_idx + i,
                detections=detections,
                inference_ms=inference_ms,
            ))

        return frame_detections
