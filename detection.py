from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
from ultralytics import YOLO


@dataclass
class Detection:
    """Container for one detector prediction."""

    bbox_xyxy: List[int]
    confidence: float
    class_id: int
    class_name: str


@dataclass
class DetectionConfig:
    model_path: str = "yolov8n.pt"
    conf_threshold: float = 0.35
    iou_threshold: float = 0.45
    image_size: int = 640
    classes: Optional[List[int]] = field(default_factory=lambda: [0])
    device: Optional[str] = None


class YOLODetector:
    """
    YOLO detector wrapper.

    This module is intentionally kept separate from tracking so the pipeline is easy to
    explain in submissions:
    1. detect objects in the frame
    2. pass detections to the tracker
    3. tracker decides whether each detection belongs to an existing ID or a new ID
    """

    def __init__(self, config: DetectionConfig) -> None:
        self.config = config
        self.device = config.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(config.model_path)

    def detect(self, frame) -> List[Detection]:
        """
        Run frame-level detection and convert the result into a tracker-friendly format.

        Returning plain python objects keeps the tracker implementation independent from
        Ultralytics internals and makes the code easier to modify later.
        """
        result = self.model.predict(
            source=frame,
            verbose=False,
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            classes=self.config.classes,
            device=self.device,
            imgsz=self.config.image_size,
        )[0]

        detections: List[Detection] = []
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().tolist()
        class_ids = result.boxes.cls.int().cpu().tolist()
        names = result.names

        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            detections.append(
                Detection(
                    bbox_xyxy=box.tolist(),
                    confidence=float(confidence),
                    class_id=class_id,
                    class_name=names.get(class_id, str(class_id)),
                )
            )

        return detections
