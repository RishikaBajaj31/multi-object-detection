from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

from detection import Detection


@dataclass
class TrackRecord:
    frame_index: int
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    center_x: int
    center_y: int
    speed_px_per_frame: float


@dataclass
class TrackerConfig:
    max_age: int = 45
    n_init: int = 2
    max_iou_distance: float = 0.7
    max_cosine_distance: float = 0.35
    nn_budget: Optional[int] = 100
    trajectory_length: int = 30
    speed_window: int = 8
    trail_thickness: int = 2


class DeepSORTTracker:
    """
    DeepSORT-based multi-object tracker.

    How IDs are assigned:
    - YOLOv8 produces detections for the current frame.
    - DeepSORT predicts where previously seen tracks should appear using motion history.
    - It then matches new detections to existing tracks using both motion and appearance.
    - If a detection matches an existing track, that track keeps the same ID.
    - If it does not match any existing track strongly enough, a new ID is created.

    How identity switching is reduced:
    - Motion consistency: the Kalman filter predicts likely next positions.
    - Appearance consistency: DeepSORT compares visual embeddings, which helps when
      nearby subjects look spatially close but still have slightly different appearance.
    - Track history: tracks are kept alive for short occlusions (`max_age`), which helps
      maintain IDs when a subject is briefly blocked or blurred.
    """

    def __init__(self, config: TrackerConfig) -> None:
        self.config = config
        self.tracker = DeepSort(
            max_age=config.max_age,
            n_init=config.n_init,
            max_iou_distance=config.max_iou_distance,
            max_cosine_distance=config.max_cosine_distance,
            nn_budget=config.nn_budget,
            embedder="mobilenet",
            half=True,
            bgr=True,
        )
        self.unique_ids = set()
        self.trajectories: Dict[int, Deque[Tuple[int, int]]] = defaultdict(
            lambda: deque(maxlen=self.config.trajectory_length)
        )
        self.position_history: Dict[int, Deque[Tuple[int, int]]] = defaultdict(
            lambda: deque(maxlen=self.config.speed_window)
        )
        self.latest_speeds: Dict[int, float] = {}
        self.heatmap_accumulator: Optional[np.ndarray] = None

    def update(self, frame, detections: List[Detection], frame_index: int) -> List[TrackRecord]:
        if self.heatmap_accumulator is None:
            self.heatmap_accumulator = np.zeros(frame.shape[:2], dtype=np.float32)

        deep_sort_detections = []
        metadata = []

        for detection in detections:
            x1, y1, x2, y2 = detection.bbox_xyxy
            width = max(0, x2 - x1)
            height = max(0, y2 - y1)
            deep_sort_detections.append(([x1, y1, width, height], detection.confidence, detection.class_name))
            metadata.append(detection)

        # DeepSORT internally combines motion prediction + appearance matching.
        # This is why it is stronger than naive frame-to-frame centroid matching when
        # subjects overlap, camera motion happens, or short occlusions occur.
        tracks = self.tracker.update_tracks(deep_sort_detections, frame=frame)

        records: List[TrackRecord] = []
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = int(track.track_id)
            ltrb = track.to_ltrb(orig=False)
            x1, y1, x2, y2 = [int(v) for v in ltrb]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = max(x1, x2)
            y2 = max(y1, y2)

            confidence = 0.0
            class_id = -1
            class_name = "object"

            # Associate DeepSORT output back to the most relevant detector prediction.
            matched_detection = self._match_detection((x1, y1, x2, y2), metadata)
            if matched_detection is not None:
                confidence = matched_detection.confidence
                class_id = matched_detection.class_id
                class_name = matched_detection.class_name

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            self.unique_ids.add(track_id)
            self.trajectories[track_id].append((center_x, center_y))
            self.position_history[track_id].append((center_x, center_y))
            self._update_heatmap(center_x, center_y, frame.shape)
            speed = self._estimate_speed(track_id)
            self.latest_speeds[track_id] = speed

            records.append(
                TrackRecord(
                    frame_index=frame_index,
                    track_id=track_id,
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    center_x=center_x,
                    center_y=center_y,
                    speed_px_per_frame=speed,
                )
            )

        return records

    def draw_tracks(self, frame, records: List[TrackRecord], fps_text: str = ""):
        annotated_frame = frame.copy()
        for record in records:
            self._draw_track(annotated_frame, record)

        self._draw_overlay(annotated_frame, fps_text)
        return annotated_frame

    def build_heatmap_overlay(self, frame):
        if self.heatmap_accumulator is None:
            return frame

        normalized = cv2.normalize(self.heatmap_accumulator, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_uint8 = normalized.astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        return cv2.addWeighted(frame, 0.65, heatmap_color, 0.35, 0)

    def _draw_track(self, frame, record: TrackRecord):
        color = self._color_for_id(record.track_id)
        label = f"ID:{record.track_id} {record.class_name} {record.confidence:.2f}"

        cv2.rectangle(frame, (record.x1, record.y1), (record.x2, record.y2), color, 2)
        cv2.rectangle(frame, (record.x1, max(0, record.y1 - 42)), (record.x2, record.y1), color, -1)
        cv2.putText(
            frame,
            label,
            (record.x1 + 4, max(16, record.y1 - 22)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Speed: {record.speed_px_per_frame:.1f}px/f",
            (record.x1 + 4, max(34, record.y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        points = self.trajectories[record.track_id]
        if len(points) > 1:
            cv2.polylines(frame, [np.array(points, dtype=np.int32)], False, color, self.config.trail_thickness)

    def _draw_overlay(self, frame, fps_text: str):
        lines = [
            "Detector: YOLOv8",
            "Tracker: DeepSORT",
            f"Unique IDs: {len(self.unique_ids)}",
        ]
        if fps_text:
            lines.append(fps_text)

        y = 30
        for line in lines:
            cv2.putText(
                frame,
                line,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y += 30

    def _estimate_speed(self, track_id: int) -> float:
        history = self.position_history[track_id]
        if len(history) < 2:
            return 0.0

        distances = []
        points = list(history)
        for prev_point, next_point in zip(points[:-1], points[1:]):
            distances.append(float(np.linalg.norm(np.array(next_point) - np.array(prev_point))))
        return sum(distances) / len(distances) if distances else 0.0

    def _update_heatmap(self, center_x: int, center_y: int, frame_shape):
        if self.heatmap_accumulator is None:
            return
        height, width = frame_shape[:2]
        x1 = max(0, center_x - 10)
        x2 = min(width, center_x + 10)
        y1 = max(0, center_y - 10)
        y2 = min(height, center_y + 10)
        self.heatmap_accumulator[y1:y2, x1:x2] += 1.0

    @staticmethod
    def _match_detection(track_box: Tuple[int, int, int, int], detections: List[Detection]) -> Optional[Detection]:
        best_detection = None
        best_iou = 0.0
        for detection in detections:
            iou = DeepSORTTracker._compute_iou(track_box, tuple(detection.bbox_xyxy))
            if iou > best_iou:
                best_iou = iou
                best_detection = detection
        return best_detection

    @staticmethod
    def _compute_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter_area
        return inter_area / union if union > 0 else 0.0

    @staticmethod
    def _color_for_id(track_id: int) -> Tuple[int, int, int]:
        return (
            (37 * track_id) % 255,
            (17 * track_id) % 255,
            (29 * track_id) % 255,
        )
