from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
import torch
import ultralytics.nn.tasks
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel])


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    label: str
    crop: np.ndarray  # YOLO가 잘라낸 숫자판 이미지


class DisplayDetector:

    def __init__(self, model_path: str, confidence: float = 0.5, imgsz: int = 320):
        path = Path(model_path)
        if not path.exists():
            print(f"모델 없음 → 자동 다운로드: {path.name}")
        self._model = YOLO(str(path))
        self._conf = confidence
        self._imgsz = imgsz

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self._model(frame, conf=self._conf, imgsz=self._imgsz, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=float(box.conf),
                    label=r.names[int(box.cls)],
                    crop=crop,
                ))
        return detections

    def draw(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        vis = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, f"{d.label} {d.confidence:.2f}",
                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
        return vis
