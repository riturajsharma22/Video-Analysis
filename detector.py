"""YOLOv8 detector wrapper."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List
from typing import Any


@dataclass(frozen=True)
class Detection:
    """Single object detection."""

    xyxy: tuple[float, float, float, float]
    confidence: float
    class_name: str


class YoloV8Detector:
    """Thin wrapper around `ultralytics.YOLO`.

    Notes:
        - We import ultralytics lazily so the project can be imported even
          if the dependency isn't installed yet.
    """

    def __init__(self, model_name: str, conf: float, iou: float, allowed_classes: Iterable[str]) -> None:
        self._log = logging.getLogger(self.__class__.__name__)
        self._model_name = model_name
        self._conf = conf
        self._iou = iou
        self._allowed = {c.strip().lower() for c in allowed_classes}

        from ultralytics import YOLO  # lazy import

        self._model = YOLO(model_name)

        # Device selection (GPU if available)
        self._device: str | int
        self._half: bool = False
        try:
            import torch

            if torch.cuda.is_available():
                self._device = 0
                self._half = True
            else:
                self._device = "cpu"
        except Exception:
            self._device = "cpu"

        # map class id -> name
        self._names: Dict[int, str] = getattr(self._model.model, "names", None) or getattr(self._model, "names", {})

    def detect(self, frame_bgr: Any) -> List[Detection]:
        """Run YOLO inference and return filtered detections."""

        import numpy as np

        results = self._model.predict(
            source=frame_bgr,
            verbose=False,
            conf=self._conf,
            iou=self._iou,
            device=self._device,
            half=self._half,
        )
        if not results:
            return []

        detections: List[Detection] = []
        r0 = results[0]
        boxes = getattr(r0, "boxes", None)
        if boxes is None:
            return []

        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
        conf = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
        cls = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls)

        for (x1, y1, x2, y2), c, cid in zip(xyxy, conf, cls):
            class_name = self._names.get(int(cid), str(int(cid)))
            if str(class_name).lower() not in self._allowed:
                continue
            detections.append(
                Detection(
                    xyxy=(float(x1), float(y1), float(x2), float(y2)),
                    confidence=float(c),
                    class_name=str(class_name),
                )
            )

        return detections
