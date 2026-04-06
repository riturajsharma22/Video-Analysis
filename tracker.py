"""DeepSORT tracking wrapper."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List
from typing import Any

from typing import Dict

from detector import Detection


@dataclass(frozen=True)
class TrackState:
    track_id: int
    class_name: str
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float


class DeepSortTracker:
    """Wrap `deep_sort_realtime` to produce stable IDs."""

    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 3,
        max_iou_distance: float = 0.7,
        nn_budget: int | None = None,
    ) -> None:
        self._log = logging.getLogger(self.__class__.__name__)

        from deep_sort_realtime.deepsort_tracker import DeepSort  # lazy import

        # Try to reduce overhead by using a lightweight embedder and GPU embeddings when available.
        use_gpu = False
        try:
            import torch

            use_gpu = bool(torch.cuda.is_available())
        except Exception:
            use_gpu = False

        try:
            self._tracker = DeepSort(
                max_age=max_age,
                n_init=n_init,
                max_iou_distance=max_iou_distance,
                nn_budget=nn_budget,
                embedder="mobilenet",
                embedder_gpu=use_gpu,
                half=use_gpu,
                bgr=True,
            )
        except TypeError:
            # Backward-compatible fallback for older deep_sort_realtime versions.
            self._tracker = DeepSort(
                max_age=max_age,
                n_init=n_init,
                max_iou_distance=max_iou_distance,
                nn_budget=nn_budget,
            )

        # Track ID -> initial (locked) class label.
        # This prevents label jitter (e.g., car<->truck) across frames.
        self._class_by_track_id: Dict[int, str] = {}

    def update(self, frame_bgr: Any, detections: List[Detection]) -> List[TrackState]:
        """Update DeepSORT and return confirmed tracks."""

        # deep-sort-realtime wants: List[Tuple[ (xmin,ymin,w,h), conf, class ]]
        ds_dets = []
        for det in detections:
            x1, y1, x2, y2 = det.xyxy
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            ds_dets.append(([x1, y1, w, h], det.confidence, det.class_name))

        tracks = self._tracker.update_tracks(ds_dets, frame=frame_bgr)

        out: List[TrackState] = []
        for t in tracks:
            if not t.is_confirmed():
                continue
            track_id = int(t.track_id)
            ltrb = t.to_ltrb()
            x1, y1, x2, y2 = map(float, ltrb)

            # Lock class label on first assignment.
            cls = getattr(t, "det_class", None)
            if track_id not in self._class_by_track_id:
                if cls is None:
                    # No class label available yet.
                    continue
                self._class_by_track_id[track_id] = str(cls)

            locked_cls = self._class_by_track_id[track_id]
            conf = float(getattr(t, "det_conf", 0.0) or 0.0)
            out.append(
                TrackState(
                    track_id=track_id,
                    class_name=locked_cls,
                    bbox_xyxy=(x1, y1, x2, y2),
                    confidence=conf,
                )
            )

        return out
