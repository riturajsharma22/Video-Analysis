"""Lane assignment logic."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LaneAssignment:
    lane: str  # 'left' or 'right'


class LaneAssigner:
    """Assign left/right lane using an x-coordinate split."""

    def __init__(
        self,
        frame_width: int,
        split_ratio: float = 0.5,
        road_x_min_ratio: float = 0.0,
        road_x_max_ratio: float = 1.0,
    ) -> None:
        """Create a lane assigner.

        Args:
            frame_width: Frame width in pixels.
            split_ratio: Split location inside the road ROI (0..1). Objects with
                center-x less than the split are assigned to `left`.
            road_x_min_ratio: Left bound of road ROI as a fraction of frame width.
            road_x_max_ratio: Right bound of road ROI as a fraction of frame width.

        Notes:
            This keeps the required ROI x-coordinate split approach while
            preventing non-road margins from skewing the split.
        """

        if not (0.05 <= split_ratio <= 0.95):
            raise ValueError("split_ratio must be between 0.05 and 0.95")
        if not (0.0 <= road_x_min_ratio < road_x_max_ratio <= 1.0):
            raise ValueError("road_x_min_ratio and road_x_max_ratio must satisfy 0 <= min < max <= 1")

        width = int(frame_width)
        roi_left = float(width) * float(road_x_min_ratio)
        roi_right = float(width) * float(road_x_max_ratio)
        self._split_x = roi_left + (roi_right - roi_left) * float(split_ratio)

    def assign(self, bbox_xyxy: tuple[float, float, float, float]) -> LaneAssignment:
        x1, y1, x2, y2 = bbox_xyxy
        cx = (x1 + x2) / 2.0
        lane = "left" if cx < self._split_x else "right"
        return LaneAssignment(lane=lane)
