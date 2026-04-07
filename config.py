"""Configuration for Traffic Video Analyzer.

Keep this file small and explicit. Override via CLI args in `main.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# Assume the video starts at this wall-clock time.
# Used to convert between wall-clock times (e.g., "3:30 PM") and seconds-from-start offsets.
VIDEO_START_TIME = "15:00:00"  # 3:00 PM


@dataclass(frozen=True)
class AnalyzerConfig:
    """Runtime configuration."""

    # Video processing
    frame_skip: int = 6  # process every Nth frame

    # Resize (applied before detection/tracking)
    resize_width: int = 640
    resize_height: int = 360

    # Optional visualization (debug only; slows down processing)
    enable_visualization: bool = False

    # Progress logging
    progress_every: int = 200  # processed frames

    # Detection
    yolo_model: str = "yolov8n.pt"  # small+fast; change to yolov8s.pt for better quality
    conf_threshold: float = 0.35
    iou_threshold: float = 0.5

    # Filter detections early to reduce tracker load
    allowed_classes: tuple[str, ...] = ("car", "truck", "bus")

    # Tracking (DeepSORT)
    max_age: int = 15
    n_init: int = 2
    max_iou_distance: float = 0.7
    nn_budget: int | None = None

    # Lane assignment
    lane_split_ratio: float = 0.5  # x < ratio*width => left, else right
    road_x_min_ratio: float = 0.0  # optional ROI (relative x) for road area
    road_x_max_ratio: float = 1.0

    # Output
    output_dir: Path = Path("outputs")
    events_csv_name: str = "events.csv"

    # Logging
    log_level: str = "INFO"

    # Queries
    run_sample_queries: bool = True
