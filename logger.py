"""Structured event logging to CSV."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd

from config import VIDEO_START_TIME


@dataclass(frozen=True)
class Event:
    timestamp_s: float
    object_id: int
    class_name: str
    lane: str


class EventLogger:
    """Accumulate structured events and write `events.csv`.

    To avoid duplicate counting and huge CSVs, we only log when a track first
    appears or when its lane changes.
    """

    def __init__(self) -> None:
        self._log = logging.getLogger(self.__class__.__name__)
        self._events: List[Event] = []
        self._last_lane: Dict[int, str] = {}  # track_id -> last lane
        self._class_by_id: Dict[int, str] = {}  # track_id -> locked initial class

    def maybe_log(self, timestamp_s: float, object_id: int, class_name: str, lane: str) -> None:
        if object_id not in self._class_by_id:
            self._class_by_id[object_id] = str(class_name)
        class_name = self._class_by_id[object_id]

        prev_lane = self._last_lane.get(object_id)
        if prev_lane == lane:
            return
        self._last_lane[object_id] = lane
        self._events.append(Event(timestamp_s=float(timestamp_s), object_id=int(object_id), class_name=str(class_name), lane=str(lane)))

    @property
    def event_count(self) -> int:
        return len(self._events)

    def to_dataframe(self) -> pd.DataFrame:
        start_dt = datetime.strptime(VIDEO_START_TIME, "%H:%M:%S")
        return pd.DataFrame(
            [
                {
                    "timestamp_s": e.timestamp_s,
                    "wall_clock_time": (start_dt + timedelta(seconds=float(e.timestamp_s))).time().strftime("%H:%M:%S"),
                    "object_id": e.object_id,
                    "class": e.class_name,
                    "lane": e.lane,
                }
                for e in self._events
            ]
        )

    def save_csv(self, path: Path) -> Path:
        df = self.to_dataframe()
        df.sort_values(["timestamp_s", "object_id"], inplace=True, ignore_index=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        self._log.info("Wrote %d events to %s", len(df), str(path))
        return path
