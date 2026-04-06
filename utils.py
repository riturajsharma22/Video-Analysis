"""Small utilities used across modules."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Optional


def setup_logging(level: str) -> None:
    """Configure root logging once."""

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


_TIME_RE = re.compile(
    r"^(?:(?P<h>\d{1,2}):)?(?P<m>\d{1,2}):(?P<s>\d{1,2})(?:\.(?P<ms>\d{1,3}))?$"
)


def parse_time_to_seconds(text: str) -> Optional[float]:
    """Parse HH:MM:SS(.ms) or MM:SS(.ms) to seconds."""

    text = text.strip()
    match = _TIME_RE.match(text)
    if not match:
        return None

    hours = int(match.group("h") or 0)
    minutes = int(match.group("m") or 0)
    seconds = int(match.group("s") or 0)
    ms = match.group("ms")
    millis = int(ms) if ms else 0

    total = hours * 3600 + minutes * 60 + seconds + millis / 1000.0
    return float(total)


def format_seconds(seconds: float) -> str:
    td = timedelta(seconds=float(seconds))
    # timedelta string is like '0:01:02.345000'
    s = str(td)
    return s


@dataclass(frozen=True)
class TimeWindow:
    start_s: Optional[float] = None
    end_s: Optional[float] = None

    def contains(self, t_s: float) -> bool:
        if self.start_s is not None and t_s < self.start_s:
            return False
        if self.end_s is not None and t_s > self.end_s:
            return False
        return True


def get_openai_api_key() -> Optional[str]:
    return os.getenv("OPENAI_API_KEY")


def get_openai_model(default: str = "gpt-4o-mini") -> str:
    """OpenAI model name for optional query parsing."""

    return os.getenv("OPENAI_MODEL", default)
