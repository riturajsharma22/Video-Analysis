"""Natural language query engine.

Rule-based parsing is mandatory.

Optional LLM support:
    - OpenAI (set `OPENAI_API_KEY`)
    - Ollama local model (set `LLM_PROVIDER=ollama` and ensure Ollama is running)

LLM is used in two ways:
    1) Fallback parsing: convert free-form NL query -> StructuredQuery
    2) Optional reasoning: produce a narrative answer from aggregated CSV stats
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from utils import TimeWindow, get_openai_api_key, get_openai_model, parse_clock_time_to_seconds, parse_time_to_seconds


@dataclass(frozen=True)
class StructuredQuery:
    intent: str  # 'count' | 'group_count'
    class_name: Optional[str] = None  # car/truck/bus/person or 'vehicle'
    lane: Optional[str] = None  # left/right
    time_window: TimeWindow = TimeWindow()
    group_by: Optional[str] = None  # 'lane'|'class'
    group_by_secondary: Optional[str] = None  # optional second group: 'lane'|'class'
    busiest: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "class_name": self.class_name,
            "lane": self.lane,
            "group_by": self.group_by,
            "group_by_secondary": self.group_by_secondary,
            "busiest": self.busiest,
            "time_window": {"start_s": self.time_window.start_s, "end_s": self.time_window.end_s},
        }


class RuleBasedQueryParser:
    """Parse a small set of query patterns into a StructuredQuery."""

    _LANE_RE = re.compile(r"\b(left|right)\b", re.IGNORECASE)
    _CLASS_RE = re.compile(r"\b(car|truck|bus|person|people|vehicle|vehicles)\b", re.IGNORECASE)
    _CLOCK_TOKEN_RE = r"[0-9:.]+(?:\s*(?:am|pm))?"
    _BETWEEN_RE = re.compile(
        rf"\bbetween\s+(?P<t1>{_CLOCK_TOKEN_RE})\s+(?:and|to)\s+(?P<t2>{_CLOCK_TOKEN_RE})\b",
        re.IGNORECASE,
    )
    _RANGE_RE = re.compile(
        rf"\b(?:from\s+)?(?P<t1>{_CLOCK_TOKEN_RE})\s+to\s+(?P<t2>{_CLOCK_TOKEN_RE})\b",
        re.IGNORECASE,
    )
    _AFTER_RE = re.compile(rf"\b(after|since)\s+(?P<t>{_CLOCK_TOKEN_RE})\b", re.IGNORECASE)
    _BEFORE_RE = re.compile(rf"\b(before|until)\s+(?P<t>{_CLOCK_TOKEN_RE})\b", re.IGNORECASE)

    def _parse_time_token_to_seconds(self, text: str) -> Optional[float]:
        t = text.strip()
        clock_s = parse_clock_time_to_seconds(t)
        if clock_s is not None:
            return float(clock_s)
        return parse_time_to_seconds(t)

    def parse(self, text: str) -> Optional[StructuredQuery]:
        q = text.strip().lower()

        lane = None
        m_lane = self._LANE_RE.search(q)
        if m_lane:
            lane = m_lane.group(1).lower()

        class_name = None
        m_cls = self._CLASS_RE.search(q)
        if m_cls:
            c = m_cls.group(1).lower()
            class_name = "person" if c == "people" else c

        # time window
        tw = TimeWindow()
        m_between = self._BETWEEN_RE.search(q)
        if m_between:
            t1 = self._parse_time_token_to_seconds(m_between.group("t1"))
            t2 = self._parse_time_token_to_seconds(m_between.group("t2"))
            if t1 is not None and t2 is not None:
                tw = TimeWindow(start_s=min(t1, t2), end_s=max(t1, t2))

        if tw.start_s is None and tw.end_s is None:
            m_range = self._RANGE_RE.search(q)
            if m_range:
                t1 = self._parse_time_token_to_seconds(m_range.group("t1"))
                t2 = self._parse_time_token_to_seconds(m_range.group("t2"))
                if t1 is not None and t2 is not None:
                    tw = TimeWindow(start_s=min(t1, t2), end_s=max(t1, t2))

        m_after = self._AFTER_RE.search(q)
        if m_after:
            t = self._parse_time_token_to_seconds(m_after.group("t"))
            if t is not None:
                tw = TimeWindow(start_s=t, end_s=tw.end_s)

        m_before = self._BEFORE_RE.search(q)
        if m_before:
            t = self._parse_time_token_to_seconds(m_before.group("t"))
            if t is not None:
                tw = TimeWindow(start_s=tw.start_s, end_s=t)

        # intent + grouping
        wants_lane = ("by lane" in q) or ("per lane" in q) or ("group by lane" in q)
        wants_class = ("by class" in q) or ("per class" in q) or ("group by class" in q)

        # 2D grouping: "by lane and class" / "by class and lane"
        if (wants_lane and wants_class) or ("by lane and class" in q) or ("by class and lane" in q):
            return StructuredQuery(
                intent="group_count",
                class_name=class_name,
                lane=lane,
                time_window=tw,
                group_by="class",
                group_by_secondary="lane",
            )
        if wants_lane:
            return StructuredQuery(intent="group_count", class_name=class_name, lane=lane, time_window=tw, group_by="lane")
        if wants_class:
            return StructuredQuery(intent="group_count", class_name=class_name, lane=lane, time_window=tw, group_by="class")

        if any(k in q for k in ["how many", "count", "number of", "total"]):
            return StructuredQuery(intent="count", class_name=class_name, lane=lane, time_window=tw)

        if "busiest lane" in q or "which lane is busiest" in q:
            return StructuredQuery(
                intent="group_count",
                class_name=class_name or "vehicle",
                lane=None,
                time_window=tw,
                group_by="lane",
                busiest=True,
            )

        return None


class QueryEngine:
    """Execute natural language queries over `events.csv`."""

    _VEHICLE_CLASSES = {"car", "truck", "bus"}

    def __init__(self, events_df: pd.DataFrame, allow_llm_fallback: bool = True) -> None:
        self._log = logging.getLogger(self.__class__.__name__)
        self._df = events_df.copy()
        self._parser = RuleBasedQueryParser()
        self._allow_llm = allow_llm_fallback

        # normalize
        if "class" not in self._df.columns:
            raise ValueError("events dataframe must include 'class'")
        if "timestamp_s" not in self._df.columns:
            raise ValueError("events dataframe must include 'timestamp_s'")

    def parse_structured(self, query: str) -> tuple[Optional[StructuredQuery], str]:
        """Parse a query into a StructuredQuery.

        Returns:
            (structured_query, source) where source is one of: rule|openai|ollama|none
        """

        structured = self._parser.parse(query)
        if structured is not None:
            return structured, "rule"

        if not self._allow_llm:
            return None, "none"

        provider = (os.getenv("LLM_PROVIDER") or "openai").strip().lower()
        structured = self._llm_to_structured(query)
        if structured is None:
            return None, "none"
        return structured, provider

    def answer(self, query: str) -> str:
        structured, _source = self.parse_structured(query)

        if structured is None:
            return "I couldn't parse that query with the rule-based engine. Try phrasing it like: 'count cars in left lane between 00:10:00 and 00:20:00'."

        result = self._execute(structured)
        return result

    def answer_with_reasoning(self, query: str) -> str:
        """Answer using an LLM over a compact summary of the CSV.

        This does not send the raw CSV; it sends aggregated statistics and a few
        small tables so the response can be grounded but efficient.

        If no LLM provider is configured, falls back to `answer()`.
        """

        provider = (os.getenv("LLM_PROVIDER") or "openai").strip().lower()
        if provider == "none":
            return self.answer(query)

        llm_text = self._llm_reason_over_summary(query)
        return llm_text or self.answer(query)

    def _apply_filters(self, sq: StructuredQuery) -> pd.DataFrame:
        df = self._df

        # time
        if sq.time_window.start_s is not None:
            df = df[df["timestamp_s"] >= sq.time_window.start_s]
        if sq.time_window.end_s is not None:
            df = df[df["timestamp_s"] <= sq.time_window.end_s]

        # lane
        if sq.lane is not None:
            df = df[df["lane"].str.lower() == sq.lane]

        # class
        if sq.class_name is not None:
            c = sq.class_name.lower()
            if c in ("vehicle", "vehicles"):
                df = df[df["class"].str.lower().isin(self._VEHICLE_CLASSES)]
            else:
                df = df[df["class"].str.lower() == c]

        return df

    def _execute(self, sq: StructuredQuery) -> str:
        df = self._apply_filters(sq)

        if sq.intent == "count":
            # count unique objects (deduped)
            n = df["object_id"].nunique()
            parts = [f"Count: {n}"]
            if sq.class_name:
                parts.append(f"class={sq.class_name}")
            if sq.lane:
                parts.append(f"lane={sq.lane}")
            if sq.time_window.start_s is not None or sq.time_window.end_s is not None:
                parts.append(f"time_window={sq.time_window}")
            return " | ".join(parts)

        if sq.intent == "group_count" and sq.group_by in ("lane", "class"):
            if sq.group_by_secondary in ("lane", "class") and sq.group_by_secondary != sq.group_by:
                # 2D breakdown (e.g., class x lane)
                pivot = (
                    df.pivot_table(
                        index=sq.group_by,
                        columns=sq.group_by_secondary,
                        values="object_id",
                        aggfunc=pd.Series.nunique,
                    )
                    .fillna(0)
                    .astype(int)
                )
                if pivot.empty:
                    return "No matching events."
                # Compact textual table
                lines = [f"Counts by {sq.group_by} and {sq.group_by_secondary}: "]
                lines.append(pivot.to_string())
                return "\n".join(lines)

            group_col = sq.group_by
            g = df.groupby(group_col)["object_id"].nunique().sort_values(ascending=False)
            if g.empty:
                return "No matching events."

            items = ", ".join([f"{idx}: {int(val)}" for idx, val in g.items()])
            if group_col == "lane" and sq.busiest:
                top_lane = str(g.index[0])
                top_count = int(g.iloc[0])
                return f"Counts by {group_col}: {items}. Busiest {group_col}: {top_lane} ({top_count})."
            return f"Counts by {group_col}: {items}"

        return "Unsupported query intent."

    def _llm_to_structured(self, query: str) -> Optional[StructuredQuery]:
        """Optional: convert free-form query to StructuredQuery using OpenAI.

        The rule-based parser remains the primary path.
        """

        provider = (os.getenv("LLM_PROVIDER") or "openai").strip().lower()
        try:
            if provider == "groq":
                return self._groq_to_structured(query)
            if provider == "ollama":
                return self._ollama_to_structured(query)
            if provider == "openai":
                return self._openai_to_structured(query)
        except Exception as e:
            self._log.warning("LLM fallback failed: %s", e)
        return None

    def _groq_to_structured(self, query: str):
        from groq import Groq

        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        system = (
            "You convert traffic video analytics questions into JSON with keys: "
            "intent (count|group_count), class_name (car|truck|bus|person|vehicle|null), "
            "lane (left|right|null), group_by (lane|class|null), "
            "time_window: {start_s: number|null, end_s: number|null}. "
            "Only output JSON, nothing else."
        )
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": query}
            ],
            temperature=0
        )
        content = resp.choices[0].message.content
        content = content.strip().strip("```json").strip("```").strip()
        data = json.loads(content)
        tw = data.get("time_window") or {}
        return StructuredQuery(
            intent=str(data.get("intent") or "count"),
            class_name=data.get("class_name"),
            lane=data.get("lane"),
            group_by=data.get("group_by"),
            time_window=TimeWindow(
                start_s=tw.get("start_s"),
                end_s=tw.get("end_s")
            ),
        )

    def _openai_to_structured(self, query: str) -> Optional[StructuredQuery]:
        api_key = get_openai_api_key()
        if not api_key:
            return None

        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        model = get_openai_model()
        system = (
            "You convert traffic video analytics questions into JSON with keys: "
            "intent (count|group_count), class_name (car|truck|bus|person|vehicle|null), "
            "lane (left|right|null), group_by (lane|class|null), "
            "time_window: {start_s: number|null, end_s: number|null}. "
            "Only output JSON."
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": query},
            ],
            temperature=0,
        )
        content = resp.choices[0].message.content
        data: Dict[str, Any] = json.loads(content)

        tw = data.get("time_window") or {}
        return StructuredQuery(
            intent=str(data.get("intent") or "count"),
            class_name=data.get("class_name"),
            lane=data.get("lane"),
            group_by=data.get("group_by"),
            time_window=TimeWindow(start_s=tw.get("start_s"), end_s=tw.get("end_s")),
        )

    def _ollama_to_structured(self, query: str) -> Optional[StructuredQuery]:
        """Local LLM parsing via Ollama HTTP API.

        Env:
          - LLM_PROVIDER=ollama
          - OLLAMA_HOST (default http://localhost:11434)
          - OLLAMA_MODEL (default llama3.1)
        """

        host = (os.getenv("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        model = os.getenv("OLLAMA_MODEL") or "llama3.1"

        system = (
            "You convert traffic video analytics questions into JSON with keys: "
            "intent (count|group_count), class_name (car|truck|bus|person|vehicle|null), "
            "lane (left|right|null), group_by (lane|class|null), "
            "time_window: {start_s: number|null, end_s: number|null}. "
            "Only output JSON."
        )

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": query},
            ],
            "stream": False,
        }
        req = urllib.request.Request(
            url=f"{host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
        data = json.loads(raw)
        content = (data.get("message") or {}).get("content")
        if not content:
            return None
        parsed: Dict[str, Any] = json.loads(content)
        tw = parsed.get("time_window") or {}
        return StructuredQuery(
            intent=str(parsed.get("intent") or "count"),
            class_name=parsed.get("class_name"),
            lane=parsed.get("lane"),
            group_by=parsed.get("group_by"),
            time_window=TimeWindow(start_s=tw.get("start_s"), end_s=tw.get("end_s")),
        )

    def _llm_reason_over_summary(self, query: str) -> Optional[str]:
        provider = (os.getenv("LLM_PROVIDER") or "openai").strip().lower()
        summary = self._build_summary()
        if provider == "groq":
            return self._groq_reason(query, summary)
        if provider == "ollama":
            return self._ollama_reason(query, summary)
        if provider == "openai":
            return self._openai_reason(query, summary)
        return None

    def _groq_reason(self, query, summary):
        from groq import Groq

        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        system = (
            "You answer questions about traffic events from a "
            "provided JSON summary. Do not invent numbers."
        )
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": 
                 f"Summary JSON:\n{json.dumps(summary)}\n\nQuestion: {query}"}
            ],
            temperature=0.2
        )
        return resp.choices[0].message.content

    def _build_summary(self) -> Dict[str, Any]:
        df = self._df

        # Core stats
        out: Dict[str, Any] = {}
        out["time_range_s"] = {
            "min": float(df["timestamp_s"].min()) if not df.empty else None,
            "max": float(df["timestamp_s"].max()) if not df.empty else None,
        }
        out["unique_objects"] = int(df["object_id"].nunique()) if not df.empty else 0

        # Counts (unique objects) by lane and class
        if not df.empty:
            out["counts_by_lane"] = (
                df.groupby("lane")["object_id"].nunique().sort_values(ascending=False).to_dict()
            )
            out["counts_by_class"] = (
                df.groupby("class")["object_id"].nunique().sort_values(ascending=False).to_dict()
            )
            out["counts_by_class_and_lane"] = (
                df.pivot_table(index="class", columns="lane", values="object_id", aggfunc=pd.Series.nunique)
                .fillna(0)
                .astype(int)
                .to_dict()
            )
        else:
            out["counts_by_lane"] = {}
            out["counts_by_class"] = {}
            out["counts_by_class_and_lane"] = {}

        return out

    def _openai_reason(self, query: str, summary: Dict[str, Any]) -> Optional[str]:
        api_key = get_openai_api_key()
        if not api_key:
            return None
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        model = get_openai_model()
        system = (
            "You answer questions about traffic events from a provided JSON summary. "
            "Do not invent numbers. If the summary lacks details, say what is missing."
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"Summary JSON:\n{json.dumps(summary)}\n\nQuestion: {query}"},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content

    def _ollama_reason(self, query: str, summary: Dict[str, Any]) -> Optional[str]:
        host = (os.getenv("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        model = os.getenv("OLLAMA_MODEL") or "llama3.1"
        system = (
            "You answer questions about traffic events from a provided JSON summary. "
            "Do not invent numbers. If the summary lacks details, say what is missing."
        )
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Summary JSON:\n{json.dumps(summary)}\n\nQuestion: {query}"},
            ],
            "stream": False,
        }
        req = urllib.request.Request(
            url=f"{host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
        data = json.loads(raw)
        return (data.get("message") or {}).get("content")
