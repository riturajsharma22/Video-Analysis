"""Traffic Video Analyzer entry point.

Usage:
    python main.py --video path/to/video.mp4

Outputs:
    - outputs/events.csv
    - Prints answers to 3 sample natural language queries
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import time
from typing import List

import cv2
import pandas as pd

from config import AnalyzerConfig
from detector import YoloV8Detector
from lane import LaneAssigner
from logger import EventLogger
from query_engine import QueryEngine
from tracker import DeepSortTracker
from utils import ensure_dir, setup_logging


def analyze_video(video_path: Path, cfg: AnalyzerConfig) -> Path:
    log = logging.getLogger("analyze_video")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    log.info("Video opened: fps=%.2f frames=%d", fps, frame_count)

    detector = YoloV8Detector(
        cfg.yolo_model,
        conf=cfg.conf_threshold,
        iou=cfg.iou_threshold,
        allowed_classes=cfg.allowed_classes,
    )
    tracker = DeepSortTracker(
        max_age=cfg.max_age,
        n_init=cfg.n_init,
        max_iou_distance=cfg.max_iou_distance,
        nn_budget=cfg.nn_budget,
    )
    lane_assigner = None
    event_logger = EventLogger()

    processed = 0
    frame_idx = -1  # absolute frame index in the original video stream
    start_wall = time.perf_counter()

    resize_w = int(cfg.resize_width) if cfg.resize_width else 0
    resize_h = int(cfg.resize_height) if cfg.resize_height else 0
    do_resize = resize_w > 0 and resize_h > 0

    frame_skip = max(1, int(cfg.frame_skip))

    while True:
        ok, frame = cap.read()  # decode only the processed frames
        if not ok:
            break
        frame_idx += 1

        if lane_assigner is None:
            h, w = frame.shape[:2]
            if do_resize:
                w = resize_w
            lane_assigner = LaneAssigner(
                frame_width=w,
                split_ratio=cfg.lane_split_ratio,
                road_x_min_ratio=cfg.road_x_min_ratio,
                road_x_max_ratio=cfg.road_x_max_ratio,
            )
            log.info(
                "Lane split initialized: width=%d road_roi=[%.2f..%.2f] split_ratio=%.2f",
                w,
                cfg.road_x_min_ratio,
                cfg.road_x_max_ratio,
                cfg.lane_split_ratio,
            )

        timestamp_s = float(frame_idx / fps)

        orig_h, orig_w = frame.shape[:2]
        work_frame = frame
        scale_x = 1.0
        scale_y = 1.0
        if do_resize and (orig_w != resize_w or orig_h != resize_h):
            work_frame = cv2.resize(frame, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
            scale_x = float(orig_w) / float(resize_w)
            scale_y = float(orig_h) / float(resize_h)

        detections = detector.detect(work_frame)
        tracks = tracker.update(work_frame, detections)

        for trk in tracks:
            lane = lane_assigner.assign(trk.bbox_xyxy).lane
            event_logger.maybe_log(timestamp_s, trk.track_id, trk.class_name, lane)

            if cfg.enable_visualization:
                x1, y1, x2, y2 = trk.bbox_xyxy
                x1o = int(x1 * scale_x)
                y1o = int(y1 * scale_y)
                x2o = int(x2 * scale_x)
                y2o = int(y2 * scale_y)
                cv2.rectangle(frame, (x1o, y1o), (x2o, y2o), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{trk.track_id}:{trk.class_name}:{lane}",
                    (max(0, x1o), max(0, y1o - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

        processed += 1
        if cfg.enable_visualization:
            cv2.imshow("Traffic Video Analyzer", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                log.info("Visualization interrupted by user")
                break

        if cfg.progress_every > 0 and processed % int(cfg.progress_every) == 0:
            wall_s = time.perf_counter() - start_wall
            proc_fps = (processed / wall_s) if wall_s > 0 else 0.0
            rt_factor = (timestamp_s / wall_s) if wall_s > 0 else 0.0
            log.info(
                "Processed %d frames | video t=%.1fs | wall=%.1fs | %.2f fps | x%.2f realtime | events=%d",
                processed,
                timestamp_s,
                wall_s,
                proc_fps,
                rt_factor,
                event_logger.event_count,
            )

        # Fast skip: advance without decoding
        if frame_skip > 1:
            for _ in range(frame_skip - 1):
                ok = cap.grab()
                if not ok:
                    break
                frame_idx += 1
            if not ok:
                break

    cap.release()
    if cfg.enable_visualization:
        cv2.destroyAllWindows()

    total_wall = time.perf_counter() - start_wall
    log.info("Finished processing: processed=%d frames | wall=%.1fs | events=%d", processed, total_wall, event_logger.event_count)

    ensure_dir(cfg.output_dir)
    out_csv = cfg.output_dir / cfg.events_csv_name
    event_logger.save_csv(out_csv)
    return out_csv


def run_queries(events_csv: Path, queries: List[str]) -> None:
    df = pd.read_csv(events_csv)
    engine = QueryEngine(df, allow_llm_fallback=True)

    print("\n=== Query Answers ===")
    for i, q in enumerate(queries, start=1):
        ans = engine.answer(q)
        print(f"Q{i}: {q}\nA{i}: {ans}\n")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Traffic Video Analyzer")
    p.add_argument("--video", help="Path to input traffic CCTV video")
    p.add_argument("--events-csv", help="Use an existing events.csv instead of processing a video")
    p.add_argument(
        "--ask",
        action="append",
        default=[],
        help="Ask a natural language question against events.csv (repeatable)",
    )
    p.add_argument(
        "--llm-reasoning",
        action="store_true",
        help="Use LLM reasoning over a compact CSV summary (optional)",
    )
    p.add_argument(
        "--debug-query",
        action="store_true",
        help="Print parsed StructuredQuery JSON before answering (query integrity)",
    )
    p.add_argument("--frame-skip", type=int, default=AnalyzerConfig.frame_skip, help="Process every Nth frame (default: 6)")
    p.add_argument("--model", default=AnalyzerConfig.yolo_model, help="YOLOv8 model name/path (default: yolov8n.pt)")
    p.add_argument("--conf", type=float, default=AnalyzerConfig.conf_threshold, help="Detection confidence threshold")
    p.add_argument("--iou", type=float, default=AnalyzerConfig.iou_threshold, help="NMS IoU threshold")
    p.add_argument("--resize-width", type=int, default=AnalyzerConfig.resize_width, help="Resize width before detection (default: 640)")
    p.add_argument("--resize-height", type=int, default=AnalyzerConfig.resize_height, help="Resize height before detection (default: 360)")
    p.add_argument("--lane-split", type=float, default=AnalyzerConfig.lane_split_ratio, help="Lane split ratio (0..1), default 0.5")
    p.add_argument(
        "--road-xmin",
        type=float,
        default=AnalyzerConfig.road_x_min_ratio,
        help="Road ROI min x ratio (0..1). Use to exclude non-road left margin.",
    )
    p.add_argument(
        "--road-xmax",
        type=float,
        default=AnalyzerConfig.road_x_max_ratio,
        help="Road ROI max x ratio (0..1). Use to exclude non-road right margin.",
    )
    p.add_argument("--output-dir", default=str(AnalyzerConfig.output_dir), help="Output directory")
    p.add_argument("--no-sample-queries", action="store_true", help="Do not run sample queries")
    p.add_argument("--visualize", action="store_true", help="Enable visualization (slower)")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    cfg = AnalyzerConfig(
        frame_skip=max(1, int(args.frame_skip)),
        yolo_model=str(args.model),
        conf_threshold=float(args.conf),
        iou_threshold=float(args.iou),
        resize_width=int(args.resize_width),
        resize_height=int(args.resize_height),
        enable_visualization=bool(args.visualize),
        lane_split_ratio=float(args.lane_split),
        road_x_min_ratio=float(args.road_xmin),
        road_x_max_ratio=float(args.road_xmax),
        output_dir=Path(args.output_dir),
        run_sample_queries=not bool(args.no_sample_queries),
    )

    setup_logging(cfg.log_level)
    log = logging.getLogger("main")

    if args.video and args.events_csv:
        raise ValueError("Provide only one of --video or --events-csv")

    if args.events_csv:
        events_csv = Path(args.events_csv)
        if not events_csv.exists():
            raise FileNotFoundError(str(events_csv))
        log.info("Using existing events CSV: %s", str(events_csv))
    else:
        if not args.video:
            raise ValueError("You must provide --video or --events-csv")
        video_path = Path(args.video)
        if not video_path.exists():
            raise FileNotFoundError(str(video_path))

        log.info("Starting analysis")
        events_csv = analyze_video(video_path, cfg)
        log.info("Events written to %s", str(events_csv))

    # Answer custom questions if provided
    if args.ask:
        df = pd.read_csv(events_csv)
        engine = QueryEngine(df, allow_llm_fallback=True)
        print("\n=== Query Answers ===")
        for i, q in enumerate(args.ask, start=1):
            if args.debug_query:
                sq, src = engine.parse_structured(q)
                print(f"[debug] parser={src} structured={sq.to_dict() if sq else None}")

            ans = engine.answer_with_reasoning(q) if args.llm_reasoning else engine.answer(q)
            print(f"Q{i}: {q}\nA{i}: {ans}\n")
        return

    if cfg.run_sample_queries:
        sample_queries = [
            "How many vehicles were detected per lane?",
            "Count cars in the left lane between 00:10:00 and 00:20:00",
            "Which lane is busiest?",
        ]
        run_queries(events_csv, sample_queries)


if __name__ == "__main__":
    main()
