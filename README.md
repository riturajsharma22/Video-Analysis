# Traffic Video Analyzer

Production-grade Python project to analyze traffic CCTV video, detect & track objects, assign lanes, log structured events, and answer natural language queries.

## Features

- Frame-by-frame video processing with configurable frame skipping (default: every 6th frame)
- Optional resize before detection for speed (default: 640x360)
- YOLOv8 detection (default classes: `car`, `truck`, `bus`)
- DeepSORT tracking to maintain consistent object IDs
- Simple lane assignment using an x-coordinate split into `left` / `right`
- Structured event log written to `outputs/events.csv`
- Natural language query engine:
  - Rule-based parsing (always available)
  - Optional OpenAI fallback (set `OPENAI_API_KEY`)
  - Optional local LLM via Ollama (`LLM_PROVIDER=ollama`)

## Models used

### Computer vision

- **Detector:** YOLOv8 via `ultralytics` (default model: `yolov8n.pt`)
- **Tracker:** DeepSORT via `deep-sort-realtime`
- **Lane logic:** ROI-based x-split into `left` (median side) vs `right` (roadside)

### LLM (optional)

LLM is integrated for query parsing and (optionally) narrative reasoning.

- **Provider: OpenAI**
  - Enable by setting `OPENAI_API_KEY`
  - Model is configurable via `OPENAI_MODEL` (default: `gpt-4o-mini`)
- **Provider: Ollama (local)**
  - Enable by setting `LLM_PROVIDER=ollama`
  - Model is configurable via `OLLAMA_MODEL` (default: `llama3.1`)

## Project structure

```
traffic_video_analyzer/
├── main.py
├── config.py
├── detector.py
├── tracker.py
├── lane.py
├── logger.py
├── query_engine.py
├── utils.py
├── requirements.txt
└── README.md
```

## Setup (Windows)

1) Create and activate a virtual environment:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies:

```powershell
pip install -r requirements.txt
```

## Setup with uv (optional)

If you prefer `uv` (fast Python package manager), you can use either:

- `requirements.txt` (pip-style), or
- `pyproject.toml` (recommended for `uv run` / `uv sync`)

```powershell
uv venv -p 3.10
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

Or, using the included `pyproject.toml`:

```powershell
uv sync
uv run python main.py --video "D:\path\to\traffic.mp4"
```

Note: `ultralytics` pulls in PyTorch. If you need a CUDA-specific build, install PyTorch following the official PyTorch instructions, then re-run the `uv pip install -r requirements.txt` step.

## Run

From inside the `traffic_video_analyzer/` folder:

```powershell
python main.py --video "D:\path\to\traffic.mp4"
```

`main.py` is the standalone entrypoint that runs the end-to-end pipeline.

### Ask questions from an existing CSV (Option B)

If you already have `outputs/events.csv` and only want to query it:

```powershell
uv run python main.py --events-csv outputs/events.csv --ask "Which lane is busiest?"
uv run python main.py --events-csv outputs/events.csv --ask "Count trucks in the right lane after 00:05:00"
```

You can repeat `--ask` multiple times.

Outputs:
- `outputs/events.csv`
- Printed answers for 3 sample queries

### Useful CLI options

- `--frame-skip 6` (default)
- `--model yolov8n.pt` (default)
- `--resize-width 640 --resize-height 360` (default)
- `--lane-split 0.5` (default)
- `--road-xmin 0.0 --road-xmax 1.0` (default; restrict split to the road ROI when the road doesn’t fill the frame)
- `--output-dir outputs`
- `--no-sample-queries`
- `--visualize` (debug; slower)

## Events format

`outputs/events.csv` columns:
- `timestamp_s`: timestamp in seconds from start of video
- `object_id`: stable DeepSORT track ID
- `class`: object class name
- `lane`: `left` or `right`

Note: events are logged only when a track is first seen or changes lane to avoid duplicate counting.

## Sample queries

These work with the rule-based engine:

- "How many vehicles were detected per lane?"
- "Count cars in the left lane between 00:10:00 and 00:20:00"
- "Count people after 00:30:00"
- "How many trucks were detected by lane?" ("by lane" / "per lane")
- "Count vehicles before 00:05:00"

## OpenAI fallback (optional)

If you want the query engine to attempt LLM parsing when rule-based parsing fails:

```powershell
$env:OPENAI_API_KEY="your_key_here"
python main.py --video "D:\path\to\traffic.mp4"
```

The system always tries rule-based parsing first.

### LLM reasoning mode (optional)

By default, the LLM (if configured) is only used to convert hard-to-parse questions into a structured query.

If you want an LLM to produce a narrative answer grounded on a compact JSON summary of the CSV (not the full CSV), use `--llm-reasoning`:

```powershell
$env:OPENAI_API_KEY="your_key_here"
uv run python main.py --events-csv outputs/events.csv --ask "Summarize traffic by lane and class" --llm-reasoning
```

### Demo: 3 distinct complex queries (LLM-integrated)

This demonstrates count + filtering + pattern/aggregation using the integrated LLM reasoning mode:

```powershell
$env:OPENAI_API_KEY="your_key_here"
uv run python main.py --events-csv outputs/events.csv --ask "Count trucks in the right lane after 00:05:00" --ask "Count cars in the left lane between 00:10:00 and 00:20:00" --ask "Summarize traffic by lane and class" --llm-reasoning
```

### Local LLM via Ollama (optional)

1) Install and run Ollama.
2) Pull a model (example): `ollama pull llama3.1`
3) Set env vars and run:

```powershell
$env:LLM_PROVIDER="ollama"
$env:OLLAMA_MODEL="llama3.1"
$env:OLLAMA_HOST="http://localhost:11434"

uv run python main.py --events-csv outputs/events.csv --ask "Which lane is busiest?" --llm-reasoning
```
LLM is used only for semantic parsing into a constrained schema; all computation is performed by deterministic Pandas transforms over events.csv.