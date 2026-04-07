# Traffic Video Analyzer

Production-grade Python project to analyze traffic CCTV video, detect & track objects, assign lanes, log structured events, and answer natural language queries using LLMs.

---

## Features

- Frame-by-frame video processing with configurable frame skipping (default: every 6th frame)
- Optional resize before detection for speed (default: 640x360)
- YOLOv8 object detection (default classes: `car`, `truck`, `bus`)
- DeepSORT tracking to maintain consistent object IDs across frames
- Lane assignment using x-coordinate split into `left` / `right`
- Structured event log written to `outputs/events.csv`
- Natural language query engine:
  - **Rule-based parsing** (always available, no API needed)
  - **Groq LLM** (recommended, free tier — set `LLM_PROVIDER=groq`)
  - Optional OpenAI fallback (set `OPENAI_API_KEY`)
  - Optional local LLM via Ollama (`LLM_PROVIDER=ollama`)

---

## Models Used

### Computer Vision

| Component | Library | Model |
|---|---|---|
| Object Detection | `ultralytics` | YOLOv8n (`yolov8n.pt`) |
| Object Tracking | `deep-sort-realtime` | DeepSORT |
| Lane Logic | Custom | ROI x-split → `left` / `right` |

### LLM (for natural language queries)

| Provider | How to enable | Model |
|---|---|---|
| **Groq** ✅ Recommended | `LLM_PROVIDER=groq` + `GROQ_API_KEY` in `.env` | `llama-3.3-70b-versatile` (free) |
| OpenAI | `OPENAI_API_KEY` in `.env` | `gpt-4o-mini` (default) |
| Ollama (local) | `LLM_PROVIDER=ollama` | `llama3.1` (default) |

> **Note:** The system always tries rule-based parsing first. LLM is only called for complex or ambiguous queries.

---

## Project Structure

```
traffic_video_analyzer/
├── main.py            ← Entry point, CLI argument handling
├── config.py          ← All configuration constants
├── detector.py        ← YOLOv8 object detection
├── tracker.py         ← DeepSORT object tracking
├── lane.py            ← Lane assignment logic
├── logger.py          ← CSV event logging
├── query_engine.py    ← Natural language query engine (rule-based + LLM)
├── utils.py           ← Shared helpers (time parsing, etc.)
├── requirements.txt   ← pip dependencies
├── pyproject.toml     ← uv/modern Python project config
├── outputs/
│   └── events.csv     ← Generated event log
└── README.md
```

---

## Setup (Windows)

### Option A — Standard pip

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Option B — uv (faster, recommended)

```powershell
uv venv -p 3.10
.\.venv\Scripts\Activate.ps1
uv sync
```

---

## Environment Setup

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
LLM_PROVIDER=groq
```

> Get a free Groq API key at: https://console.groq.com
> The app auto-loads `.env` at startup via `python-dotenv`.

---

## Running the Analyzer

### Step 1 — Process a video file

```powershell
uv run python main.py --video "D:\path\to\traffic.mp4"
```

This will:
1. Open the video frame by frame
2. Detect vehicles using YOLOv8
3. Track them using DeepSORT
4. Assign each to left/right lane
5. Save all events to `outputs/events.csv`
6. Run 3 sample queries automatically

### Step 2 — Query an existing events CSV

If you already have `outputs/events.csv`:

```powershell
uv run python main.py --events-csv outputs/events.csv --ask "Which lane is busiest?"
```

---

## Sample Queries and Results

These are real outputs generated from a 34-minute traffic CCTV video.

### Rule-based queries (no LLM needed)

| Query | Answer |
|---|---|
| "How many vehicles were detected per lane?" | right: 836, left: 151 |
| "how many cars passed between 3pm and 3:30pm?" | 729 cars |
| "Count vehicles before 00:05:00" | 109 vehicles |
| "How many trucks were detected by lane?" | right: 127, left: 52 |

### Complex LLM queries using Groq (`--llm-reasoning`)

| Query | Answer |
|---|---|
| "Which lane has more trucks compared to cars?" | Left lane has higher truck-to-car ratio (0.667 vs 0.195) |
| "Summarize overall traffic patterns by vehicle type and lane" | Cars 88% right, Trucks 59% right, Buses 64% right |
| "Which vehicle class dominates the right lane and by how much?" | Cars dominate with 652 vehicles — 525 more than trucks |

### Run all 3 complex queries at once

```powershell
uv run python main.py --events-csv outputs/events.csv `
  --ask "Which lane has more trucks compared to cars?" `
  --ask "Summarize overall traffic patterns by vehicle type and lane" `
  --ask "Which vehicle class dominates the right lane and by how much?" `
  --llm-reasoning
```

---

## Events CSV Format

`outputs/events.csv` columns:

| Column | Type | Description |
|---|---|---|
| `timestamp_s` | float | Seconds from video start |
| `object_id` | int | Stable DeepSORT track ID |
| `class` | string | `car`, `truck`, or `bus` |
| `lane` | string | `left` or `right` |

> Events are only logged when a track is **first seen** or **changes lane** to avoid duplicate counting.

### Sample CSV output:
```
timestamp_s,object_id,class,lane
0.2002,1,car,right
0.2002,3,car,right
0.6006,6,car,left
16.8168,45,truck,left
20.0200,72,bus,left
```

---

## CLI Options Reference

| Option | Default | Description |
|---|---|---|
| `--video` | — | Path to input video file |
| `--events-csv` | — | Path to existing events CSV (skip processing) |
| `--ask` | — | Natural language query (repeatable) |
| `--llm-reasoning` | off | Use LLM for narrative answers |
| `--frame-skip` | 6 | Process every Nth frame |
| `--model` | yolov8n.pt | YOLOv8 model to use |
| `--resize-width` | 640 | Resize frame width before detection |
| `--resize-height` | 360 | Resize frame height before detection |
| `--lane-split` | 0.5 | X-axis split point for lane assignment |
| `--output-dir` | outputs | Directory for output files |
| `--visualize` | off | Show live detection window (slower) |
| `--no-sample-queries` | off | Skip automatic sample queries |

---

## How It Works

### 1. Object Detection (YOLOv8)
Each frame is passed through YOLOv8 which returns bounding boxes and class labels for detected vehicles. Only configured classes (`car`, `truck`, `bus`) are kept.

### 2. Object Tracking (DeepSORT)
DeepSORT assigns a consistent ID to each vehicle across frames, so the same car keeps the same ID even as it moves through the frame.

### 3. Lane Assignment
The frame is split at the midpoint (x = 0.5 by default). Vehicles with center x < 0.5 go to `left` lane. Vehicles with center x >= 0.5 go to `right` lane.

### 4. Event Logging
An event is written to CSV only when a vehicle is first detected or changes lane. This prevents counting the same vehicle multiple times.

### 5. Query Engine Flow
```
User question (plain English)
        ↓
Rule-based parser tries first (fast, no API call)
        ↓ (if rule-based fails)
Groq LLM converts question to structured JSON
        ↓
Pandas filters events.csv using the structured query
        ↓
Clean answer returned to user
```

---

## LLM Provider Setup

### Groq (Recommended — Free)
```
GROQ_API_KEY=your_key_here
LLM_PROVIDER=groq
```
Get key at: https://console.groq.com

### OpenAI
```
OPENAI_API_KEY=your_key_here
LLM_PROVIDER=openai
```

### Ollama (Local, no internet needed)
```powershell
ollama pull llama3.1

$env:LLM_PROVIDER="ollama"
$env:OLLAMA_MODEL="llama3.1"
uv run python main.py --events-csv outputs/events.csv `
  --ask "Which lane is busiest?" `
  --llm-reasoning
```

---

## Requirements

- Python 3.10+
- Windows / Mac / Linux
- CUDA GPU (optional — speeds up YOLOv8 significantly)

Key dependencies:

```
ultralytics          # YOLOv8 object detection
deep-sort-realtime   # DeepSORT object tracking
groq                 # Groq LLM API (free tier)
pandas               # CSV querying and aggregation
python-dotenv        # Auto-loads .env at startup
opencv-python        # Video frame processing
```