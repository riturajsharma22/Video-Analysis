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

### Install Playwright browser (for recording)

```powershell
playwright install chromium
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
python main.py --video "D:\path\to\traffic.mp4"
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
python main.py --events-csv outputs/events.csv --ask "Which lane is busiest?"
```

---

## Query Examples

### Simple queries (rule-based, no LLM needed)

```powershell
python main.py --events-csv outputs/events.csv --ask "How many vehicles were detected per lane?"
python main.py --events-csv outputs/events.csv --ask "Count cars in the left lane between 00:10:00 and 00:20:00"
python main.py --events-csv outputs/events.csv --ask "How many trucks were detected by lane?"
python main.py --events-csv outputs/events.csv --ask "Count vehicles before 00:05:00"
```

### Time-based queries (supports real clock time)

```powershell
python main.py --events-csv outputs/events.csv --ask "how many cars passed between 3pm and 3:30pm?"
python main.py --events-csv outputs/events.csv --ask "Count trucks after 00:30:00"
```

### Complex queries using Groq LLM (`--llm-reasoning`)

```powershell
uv run python main.py --events-csv outputs/events.csv `
  --ask "Which lane has more trucks compared to cars?" `
  --ask "Summarize overall traffic patterns by vehicle type and lane" `
  --ask "Which vehicle class dominates the right lane and by how much?" `
  --llm-reasoning
```

---

## Sample Query Results

These are real outputs from a 34-minute traffic video (2083 seconds):

### Q1 — Filtering + Comparison
**Query:** "Which lane has more trucks compared to cars?"

**Answer:**
```
Left lane:  52 trucks / 78 cars  = 0.667 ratio (2/3 trucks per car)
Right lane: 127 trucks / 652 cars = 0.195 ratio (1/5 trucks per car)
→ The LEFT lane has a higher truck-to-car ratio.
```

### Q2 — Pattern Recognition + Aggregation
**Query:** "Summarize overall traffic patterns by vehicle type and lane"

**Answer:**
```
Cars:   655 total → 78 (12%) left lane, 577 (88%) right lane
Trucks: 127 total → 52 (41%) left lane, 75  (59%) right lane
Buses:  58  total → 21 (36%) left lane, 37  (64%) right lane
Overall: Right lane dominates with 836 vehicles vs 151 in left lane.
```

### Q3 — Complex Reasoning
**Query:** "Which vehicle class dominates the right lane and by how much?"

**Answer:**
```
Cars dominate the right lane with 652 vehicles.
Margin over next class (trucks): 525 vehicles.
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
The frame is split at the midpoint (x = 0.5 by default). Vehicles with center x < 0.5 → `left` lane. Vehicles with center x >= 0.5 → `right` lane.

### 4. Event Logging
An event is written to CSV only when a vehicle is first detected or changes lane. This prevents counting the same vehicle multiple times.

### 5. Query Engine
```
User question (English)
        ↓
Rule-based parser tries first
        ↓ (if fails)
Groq LLM converts to structured JSON
        ↓
Pandas filters events.csv
        ↓
Answer returned
```

---

## LLM Provider Setup

### Groq (Recommended — Free)
```
GROQ_API_KEY=your_key_here
LLM_PROVIDER=groq
```
Get key: https://console.groq.com

### OpenAI
```
OPENAI_API_KEY=your_key_here
LLM_PROVIDER=openai
```

### Ollama (Local, no internet needed)
```powershell
# Install Ollama, then:
ollama pull llama3.1

$env:LLM_PROVIDER="ollama"
$env:OLLAMA_MODEL="llama3.1"
uv run python main.py --events-csv outputs/events.csv --ask "Which lane is busiest?" --llm-reasoning
```

---

## Requirements

- Python 3.10+
- Windows / Mac / Linux
- CUDA GPU (optional, speeds up YOLOv8 significantly)

Key dependencies:
```
ultralytics       # YOLOv8
deep-sort-realtime # DeepSORT tracking
groq              # Groq LLM API
pandas            # CSV querying
python-dotenv     # .env loading
opencv-python     # Video processing
```