# Video Surveillance: Detection, Tracking & Event Recognition

A modular video surveillance pipeline that processes security camera footage to detect people, track them across frames, and identify events of interest (zone intrusions and loitering).

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    CLI / API Entry Point                        │
│                        (run.py)                                 │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Pipeline Orchestrator                          │
│                   (src/pipeline.py)                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │  Video   │→ │ Detect + │→ │  Event   │→ │  Visualize   │   │
│  │  Input   │  │  Track   │  │  Detect  │  │  & Output    │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────────────┘
        │              │              │               │
        ▼              ▼              ▼               ▼
  ┌──────────┐  ┌──────────────┐ ┌──────────┐  ┌──────────────┐
  │ OpenCV   │  │  YOLOv8 +    │ │  Zone    │  │  Annotated   │
  │ VideoI/O │  │  ByteTrack   │ │  Manager │  │  Video +     │
  │          │  │              │ │ (Shapely)│  │  Event Logs  │
  └──────────┘  └──────────────┘ └──────────┘  └──────────────┘
```

### Pipeline Stages

1. **Video Input** — OpenCV captures frames from video files. Supports frame skipping for faster processing.
2. **Person Detection** — YOLOv8 detects people in each frame with bounding boxes and confidence scores.
3. **Multi-Object Tracking** — ByteTrack assigns persistent IDs across frames, handling temporary occlusions and re-identification.
4. **Event Detection** — Zone-based analysis detects intrusions (entering restricted areas) and loitering (remaining stationary beyond a threshold). Uses a state machine per (track, zone) pair with cooldown-based deduplication.
5. **Visualization & Output** — Annotated video with bounding boxes, track IDs, trajectory trails, zone overlays, and event indicators. Structured JSON/CSV event logs.

## Model Choices

### Detection: YOLOv8 (ultralytics)

| Alternative      | Speed     | Accuracy  | Why Not Chosen                                     |
|-----------------|-----------|-----------|---------------------------------------------------|
| **YOLOv8**      | ★★★★★     | ★★★★      | **Selected** — best speed/accuracy balance         |
| Faster R-CNN    | ★★        | ★★★★★     | 3-5× slower, overkill for surveillance            |
| YOLOv5          | ★★★★      | ★★★       | Superseded by v8 in both metrics                  |
| DETR            | ★★        | ★★★★      | Transformer overhead, poor real-time performance  |
| YOLOv9/v10      | ★★★★★     | ★★★★★     | Newer but less ecosystem maturity                 |

**Why YOLOv8:**
- Pre-trained on COCO (80 classes including `person` as class 0)
- Multiple sizes: `yolov8n` (nano, 3.2M params, ~80 FPS) through `yolov8x` (68.2M params, ~15 FPS)
- Native batch inference and GPU acceleration
- Built-in tracker integration (ByteTrack/BoT-SORT)
- Easy ONNX/TensorRT export for production

### Tracking: ByteTrack

| Alternative      | Re-ID     | Speed     | Why Not Chosen                                     |
|-----------------|-----------|-----------|---------------------------------------------------|
| **ByteTrack**   | Motion    | ★★★★★     | **Selected** — robust, fast, no extra model needed |
| DeepSORT        | Appearance| ★★★       | Requires separate Re-ID CNN, slower                |
| BoT-SORT        | Both      | ★★★★      | Available as alternative (configurable)            |
| StrongSORT      | Appearance| ★★        | Heavy compute for marginal accuracy gain           |
| OC-SORT         | Motion    | ★★★★★     | Good but ByteTrack better tested                   |

**Why ByteTrack:**
- Two-stage association handles both high and low-confidence detections
- Pure motion-based (Kalman filter) — no appearance model overhead
- Robust to occlusion through second-round matching of unmatched tracks
- Track buffer maintains lost tracks for re-identification when subjects re-enter

## Setup Instructions

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended, not required — CPU inference works)

### Install

```bash
# Clone the repository
git clone <repo-url>
cd video-surveillance

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Docker

```bash
# Build
docker build -t video-surveillance .

# Run
docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/results:/app/results \
    video-surveillance --video data/input.mp4 --zones data/zones.json --output results/
```

## Usage

### Basic Run

```bash
python run.py --video input.mp4 --zones config/zones_example.json --output results/
```

### With Custom Model and Options

```bash
python run.py \
    --video surveillance_clip.mp4 \
    --zones config/zones_example.json \
    --output results/ \
    --model yolov8s.pt \
    --tracker bytetrack \
    --confidence 0.4 \
    --device cuda
```

### Process Faster (skip frames)

```bash
python run.py --video long_video.mp4 --output results/ --frame-skip 2
```

### Using a Config File

```bash
python run.py --video input.mp4 --config config/pipeline_config.json --output results/
```

### CLI Options

| Flag                 | Default        | Description                                     |
|---------------------|----------------|------------------------------------------------|
| `--video`           | (required)     | Input video file path                          |
| `--zones`           | auto-generated | Zone config JSON path                          |
| `--output`          | `results/`     | Output directory                               |
| `--model`           | `yolov8n.pt`   | YOLO model (n/s/m/l/x)                        |
| `--tracker`         | `bytetrack`    | Tracker: `bytetrack` or `botsort`              |
| `--confidence`      | `0.35`         | Detection confidence threshold                 |
| `--device`          | auto           | `cuda`, `cpu`, or auto-detect                  |
| `--frame-skip`      | `0`            | Process every Nth frame                        |
| `--loiter-threshold`| `50.0`         | Loitering displacement threshold (pixels)      |
| `--zone-method`     | `bottom_center`| Zone check: `bottom_center`/`center`/`intersect`/`overlap` |
| `--no-video`        | false          | Skip annotated video output                    |
| `--no-trajectories` | false          | Disable trajectory trails                      |

## Configuration

### Zone Config Format (`zones.json`)

```json
{
  "zones": [
    {
      "zone_id": "restricted_entrance",
      "name": "Restricted Entrance",
      "type": "restricted",
      "polygon": [[100, 200], [400, 200], [400, 500], [100, 500]],
      "color": [0, 0, 255],
      "loiter_threshold_sec": 30.0,
      "enabled": true
    }
  ]
}
```

**Zone types:**
- `restricted` — Intrusions generate HIGH severity events
- `monitored` — Intrusions generate MEDIUM severity events

**Creating zones interactively:**

```bash
python scripts/create_zones.py --video input.mp4 --output my_zones.json
```

This opens the first frame of the video. Left-click to add polygon points, right-click to finish a zone, and press `s` to save.

### Pipeline Config (`pipeline_config.json`)

All CLI parameters can be set in a JSON config file. See `config/pipeline_config.json` for the full template.

## Output

The pipeline generates these files in the output directory:

| File                          | Description                                    |
|------------------------------|------------------------------------------------|
| `{video}_annotated.mp4`     | Annotated video with detections, tracks, zones |
| `{video}_events.json`       | Structured event log (JSON)                    |
| `{video}_events.csv`        | Event log in CSV format                        |
| `{video}_stats.json`        | Pipeline statistics and track histories         |
| `{video}_summary.json`      | High-level summary with counts and timing       |

### Event Log Entry Example

```json
{
  "event_id": "EVT-000001",
  "event_type": "zone_intrusion",
  "severity": "high",
  "track_id": 3,
  "zone_id": "restricted_entrance",
  "zone_name": "Restricted Entrance",
  "frame_idx": 142,
  "timestamp_sec": 4.733,
  "bbox": [234.5, 156.2, 298.1, 345.8],
  "confidence": 0.87,
  "metadata": {
    "zone_type": "restricted"
  }
}
```

## Evaluation

Compare tracking results against MOTChallenge ground truth:

```bash
python scripts/evaluate.py --gt path/to/gt.txt --pred results/video_stats.json
```

Metrics computed: MOTA, MOTP, Precision, Recall, ID Switches.

## Sample Datasets

Download sample videos for testing:

```bash
python scripts/download_samples.py
```

Recommended clips:
- **MOT17**: 2-3 sequences (clean ground truth for tracking evaluation)
- **UCF-Crime**: 1-2 clips (real-world CCTV, varied quality)
- **VIRAT**: Outdoor surveillance with activity annotations
- **VisDrone**: Drone footage with multi-scale challenges

## Known Limitations

1. **ID Switches in Crowds**: ByteTrack relies on motion prediction; dense crowds with similar trajectories cause ID swaps. BoT-SORT (`--tracker botsort`) adds appearance features to mitigate this.

2. **Small/Distant Persons**: YOLOv8n struggles with persons below ~30px height. Use `yolov8s.pt` or `yolov8m.pt` and increase `--img-size 1280` for better small-object detection.

3. **Re-ID After Long Absence**: Track buffer (`--track-buffer`) controls how many frames a lost track is maintained. Extended absences (>2 seconds) will get a new ID.

4. **Lighting Changes**: Sudden lighting shifts (day→night, camera auto-exposure) can temporarily degrade detection confidence. The confidence threshold should be tuned per deployment.

5. **Camera Shake**: Unstable footage causes all bounding boxes to jitter, potentially triggering false loitering exits. Video stabilization preprocessing would help.

6. **Zone Accuracy on Non-Planar Scenes**: Zone polygons are defined in image space. For angled camera views, the bottom-center check approximates ground-plane position but isn't a true perspective projection.

7. **Memory on Long Videos**: Track histories accumulate in memory. For multi-hour videos, consider periodic pruning of completed tracks or streaming output to disk.

## Performance Notes

| Model      | GPU (RTX 3080) | CPU (i9-12900K) | Params  |
|-----------|----------------|-----------------|---------|
| YOLOv8n   | ~80 FPS        | ~15 FPS         | 3.2M    |
| YOLOv8s   | ~55 FPS        | ~8 FPS          | 11.2M   |
| YOLOv8m   | ~35 FPS        | ~4 FPS          | 25.9M   |
| YOLOv8l   | ~22 FPS        | ~2 FPS          | 43.7M   |

- Memory usage: ~500MB GPU VRAM for YOLOv8n, ~1.5GB for YOLOv8m
- CPU inference is viable for offline processing with frame skipping
- Batch inference (`detect_batch`) improves GPU utilization by ~20-30%

## Project Structure

```
├── run.py                      # CLI entry point
├── src/
│   ├── __init__.py
│   ├── detector.py             # YOLOv8 person detection
│   ├── tracker.py              # ByteTrack/BoT-SORT multi-object tracking
│   ├── zone_manager.py         # Zone polygon management (Shapely)
│   ├── event_detector.py       # Zone intrusion + loitering detection
│   ├── visualizer.py           # Frame annotation and rendering
│   └── pipeline.py             # Main pipeline orchestrator
├── config/
│   ├── zones_example.json      # Example zone configuration
│   └── pipeline_config.json    # Full pipeline config template
├── scripts/
│   ├── download_samples.py     # Download sample datasets
│   ├── evaluate.py             # MOT metrics evaluation
│   └── create_zones.py         # Interactive zone drawing tool
├── results/                    # Output directory
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container build
└── README.md                   # This file
```

## Future Improvements

With more time, these enhancements would strengthen the system:

- **Multi-camera support**: Process multiple feeds with cross-camera Re-ID
- **TensorRT optimization**: Export YOLOv8 to TensorRT for 2-3× inference speedup
- **Web dashboard**: Real-time monitoring UI with WebSocket streaming
- **Alert system**: Webhook/email notifications on critical events
- **Perspective correction**: Camera calibration for accurate ground-plane zone mapping
- **Activity recognition**: Extend beyond intrusion/loitering to detect fighting, falling, running
- **Database backend**: Store events in PostgreSQL/TimescaleDB for historical analysis
- **Video stabilization**: Preprocessing for shaky camera feeds
- **Distributed processing**: Multi-GPU or multi-node pipeline for large camera networks
