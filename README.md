# Object Detection with YOLO

Real-time object detection and tracking on videos, images, or a live webcam feed, powered by [Ultralytics YOLO](https://github.com/ultralytics/ultralytics).

## Features

- Detect objects in videos, images, or webcam streams
- Optional object **tracking** with persistent IDs across frames
- Filter detections to specific classes (e.g. only `person`, `car`)
- Adjustable confidence and IoU thresholds
- Automatic GPU detection (falls back to CPU)
- Per-run object count summary

## Setup

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
```

## Usage

Run detection on a video file:
```bash
python detect.py --source path/to/video.mp4
```

Run on an image:
```bash
python detect.py --source path/to/image.jpg
```

Run on a webcam:
```bash
python detect.py --source 0
```

Enable tracking (assigns consistent object IDs across frames):
```bash
python detect.py --source video.mp4 --track
```

Only detect specific classes:
```bash
python detect.py --source video.mp4 --classes person car
```

Full options:
```bash
python detect.py --help
```

| Flag           | Description                                   | Default        |
|----------------|------------------------------------------------|----------------|
| `--source`     | Video/image path, or `0` for webcam            | required       |
| `--model`      | YOLO weights to use                            | `yolo11n.pt`   |
| `--conf`       | Confidence threshold                           | `0.5`          |
| `--iou`        | IoU threshold for NMS                          | `0.45`         |
| `--device`     | `cpu`, `cuda`, or auto-detected                | auto           |
| `--classes`    | Restrict to specific class names               | all classes    |
| `--track`      | Enable object tracking                         | off            |
| `--output-dir` | Where annotated output is saved                | `runs/detect`  |
| `--show`       | Display a live preview window while processing | off            |

Output videos/images are saved under `runs/detect/exp/`.

## Requirements

- Python 3.8+
- See `requirements.txt`

## Roadmap

- [ ] Streamlit/Gradio web UI for drag-and-drop uploads
- [ ] Export detection counts to CSV
- [ ] Dockerfile for containerized deployment
- [ ] Config file (YAML) support alongside CLI flags

## License

MIT (add a `LICENSE` file if you want this to apply).
