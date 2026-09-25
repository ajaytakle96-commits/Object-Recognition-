"""
Object Detection with YOLO
---------------------------
Run YOLO object detection/tracking on a video, image, or webcam feed.

Usage:
    python detect.py --source video.mp4
    python detect.py --source image.jpg
    python detect.py --source 0                     # webcam
    python detect.py --source video.mp4 --track      # with object tracking + IDs
    python detect.py --source video.mp4 --classes person car --conf 0.6
"""

import argparse
import logging
import sys
from pathlib import Path

from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO object detection/tracking")
    parser.add_argument(
        "--source", required=True,
        help="Path to video/image file, or '0' for webcam"
    )
    parser.add_argument(
        "--model", default="yolo11n.pt",
        help="YOLO model weights to use (default: yolo11n.pt)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.5,
        help="Confidence threshold (default: 0.5)"
    )
    parser.add_argument(
        "--iou", type=float, default=0.45,
        help="IoU threshold for NMS (default: 0.45)"
    )
    parser.add_argument(
        "--device", default=None,
        help="Device to run on: 'cpu', 'cuda', or leave unset for auto-detect"
    )
    parser.add_argument(
        "--classes", nargs="*", default=None,
        help="Only detect these class names, e.g. --classes person car"
    )
    parser.add_argument(
        "--track", action="store_true",
        help="Enable object tracking (assigns consistent IDs across frames)"
    )
    parser.add_argument(
        "--output-dir", default="runs/detect",
        help="Directory to save annotated output (default: runs/detect)"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display output in a live window while processing"
    )
    return parser.parse_args()


def resolve_device(requested):
    if requested:
        return requested
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def resolve_class_indices(model, class_names):
    if not class_names:
        return None
    name_to_id = {v: k for k, v in model.names.items()}
    indices = []
    for name in class_names:
        if name not in name_to_id:
            logger.warning("Class '%s' not found in model classes, skipping.", name)
            continue
        indices.append(name_to_id[name])
    return indices or None


def main():
    args = parse_args()

    # webcam index vs file path
    source = int(args.source) if args.source.isdigit() else args.source
    if isinstance(source, str) and not Path(source).exists():
        logger.error("Source file not found: %s", source)
        sys.exit(1)

    device = resolve_device(args.device)
    logger.info("Loading model '%s' on device '%s'...", args.model, device)
    model = YOLO(args.model)

    class_indices = resolve_class_indices(model, args.classes)

    logger.info("Running %s on source: %s", "tracking" if args.track else "detection", source)

    run_fn = model.track if args.track else model.predict
    kwargs = dict(
        source=source,
        conf=args.conf,
        iou=args.iou,
        device=device,
        classes=class_indices,
        save=True,
        show=args.show,
        project=args.output_dir,
        name="exp",
    )
    if args.track:
        kwargs["persist"] = True

    results = run_fn(**kwargs)

    # Simple per-run object count summary
    if results:
        last = results[-1]
        if last.boxes is not None and len(last.boxes) > 0:
            counts = {}
            for cls_id in last.boxes.cls.tolist():
                name = model.names[int(cls_id)]
                counts[name] = counts.get(name, 0) + 1
            summary = ", ".join(f"{v} {k}" for k, v in counts.items())
            logger.info("Detected in final frame: %s", summary)

    logger.info("Done. Output saved under: %s", args.output_dir)


if __name__ == "__main__":
    main()
