#!/usr/bin/env python3
"""Annotate people in a video with detection, tracking, pose, and height measurements.

The default measurement is bounding-box height in pixels. A metric estimate is
only enabled when a calibration reference is supplied:

    estimated_height_m = person_height_px / reference_height_px * reference_height_m

For a useful metric estimate, the reference and target person should be on the
same ground plane and viewed by the same camera at roughly the same time.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from statistics import median
from typing import Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Silent annotated MP4 output path")
    parser.add_argument("--csv", required=True, help="Frame-level measurements CSV path")
    parser.add_argument("--model", default="yolo11n-pose.pt", help="Ultralytics pose model")
    parser.add_argument("--conf", type=float, default=0.30, help="Detection confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--max-frames", type=int, default=0, help="Process at most this many frames; 0 means all")
    parser.add_argument("--reference-height-m", type=float, default=None, help="Known reference height in meters")
    parser.add_argument("--reference-pixel-height-px", type=float, default=None, help="Known reference height in pixels")
    parser.add_argument("--reference-track-id", type=int, default=None, help="Track ID of a known-height reference person")
    parser.add_argument("--reference-window", type=int, default=300, help="Number of reference pixel heights kept for auto-calibration")
    args = parser.parse_args()
    # Calibration needs a real height plus a pixel reference (fixed or from a
    # tracked person). Half a calibration would silently fall back to pixels.
    has_pixel_reference = args.reference_pixel_height_px is not None or args.reference_track_id is not None
    if args.reference_height_m is not None and not has_pixel_reference:
        parser.error("--reference-height-m needs --reference-pixel-height-px or --reference-track-id")
    if has_pixel_reference and args.reference_height_m is None:
        parser.error("--reference-pixel-height-px / --reference-track-id need --reference-height-m")
    return args


# Track ID for a detection the tracker has not confirmed yet.
UNTRACKED = -1


def safe_int(value: float) -> int:
    return int(round(float(value)))


def make_color(track_id: int) -> tuple[int, int, int]:
    # Deterministic BGR color per track ID.
    rng = np.random.default_rng(abs(int(track_id)) + 2026)
    color = rng.integers(55, 235, size=3).tolist()
    return int(color[0]), int(color[1]), int(color[2])


def draw_label(frame: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max(0, x)
    y = max(th + baseline + 2, y)
    cv2.rectangle(frame, (x, y - th - baseline - 4), (x + tw + 6, y + 2), color, -1)
    cv2.putText(frame, text, (x + 3, y - 2), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {args.output}")

    model = YOLO(args.model)
    device = 0 if torch.cuda.is_available() else "cpu"
    reference_pixels: list[float] = []
    fixed_reference_px = args.reference_pixel_height_px
    use_metric = args.reference_height_m is not None and (
        fixed_reference_px is not None or args.reference_track_id is not None
    )
    rows: list[dict] = []
    frame_index = 0

    # One tracker session is maintained by persist=True across frames.
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_frames and frame_index >= args.max_frames:
            break

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],  # COCO person class
            conf=args.conf,
            imgsz=args.imgsz,
            device=device,
            verbose=False,
        )
        result = results[0]

        # Draw pose keypoints/skeleton first, then add our measurement boxes/labels.
        try:
            annotated = result.plot(boxes=False, labels=False, conf=False, kpt_line=True, kpt_radius=2, line_width=1)
        except Exception:
            annotated = frame.copy()

        boxes = result.boxes
        xyxy = boxes.xyxy.detach().cpu().numpy() if boxes is not None and boxes.xyxy is not None else np.empty((0, 4))
        confs = boxes.conf.detach().cpu().numpy() if boxes is not None and boxes.conf is not None else np.empty((0,))
        # The tracker needs a frame or two to confirm new people; until then boxes
        # have no ID. Mark them -1 rather than numbering them 0..n, which would
        # collide with real track IDs (and could feed the reference calibration).
        ids = boxes.id.detach().cpu().numpy().astype(int) if boxes is not None and boxes.id is not None else np.full(len(xyxy), UNTRACKED, dtype=int)

        # If the user supplied a reference track ID, estimate its pixel height online.
        for bbox, tid in zip(xyxy, ids):
            if args.reference_track_id is not None and int(tid) == int(args.reference_track_id):
                ref_px = float(max(0.0, bbox[3] - bbox[1]))
                if ref_px > 1:
                    reference_pixels.append(ref_px)
                    if len(reference_pixels) > max(1, args.reference_window):
                        reference_pixels.pop(0)

        reference_px = fixed_reference_px
        if reference_px is None and reference_pixels:
            reference_px = float(median(reference_pixels))

        time_s = frame_index / fps
        for i, bbox in enumerate(xyxy):
            x1, y1, x2, y2 = [float(v) for v in bbox]
            bbox_h = max(0.0, y2 - y1)
            tid = int(ids[i]) if i < len(ids) else UNTRACKED
            id_text = f"ID {tid}" if tid != UNTRACKED else "ID ?"
            score = float(confs[i]) if i < len(confs) else float("nan")
            height_m: Optional[float] = None
            if use_metric and reference_px is not None and reference_px > 1:
                height_m = bbox_h / reference_px * float(args.reference_height_m)

            color = make_color(tid)
            p1 = (safe_int(x1), safe_int(y1))
            p2 = (safe_int(x2), safe_int(y2))
            cv2.rectangle(annotated, p1, p2, color, 2)
            cv2.circle(annotated, (safe_int((x1 + x2) / 2), safe_int(y2)), 3, color, -1)

            if height_m is None:
                label = f"{id_text} | Hpx {bbox_h:.0f} | {score:.2f}"
            else:
                label = f"{id_text} | H {height_m:.2f} m | {score:.2f}"
            draw_label(annotated, label, safe_int(x1), safe_int(y1) - 2, color)

            rows.append(
                {
                    "frame": frame_index,
                    "time_s": f"{time_s:.3f}",
                    "track_id": tid if tid != UNTRACKED else "",
                    "confidence": f"{score:.4f}",
                    "x1": f"{x1:.2f}",
                    "y1": f"{y1:.2f}",
                    "x2": f"{x2:.2f}",
                    "y2": f"{y2:.2f}",
                    "height_px": f"{bbox_h:.2f}",
                    "height_m": "" if height_m is None else f"{height_m:.4f}",
                    "reference_px_used": "" if reference_px is None else f"{reference_px:.2f}",
                }
            )

        # Small header gives context even when labels are not visible.
        header = "Person detection + tracking + height"
        if use_metric and reference_px is not None:
            header += f" | calibrated using {reference_px:.0f}px reference"
        else:
            header += " | metric height unavailable: provide calibration"
        cv2.rectangle(annotated, (0, 0), (width, 28), (20, 20, 20), -1)
        cv2.putText(annotated, header, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 240), 1, cv2.LINE_AA)

        writer.write(annotated)
        frame_index += 1
        if frame_index % 300 == 0:
            print(f"Processed {frame_index}/{total_frames or '?'} frames", flush=True)

    cap.release()
    writer.release()

    fieldnames = [
        "frame", "time_s", "track_id", "confidence", "x1", "y1", "x2", "y2",
        "height_px", "height_m", "reference_px_used",
    ]
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    print(f"Wrote {frame_index} frames to {args.output}")
    print(f"Wrote {len(rows)} person observations to {args.csv}")


if __name__ == "__main__":
    main()
