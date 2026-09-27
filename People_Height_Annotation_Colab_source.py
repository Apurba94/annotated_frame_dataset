
# ===== Colab code cell 1 =====
# Install dependencies. Select Runtime > Change runtime type > T4 GPU for faster inference.
# !pip -q install ultralytics lap opencv-python-headless pandas tqdm

import os, csv, subprocess
from collections import defaultdict
from statistics import median
from typing import Optional

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm
from ultralytics import YOLO
from google.colab import files

# ===== Colab code cell 2 =====
# Upload the source MP4.
uploaded = files.upload()
INPUT_VIDEO = next(iter(uploaded))

# Output filenames.
SILENT_OUTPUT_VIDEO = 'annotated_people_height_silent.mp4'
OUTPUT_VIDEO = 'annotated_people_height.mp4'
OUTPUT_CSV = 'people_height_measurements.csv'

# Detector settings.
MODEL_NAME = 'yolo11n-pose.pt'
CONFIDENCE = 0.30
IMAGE_SIZE = 640

# Calibration settings. Leave all calibration values as None to report pixels only.
REFERENCE_HEIGHT_M = None          # Example: 1.75
REFERENCE_PIXEL_HEIGHT_PX = None  # Example: 182.0, measured in the same view
REFERENCE_TRACK_ID = None         # Example: 3, if a known-height person is tracked
REFERENCE_WINDOW = 300

DEVICE = 0 if torch.cuda.is_available() else 'cpu'
print('Device:', 'GPU' if torch.cuda.is_available() else 'CPU')

# ===== Colab code cell 3 =====
def safe_int(value):
    return int(round(float(value)))

def make_color(track_id):
    rng = np.random.default_rng(abs(int(track_id)) + 2026)
    color = rng.integers(55, 235, size=3).tolist()
    return int(color[0]), int(color[1]), int(color[2])

def draw_label(frame, text, x, y, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max(0, x)
    y = max(th + baseline + 2, y)
    cv2.rectangle(frame, (x, y - th - baseline - 4), (x + tw + 6, y + 2), color, -1)
    cv2.putText(frame, text, (x + 3, y - 2), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

def annotate_video(source, silent_output, csv_output, model_name=MODEL_NAME, conf=CONFIDENCE, imgsz=IMAGE_SIZE,
                   reference_height_m=REFERENCE_HEIGHT_M, reference_pixel_height_px=REFERENCE_PIXEL_HEIGHT_PX,
                   reference_track_id=REFERENCE_TRACK_ID, reference_window=REFERENCE_WINDOW):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video: {source}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(silent_output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f'Could not create output video: {silent_output}')
    model = YOLO(model_name)
    reference_pixels = []
    frame_index = 0
    rows = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        results = model.track(frame, persist=True, tracker='bytetrack.yaml', classes=[0], conf=conf, imgsz=imgsz, device=DEVICE, verbose=False)
        result = results[0]
        try:
            annotated = result.plot(boxes=False, labels=False, conf=False, kpt_line=True, kpt_radius=2, line_width=1)
        except Exception:
            annotated = frame.copy()
        boxes = result.boxes
        xyxy = boxes.xyxy.detach().cpu().numpy() if boxes is not None and boxes.xyxy is not None else np.empty((0, 4))
        confs = boxes.conf.detach().cpu().numpy() if boxes is not None and boxes.conf is not None else np.empty((0,))
        # Unconfirmed detections have no track ID yet: mark them -1 so they never
        # collide with a real ID or feed the reference calibration.
        ids = boxes.id.detach().cpu().numpy().astype(int) if boxes is not None and boxes.id is not None else np.full(len(xyxy), -1, dtype=int)
        for bbox, tid in zip(xyxy, ids):
            if reference_track_id is not None and int(tid) == int(reference_track_id):
                ref_px = float(max(0.0, bbox[3] - bbox[1]))
                if ref_px > 1:
                    reference_pixels.append(ref_px)
                    if len(reference_pixels) > max(1, reference_window):
                        reference_pixels.pop(0)
        reference_px = reference_pixel_height_px
        if reference_px is None and reference_pixels:
            reference_px = float(median(reference_pixels))
        for i, bbox in enumerate(xyxy):
            x1, y1, x2, y2 = [float(v) for v in bbox]
            bbox_h = max(0.0, y2 - y1)
            tid = int(ids[i]) if i < len(ids) else -1
            id_text = f"ID {tid}" if tid != -1 else "ID ?"
            score = float(confs[i]) if i < len(confs) else float('nan')
            height_m = None
            if reference_height_m is not None and reference_px is not None and reference_px > 1:
                height_m = bbox_h / reference_px * float(reference_height_m)
            color = make_color(tid)
            cv2.rectangle(annotated, (safe_int(x1), safe_int(y1)), (safe_int(x2), safe_int(y2)), color, 2)
            cv2.circle(annotated, (safe_int((x1 + x2) / 2), safe_int(y2)), 3, color, -1)
            label = (f'{id_text} | H {height_m:.2f} m | {score:.2f}' if height_m is not None else f'{id_text} | Hpx {bbox_h:.0f} | {score:.2f}')
            draw_label(annotated, label, safe_int(x1), safe_int(y1) - 2, color)
            rows.append({'frame': frame_index, 'time_s': f'{frame_index / fps:.3f}', 'track_id': tid if tid != -1 else '', 'confidence': f'{score:.4f}',
                          'x1': f'{x1:.2f}', 'y1': f'{y1:.2f}', 'x2': f'{x2:.2f}', 'y2': f'{y2:.2f}',
                          'height_px': f'{bbox_h:.2f}', 'height_m': '' if height_m is None else f'{height_m:.4f}',
                          'reference_px_used': '' if reference_px is None else f'{reference_px:.2f}'})
        header = 'Person detection + tracking + height'
        header += (f' | calibrated using {reference_px:.0f}px reference' if reference_height_m is not None and reference_px is not None else ' | metric height unavailable: provide calibration')
        cv2.rectangle(annotated, (0, 0), (width, 28), (20, 20, 20), -1)
        cv2.putText(annotated, header, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 240), 1, cv2.LINE_AA)
        writer.write(annotated)
        frame_index += 1
        if frame_index % 300 == 0:
            print(f"Processed {frame_index}/{total_frames or '?'} frames")
    cap.release()
    writer.release()
    fields = ['frame', 'time_s', 'track_id', 'confidence', 'x1', 'y1', 'x2', 'y2', 'height_px', 'height_m', 'reference_px_used']
    with open(csv_output, 'w', newline='', encoding='utf-8') as f:
        out = csv.DictWriter(f, fieldnames=fields)
        out.writeheader()
        out.writerows(rows)
    return frame_index, len(rows)

frames_done, observations = annotate_video(INPUT_VIDEO, SILENT_OUTPUT_VIDEO, OUTPUT_CSV)
print(f'Wrote {frames_done} frames and {observations} observations.')

# ===== Colab code cell 4 =====
# Reattach the original audio stream. OpenCV writes the annotated video stream but does not preserve audio.
subprocess.run([
    'ffmpeg', '-y', '-i', SILENT_OUTPUT_VIDEO, '-i', INPUT_VIDEO,
    '-map', '0:v:0', '-map', '1:a:0?',
    # OpenCV writes MPEG-4 Part 2 (mp4v), which browsers and many phones cannot play; H.264 plays everywhere.
    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', '-c:a', 'aac', '-shortest', OUTPUT_VIDEO
], check=True)
print('Created:', OUTPUT_VIDEO)
print('Created:', OUTPUT_CSV)
files.download(OUTPUT_VIDEO)
files.download(OUTPUT_CSV)
