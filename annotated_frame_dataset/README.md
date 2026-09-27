# Annotated Frame Dataset

This package describes every frame of the annotated sample video: one YOLO label file per frame plus a manifest and summary. The frames themselves show real people and are not published in this repository.

## Contents

- `images/`: not published. Locally, one JPEG per annotated frame, `frame_000000.jpg` through `frame_005948.jpg`.
- `labels/`: one YOLO-format label file per image. Each row is `class x_center y_center width height`; class `0` is `person`.
- `../people_height_measurements.csv`: original frame-level measurements including track ID, confidence, box coordinates, and pixel height.
- `frame_manifest.csv`: image-to-label-to-frame/time mapping.
- `data.yaml`: YOLO dataset configuration.
- `dataset_summary.json`: machine-readable package summary.

> The frames are annotated/visualization images, so the labels and the drawn boxes are intentionally visible. For training a new detector, use clean original frames rather than these burned-in images, because the overlay can leak the target labels into the input.

Absolute height in meters is not included because the source video has no camera calibration or known-height scale reference. The CSV stores reliable pixel height and leaves metric height blank unless calibration is provided.
