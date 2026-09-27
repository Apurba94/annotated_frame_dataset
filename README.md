# People Height Annotation

Detect, track and measure people in a video with a YOLO pose model. Each person
gets a bounding box, a temporary track ID, a pose skeleton and a height reading,
drawn onto the video and exported frame by frame to CSV.

```
ID 12 | Hpx 231 | 0.87        # track ID | height in pixels | detector confidence
ID 12 | H 1.74 m | 0.87       # the same, once a calibration reference is given
```

`ID n` is an anonymous tracker ID for one person across nearby frames. It is not
a name, face recognition or biometric identification. People the tracker has not
confirmed yet (usually for a frame or two after they appear) show as `ID ?`.

## Use it

### Command line

```bash
pip install -r requirements.txt
python People_Height_Annotation.py --source input.mp4 --output annotated.mp4 --csv measurements.csv
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--model` | `yolo11n-pose.pt` | Any Ultralytics pose model (downloaded on first use) |
| `--conf` | `0.30` | Detection confidence threshold |
| `--imgsz` | `640` | Inference image size |
| `--max-frames` | all | Stop after this many frames |
| `--reference-height-m` | none | Real height of a reference person or object, in metres |
| `--reference-pixel-height-px` | none | That reference's height in pixels, measured in the same view |
| `--reference-track-id` | none | Or: the track ID of a known-height person (their median pixel height is used) |
| `--reference-window` | `300` | Observations kept for the track-ID reference |

The script writes a silent video; OpenCV does not carry audio. The notebook below
also copies the original audio across.

### Google Colab

Open `People_Height_Annotation_Colab.ipynb` in [Google Colab](https://colab.research.google.com/),
select a GPU runtime if available, and run the cells: install, upload your MP4,
annotate, then re-attach the audio and download the MP4 and CSV.
`People_Height_Annotation_Colab_source.py` is the same code as a plain script.

## Heights: pixels unless calibrated

A video has no built-in pixel-to-metre scale, so heights are reported in pixels
by default. Pixel heights compare people within one frame; they are not physical
heights. For approximate metres, supply a reference:

```text
estimated_height_m = person_height_px / reference_height_px * reference_height_m
```

| Calibration | Settings |
| --- | --- |
| Fixed reference | `--reference-height-m 1.75 --reference-pixel-height-px 182` |
| Reference person | `--reference-height-m 1.75 --reference-track-id 3` |
| None | leave all three unset: pixels only |

The estimate only holds when the reference and the person measured stand on
about the same ground plane at a similar distance from the camera. It breaks
down for people crouching, jumping, partly hidden, cut off by the frame, or too
far away to locate head and feet.

## Data in this repository

| File | Contents |
| --- | --- |
| `people_height_measurements.csv` | 15,817 person observations from a 198-second, 5,949-frame sample video: frame, time, track ID, confidence, box, pixel height |
| `annotated_frame_dataset/labels/` | One YOLO label file per frame (class 0 = person, normalised boxes) |
| `annotated_frame_dataset/frame_manifest.csv`, `dataset_summary.json`, `data.yaml` | Frame index and dataset description |

The sample video, its extracted frames and contact sheets show real people and
are **not published**. The labels and measurements contain no images, only box
coordinates, pixel heights and anonymous track IDs. `data.yaml` expects the
frames in `annotated_frame_dataset/images/`; regenerate them from your own
footage to use it.

## How it works

Tracking uses Ultralytics YOLO with `persist=True` across frames and the
ByteTrack tracker ([tracking docs](https://docs.ultralytics.com/modes/track/));
the pose model supplies the keypoints drawn as the skeleton
([pose docs](https://docs.ultralytics.com/tasks/pose/)). Ultralytics is licensed
under AGPL-3.0; check its terms before building a closed-source product on it.

## Credits

Created by **Janin A Apurba**. © 2026 Janin A Apurba. All rights reserved; no
open-source licence has been chosen yet.
