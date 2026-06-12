# Ego Homes

Ego Homes is a first-person dataset of everyday household activities captured in
native home environments. Using head-mounted and body-worn cameras, we record
unscripted interactions across the entire home — kitchen, laundry, living spaces —
annotated with hand pose, segmentation, and action labels.

![sample clip](assets/preview.gif)

## HomeHands-50

The core dataset: egocentric RGB footage of everyday household activities captured
from a first-person wearable camera.

| | |
|---|---|
| Hours | 50 |
| Videos | 2,700 |
| Frames | 5.4M |
| Tasks | 25 |
| Action classes | 100 |
| Narrations | 7,600 |
| Hands tracked | 1.8M |
| Segmentation masks | 8.9M |

Example activities: washing a cup, cutting a banana, folding clothes, making tea,
mopping, sweeping, arranging shoes, pouring water, and more.

## Derived Datasets

- **HomeTrace** — 21-point hand skeleton (MediaPipe HandLandmarker), tracked per frame for both hands.
- **HomeMask** — per-frame instance segmentation masks for hands, objects, and surfaces.
- **HomeVoice** — live-narrated recordings with burned-in captions and subtitle files.
- **HomeDepth** — per-frame monocular depth maps.

## Contact

Questions about the dataset, pipeline, or collaboration: **aneessaheba04@gmail.com**
