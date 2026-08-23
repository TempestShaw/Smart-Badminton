# Architecture

## Pipeline

```text
source video
  -> audio / motion / pose features
  -> rally probability model
  -> trajectory-aware segmenter
  -> editable timeline
  -> render, score and analytics

shuttle YOLO + TrackNetV3 + InpaintNet
  -> hybrid detections
  -> owned shuttle flights
  -> boundary evidence, score evidence and trail overlay
```

The classifier produces probabilities. `segmenter.py` owns cuts. Scoring and analytics consume reviewed cuts but never edit them.

## Code boundaries

| Area | Owner |
| --- | --- |
| Project paths and artifact names | `project_layout.py` |
| Model roles and availability | `model_registry.py` |
| Feature extraction | `features.py`, `audio.py` |
| Rally prediction | `model.py` |
| Cut decisions | `segmenter.py`, `rally_evidence.py` |
| Shuttle detection and tracking | `shuttle.py`, `tracknet.py`, `hybrid.py`, `trajectory.py` |
| Score | `court_events.py`, `scoring.py`, `score_learning.py`, `score_labeling.py` |
| Studio API and job coordination | `studio.py` |
| Studio UI | `studio-web/` |

Algorithms stay outside `studio.py`. The frontend never reads the filesystem or runs media tools directly.

## Model registry

The current runtime has eight configured model entries:

| Role | Runtime | Scope |
| --- | --- | --- |
| Rally probability | scikit-learn v4 checkpoint | local |
| Player pose | YOLO11n Pose | local |
| Shuttle candidate detection | YOLO11s Ball | local |
| Shuttle sequence tracking | TrackNetV3 | local |
| Missing-point repair | InpaintNet | local helper |
| Score evidence | versioned JSON rules | video library |
| Score image labeling | GPT-5.6 Luna | remote |
| Score cross-check | Qwen3-VL 32B | remote |

Hybrid tracking, the cut state machine and per-video adapters are algorithms or parameters, not additional models.

## Files

```text
library/
  MatchN/
    Original/                         optional source-video folder
    *.mp4                             source video may also be here
    Metadata/
      rallies-studio-review.csv       editable timeline
      rallies-ground-truth.csv        optional locked truth
      score-corrections.csv            human score labels
      segmentation-adapter.json        per-video cut parameters
      Backups/
    Analysis/                          generated, rebuildable
      smart-features.csv
      rally-probabilities.csv
      rallies-auto.csv
      shuttle-*.csv
      shuttle-detection.json
      Shuttle_Annotations/
      Score_Labeling/
      Pose_Overlays/
    Proxy/                             generated preview media
    Edited/                            final exports
  .smart-badminton/
    models/                            current library-wide learned rules
    jobs/                              durable job state
    archive/                           superseded runtime artifacts

repo/
  models/                              user-supplied local checkpoints
  ThirdParty/                          external runtimes and their checkpoints
  local_configs/                       camera calibrations
```

`Metadata/` and source video are user data. `Analysis/` and `Proxy/` are caches. `.smart-badminton/` is library runtime state. Final renders always read the source video.

## Update rules

- Save timeline edits atomically and retain a backup.
- Never overwrite human truth during prediction or training.
- Keep machine score labels under `Analysis/Score_Labeling/`.
- Keep global learned rules under `.smart-badminton/models/`.
- Archive superseded learned artifacts under `.smart-badminton/archive/`.
- A shuttle track may protect a cut only after it has current-court ownership evidence.
