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

YOLO pose (one crop per court half)
  -> player tracker: 1 or 2 identity slots per half (singles / doubles)
  -> per-player kinematics, player-tracks.csv, hitter attribution
  -> per-side features for the rally model

camera / stream / phone (live)
  -> newest frame only
  -> motion + tracked pose + online shuttle + streaming hit sounds
  -> lagged rally model + shuttle flight -> online rally state
  -> live stats and an annotated MJPEG view
```

The classifier produces probabilities. `segmenter.py` owns cuts. Scoring and analytics consume reviewed cuts but never edit them.

## Code boundaries

| Area | Owner |
| --- | --- |
| Project paths and artifact names | `project_layout.py` |
| Model roles and availability | `model_registry.py` |
| Feature extraction | `features.py`, `audio.py` |
| Player identity and kinematics | `players.py` |
| Live analysis | `live.py`, `studio/live_routes.py` |
| Rally prediction | `model.py` |
| Cut decisions | `segmenter.py`, `rally_evidence.py` |
| Shuttle detection and tracking | `shuttle.py`, `tracknet.py`, `hybrid.py`, `trajectory.py` |
| Score | `court_events.py`, `scoring.py`, `score_learning.py`, `score_labeling.py` |
| Studio API and job coordination | `studio/` |
| Studio UI | `studio-web/` |

Algorithms stay outside `studio/`. The frontend never reads the filesystem or runs media tools directly.

Player identities are bounded slots: at most `players_per_side` (1 in singles, 2 in doubles) per half, and nobody
crosses the net. A slot follows its player with a capped constant-velocity prediction in court metres, an exact
per-side assignment and a torso colour histogram; a lost slot keeps its colour so the player is re-identified rather
than renumbered. After extraction, an offline pass undoes partner swaps that happened while two players overlapped.
Wrist, foot and lunge speeds are differenced within one slot, never across two people. `side_features` folds the
slots back into the per-side columns the rally model was trained on, picking the partner whose wrist is nearest the
shuttle as the active player.

Court coordinates span the doubles court: x from 0 to 6.1 m (near-left doubles sideline to near-right), y from 0 at
the near baseline to 13.4 m. Calibration asks for the four singles corners, which map to x = 0.46 and 5.64 m.

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
  Latest_Auto_Cut/                    latest validated automatic timelines
    MatchN-rallies.csv
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
      vision-features.csv
      player-tracks.csv                tracked players: ID, box, court position, speed, swing
      rally-probabilities.csv
      rallies-auto.csv
      shuttle-*.csv
      shuttle-detection.json
      Shuttle_Annotations/
      Score_Labeling/
      Pose_Overlays/
    Proxy/                             generated preview media
    Edited/                            final exports
  Live/                                live-session recordings (phone uploads, stream copies)
  Calibration/
    court-config.json                  Studio calibration
    live-court-config.json             live-camera calibration
  .smart-badminton/
    models/                            current library-wide learned rules
    jobs/                              durable job state
    tls/                               self-signed certificate for --https

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
