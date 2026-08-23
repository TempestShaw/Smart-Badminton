# Smart Badminton

Smart Badminton removes pickup, waiting, walking and conversation from fixed-camera badminton footage while conservatively preserving every rally ending.

The segmenter combines court-specific optical flow, player pose, audio transients and optional shuttle detections. Audio or a missing shuttle detection can support a decision, but neither is allowed to end a rally by itself.

> Status: early research release. Calibrate and validate on representative footage before unattended batch processing.

## Highlights

- normalized active-court, near-player, far-player, net and aerial-space masks;
- court-gated, dual-ROI pose inference with temporal continuity and keypoint smoothing;
- motion, optical flow, pose, audio and optional shuttle features;
- active-wrist/shuttle contact candidates with explicit fallback and uncertainty;
- stationary-false-positive suppression, contact-assisted competition and optical-flow-recovered shuttle tracklets;
- corrected rally CSV files as lightweight supervision;
- leave-one-video-out validation for multi-video training and a less misleading transfer estimate;
- precision-oriented end-pending state machine with adjustable pre-roll and post-roll;
- contextual suppression of weak shuttle-handoff actions between real exchanges;
- automatic removal of overlapping padded clips;
- low-confidence boundary review clips;
- MP4 rendering through system FFmpeg or `imageio-ffmpeg`.

## Install

Python 3.10 or newer is required.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Player pose and shuttle YOLO support are optional:

```bash
python -m pip install -r requirements.txt
```

See [Third-party software and model licensing](docs/third-party.md) before enabling the vision extra or redistributing model weights.

Create a project layout and inspect hardware/model provenance before the first run:

```bash
smart-badminton init-project --directory my-match
smart-badminton doctor --config my-match/Calibration/court-config.json --encoder auto --packages /optional/local/site-packages
```

`doctor` actually starts a one-frame encode before reporting an H.264 encoder as ready. Video jobs retry NVIDIA NVENC, Intel QSV, Apple VideoToolbox, `libx264`, and `libopenh264` in that order when those implementations are compiled in, so a missing hardware device does not turn a visible Studio button into a dead action. A model is release-safe only when its checksum matches a `<model>.license.json` sidecar; missing provenance never silently passes.

## Quick start

Start by copying `smart_badminton/configs/example_fixed_camera.json` and adjusting its normalized polygons for the camera angle.

```bash
smart-badminton preview-geometry --video input.mp4 --config camera.json --time 60 --output output/court-preview.jpg
smart-badminton analyze-audio --input input.mp4 --output output/audio-events.csv
smart-badminton extract-features --video input.mp4 --config camera.json --audio-events output/audio-events.csv --output output/features.csv
smart-badminton train --features output/features.csv --rallies corrected-rallies.csv --model output/rally-state.joblib --report output/training-report.json
smart-badminton predict --features output/features.csv --model output/rally-state.joblib --output output/probabilities.csv
smart-badminton segment --features output/features.csv --probabilities output/probabilities.csv --output output/rallies.csv
smart-badminton render --video input.mp4 --rallies output/rallies.csv --output output/edited.mp4
smart-badminton analyze-actions --features output/features.csv --rallies output/rallies.csv --output output/actions.csv --summary output/action-summary.json
```

For two or more reviewed videos, prefer a dataset manifest and whole-video holdout validation:

```json
{
  "sources": [
    {"id": "match-a", "features": "a/features.csv", "rallies": "a/corrected.csv"},
    {"id": "match-b", "features": "b/features.csv", "rallies": "b/corrected.csv"}
  ]
}
```

```bash
smart-badminton train-multi --dataset dataset.json --model output/rally-state.joblib --report output/training-report.json
```

Before freezing a release model, the optional benchmark compares histogram gradient boosting, extra trees, random forest and standardized logistic regression. Every fold holds out a complete video and runs the real segmenter, so selection is not based only on correlated frame accuracy:

```bash
smart-badminton benchmark-models --dataset dataset.json --report output/model-benchmark.json
smart-badminton train-multi --dataset dataset.json --family hist_gradient_boosting --model output/rally-state.joblib --report output/training-report.json
smart-badminton regression-gate --dataset dataset.json --baseline-model models/rally-state.joblib --report output/regression.json
smart-badminton train-guarded --dataset dataset.json --baseline-model models/rally-state.joblib --candidate-model output/candidate.joblib --gate-report output/regression.json
```

Every regression source includes a SHA-256 for its reviewed rally CSV. `train-guarded` writes a candidate only after every held-out match passes recall, complete-rally coverage and premature-cut checks.

The development footage used a locally frozen rally-state checkpoint trained from human-reviewed sources. That private dataset, model card and checkpoint are intentionally not published. Public users can train a camera-specific checkpoint with `train` or `train-multi`, then pass it with `--model`. Studio reports automatic analysis as unavailable when the file is absent; it never presents a missing bundled model as a working feature. Later shuttle tracking and action analysis remain independent of the rally-state artifact.

On Windows, `smart_badminton/run_pipeline.ps1` runs the same sequence. Supply an existing model for footage from a calibrated camera, or `-CorrectedTimeline` to train from reviewed ranges.

## Local Studio

Install the local UI and open a reviewed or automatic timeline:

```bash
python -m pip install -r requirements.txt
smart-badminton studio \
  --video input.mp4 \
  --proxy browser-friendly-proxy.mp4 \
  --rallies output/rallies.csv \
  --output output/edited.mp4
```

Or launch a whole recording folder and choose the video inside Studio:

```bash
smart-badminton studio \
  --library /path/to/recordings \
  --config camera.json \
  --model output/rally-state.joblib \
  --pose-model yolo11n-pose.pt
```

The folder launcher recursively finds source videos, while ignoring derivative folders such as `Proxy`, `Edited` and `Analysis`. A `Match*` folder becomes one project with its own `Analysis/`, `Proxy/`, `Metadata/` and `Edited/` outputs. Top-level recordings keep editable timelines under `Analysis/Studio/`. If `Standard_Answers/<video-number>_rallies_ground_truth.csv` exists, Studio seeds a separate editable copy without modifying the locked ground truth.

When a config and model are supplied, Studio can analyze the current video or queue every unedited match. It generates a browser-friendly 720p proxy once, extracts multimodal features from the source master, predicts rally state and writes an editable timeline. Existing reviewed timelines and completed-match markers are skipped by the batch command. The generated proxy is used only for browser playback; saved timestamps and final rendering still target the source video.

The default `precision-v3` preset keeps 0.35 seconds before a detected rally and 0.55 seconds after it. Studio exposes both values beside the Auto Cut controls. **Filter shuttle handoffs** is enabled by default: it uses surrounding activity, event strength and the stronger exchange that follows to suppress a weak action whose purpose is probably returning the shuttle to the next server. Disable the option for maximum recall or when players intentionally begin points with similarly gentle one-hit exchanges. Reviewed timelines are never silently rewritten; rerunning analysis requires confirmation and creates a backup.

The Studio provides a zoomable source timeline, low-I/O scrub preview, instant cut-preview playback that skips removed gaps without rendering, draggable start/end handles, frame nudging, split/add/delete, undo/redo, automatic CSV backup, analysis progress, a model/flight/ready/event evidence lane, selected-boundary explanations, per-rally entertainment analytics, and local rendering. **分析整段视觉（姿态＋羽球）** is a real background preprocessing job outside the player: it creates a full-length pose video and shuttle trajectory cache without changing the edit timeline. Once current, two controls on the main player toggle that evidence in place at any source timestamp: **YOLO 姿态** overlays player boxes, skeletons and wrist trails, while **羽球轨迹** draws the timestamp-synchronized primary flight tail. Either overlay can be turned off without changing the edit timeline. The player and Clip Inspector share a viewport-bounded row; long inspector content scrolls inside its panel so the timeline remains reachable while watching the video. It binds to `127.0.0.1` by default and does not upload the video.

### Everyday one-camera workflow

One fixed phone is enough. Multiple cameras are required only by developers who want to train a shuttle detector that generalizes to previously unseen camera angles; they are not simultaneous inputs to Studio.

1. Use **输入视频文件夹 → 选择** to browse a local folder (or paste a path and scan it), choose one recording, and open it. Set **成片输出文件夹** and an `.mp4` filename in the same project bar; the exact render target is shown before output.
2. For a new camera position or zoom, expand **Court and trajectory tools**, open **Court calibration**, and roughly outline the active court. Reuse that calibration while the tripod position and zoom remain unchanged.
3. Keep **Automatic cut** expanded, run the current-video analysis, then scrub anywhere on the ruler and drag either edge of each kept segment. **Cut preview** plays the unsaved timeline while skipping rejected gaps.
4. Save the timeline, then render the final video. Rendering always uses source timestamps and the source master, even when playback uses a proxy.

The main automatic-cut controls stay open. Court/trajectory tools and developer-only multi-camera/license settings are shadcn accordions collapsed by default so the normal one-camera workflow is not buried under advanced options.

The maintained UI source is a statically exported React 19 + Next.js 16 app in `studio-web/`. FastAPI remains the local media and project API and serves the exported files from `smart_badminton/studio_static/`. To work on the UI with hot reload, run the backend on port 8765, then:

```powershell
cd studio-web
$env:NEXT_PUBLIC_STUDIO_API_ORIGIN = "http://127.0.0.1:8765"
npm install
npm run dev
```

Before packaging Python, publish a production export back into the package:

```bash
cd studio-web
npm ci
npm run check
npm run build
```

Create a wheel through the checked release builder. It removes only the repository's generated `build/` directory,
then rejects stale React hashes, missing license files, private media or bundled model weights:

```bash
python scripts/build_wheel.py --output-directory dist
```

The frontend checks the backend schema at startup and refuses an incompatible API instead of failing later during editing. It also displays the resolved FFmpeg capability and keeps video-analysis, pose-preview and render actions disabled if that runtime is missing; the backend repeats the same preflight before every job. The local filesystem browser, project selection, output target, rendering, calibration, annotations and score corrections are real FastAPI operations; the React app contains no mock controls or placeholder processing. See [Studio API](docs/studio-api.md) for the boundary and status semantics.

Camera configuration does not require hand-editing JSON. Open **Court Calibration** in Studio, pause on a clear frame, and roughly outline the red active court. **Generate helper regions from active court** creates initial near/far foot zones, net band, shuttle airspace and a two-point cyan shuttle-perspective axis; every point remains draggable. Put the first axis point near the visual centre of the highest plausible clear and the second near the centre of the court floor. The tracker extrapolates this axis to a virtual vanishing point and admits pixels whose perspective ray can terminate on the selected court, so high clears may rise above the old 2D airspace without opening the neighbouring courts. For metric ground coordinates, select the optional orange four-corner region and click near-left, near-right, far-right, far-left. Studio reports how a three-pixel click perturbation propagates into court-coordinate uncertainty; four clicked points alone must not be described with a misleading zero reprojection error. You can also draw each region directly and add optional background exclusions. Studio validates normalized geometry, creates `Calibration/court-config.json` when no config was supplied, and backs up an existing config before replacing it. Automatic analysis remains disabled until all five required regions are valid; the perspective axis and homography are optional and fall back to the legacy aerial polygon/no metric event coordinates when omitted.

`active_court_polygon` is the hard ground-plane gate for player feet. The blue/green near/far polygons classify a player only by their ankle/foot point; they are not body crop boundaries. Pose runs on separate perspective-aware regions that extend upward, then maps the full skeleton back to the source frame. A near player's head, shoulder and wrist may therefore cross the far-zone image band without being clipped, while a spectator cannot become a court player unless their foot point passes the active-court gate.

Shuttle tracking supports `yolo`, `tracknet`, and `hybrid`. Hybrid fuses YOLO boxes with TrackNetV3 temporal heatmaps, then applies the calibrated flight volume, stationary-object suppression, contact evidence, track competition, and manual corrections.

```bash
smart-badminton detect-shuttle --video input.mp4 --config camera.json --model shuttle.pt --output output/shuttle-raw.csv
smart-badminton detect-tracknet --video input.mp4 --config camera.json --tracknet-model TrackNet_best.pt --inpaint-model InpaintNet_best.pt --output output/tracknet-raw.csv
smart-badminton fuse-shuttle --yolo output/shuttle-raw.csv --tracknet output/tracknet-raw.csv --config camera.json --output output/hybrid-raw.csv
smart-badminton track-shuttle --video input.mp4 --config camera.json --detections output/hybrid-raw.csv --contact-features output/features.csv --annotations output/user-shuttle-points.csv --output output/shuttle-track.csv --preview output/shuttle-preview.mp4
smart-badminton segment --features output/features.csv --probabilities output/probabilities.csv --shuttle-trajectory output/shuttle-track.csv --output output/rallies.csv
smart-badminton fit-adapter --features output/features.csv --probabilities output/probabilities.csv --truth reviewed.csv --shuttle-trajectory output/shuttle-track.csv --base-adapter segmentation-adapter.json --output segmentation-adapter-next.json
```

TrackNet uses an airspace crop and background mask to retain high clears while suppressing neighbouring courts. Inpainted points may bridge a short occlusion but cannot independently create landing or clip-boundary evidence.

Detection and trajectory CSVs include the pixel coordinate dimensions used by YOLO. This lets an advanced workflow detect on a proxy while extracting pose from the source master without shifting the shuttle coordinates. Strict racket-contact candidates require the shuttle to approach an active wrist plus independent swing, direction-change or audio support. Terminal-event analysis deliberately prefers `unknown` to a fabricated landing or score:

```bash
smart-badminton analyze-contacts --features output/features.csv --rallies output/rallies.csv --output output/contacts.csv --summary output/contact-summary.json
smart-badminton analyze-events --video input.mp4 --config camera.json --rallies output/rallies.csv --trajectory output/shuttle-track.csv --contacts output/contacts.csv --output output/events.csv --summary output/event-summary.json
smart-badminton analyze-actions --features output/features.csv --rallies output/rallies.csv --contacts output/contacts.csv --events output/events.csv --trajectory output/shuttle-track.csv --config camera.json --output output/actions.csv --summary output/action-summary.json
smart-badminton train-score-evidence --library matches
smart-badminton analyze-score --rallies output/rallies.csv --events output/events.csv --corrections output/score-corrections.csv --model matches/.smart-badminton/models/score-evidence-v3.json --output output/score.csv --summary output/score-summary.json
```

Launch Studio with the detector paths, then choose the mode under **场地与轨迹**:

```bash
smart-badminton studio --library matches --config camera.json --shuttle-model shuttle.pt --tracknet-model TrackNet_best.pt --inpaint-model InpaintNet_best.pt --shuttle-mode hybrid
```

Studio point corrections update the current trajectory. They can also be exported as TrackNet point labels and used to fine-tune a local TrackNet checkpoint:

```bash
smart-badminton export-tracknet-labels --video input.mp4 --trajectory output/shuttle-track.csv --annotations output/user-shuttle-points.csv --output output/tracknet-labels.csv
smart-badminton train-tracknet --video input.mp4 --config camera.json --labels output/tracknet-labels.csv --base-model TrackNet_best.pt --output output/TrackNet-finetuned.pt
```

To build a future redistributable shuttle detector, start from [the manifest template](examples/shuttle-training-manifest.example.json). The audit requires at least two camera identities, paired class-0 YOLO labels and explicit source/license/redistribution fields. Validation holds out a complete camera:

```bash
smart-badminton audit-shuttle-dataset --manifest shuttle-training-manifest.json
smart-badminton train-shuttle --manifest shuttle-training-manifest.json --base-model licensed-base.pt --work-directory output/shuttle-training --output output/shuttle-multi-angle.pt --checkpoint-license AGPL-3.0
```

Training also requires a checksum-matched license sidecar for the base checkpoint and writes one for the derived checkpoint. This repository intentionally does not manufacture or redistribute a weight file without a real licensed multi-angle dataset.

## Rally CSV

Corrected labels require only three columns:

```csv
rally,start_seconds,end_seconds
1,2.600,8.750
2,11.770,25.800
```

Generated timelines also contain boundary confidence, review status and the boundary reason.

## Design principle

The detector still waits for visual inactivity rather than cutting at the last audible impact, preserving quiet net shots, misses and slow falls. Its current default favors a compact first draft: short buffers and contextual handoff suppression reduce cleanup, while Studio makes it cheap to drag a boundary outward or disable handoff filtering when maximum recall is preferred.

See [Architecture](docs/architecture.md), [Repository audit](docs/repository-audit.md), and the package [command reference](smart_badminton/README.md).

## Data and models

No match video, extracted frame, venue image, trained model or third-party checkpoint is included. Keep those artifacts outside Git; the provided `.gitignore` excludes the local working directories used during development.

[`examples/privacy_safe_sample`](examples/privacy_safe_sample) is a synthetic, metadata-only fixture containing calibration, trajectory statuses, user shuttle corrections and a reviewed timeline. It contains no recorded person, venue or audio and is exercised by the test suite.

## License

The project source code is licensed under [Apache-2.0](LICENSE). No match media, training dataset, model weight or Ultralytics package is covered merely because the surrounding source repository uses Apache-2.0. Review [third-party software and model licensing](docs/third-party.md) before distributing optional checkpoints or a build that bundles Ultralytics.
