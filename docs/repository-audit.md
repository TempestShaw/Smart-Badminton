# Repository audit

## Core, exercised in the validated pipeline

- `smart_badminton/geometry.py`: court masks and preview rendering;
- `smart_badminton/audio.py`: transient audio candidates;
- `smart_badminton/features.py`: motion, flow, pose, audio and optional shuttle features;
- `smart_badminton/model.py`: feature engineering, training and prediction;
- `smart_badminton/segmenter.py`: conservative boundaries and overlap removal;
- `smart_badminton/rally_evidence.py`: post-model readiness, serve, shuttle-flight and boundary evidence;
- `smart_badminton/trajectory.py`: primary-flight selection, contact support and endpoint-validated gap recovery;
- `smart_badminton/shuttle_annotations.py`: normalized user corrections with atomic writes and backups;
- `smart_badminton/contacts.py` and `court_events.py`: conservative contact and terminal-event candidates;
- `smart_badminton/analytics.py`: post-boundary entertainment estimates and explainable fallbacks;
- `smart_badminton/evaluate.py`: per-rally coverage regression;
- `smart_badminton/render.py`: final MP4 rendering;
- `smart_badminton/studio.py`: local project/media API and job orchestration;
- `studio-web/`: maintainable React/Next.js timeline editor source;
- `smart_badminton/studio_static/`: generated, packaged static Studio build;
- `smart_badminton/cli.py`: public command-line interface.

## Supported but optional

- `smart_badminton/shuttle.py`: supplied YOLO-compatible shuttle checkpoint;
- `smart_badminton/review.py`: low-confidence boundary clips;
- `smart_badminton/run_pipeline.ps1`: Windows orchestration helper.

## Local-only and intentionally excluded

- dated match directories: private masters, proxies, edits, contact sheets and extracted features;
- unverified files under `models/`;
- `.tools/`: the development Python/runtime environment;
- `ThirdParty/`: exploratory upstream checkouts;
- `tools/`: one-off scripts used during investigation and superseded by package commands.

## Third-party experiments

- Good-Badminton informed the optional shuttle detector interface. Its full source is not vendored. The tested checkpoint produced useful candidates but also persistent venue false positives, so shuttle absence is never an end condition.
- TrackNetV3 is an optional Hybrid detector input and remains a separately downloaded checkpoint.

## Publication gates

- project source is Apache-2.0; optional Ultralytics packages and all model weights keep their own terms and are not
  relicensed by this repository;
- CI must rebuild the React export, install the Python package, run tests and lint, and produce a wheel from a clean
  checkout; `scripts/build_wheel.py` also makes repeated local builds deterministic by rejecting stale package data;
- `scripts/repository_guard.py` rejects private runtime data, credentials and unlicensed model files;
- add only media for which publication permission is explicit.
