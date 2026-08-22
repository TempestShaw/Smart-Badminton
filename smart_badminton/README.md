# Command reference

Run `smart-badminton --help` or `python -m smart_badminton --help` after installation.

| Command | Purpose |
| --- | --- |
| `init-project` | Create the first-run calibration, analysis, metadata, output and model-license layout. |
| `doctor` | Check Python, optional packages, FFmpeg encoder fallback and checkpoint license sidecars. |
| `preview-geometry` | Draw configured court and exclusion polygons on a video frame. |
| `analyze-audio` | Extract transient impact candidates from a WAV file or video. |
| `extract-features` | Build the sampled multimodal feature timeline. |
| `detect-shuttle` | Optionally run a supplied Ultralytics-compatible shuttle checkpoint. |
| `detect-tracknet` | Run a TrackNetV3 checkpoint with airspace masking and optional InpaintNet repair. |
| `fuse-shuttle` | Fuse YOLO and TrackNet candidates into the shared trajectory format. |
| `export-tracknet-labels` | Export reviewed trajectory points as frame-level TrackNet labels. |
| `train-tracknet` | Fine-tune a local TrackNet checkpoint from exported labels. |
| `audit-shuttle-dataset` | Validate YOLO labels, multi-camera separation and redistribution rights before training. |
| `train-shuttle` | Fine-tune with a camera-disjoint holdout and write a checksum-bound checkpoint license manifest. |
| `track-shuttle` | Apply the projected court volume, contact-assisted competition, short optical-flow recovery and optional user annotations, then render the primary shuttle path. |
| `train` | Train a rally-state classifier from corrected rally ranges. |
| `train-multi` | Train across multiple reviewed videos with leave-one-video-out validation. |
| `benchmark-models` | Compare supported model families with whole-video holdouts and real segmented-rally metrics. |
| `predict` | Produce the rally probability timeline. |
| `segment` | Produce conservative, non-overlapping rally clips. |
| `evaluate` | Compare a proposed timeline with reviewed ranges. |
| `review` | Render short clips around low-confidence boundaries. |
| `analyze-actions` | Estimate per-rally pace, shots, movement, action candidates and entertainment tags. |
| `analyze-contacts` | Fuse wrist proximity, swing, shuttle direction change and audio into contact candidates. |
| `analyze-events` | Produce conservative terminal-event and last-hitter candidates without changing score. |
| `analyze-score` | Maintain independent rally-score/server state from safe events and manual corrections. |
| `evaluate-score` | Report scoring coverage and covered accuracy without mixing in editing metrics. |
| `render` | Concatenate rally ranges into an MP4. |
| `studio` | Open one video or a recursive Match library, run automatic analysis and edit the timeline. |

The repository-level [README](../README.md) contains installation and end-to-end examples.
