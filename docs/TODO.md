# Smart Badminton roadmap

This file is the single source of truth for unfinished product work. The current rally-state checkpoint is
`models/rally-state-final-v4-frozen.joblib`; reviewed timelines remain independent from automatic candidates.

Status: `[x]` complete, `[~]` implemented but awaiting broader video validation, `[ ]` not implemented.

## P0 - Rally boundary reliability

- [x] Fuse model probability, motion, audio and pose instead of letting audio end a rally alone.
- [x] Keep a rally open while an accepted shuttle trajectory is still visible or descending.
- [x] Treat a short trajectory disappearance with continued player activity as occlusion, not an ending.
- [x] Start from a qualified flight after both players become ready, and lock the ending after a confirmed landing.
- [x] Allow a descending track followed by player inactivity to produce an explainable landing end reason.
- [~] Validate trajectory-protected endings on Match5, Match6 and Match7-1 and tune only the post-model rules.
- [~] Associate shuttle points with wrist/racket contact so an unrelated moving object cannot protect an ending.
- [~] Distinguish floor landing, net impact and out-of-frame flight before assigning an end reason.
- [x] Add an evidence lane in Studio showing model, trajectory, occlusion, ready stance and chosen boundary reason.
- [x] Let a reviewed project generate current-video shuttle tracks without replacing its editable timeline.

## P0 - Handoff versus formal serve

- [x] Export near/far player foot position, foot speed and stance width from pose extraction.
- [x] Detect a stable two-player ready phase before a serve candidate.
- [x] Trim an opening handoff to the later formal serve when the model contains no probability valley.
- [x] Split a merged previous rally from the next serve when a landing and ready phase are both observed.
- [x] Keep complete shuttle-return flights inside a between-points phase until a qualified formal serve.
- [x] Learn an inter-rally gap prior from reviewed timelines and let qualified serves bypass it.
- [~] Validate the phase rules against manually corrected handoff examples from several matches.
- [~] Infer which player is serving/receiving and require the receiver, specifically, to become ready.
- [~] Adapt near/far readiness thresholds to the current video's observed stance and foot-speed distributions.
- [x] Save user corrections as phase examples without retraining the frozen rally-state model.
- [x] Fit per-video thresholds and soft gap behavior in a metadata adapter without changing the global model.
- [x] Filter handoff intervals with per-video quality features while preserving recall and complete-rally coverage.
- [x] Learn formal-start timing and sustained quiet gaps locally to split merged rallies without changing v4.

## P1 - One continuous shuttle flight

- [x] Reject persistent venue lights and other stationary detections.
- [x] Use a perspective flight volume so high clears can leave the old 2D aerial polygon.
- [x] Reject simultaneous neighbouring-court candidates through single-shuttle competition.
- [x] Require player contact, net crossing, active-court continuity or manual confirmation before a track can affect editing.
- [x] Stitch accepted tracklets into one `flight_id` across short occlusions.
- [~] Use bounded ballistic prediction plus endpoint-validated optical-flow recovery for motion blur and short gaps.
- [~] Use player/racket contact to choose between ambiguous tracklets that both pass the court projection.
- [x] Build a Studio annotation tool for missed shuttle points and wrong neighbouring-court points.
- [x] Add a rights-audited, camera-disjoint multi-angle fine-tuning command and checksum-bound checkpoint license sidecar.
- [~] Publish a broadly validated shuttle checkpoint after a real multi-angle labelled dataset with confirmed redistribution rights is available.

## P2 - Score and court events

- [x] Calibrate a court homography and expose click-sensitivity uncertainty when the four ground corners are approximate.
- [x] Estimate apparent net-plane crossing while preserving the single-camera height ambiguity.
- [~] Infer the last hitter from shuttle direction change plus racket/contact evidence.
- [~] Classify the terminal event as in, out, net or unknown with a confidence score.
- [x] Maintain score/server state and provide one-click manual correction in Studio.
- [x] Calculate the full score on demand from the saved final timeline and invalidate stale results after edits.
- [x] Infer a previous rally winner from the following high-confidence formal server, while keeping ambiguous and final rallies unresolved.
- [x] Label partial score coverage honestly in Studio instead of presenting unresolved rallies as zero points.
- [x] Evaluate scoring separately from editing; uncertain events never change a cut automatically.
- [x] Disable next-server scoring for a video when manual results expose a camera-side bias.
- [x] Calibrate terminal scoring rules from structured manual outcomes without mutating reviewed timelines.

## P2 - Action and entertainment features

- [~] Replace motion-peak shot counts with racket-contact counts.
- [x] Add candidate shot types, court heatmap, rally pace and explainable highlight cards.

## P1 - Open-source readiness

- [x] Publish the frozen rally-state v4 checkpoint with a model card, checksum and redistributable license.
- [x] Gate candidate rally models with truth hashes and leave-one-match-out edit-quality regression.
- [x] Measure merged rallies, fragmented rallies, extra context and overlap in the promotion loss.

- [x] License the project source under Apache-2.0 while keeping model and optional runtime rights separate.
- [x] Exclude exploratory/private artifacts and publish a clear repository/data directory policy.
- [x] Add first-run project setup, hardware fallback and third-party checkpoint license checks.
- [x] Provide a small privacy-safe metadata sample with calibration, trajectory, annotations and reviewed timeline.
- [x] Add CI for tests, linting, CLI startup and a reproducible wheel artifact; validate a local wheel build.
- [x] Replace the monolithic browser script with a typed React/Next.js Studio and reproducible static export.
- [x] Use shadcn primitives for maintained controls, confirmations, dialogs and collapsible workflows.
- [x] Expose real local input/output directory selection and a version-checked FastAPI boundary.
- [x] Expose the one-camera workflow first and collapse calibration/training options until requested.

## Code ownership

- `features.py`: raw per-frame motion, pose, audio and shuttle measurements.
- `model.py`: frozen rally-state classifier training, benchmarking and inference.
- `rally_evidence.py`: pure post-model evidence fusion and phase signals.
- `start_quality.py`: per-video formal-start feature models.
- `local_boundaries.py`: local start timing, boundary coalescing and quiet-gap splitting.
- `segmenter.py`: conservative rally state machine and boundary decisions.
- `trajectory.py`: false-positive removal, tracklets, competition and flight stitching.
- `shuttle_annotations.py`: validation, atomic persistence and backups for user shuttle points/rejections.
- `contacts.py`: strict shuttle-to-active-wrist contact candidates and last-hitter evidence.
- `court_events.py`: conservative terminal-event candidates and score-safety gating.
- `scoring.py`: score/server state, manual corrections and scoring-only evaluation.
- `shot_analysis.py`: trajectory-supported candidate shot types; never official 3D statistics.
- `analytics.py`: post-boundary entertainment estimates, heatmap and explainable highlight cards.
- `phase_examples.py`: boundary corrections plus local phase evidence for future post-model tuning.
- `shuttle_training.py`: dataset rights audit, camera-disjoint split and licensed checkpoint packaging.
- `doctor.py`: first-run environment, encoder fallback and checkpoint provenance checks.
- `studio.py`: orchestration and file management; it should not contain detection algorithms.
- `studio-web/`: typed React/Next.js UI only; media processing and filesystem writes stay behind the Studio API.

New algorithms should be added to the narrowest module above. Reviewed timelines live under `Metadata/`, generated
candidates under `Analysis/`, locked labels under `Standard_Answers/`, and versioned training artifacts under
`Training/`.
