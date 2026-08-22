# Architecture

```text
video
  +-- audio transient candidates
  +-- calibrated court motion and optical flow
  +-- optional player pose
  +-- optional shuttle candidates
             |
             v
       sampled feature CSV
             |
             v
  gradient-boosted rally probability
             |
             v
 conservative state machine
  +-- pre/post roll
  +-- quiet-exchange protection
  +-- overlap deduplication
  +-- boundary confidence
             |
             v
      rally CSV -> review/render
             |
             v
 independent score + entertainment analytics
  +-- shuttle-to-active-wrist contact candidates
  +-- estimated hit count and pace
  +-- smash and lunge candidates
  +-- last hitter and conservative terminal event
  +-- near/far movement scores
  +-- highlight score and fun tags
  +-- candidate shot types and court heatmap
```

The frozen classifier produces probability only. `rally_evidence.py` aligns pose readiness and accepted shuttle
flights to that probability timeline; `segmenter.py` alone owns start/end decisions. This separation allows boundary
safety to improve without retraining or silently replacing the versioned classifier. The maintained work queue is in
[`TODO.md`](TODO.md).

## Studio

The optional local Studio is split at a narrow, versioned HTTP boundary. `studio-web/` contains the typed React/Next.js interface, shadcn primitives and a static export; `studio.py` owns the FastAPI project/media API, local background jobs and file management, then serves that export from `studio_static/` on the loopback interface. In development the Next server may run on port 3000 and call the loopback FastAPI service through its configured origin and CORS policy. React components are divided by product area (project browser, workflow, video tools, inspector, timeline and clip bin), while `use-studio-controller.ts` owns shared project state and API mutations. No media processing algorithm or direct filesystem access lives in the frontend.

A recursive library scanner treats each `Match*` directory as a project and keeps analysis, proxy, metadata and final-output files inside it. Automatic-analysis jobs run sequentially in a local background queue. Its browser timeline edits seconds, not copied media. Cut-preview mode plays those in-memory ranges in order and seeks across rejected gaps, so unsaved trims can be reviewed before rendering. The evidence lane displays compact model, flight, occlusion, ready-stance, serve/contact and terminal-event spans below the editable cuts; the inspector explains the selected boundary. Saving validates ordering and overlap, writes the CSV atomically and keeps a `.bak` copy. It also records the automatic boundary, human correction and local probability/ready/serve/handoff/flight evidence under `Analysis/Boundary_Corrections/`; these examples do not retrain the frozen classifier. Score corrections live in a separate CSV and cannot mutate the editing timeline. Rendering uses the source master even when the browser plays a lower-resolution H.264 proxy.

Shuttle annotation is a separate, frame-local correction workflow. Users can add a missed point or reject a nearby false candidate directly over the paused video. Normalized annotations are atomically stored under `Analysis/Shuttle_Annotations/user-shuttle-points.csv`, with a `.bak` on replacement. They influence the next generated trajectory but never mutate a reviewed timeline by themselves. A separate current-video trajectory job can generate or regenerate only `shuttle-raw.csv` and `shuttle-track.csv`; it shares analysis progress reporting but has no code path to the timeline writer. The UI distinguishes detector configuration from a generated track and renders a bounded recent tail from one active `flight_id`.

## Court geometry

Coordinates in a camera config are normalized to `[0, 1]`, so one config works at multiple resolutions with the same camera pose. The ground polygon limits general motion. Near/far player zones assign pose detections. A legacy aerial polygon remains available as a fallback.

Shuttle detection has YOLO, TrackNetV3, and Hybrid backends. TrackNet processes overlapping frame sequences through an airspace crop whose excluded pixels are replaced by the median background. Hybrid merges agreeing observations, retains source metadata, and gives InpaintNet repairs a low evidence weight. The existing stationary filter, track competition, contact support, optical-flow recovery, and manual corrections operate on the unified candidate schema.

Accepted tracklets inside one physically plausible temporal component receive a shared `flight_id`. Each point also carries an ownership confidence and one of four credentials: player contact, net crossing, active-court continuity or manual confirmation. Unknown tracks remain available for detector review but cannot protect an edit boundary. The evidence layer keeps moving components with at least three observed points and independent action or model support, then emits flight start, visible, descending, occluded, landing and flight-end signals on the 10 Hz feature timeline. Pose extraction also emits player foot position, foot speed and stance width. Near/far readiness thresholds adapt to the observed camera-relative distributions; a trajectory-backed serve requires a recent two-player ready phase. These remain post-model evidence and do not become new inputs to the frozen classifier.

Detection and trajectory CSVs carry their coordinate width and height. This keeps normalized shuttle positions correct when detection is deliberately run on a proxy but pose and motion features come from the source master. Older CSVs without the metadata remain readable and assume the feature video's dimensions.

An optional ordered four-point homography maps image coordinates to the 6.1 m by 13.4 m badminton court. Four points have zero fitting residual by construction, so the UI reports sensitivity to simulated click error rather than misleading reprojection error. Apparent crossings of the 6.7 m ground-plane line are retained as low-confidence candidates with `height_ambiguous=yes`. The projected flight volume is a perspective gate rather than metric 3D reconstruction; exact shuttle height and depth are never claimed from one camera.

## Boundary safety

The classifier estimates `RALLY_ACTIVE`; the state machine owns the cut. A trajectory-backed serve can recover a missed opening after both players become ready. During play, a qualified flight keeps the interval open through descent and short occlusion. A confirmed landing locks the ending so the following handoff cannot extend it. Every accepted interval receives adjustable pre-roll and post-roll (the precision preset defaults to 0.35 s and 0.55 s). The handoff filter suppresses weak isolated actions, while trajectory-backed serve boundaries are preserved. Neighboring buffers share one quiet boundary instead of replaying the same source range.

## Entertainment analytics and score evidence

Action analytics reuse the sampled pose, motion, audio and shuttle features after rally boundaries are known. `contacts.py` emits a strict contact only when the shuttle is near an active wrist and swing, direction change or audio provides independent support. Strict contacts supply last-hitter evidence, but their recall can be low during blur and occlusion; hit count therefore fuses them with isolated audio and pose peaks instead of letting one surviving contact erase several supported hits. `court_events.py` then emits conservative `in`, `out`, `net`, `flight_lost` or `unknown` candidates. `shot_analysis.py` labels only trajectory-supported candidates such as net, smash, clear/lift or drive. The Studio heatmap plots mapped terminal candidates, and every highlight card lists the evidence that raised its rank. Monocular ambiguity defaults to `unknown`.

`scoring.py` is independent from the editing state machine. It accepts an explicitly score-safe terminal event, or infers rally N's winner from rally N+1's high-confidence visually detected formal server because the rally winner serves next. Three or more manual results calibrate that inference for the current camera; a biased server detector is disabled for that video. Manual winner/server corrections are stored separately and no score operation can change a cut. Hit counts, shot labels, smash candidates and lunge counts remain entertainment estimates.

## Shuttle model training

`shuttle_training.py` requires a JSON manifest with at least two distinct camera identities. Each source declares its URL, license and confirmed redistribution permission, and every image has a normalized class-0 YOLO label. Validation holds out an entire camera instead of random frames. Training refuses an unverified base checkpoint, refuses to relabel an AGPL base as non-AGPL, and binds the derived checkpoint manifest to its SHA-256. The command and safeguards are complete; a public checkpoint still requires real labelled multi-angle footage whose data rights have been confirmed.

## Training and evaluation

Reviewed start/end ranges label the feature timeline without frame-by-frame shuttle annotation. Training stores the reviewed inter-rally gap distribution with the model; segmentation uses it as a soft restart prior while qualified serves remain eligible immediately. `regression-gate` verifies truth hashes and holds out every match in turn. A candidate cannot advance if any match loses recall, complete-rally coverage or ending safety, and its mean edit-quality loss must improve. `fit-adapter` searches a small per-video parameter layer and saves it beside the project metadata; it never changes the classifier or reviewed timeline.
