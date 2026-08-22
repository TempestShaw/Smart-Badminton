# Studio API

Smart Badminton Studio has two independently maintainable layers:

- `studio-web/`: React 19, Next.js 16 and shadcn UI. It edits typed project state and never reads local files directly.
- `smart_badminton/studio.py`: FastAPI, media streaming, local filesystem access, background analysis and rendering.

Production builds export the frontend into the Python package. During development, set
`NEXT_PUBLIC_STUDIO_API_ORIGIN=http://127.0.0.1:8765` and run Next on port 3000. The backend allows only those
loopback development origins. `GET /api/health` exposes `api_schema_version`; the frontend refuses to load if its
expected version differs. The same response and `GET /api/project` expose `runtime.ffmpeg`: video-analysis, pose and
render controls stay disabled when the executable cannot be resolved, and their POST endpoints independently return
HTTP 503 rather than queuing a job that can only fail later.

## Files and projects

- `GET /api/filesystem/directories?path=...` lists real child directories, roots, home and parent. It returns paths,
  not file contents, and is intended only for the loopback app.
- `POST /api/library` selects an existing folder containing supported videos and returns its recursive video index.
- `POST /api/project/open` switches to one indexed source video.
- `PUT /api/output` validates an existing writable directory, a basename-only `.mp4` filename and the active project
  id before changing the render target.
- `GET /media/video` streams the browser preview; rendering still uses the source master.

## Editing and long-running jobs

- `PUT /api/timeline` validates sorted non-overlapping seconds, writes atomically and preserves a `.bak`.
- `POST /api/analyze` and `POST /api/analyze/batch` start real local jobs. `GET /api/analyze` reports `idle`,
  `running`, `complete` or `error` plus progress.
- `POST /api/analyze/shuttle` runs detection and trajectory linking only for the active project. It validates the
  submitted `project_id`, reports `mode=shuttle` through the same status endpoint, and never writes the editable cut
  timeline. `GET /api/project` exposes configured/generated state plus real point and flight counts; configured does
  not imply generated.
- `PUT /api/settings/shuttle-mode` selects `yolo`, `tracknet`, or `hybrid` from the model backends configured at
  Studio startup. A mode change marks the existing detection cache stale.
- `POST /api/analyze/visual` precomputes the active video's full-length pose overlay and shuttle trajectory in one
  background job. It reuses current artifacts, reports `mode=visual`, and never edits the cut timeline. The resulting
  pose media is streamed by `GET /media/pose-overlay/full`; `pose_analysis.current` and
  `shuttle_analysis.current` are the only states that enable the main player's overlay switches.
- `POST /api/render` starts source-quality rendering. `GET /api/render` reports the same explicit lifecycle and exact
  output path. Project state also reports whether that file currently exists, so success remains visible after a toast
  disappears or the page reloads.
- Calibration, shuttle annotations and score corrections each have separate typed endpoints and backups. They cannot
  silently modify the cut timeline.
- `POST /api/score/analyze` calculates score from the saved final timeline. A later timeline or evidence update marks
  the cached score stale until the user runs it again.

## Score semantics

Automatic score rows have an evidence source: `automatic-terminal` or `automatic-next-serve`. A next-serve inference
requires a high-confidence formal serve at the beginning of the following rally. `unresolved` rows do not add a point,
and any total after an unresolved rally is labelled partial through `score_complete=false`. `manual` and
`manual-no-point` are explicit user corrections, never automatic estimates.
