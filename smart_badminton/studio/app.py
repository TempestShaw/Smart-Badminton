from __future__ import annotations

import sys
import threading
import webbrowser
from importlib.resources import files
from pathlib import Path
from typing import Any

from ..io import load_rallies
from ..model_assets import resolve_model
from ..phase_examples import capture_phase_examples
from ..pose_overlay import render_pose_overlay
from ..project_layout import ProjectLayout
from ..score_labeling import load_machine_labels, save_machine_label_review
from ..scoring import load_score_corrections, save_score_corrections, validate_score_corrections
from ..shuttle_annotations import save_shuttle_annotations, validate_shuttle_annotations
from . import jobs
from .calibration import calibration_payload, calibration_ready, default_calibration_path, save_calibration
from .insights import (
    analytics_payload,
    calculate_score_payload,
    evidence_payload,
    score_label_paths,
    score_labeling_payload,
    score_payload,
)
from .media import (
    NO_AUDIO_MESSAGE,
    audio_available,
    ffmpeg_available,
    full_pose_overlay_path,
    pose_overlay_path,
    pose_status_payload,
    runtime_payload,
)
from .project import (
    activate_video,
    directory_browser_payload,
    discover_videos,
    ensure_working_timeline,
    library_payload,
    output_payload,
    project_has_completed_marker,
    public_segments,
    save_segments,
    timeline_has_segments,
    timeline_rows,
    video_id,
    video_metadata,
)
from .shuttle import (
    RAW_INVALIDATING_REASONS,
    available_shuttle_modes,
    model_registry,
    shuttle_annotation_payload,
    shuttle_status_payload,
)
from .state import API_SCHEMA_VERSION, StudioState, restore_analysis_status

LOOPBACK_HOSTS = ["127.0.0.1", "localhost", "[::1]"]
MISSING_STUDIO_DEPENDENCIES = 'Studio dependencies are missing. Install with: pip install -e ".[studio]"'


def create_studio_app(state: StudioState, allowed_hosts: list[str] | None = None):
    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.middleware.trustedhost import TrustedHostMiddleware
        from fastapi.responses import FileResponse, JSONResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as error:
        raise RuntimeError(MISSING_STUDIO_DEPENDENCIES) from error

    static_root = Path(str(files("smart_badminton").joinpath("studio_static")))
    restore_analysis_status(state)
    app = FastAPI(title="Smart Badminton Studio", docs_url=None, redoc_url=None)
    # The API can browse and write local folders, so reject DNS-rebinding requests whose Host is not this machine.
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts or LOOPBACK_HOSTS)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:3000", "http://localhost:3000"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.mount("/static", StaticFiles(directory=static_root), name="static")

    async def bad_request(_request: Request, error: Exception) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(error)})

    for error_type in (KeyError, OSError, RuntimeError, TypeError, ValueError):
        app.add_exception_handler(error_type, bad_request)

    def active_project() -> str:
        return video_id(state.library_root, state.video)

    def require_project(payload: dict[str, Any] | None) -> str:
        """Reject writes from a tab that still shows a different video."""
        requested = str((payload or {}).get("project_id", "")).strip()
        if requested != active_project():  # A missing id is as stale as a wrong one.
            raise HTTPException(status_code=409, detail="The open project changed; reload Studio")
        return active_project()

    def require_idle(render: bool = False, analysis: bool = False) -> None:
        if render and state.render_status.get("state") == "running":
            raise HTTPException(status_code=409, detail="Wait for the current render to finish")
        if analysis and state.analysis_status.get("state") == "running":
            raise HTTPException(status_code=409, detail="Wait for the current analysis job to finish")

    def analysis_configured() -> bool:
        return bool(calibration_ready(state) and state.model and audio_available(state))

    def require_audio(videos: list[Path]) -> None:
        silent = [video.name for video in videos if not audio_available(state, video)]
        if len(videos) == 1 and silent:
            raise HTTPException(status_code=400, detail=NO_AUDIO_MESSAGE)
        if silent:
            raise HTTPException(status_code=400, detail=f"这些视频没有音轨，无法自动分析：{'、'.join(silent)}")

    def acquire_analysis() -> None:
        if not state.analysis_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="Another analysis job is already running")

    @app.get("/")
    def index():
        return FileResponse(static_root / "index.html")

    @app.get("/media/video")
    def video():
        return FileResponse(state.proxy or state.video, media_type="video/mp4")

    @app.get("/api/health")
    def health():
        return {"ok": True, "api_schema_version": API_SCHEMA_VERSION, "runtime": runtime_payload(state)}

    @app.get("/api/tutorial")
    def tutorial(language: str = "zh"):
        filename = "studio-tutorial.en.md" if language == "en" else "studio-tutorial.zh-CN.md"
        tutorial_path = static_root / filename
        if not tutorial_path.exists():
            # Source checkouts can run the API before the frontend copies the manual into the static bundle.
            tutorial_path = Path(__file__).resolve().parents[2] / "docs" / filename
        return {"markdown": tutorial_path.read_text(encoding="utf-8")}

    def project_payload() -> dict[str, Any]:
        runtime = runtime_payload(state)
        shuttle_analysis = shuttle_status_payload(state)
        ffmpeg_ok = ffmpeg_available(state)
        has_audio = audio_available(state)
        ready = calibration_ready(state)
        return {
            "id": active_project(),
            "video": video_metadata(state.video),
            "preview": video_metadata(state.proxy or state.video),
            "segments": public_segments(state.rallies),
            "timeline_path": str(state.rallies),
            "output_path": str(state.output),
            "output": output_payload(state),
            "runtime": runtime,
            "models": model_registry(state).public_payload(),
            "shuttle_analysis": shuttle_analysis,
            "pose_analysis": pose_status_payload(state),
            "api_schema_version": API_SCHEMA_VERSION,
            "automatic_analysis": {
                "configured": analysis_configured(),
                "pose_overlay_configured": bool(ready and state.pose_model and ffmpeg_ok),
                "audio_available": has_audio,
                "configuration_issue": (
                    runtime["ffmpeg"]["reason"] if not ffmpeg_ok else None if has_audio else NO_AUDIO_MESSAGE
                ),
                "model": state.model.name if state.model else None,
                "shuttle_model": state.shuttle_model.name if state.shuttle_model else None,
                "tracknet_model": state.tracknet_model.name if state.tracknet_model else None,
                "inpaint_model": state.inpaint_model.name if state.inpaint_model else None,
                "shuttle_mode": shuttle_analysis["mode"],
                "available_shuttle_modes": shuttle_analysis["available_modes"],
                "trajectory_boundary_protection": bool(shuttle_analysis["current"]),
                "config": state.config.name if state.config else None,
                "preset": "precision-v3",
                "settings": dict(state.analysis_options),
            },
            "calibration": {"ready": ready, "path": str(state.config or default_calibration_path(state))},
            "analytics": analytics_payload(state),
            "score": score_payload(state),
            "score_labeling": score_labeling_payload(state),
            "evidence": evidence_payload(state),
        }

    @app.get("/api/project")
    def project():
        return project_payload()

    @app.get("/api/library")
    def library():
        return library_payload(state)

    @app.get("/api/filesystem/directories")
    def browse_directories(path: str | None = None):
        return directory_browser_payload(path, state.library_root)

    @app.post("/api/library")
    def change_library(payload: dict[str, Any]):
        require_idle(analysis=True)
        library = Path(str(payload["path"])).expanduser().resolve()
        if not library.is_dir() or not discover_videos(library):
            raise ValueError(f"No supported videos found in {library}")
        state.library = library
        return library_payload(state)

    @app.post("/api/project/open")
    def open_project(payload: dict[str, Any]):
        require_idle(render=True, analysis=True)
        requested = str(payload.get("id", ""))
        match = next(
            (
                video
                for video in discover_videos(state.library_root)
                if video_id(state.library_root, video) == requested or video.stem == requested
            ),
            None,
        )
        if match is None:
            raise HTTPException(status_code=404, detail="Video not found in the selected folder")
        activate_video(state, match)
        return project_payload()

    @app.put("/api/output")
    def change_output(payload: dict[str, Any]):
        require_project(payload)
        require_idle(render=True)
        directory = Path(str(payload["directory"])).expanduser().resolve()
        state.output_directory = directory
        # Keep only the base name so a filename cannot point outside the chosen folder.
        state.output = directory / Path(str(payload.get("filename") or state.output.name).strip()).name
        state.render_status = {"state": "idle"}
        return {"ok": True, **output_payload(state)}

    @app.put("/api/timeline")
    def save_timeline(payload: dict[str, Any]):
        project_id = require_project(payload)
        rows = timeline_rows(payload["segments"], float(video_metadata(state.video)["duration"]))
        backup = save_segments(state.rallies, rows)
        artifacts = state.layout().analysis
        phase_examples = capture_phase_examples(
            project_id,
            rows,
            artifacts.automatic_rallies,
            artifacts.root / "Boundary_Corrections" / "user-phase-examples.csv",
            artifacts.features,
            artifacts.probabilities,
            artifacts.shuttle_track,
        )
        return {
            "ok": True,
            "segments": public_segments(state.rallies),
            "backup": backup.name,
            "analytics": analytics_payload(state),
            "score": score_payload(state),
            "phase_examples": phase_examples,
        }

    @app.post("/api/render")
    def start_render(payload: dict[str, Any] | None = None):
        payload = payload or {}
        require_idle(analysis=True)
        options = (
            bool(payload.get("include_trajectory", False)),
            str(payload.get("winner_filter", "all")),
            bool(payload.get("include_score", False)),
        )
        with state.render_lock:
            if state.render_status.get("state") == "running":
                return state.render_status
            state.render_status = {
                "state": "running",
                "output": str(state.output),
                "include_trajectory": options[0],
                "winner_filter": options[1],
                "include_score": options[2],
            }
            threading.Thread(target=jobs.render_job, args=(state, *options), daemon=True).start()
        return state.render_status

    @app.get("/api/render")
    def render_status():
        return state.render_status

    @app.get("/api/analytics")
    def analytics():
        return analytics_payload(state)

    @app.get("/api/score")
    def score():
        return score_payload(state)

    @app.post("/api/score/analyze")
    def calculate_score(payload: dict[str, Any]):
        require_project(payload)
        require_idle(analysis=True)
        analytics_payload(state)
        return calculate_score_payload(state)

    @app.put("/api/score")
    def update_score(payload: dict[str, Any]):
        require_project(payload)
        rows = validate_score_corrections(payload.get("corrections"), len(load_rallies(state.rallies)))
        backup = save_score_corrections(state.layout().score_corrections, rows)
        return {**calculate_score_payload(state), "backup": str(backup) if backup else None}

    @app.post("/api/score-labels/analyze")
    def start_score_labeling(payload: dict[str, Any] | None = None):
        project_id = require_project(payload)
        require_idle(render=True)
        score = score_payload(state)
        if not score.get("available"):
            score = calculate_score_payload(state)
        review = {int(row.get("rally", 0)) for row in load_machine_labels(score_label_paths(state)[2]) if row.get("status") == "review"}
        unresolved = {int(row["rally"]) for row in score.get("rallies", []) if row.get("winner_source") == "unresolved"}
        rally_ids = sorted(review | unresolved)
        if not rally_ids:
            state.analysis_status = {
                "state": "complete",
                "mode": "score-labels",
                "stage": "complete",
                "label": "没有待判断的回合",
                "progress": 1.0,
                "completed": 0,
                "total": 0,
                "results": [],
            }
            return state.analysis_status
        acquire_analysis()
        source = state.video
        return jobs.start_analysis_job(
            state,
            "score-labels",
            "终局标注失败",
            lambda: jobs.score_labeling_job(state, source, rally_ids),
            label="终局标注已排队",
            total=len(rally_ids),
            project_id=project_id,
            project_name=source.name,
        )

    @app.put("/api/score-labels/review")
    def review_score_label(payload: dict[str, Any]):
        require_project(payload)
        rally, decision = int(payload["rally"]), str(payload["decision"])
        labels_path = score_label_paths(state)[2]
        label = {int(row.get("rally", 0)): row for row in load_machine_labels(labels_path)}[rally]
        if decision == "accepted":
            corrections_path = state.layout().score_corrections
            corrections = load_score_corrections(corrections_path)
            current = next((row for row in corrections if int(row["rally"]) == rally), None) or {
                "rally": rally,
                "winner": "auto",
                "server_override": "unknown",
                "server": "unknown",
                "last_hitter": "unknown",
                "terminal_event": "unknown",
                "landing_side": "unknown",
                "post_rally_event": "unknown",
                "note": "",
            }
            suggestion = label.get("suggestion", {})
            replacement = {
                **current,
                **{
                    field: str(suggestion[field])
                    for field in ("winner", "last_hitter", "terminal_event", "landing_side", "post_rally_event")
                    if str(suggestion.get(field, "unknown")) != "unknown"
                },
                "note": "multimodal review",
            }
            rows = [row for row in corrections if int(row["rally"]) != rally] + [replacement]
            save_score_corrections(corrections_path, validate_score_corrections(rows, len(load_rallies(state.rallies))))
        save_machine_label_review(labels_path, rally, decision)
        return {"score": calculate_score_payload(state), "score_labeling": score_labeling_payload(state)}

    @app.get("/api/shuttle-annotations")
    def shuttle_annotations():
        return shuttle_annotation_payload(state)

    @app.put("/api/shuttle-annotations")
    def update_shuttle_annotations(payload: dict[str, Any]):
        require_project(payload)
        metadata = video_metadata(state.video)
        rows = validate_shuttle_annotations(
            payload.get("annotations"),
            float(metadata["duration"]),
            float(metadata["fps"]),
            int(metadata["width"]),
            int(metadata["height"]),
        )
        backup = save_shuttle_annotations(state.layout().analysis.shuttle_annotations, rows)
        return {**shuttle_annotation_payload(state), "backup": str(backup) if backup else None}

    @app.get("/api/calibration")
    def calibration():
        return calibration_payload(state)

    @app.put("/api/calibration")
    def update_calibration(payload: dict[str, Any]):
        require_project(payload)
        path, backup = save_calibration(state, payload)
        ready, ffmpeg_ok = calibration_ready(state), ffmpeg_available(state)
        return {
            **calibration_payload(state),
            "ok": True,
            "path": str(path),
            "backup": str(backup) if backup else None,
            "automatic_analysis_configured": analysis_configured(),
            "pose_overlay_configured": bool(ready and state.pose_model and ffmpeg_ok),
            "shuttle_analysis": shuttle_status_payload(state),
        }

    @app.post("/api/pose-overlay")
    def pose_overlay(payload: dict[str, Any]):
        require_project(payload)
        rally_number = int(payload["rally"])
        output, segment = pose_overlay_path(state, rally_number)
        with state.pose_lock:
            inputs = (state.video, state.rallies, state.config, state.pose_model)
            if not output.exists() or output.stat().st_mtime < max(path.stat().st_mtime for path in inputs):
                render_pose_overlay(
                    state.video,
                    float(segment["start"]),
                    float(segment["end"]),
                    output,
                    state.pose_model,
                    state.config,
                    state.packages,
                    state.ffmpeg,
                    str(runtime_payload(state)["ffmpeg"]["selected_encoder"]),
                )
        return {
            "ok": True,
            "rally": rally_number,
            "url": f"/media/pose-overlay?rally={rally_number}&v={output.stat().st_mtime_ns}",
            "output": str(output),
        }

    @app.get("/media/pose-overlay")
    def pose_overlay_media(rally: int):
        return FileResponse(pose_overlay_path(state, rally)[0], media_type="video/mp4")

    @app.get("/media/pose-overlay/full")
    def full_pose_overlay_media():
        return FileResponse(full_pose_overlay_path(state), media_type="video/mp4")

    @app.post("/api/analyze/shuttle")
    def start_shuttle_analysis(payload: dict[str, Any] | None = None):
        project_id = require_project(payload)
        require_idle(render=True, analysis=True)
        force = bool((payload or {}).get("force", False))
        current = shuttle_status_payload(state)
        if current["current"] and not force:
            state.analysis_status = {
                "state": "complete",
                "mode": "shuttle",
                "stage": "complete",
                "label": "当前视频已经有羽球轨迹",
                "progress": 1.0,
                "completed": 1,
                "total": 1,
                "results": [
                    {
                        "video": project_id,
                        "points": current["point_count"],
                        "flights": current["flight_count"],
                        "track": current["track_path"],
                        "timeline_unchanged": str(state.rallies),
                    }
                ],
            }
            return state.analysis_status
        stale_reason = str(current.get("stale_reason") or "")
        force = force or any(reason in stale_reason for reason in RAW_INVALIDATING_REASONS)
        acquire_analysis()
        source = state.video
        return jobs.start_analysis_job(
            state,
            "shuttle",
            "羽球轨迹分析失败",
            lambda: jobs.shuttle_job(state, source, force),
            label="羽球轨迹分析已排队；剪辑时间表不会改变",
            total=1,
            project_id=project_id,
            project_name=source.name,
        )

    @app.put("/api/settings/shuttle-mode")
    def update_shuttle_mode(payload: dict[str, Any]):
        mode = str(payload["mode"]).strip().lower()
        if mode not in available_shuttle_modes(state):
            raise ValueError(f"Shuttle mode {mode!r} is not configured; available: {', '.join(available_shuttle_modes(state))}")
        state.shuttle_mode = mode
        return shuttle_status_payload(state)

    @app.post("/api/analyze/visual")
    def start_visual_analysis(payload: dict[str, Any] | None = None):
        project_id = require_project(payload)
        require_idle(render=True, analysis=True)
        require_audio([state.video])
        pose_status = pose_status_payload(state)
        shuttle_status = shuttle_status_payload(state)
        force = bool((payload or {}).get("force", False))
        run_pose = bool(pose_status["configured"] and (force or not pose_status["current"]))
        run_shuttle = bool(shuttle_status["configured"] and (force or not shuttle_status["current"]))
        if not run_pose and not run_shuttle:
            state.analysis_status = {
                "state": "complete",
                "mode": "visual",
                "stage": "complete",
                "label": "整段姿态与羽球分析已经是最新版本",
                "progress": 1.0,
                "completed": 2,
                "total": 2,
                "results": [{"video": project_id, "pose": pose_status, "shuttle": shuttle_status}],
            }
            return state.analysis_status
        acquire_analysis()
        rerun_raw = bool(force or shuttle_status["stale"] or not shuttle_status["raw_generated"])
        source = state.video
        return jobs.start_analysis_job(
            state,
            "visual",
            "整段视觉分析失败",
            lambda: jobs.visual_job(state, source, run_pose, run_shuttle, rerun_raw),
            label="整段视觉分析已排队；不会修改剪辑时间表",
            total=int(run_pose) + int(run_shuttle),
            project_id=project_id,
            project_name=source.name,
        )

    def launch_analysis(videos: list[Path], mode: str, options: dict[str, Any] | None) -> dict[str, Any]:
        require_idle(render=True)
        require_audio(videos)
        requested = options or {}

        def padding(name: str) -> float:
            value = float(requested.get(name, state.analysis_options[name]))
            if not 0 <= value <= 2:  # NaN fails too
                raise ValueError(f"{name} must be between 0 and 2 seconds")
            return value

        # Parse before taking the lock: only the job's worker releases it.
        analysis_options = {
            **state.analysis_options,
            "preroll": padding("preroll"),
            "postroll": padding("postroll"),
            "suppress_handoffs": bool(requested.get("suppress_handoffs", True)),
        }
        acquire_analysis()
        state.analysis_options = analysis_options
        single = len(videos) == 1
        return jobs.start_analysis_job(
            state,
            mode,
            "自动分析失败",
            lambda: jobs.batch_analysis_job(state, videos, mode),
            label="自动分析已排队",
            total=len(videos),
            project_id=video_id(state.library_root, videos[0]) if single else "batch",
            project_name=videos[0].name if single else f"{len(videos)} 个比赛",
        )

    @app.post("/api/analyze")
    def start_analysis(payload: dict[str, Any] | None = None):
        return launch_analysis([state.video], "single", payload)

    @app.post("/api/analyze/batch")
    def start_batch_analysis(payload: dict[str, Any] | None = None):
        library_root = state.library_root
        pending = [
            video
            for video in discover_videos(library_root)
            if not timeline_has_segments(ProjectLayout.for_video(library_root, video).working_timeline)
            and not project_has_completed_marker(library_root, video)
        ]
        if not pending:
            return {
                "state": "complete",
                "mode": "batch",
                "label": "没有待分析的比赛",
                "progress": 1.0,
                "completed": 0,
                "total": 0,
                "results": [],
            }
        return launch_analysis(pending, "batch", payload)

    @app.get("/api/analyze")
    def analysis_status():
        return state.analysis_status

    return app


def run_studio(
    video: Path | None,
    rallies: Path | None,
    output: Path | None = None,
    proxy: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    ffmpeg: Path | None = None,
    encoder: str = "auto",
    quality: int = 21,
    library: Path | None = None,
    config: Path | None = None,
    model: Path | None = None,
    pose_model: Path | None = None,
    packages: Path | None = None,
    shuttle_model: Path | None = None,
    tracknet_model: Path | None = None,
    inpaint_model: Path | None = None,
    shuttle_mode: str = "hybrid",
    tracknet_packages: Path | None = None,
) -> None:
    if tracknet_packages is not None and str(tracknet_packages) not in sys.path:
        sys.path.insert(0, str(tracknet_packages))
    try:
        import uvicorn
    except ImportError as error:
        raise RuntimeError(MISSING_STUDIO_DEPENDENCIES) from error
    if video is None:
        videos = discover_videos(library)
        if not videos:
            raise ValueError(f"No supported video files found in: {library}")
        video = videos[0]
    library = (library or video.parent).resolve()
    layout = ProjectLayout.for_video(library, video)
    state = StudioState(
        video=video,
        rallies=rallies or ensure_working_timeline(library, video),
        output=output.expanduser().resolve() if output is not None else layout.default_output,
        proxy=proxy or layout.existing_proxy(),
        ffmpeg=ffmpeg,
        encoder=encoder,
        quality=quality,
        library=library,
        output_directory=output.expanduser().resolve().parent if output is not None else None,
        config=config,
        model=resolve_model(model, "rally", library),
        pose_model=resolve_model(pose_model, "pose", library),
        shuttle_model=resolve_model(shuttle_model, "shuttle", library),
        tracknet_model=resolve_model(tracknet_model, "tracknet", library),
        inpaint_model=resolve_model(inpaint_model, "inpaint", library),
        shuttle_mode=shuttle_mode,
        packages=packages,
        tracknet_packages=tracknet_packages,
    )
    # Binding beyond loopback is an explicit opt-in to LAN access, so accept that interface's Host header too.
    allowed_hosts = LOOPBACK_HOSTS if host in LOOPBACK_HOSTS else [*LOOPBACK_HOSTS, host if host != "0.0.0.0" else "*"]
    app = create_studio_app(state, allowed_hosts)
    url = f"http://{host}:{port}"
    print(f"Smart Badminton Studio: {url}")
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=host, port=port, log_level="info")
