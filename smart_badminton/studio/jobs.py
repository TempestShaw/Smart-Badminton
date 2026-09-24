"""Background jobs. Each runs on a daemon thread and reports progress through ``state.analysis_status``."""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..audio import analyze_audio
from ..features import extract_features
from ..pose_overlay import render_pose_overlay
from ..render import render_rallies
from ..score_labeling import label_score_evidence, prepare_score_evidence, score_labeling_config
from ..score_learning import default_score_model_path, fit_score_evidence_model
from .insights import calculate_score_payload, score_label_paths
from .media import full_pose_overlay_path, pose_status_payload, record_encoder, runtime_payload
from .pipeline import analyze_video, has_pose_association, preserve_state_features
from .project import video_id, video_metadata
from .shuttle import features_for_trajectory, run_shuttle_detection, run_shuttle_trajectory, shuttle_status_payload
from .state import StudioState, begin_analysis_status, set_analysis_status


def start_analysis_job(
    state: StudioState, mode: str, error_label: str, work: Callable[[], None], **status: Any
) -> dict[str, Any]:
    """Run ``work`` in the background; the caller must already hold ``state.analysis_lock``."""

    def run() -> None:
        try:
            work()
        except Exception as error:  # noqa: BLE001 - every background failure must reach the local UI
            set_analysis_status(state, state="error", mode=mode, label=error_label, message=str(error))
        finally:
            state.analysis_lock.release()

    begin_analysis_status(state, mode=mode, stage="queued", progress=0.0, completed=0, **status)
    threading.Thread(target=run, daemon=True).start()
    return state.analysis_status


def complete(state: StudioState, mode: str, label: str, total: int, results: list[dict[str, Any]]) -> dict[str, Any]:
    status = {
        "state": "complete",
        "mode": mode,
        "stage": "complete",
        "label": label,
        "progress": 1.0,
        "completed": total,
        "total": total,
        "results": results,
    }
    set_analysis_status(state, **status)
    return state.analysis_status


def render_job(state: StudioState, include_trajectory: bool, winner_filter: str, include_score: bool) -> None:
    try:
        with state.project_lock:
            video, timeline, output = state.video, state.rallies, state.output
            layout = state.layout()
        used_encoder = render_rallies(
            video,
            timeline,
            output,
            state.ffmpeg,
            state.encoder,
            state.quality,
            trajectory_csv=layout.analysis.shuttle_track if include_trajectory else None,
            score_csv=layout.analysis.score_state if include_score or winner_filter != "all" else None,
            winner_filter=winner_filter,
            include_score=include_score,
        )
        record_encoder(state, used_encoder, "render")
        state.render_status = {
            "state": "complete",
            "output": str(output),
            "encoder": used_encoder,
            "include_trajectory": include_trajectory,
            "winner_filter": winner_filter,
            "include_score": include_score,
        }
    except Exception as error:  # noqa: BLE001 - every renderer failure must reach the local UI
        state.render_status = {"state": "error", "message": str(error)}


def score_labeling_job(state: StudioState, video: Path, rally_ids: list[int]) -> None:
    root, manifest_path, labels_path = score_label_paths(state)
    config = score_labeling_config()
    total = len(rally_ids)

    def progress(stage: str, label: str, start: float, span: float) -> Callable[[int, int], None]:
        return lambda completed, count: set_analysis_status(
            state,
            state="running",
            stage=stage,
            label=label,
            progress=start + span * completed / max(1, count),
            completed=completed,
            total=count,
        )

    prepare_score_evidence(
        video,
        state.rallies,
        root,
        rally_ids,
        trajectory_csv=root.parent / "shuttle-track.csv",
        progress_callback=progress("frames", "正在提取终局画面", 0.05, 0.35),
    )
    if not config["configured"]:
        complete(state, "score-labels", "待标注素材已生成", total, [{"directory": str(root), "rallies": rally_ids, "labeled": 0}])
        return
    result = label_score_evidence(
        manifest_path,
        labels_path,
        list(config["models"]),
        str(config["endpoint"]),
        str(config["api_key"]),
        progress_callback=progress("label", "正在判断终局", 0.40, 0.60),
    )
    model_payload = fit_score_evidence_model(state.library_root, default_score_model_path(state.library_root))
    calculate_score_payload(state)
    targets = set(rally_ids)
    consensus = sum(row.get("status") == "consensus" and int(row.get("rally", 0)) in targets for row in result["labels"])
    complete(
        state,
        "score-labels",
        "终局建议已生成",
        total,
        [
            {
                "directory": str(root),
                "rallies": rally_ids,
                "labeled": total,
                "consensus": consensus,
                "pseudo_examples": model_payload.get("pseudo_examples", 0),
            }
        ],
    )


def shuttle_job(state: StudioState, video: Path, force: bool) -> None:
    artifacts = state.layout(video).analysis
    artifacts.root.mkdir(parents=True, exist_ok=True)

    def running(stage: str, label: str, progress: float) -> None:
        set_analysis_status(state, state="running", mode="shuttle", stage=stage, label=label, progress=progress)

    detection: dict[str, Any] = {"reused": True}
    if force or not artifacts.shuttle_raw.exists():
        running("shuttle-detect", "正在逐帧检测本场羽球；剪辑时间表不会改变", 0.12)
        detection = run_shuttle_detection(
            state,
            video,
            artifacts.root,
            force,
            lambda value, detector: running("shuttle-detect", f"正在运行 {detector} 羽球检测", 0.12 + value * 0.68),
        )
    running("shuttle-track", "正在过滤隔壁场目标并连接连续飞行轨迹", 0.82)
    run_shuttle_trajectory(state, video, features_for_trajectory(artifacts.root))
    status = shuttle_status_payload(state)
    complete(
        state,
        "shuttle",
        "羽球轨迹分析完成；剪辑时间表未改变",
        1,
        [
            {
                "video": video_id(state.library_root, video),
                "detector": detection.get("mode"),
                "points": status["point_count"],
                "flights": status["flight_count"],
                "track": status["track_path"],
                "timeline_unchanged": str(state.rallies),
            }
        ],
    )


def visual_job(state: StudioState, video: Path, run_pose: bool, run_shuttle: bool, rerun_shuttle_raw: bool) -> None:
    artifacts = state.layout(video).analysis
    artifacts.root.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {"video": video_id(state.library_root, video)}

    def running(stage: str, label: str, progress: float) -> None:
        set_analysis_status(state, state="running", mode="visual", stage=stage, label=label, progress=progress)

    if run_pose:
        pose_label = "正在分析整段人物姿态并生成可开关覆盖层"
        render_pose_overlay(
            video,
            0.0,
            float(video_metadata(video)["duration"]),
            full_pose_overlay_path(state),
            state.pose_model,
            state.config,
            state.packages,
            state.ffmpeg,
            str(runtime_payload(state)["ffmpeg"]["selected_encoder"]),
            progress_callback=lambda value: running("pose", pose_label, 0.05 + value * (0.52 if run_shuttle else 0.9)),
        )
        results["pose"] = pose_status_payload(state)

    if run_shuttle:
        detection: dict[str, Any] = {"reused": True}
        if rerun_shuttle_raw or not artifacts.shuttle_raw.exists():
            start = 0.60 if run_pose else 0.12
            running("shuttle-detect", "正在分析整段羽球候选；高远球与隔壁场会交给轨迹器复核", start)
            detection = run_shuttle_detection(
                state,
                video,
                artifacts.root,
                rerun_shuttle_raw,
                lambda value, detector: running(
                    "shuttle-detect", f"正在运行 {detector} 羽球检测", start + value * (0.74 - start)
                ),
            )
        needs_pose_association = bool(state.pose_model and not has_pose_association(artifacts.vision_features))
        if needs_pose_association:
            running("shuttle-track-initial", "正在先排除长期静止杂物并建立初始羽球轨迹", 0.76)
            run_shuttle_trajectory(state, video, None)
            if not artifacts.audio_events.exists():
                analyze_audio(video, artifacts.audio_events, ffmpeg=state.ffmpeg)
            extract_features(
                video,
                state.config,
                artifacts.vision_features,
                artifacts.audio_events,
                artifacts.shuttle_track,
                state.pose_model,
                state.packages,
                None,
                0.0,
                None,
                lambda value: running("player-association", "正在提取手腕、捡球与持球证据", 0.79 + value * 0.16),
            )
            preserve_state_features(artifacts.features, artifacts.vision_features)
        running(
            "shuttle-track",
            "正在结合静态杂物、飞行连续性和球员交互复核羽球轨迹",
            0.96 if needs_pose_association else 0.93,
        )
        run_shuttle_trajectory(
            state, video, artifacts.vision_features if artifacts.vision_features.exists() else None
        )
        results["shuttle"] = {**shuttle_status_payload(state), "detector": detection.get("mode")}

    total = int(run_pose) + int(run_shuttle)
    complete(state, "visual", "整段视觉分析完成；播放器可以随时开关姿态和球路", total, [results])


def batch_analysis_job(state: StudioState, videos: list[Path], mode: str) -> None:
    results = [analyze_video(state, source, index, len(videos)) for index, source in enumerate(videos, 1)]
    set_analysis_status(
        state,
        state="complete",
        mode=mode,
        label="自动分析完成，时间表已可校对",
        progress=1.0,
        completed=len(results),
        total=len(videos),
        results=results,
    )
