"""The automatic analysis hot path: proxy → audio → shuttle → features → model → editable timeline."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

from ..audio import analyze_audio
from ..features import extract_features
from ..model import predict_model
from ..segmenter import segment_rallies
from .media import make_proxy
from .project import public_segments, save_segments, timeline_rows, video_id, video_metadata
from .shuttle import effective_shuttle_mode, features_for_trajectory, run_shuttle_detection, run_shuttle_trajectory
from .state import StudioState, set_analysis_status

POSE_ASSOCIATION_FIELDS = {
    "near_foot_speed_normalized",
    "far_foot_speed_normalized",
    "near_stance_width_normalized",
    "far_stance_width_normalized",
    "near_active_wrist_x_normalized",
    "near_active_wrist_y_normalized",
    "far_active_wrist_x_normalized",
    "far_active_wrist_y_normalized",
    "near_active_wrist_visible",
    "far_active_wrist_visible",
}


def has_pose_association(features: Path) -> bool:
    if not features.exists():
        return False
    with features.open(encoding="utf-8-sig") as handle:
        return POSE_ASSOCIATION_FIELDS.issubset(next(csv.reader(handle), []))


def preserve_state_features(state_features: Path, vision_features: Path) -> None:
    """Seed a new project once without replacing an existing state-model input."""
    if not state_features.exists() and vision_features.exists():
        shutil.copy2(vision_features, state_features)


def trajectory_needs_refinement(features_path: Path, trajectory_path: Path, annotations_path: Path) -> bool:
    if not trajectory_path.exists():
        return False
    trajectory_mtime = trajectory_path.stat().st_mtime
    return any(path.exists() and path.stat().st_mtime >= trajectory_mtime for path in (features_path, annotations_path))


def analyze_video(state: StudioState, video: Path, index: int, total: int) -> dict[str, Any]:
    layout = state.layout(video)
    artifacts = layout.analysis
    artifacts.root.mkdir(parents=True, exist_ok=True)
    current = video_id(state.library_root, video)
    shuttle_enabled = effective_shuttle_mode(state) is not None

    def stage(name: str, label: str, progress: float) -> None:
        set_analysis_status(
            state,
            state="running",
            stage=name,
            label=label,
            progress=((index - 1) + progress) / total,
            current=current,
            current_name=video.name,
            completed=index - 1,
            total=total,
        )

    stage("proxy", "正在建立流畅预览代理", 0.05)
    proxy = make_proxy(state, video)
    stage("audio", "正在分析击球声音", 0.18)
    if not artifacts.audio_events.exists():
        analyze_audio(video, artifacts.audio_events, ffmpeg=state.ffmpeg)
    if shuttle_enabled:
        stage("shuttle", "正在检测并连接本场羽球轨迹", 0.24)
        if not artifacts.shuttle_raw.exists():
            run_shuttle_detection(
                state,
                video,
                artifacts.root,
                False,
                lambda value, detector: stage("shuttle", f"正在运行 {detector} 羽球检测", 0.24 + value * 0.05),
            )
        if not artifacts.shuttle_track.exists():
            run_shuttle_trajectory(state, video, features_for_trajectory(artifacts.root))
    stage("features", "正在理解球场、球员动作与运动轨迹", 0.30)
    if not has_pose_association(artifacts.vision_features):
        extract_features(
            video,
            state.config,
            artifacts.vision_features,
            artifacts.audio_events,
            artifacts.shuttle_track if artifacts.shuttle_track.exists() else None,
            state.pose_model,
            state.packages,
            None,
            0.0,
            None,
            lambda progress: stage(
                "features", "正在理解球场、球员动作与运动轨迹", 0.30 + max(0.0, min(1.0, progress)) * 0.52
            ),
        )
    if (
        shuttle_enabled
        and artifacts.shuttle_raw.exists()
        and trajectory_needs_refinement(artifacts.vision_features, artifacts.shuttle_track, artifacts.shuttle_annotations)
    ):
        stage("shuttle-contact", "正在用球拍接触证据复核竞争轨迹", 0.83)
        run_shuttle_trajectory(state, video, artifacts.vision_features)
    preserve_state_features(artifacts.features, artifacts.vision_features)
    stage("predict", "正在用标准答案模型判断每一球", 0.84)
    predict_model(artifacts.features, state.model, artifacts.probabilities)
    stage("segment", "正在生成保守、不漏球的时间表", 0.94)
    model_adapter = state.model.with_suffix(".adapter.json")
    adapter = layout.segmentation_adapter if layout.segmentation_adapter.exists() else model_adapter
    trajectory_policy = "integrated"
    if model_adapter.exists():
        trajectory_policy = str(json.loads(model_adapter.read_text(encoding="utf-8")).get("trajectory_policy", "integrated"))
    options = state.analysis_options
    intervals = segment_rallies(
        artifacts.features,
        artifacts.probabilities,
        artifacts.automatic_rallies,
        start_threshold=0.56,
        keep_threshold=0.30,
        preroll=float(options["preroll"]),
        postroll=float(options["postroll"]),
        end_pending=float(options["end_pending"]),
        maximum_internal_gap=float(options["maximum_internal_gap"]),
        suppress_handoffs=bool(options["suppress_handoffs"]),
        shuttle_trajectory_csv=artifacts.shuttle_track if artifacts.shuttle_track.exists() else None,
        adapter=adapter if adapter.exists() else None,
        trajectory_policy=trajectory_policy,
    )
    timeline = layout.working_timeline
    duration = float(video_metadata(video)["duration"])
    save_segments(timeline, timeline_rows(public_segments(artifacts.automatic_rallies), duration))
    with state.project_lock:
        if state.video.resolve() == video.resolve():
            state.rallies = timeline
            state.proxy = proxy
    return {
        "video": current,
        "timeline": str(timeline),
        "automatic_timeline": str(artifacts.automatic_rallies),
        "rallies": len(intervals),
        "proxy": str(proxy),
    }
