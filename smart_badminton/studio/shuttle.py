from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..hybrid import fuse_shuttle_detections
from ..io import read_rows
from ..model_registry import ModelRegistry
from ..score_labeling import score_labeling_config
from ..score_learning import default_score_model_path
from ..shuttle import detect_shuttle
from ..shuttle_annotations import load_shuttle_annotations
from ..tracknet import TrackNetRuntimeConfig, detect_tracknet
from ..trajectory import analyze_shuttle_trajectory
from .project import video_id, video_metadata
from .state import StudioState, cached_payload

VISIBLE_SHUTTLE_STATUSES = {"tracked", "recovered", "competing", "manual"}
# A calibration or detector change invalidates raw detections, not only the linked track.
RAW_INVALIDATING_REASONS = ("球场校准已更改", "检测模式已更改", "检测缓存需要更新")


def model_registry(state: StudioState) -> ModelRegistry:
    return ModelRegistry(
        rally_state=state.model,
        pose=state.pose_model,
        shuttle_yolo=state.shuttle_model,
        tracknet=state.tracknet_model,
        inpaint=state.inpaint_model,
        score_evidence=default_score_model_path(state.library_root),
        vision_models=tuple(score_labeling_config()["models"]),
    )


def available_shuttle_modes(state: StudioState) -> list[str]:
    return model_registry(state).available_shuttle_modes()


def effective_shuttle_mode(state: StudioState) -> str | None:
    available = available_shuttle_modes(state)
    if state.shuttle_mode in available:
        return state.shuttle_mode
    if "hybrid" in available:
        return "hybrid"
    return available[0] if available else None


def features_for_trajectory(analysis_root: Path) -> Path | None:
    """Prefer vision features (with pose/contact columns); fall back to state-model features."""
    for name in ("vision-features.csv", "smart-features.csv"):
        if (analysis_root / name).exists():
            return analysis_root / name
    return None


def run_shuttle_trajectory(state: StudioState, video: Path, features: Path | None) -> None:
    layout = state.layout(video)
    analyze_shuttle_trajectory(
        layout.analysis.shuttle_raw,
        layout.analysis.shuttle_track,
        features,
        video,
        layout.analysis.shuttle_annotations,
        state.config,
    )


def run_shuttle_detection(
    state: StudioState,
    video: Path,
    analysis_root: Path,
    force: bool,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    mode = effective_shuttle_mode(state)
    yolo_output = analysis_root / "shuttle-yolo-raw.csv"
    tracknet_output = analysis_root / "shuttle-tracknet-raw.csv"
    final_path = analysis_root / "shuttle-raw.csv"

    def progress(value: float, label: str) -> None:
        if progress_callback is not None:
            progress_callback(max(0.0, min(1.0, value)), label)

    results: dict[str, Any] = {"mode": mode, "output": str(final_path)}
    if mode in {"yolo", "hybrid"}:
        output = yolo_output if mode == "hybrid" else final_path
        if force or not output.exists():
            scale = 0.38 if mode == "hybrid" else 0.92
            results["yolo"] = detect_shuttle(
                video,
                state.config,
                state.shuttle_model,
                state.packages,
                output,
                sample_fps=15.0,
                confidence=0.04,
                image_size=1280,
                progress_callback=lambda value: progress(value * scale, "YOLO"),
            )
    if mode in {"tracknet", "hybrid"}:
        output = tracknet_output if mode == "hybrid" else final_path
        if force or not output.exists():
            start, scale = (0.40, 0.52) if mode == "hybrid" else (0.0, 0.92)
            results["tracknet"] = detect_tracknet(
                video,
                state.config,
                state.tracknet_model,
                output,
                state.inpaint_model if state.inpaint_model and state.inpaint_model.exists() else None,
                state.tracknet_packages,
                TrackNetRuntimeConfig(),
                progress_callback=lambda value: progress(start + value * scale, "TrackNet"),
            )
    if mode == "hybrid":
        progress(0.94, "融合")
        results["hybrid"] = fuse_shuttle_detections(yolo_output, tracknet_output, state.config, final_path)

    def mtime(path: Path | None) -> int | None:
        return path.stat().st_mtime_ns if path else None

    metadata = {
        "mode": mode,
        "video": str(video),
        "config": str(state.config),
        "yolo_model": str(state.shuttle_model) if state.shuttle_model else None,
        "tracknet_model": str(state.tracknet_model) if state.tracknet_model else None,
        "inpaint_model": str(state.inpaint_model) if state.inpaint_model else None,
        "yolo_model_mtime_ns": mtime(state.shuttle_model),
        "tracknet_model_mtime_ns": mtime(state.tracknet_model),
        "inpaint_model_mtime_ns": mtime(state.inpaint_model),
        "completed_at": time.time(),
    }
    (analysis_root / "shuttle-detection.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    progress(1.0, "完成")
    return results


def _build_shuttle_status_payload(state: StudioState) -> dict[str, Any]:
    artifacts = state.layout().analysis
    raw_path, trajectory_path = artifacts.shuttle_raw, artifacts.shuttle_track
    metadata_path = artifacts.shuttle_detection_metadata
    accepted_rows = []
    if trajectory_path.exists():
        accepted_rows = [row for row in read_rows(trajectory_path) if row.get("status") in VISIBLE_SHUTTLE_STATUSES]
    flight_ids = {str(row.get("flight_id", "")).strip() for row in accepted_rows} - {""}
    mode = effective_shuttle_mode(state)
    stale_sources: list[str] = []
    if trajectory_path.exists():
        trajectory_mtime = trajectory_path.stat().st_mtime_ns
        stale_sources = [
            reason
            for path, reason in (
                (state.config, "球场校准已更改"),
                (raw_path, "原始检测已更新"),
                (features_for_trajectory(artifacts.root), "动作特征已更新"),
                (artifacts.shuttle_annotations, "人工羽球标注已更新"),
            )
            if path is not None and path.exists() and path.stat().st_mtime_ns > trajectory_mtime
        ]
    detection = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    if raw_path.exists() and detection.get("mode") != mode:
        stale_sources.append("检测模式已更改")
    if raw_path.exists() and not metadata_path.exists():
        stale_sources.append("检测缓存需要更新")
    uses_tracknet = mode in {"tracknet", "hybrid"}
    for model_path, key, reason in (
        (state.shuttle_model if mode in {"yolo", "hybrid"} else None, "yolo_model_mtime_ns", "YOLO 模型已更新"),
        (state.tracknet_model if uses_tracknet else None, "tracknet_model_mtime_ns", "TrackNet 模型已更新"),
        (state.inpaint_model if uses_tracknet else None, "inpaint_model_mtime_ns", "Inpaint 模型已更新"),
    ):
        if model_path is not None and detection.get(key) not in {None, model_path.stat().st_mtime_ns}:
            stale_sources.append(reason)
    return {
        "configured": bool(state.config_ready() and mode),
        "generated": trajectory_path.exists(),
        "stale": bool(stale_sources),
        "current": bool(trajectory_path.exists() and not stale_sources),
        "stale_reason": "；".join(stale_sources) or None,
        "raw_generated": raw_path.exists(),
        "point_count": len(accepted_rows),
        "flight_count": len(flight_ids),
        "track_path": str(trajectory_path),
        "mode": mode,
        "requested_mode": state.shuttle_mode,
        "available_modes": available_shuttle_modes(state),
        "tracknet_configured": bool(state.tracknet_model and state.tracknet_model.exists()),
        "inpaint_configured": bool(state.inpaint_model and state.inpaint_model.exists()),
    }


def shuttle_status_payload(state: StudioState) -> dict[str, Any]:
    artifacts = state.layout().analysis
    paths = (
        state.config,
        state.shuttle_model,
        state.tracknet_model,
        state.inpaint_model,
        artifacts.shuttle_raw,
        artifacts.shuttle_track,
        artifacts.features,
        artifacts.vision_features,
        artifacts.shuttle_annotations,
        artifacts.shuttle_detection_metadata,
    )
    return cached_payload(
        state, f"shuttle-status:{state.shuttle_mode}", paths, lambda: _build_shuttle_status_payload(state)
    )


def shuttle_annotation_payload(state: StudioState) -> dict[str, Any]:
    artifacts = state.layout().analysis
    status = shuttle_status_payload(state)
    metadata = video_metadata(state.video)
    detections = []
    if artifacts.shuttle_track.exists():
        for row in read_rows(artifacts.shuttle_track):
            if row.get("status") not in VISIBLE_SHUTTLE_STATUSES:
                continue
            width = float(row.get("source_width") or metadata["width"])
            height = float(row.get("source_height") or metadata["height"])
            detections.append(
                {
                    "time": round(float(row["time_seconds"]), 4),
                    "frame": int(row["frame"]),
                    "x": round(float(row["center_x"]) / width, 7),
                    "y": round(float(row["center_y"]) / height, 7),
                    "confidence": round(float(row.get("confidence", 0.0)), 3),
                    "status": row.get("status", "unknown"),
                    "source": row.get("source", "unknown"),
                    "detection_status": row.get("detection_status", "detected"),
                    "evidence_weight": round(float(row.get("evidence_weight") or 1.0), 3),
                    "flight_id": str(row.get("flight_id", "")).strip() or None,
                }
            )
    annotation_fields = ("id", "time_seconds", "x_normalized", "y_normalized", "action", "note")
    return {
        "project_id": video_id(state.library_root, state.video),
        "available": status["generated"],
        **status,
        "path": str(artifacts.shuttle_annotations),
        "detections": detections,
        "annotations": [
            {key: row[key] for key in annotation_fields}
            for row in load_shuttle_annotations(artifacts.shuttle_annotations)
        ],
    }
