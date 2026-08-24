from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from importlib.resources import files
from itertools import pairwise
from pathlib import Path
from string import ascii_uppercase
from typing import Any

import cv2
import pandas as pd

from .analytics import ANALYTICS_SCHEMA_VERSION, analyze_rally_actions
from .audio import analyze_audio
from .contacts import analyze_contacts
from .court_events import analyze_terminal_events
from .encoding import (
    choose_working_video_encoder,
    h264_encoding_arguments,
    run_ffmpeg_with_encoder_fallback,
)
from .features import extract_features
from .geometry import CourtGeometry
from .hybrid import fuse_shuttle_detections
from .io import load_rallies, read_rows, resolve_ffmpeg
from .model import predict_model
from .model_assets import resolve_model
from .model_registry import ModelRegistry
from .phase_examples import capture_phase_examples
from .pose_overlay import render_pose_overlay
from .project_layout import LibraryLayout, ProjectLayout, find_project_root, video_key
from .rally_evidence import build_rally_evidence
from .render import render_rallies
from .score_labeling import (
    label_score_evidence,
    load_machine_labels,
    prepare_score_evidence,
    save_machine_label_review,
    score_labeling_config,
)
from .score_learning import default_score_model_path, fit_score_evidence_model
from .scoring import (
    analyze_score,
    load_score_corrections,
    save_score_corrections,
    validate_score_corrections,
)
from .segmenter import segment_rallies
from .serve_inference import infer_serve_observations
from .shuttle import detect_shuttle
from .shuttle_annotations import (
    load_shuttle_annotations,
    save_shuttle_annotations,
    validate_shuttle_annotations,
)
from .tracknet import TrackNetRuntimeConfig, detect_tracknet
from .trajectory import analyze_shuttle_trajectory


def _run_shuttle_trajectory(
    detections_csv: Path,
    output_csv: Path,
    contact_features_csv: Path | None,
    video: Path | None,
    annotations_csv: Path | None,
    config: Path | None,
) -> dict[str, Any]:
    return analyze_shuttle_trajectory(
        detections_csv,
        output_csv,
        contact_features_csv,
        video,
        annotations_csv,
        config,
    )


@dataclass
class StudioState:
    video: Path
    rallies: Path
    output: Path
    proxy: Path | None = None
    ffmpeg: Path | None = None
    encoder: str = "auto"
    quality: int = 21
    library: Path | None = None
    output_directory: Path | None = None
    config: Path | None = None
    model: Path | None = None
    pose_model: Path | None = None
    shuttle_model: Path | None = None
    tracknet_model: Path | None = None
    inpaint_model: Path | None = None
    shuttle_mode: str = "hybrid"
    packages: Path | None = None
    tracknet_packages: Path | None = None
    analysis_options: dict[str, float | bool] = field(
        default_factory=lambda: {
            "preroll": 0.35,
            "postroll": 0.55,
            "end_pending": 0.55,
            "maximum_internal_gap": 1.2,
            "suppress_handoffs": True,
        }
    )
    render_status: dict[str, Any] = field(default_factory=lambda: {"state": "idle"})
    analysis_status: dict[str, Any] = field(default_factory=lambda: {"state": "idle"})
    render_lock: threading.Lock = field(default_factory=threading.Lock)
    analysis_lock: threading.Lock = field(default_factory=threading.Lock)
    project_lock: threading.Lock = field(default_factory=threading.Lock)
    pose_lock: threading.Lock = field(default_factory=threading.Lock)
    runtime_cache: dict[str, Any] = field(default_factory=dict)
    payload_cache: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] = field(default_factory=dict)
    payload_cache_lock: threading.Lock = field(default_factory=threading.Lock)


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi"}
API_SCHEMA_VERSION = 8
EVIDENCE_SCHEMA_VERSION = 1
VISIBLE_SHUTTLE_STATUSES = {"tracked", "recovered", "competing", "manual"}
SHUTTLE_MODES = {"yolo", "tracknet", "hybrid"}
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


def _path_signature(paths: tuple[Path | None, ...]) -> tuple[Any, ...]:
    signature: list[Any] = []
    for path in paths:
        if path is None:
            signature.append(None)
            continue
        resolved = path.resolve()
        try:
            stat = resolved.stat()
            signature.append((str(resolved), stat.st_mtime_ns, stat.st_size))
        except FileNotFoundError:
            signature.append((str(resolved), None, None))
    return tuple(signature)


def _cached_payload(
    state: StudioState,
    key: str,
    paths: tuple[Path | None, ...],
    builder: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    signature = _path_signature(paths)
    with state.payload_cache_lock:
        cached = state.payload_cache.get(key)
        if cached is not None and cached[0] == signature:
            return cached[1]
    payload = builder()
    with state.payload_cache_lock:
        state.payload_cache[key] = (_path_signature(paths), payload)
    return payload


CALIBRATION_REGION_DEFINITIONS = {
    "active_court_polygon": {"label": "有效比赛场地", "color": "#ff4545", "required": True},
    "near_player_zone": {"label": "近场脚点区域", "color": "#6ee78f", "required": True},
    "far_player_zone": {"label": "远场脚点区域", "color": "#62c8ff", "required": True},
    "net_band": {"label": "球网区域", "color": "#e97cff", "required": True},
    "shuttle_airspace_polygon": {"label": "羽球飞行空域", "color": "#ffd84d", "required": True},
}
PERSPECTIVE_AXIS_DEFINITION = {
    "label": "羽球透视轴（高空中心 → 场地中心）",
    "color": "#3bf2df",
    "required": False,
    "minimum_points": 2,
    "maximum_points": 2,
}
COURT_CORNERS_DEFINITION = {
    "label": "单打场地四角（近左 → 近右 → 远右 → 远左）",
    "color": "#ffb45c",
    "required": False,
    "minimum_points": 4,
    "maximum_points": 4,
}
EXCLUDED_VIDEO_DIRECTORIES = {
    ".git",
    ".tools",
    "analysis",
    "archive",
    "documentation",
    "edited",
    "metadata",
    "models",
    "proxy",
    "standard_answers",
    "thirdparty",
}
EXCLUDED_VIDEO_MARKERS = ("_edited", "_final", "_proxy", "_review")


def _video_key(video: Path) -> str:
    return video_key(video)


def _discover_videos(library: Path) -> list[Path]:
    if not library.is_dir():
        raise ValueError(f"Video folder does not exist: {library}")
    videos = []
    for path in library.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        relative = path.relative_to(library)
        if any(part.lower() in EXCLUDED_VIDEO_DIRECTORIES for part in relative.parts[:-1]):
            continue
        if any(marker in path.stem.lower() for marker in EXCLUDED_VIDEO_MARKERS):
            continue
        videos.append(path)
    return sorted(videos, key=lambda path: path.relative_to(library).as_posix().lower())


def _project_root(library: Path, video: Path) -> Path:
    return find_project_root(library, video)


def _video_id(library: Path, video: Path) -> str:
    return video.relative_to(library).as_posix()


def _proxy_for(video: Path, library: Path | None = None) -> Path | None:
    return ProjectLayout.for_video(library or video.parent, video).existing_proxy()


def _working_timeline_path(library: Path, video: Path) -> Path:
    return ProjectLayout.for_video(library, video).working_timeline


def _ground_truth_path(library: Path, video: Path) -> Path:
    return ProjectLayout.for_video(library, video).ground_truth


def _latest_auto_cut_path(library: Path, video: Path) -> Path:
    return ProjectLayout.for_video(library, video).latest_auto_cut


def _timeline_has_segments(path: Path) -> bool:
    try:
        return path.exists() and bool(read_rows(path))
    except (OSError, KeyError):
        return False


def _project_has_completed_marker(library: Path, video: Path) -> bool:
    project_root = _project_root(library, video)
    if project_root == library:
        return _ground_truth_path(library, video).exists()
    legacy_timeline = project_root / "Metadata" / "rallies.csv"
    edited = project_root / "Edited"
    return legacy_timeline.exists() or (edited.exists() and any(path.suffix.lower() in VIDEO_EXTENSIONS for path in edited.iterdir()))


def _ensure_working_timeline(library: Path, video: Path) -> Path:
    timeline = _working_timeline_path(library, video)
    if timeline.exists():
        return timeline
    timeline.parent.mkdir(parents=True, exist_ok=True)
    seed = next(
        (
            candidate
            for candidate in (
                _latest_auto_cut_path(library, video),
                _ground_truth_path(library, video),
            )
            if candidate.exists()
        ),
        None,
    )
    if seed is not None:
        shutil.copy2(seed, timeline)
        timeline.chmod(timeline.stat().st_mode | 0o200)
    else:
        _save_segments(timeline, [])
    return timeline


def _default_output(library: Path, video: Path) -> Path:
    return ProjectLayout.for_video(library, video).default_output


def _filesystem_roots() -> list[Path]:
    if os.name == "nt":
        return [Path(f"{letter}:\\") for letter in ascii_uppercase if Path(f"{letter}:\\").is_dir()]
    roots = [Path("/"), Path.home().resolve()]
    return list(dict.fromkeys(roots))


def _nearest_existing_directory(path: Path) -> Path:
    candidate = path.expanduser()
    while not candidate.is_dir() and candidate != candidate.parent:
        candidate = candidate.parent
    if candidate.is_dir():
        return candidate.resolve()
    return Path.home().resolve()


def _directory_browser_payload(requested: str | None, fallback: Path) -> dict[str, Any]:
    current = _nearest_existing_directory(Path(requested) if requested else fallback)
    directories: list[dict[str, str]] = []
    try:
        children = sorted(current.iterdir(), key=lambda path: path.name.casefold())
    except OSError as error:
        raise ValueError(f"Cannot read directory: {current}: {error}") from error
    for child in children:
        try:
            if child.is_dir():
                directories.append({"name": child.name, "path": str(child.resolve())})
        except OSError:
            continue
    parent = current.parent if current.parent != current else None
    return {
        "path": str(current),
        "parent": str(parent) if parent else None,
        "home": str(Path.home().resolve()),
        "roots": [{"name": path.anchor or str(path), "path": str(path)} for path in _filesystem_roots()],
        "directories": directories,
    }


def _output_payload(state: StudioState) -> dict[str, Any]:
    directory = state.output.parent
    return {
        "directory": str(directory),
        "filename": state.output.name,
        "path": str(state.output),
        "directory_exists": directory.is_dir(),
        "file_exists": state.output.exists(),
    }


def _runtime_payload(state: StudioState) -> dict[str, Any]:
    if state.runtime_cache:
        return state.runtime_cache
    ffmpeg: Path | None = None
    try:
        ffmpeg = resolve_ffmpeg(state.ffmpeg)
        selected_encoder, warning, probe_failures = choose_working_video_encoder(state.encoder, ffmpeg)
        state.runtime_cache = {
            "ffmpeg": {
                "available": True,
                "path": str(ffmpeg),
                "reason": None,
                "requested_encoder": state.encoder,
                "selected_encoder": selected_encoder,
                "warning": warning,
                "probe_failures": probe_failures,
            }
        }
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        state.runtime_cache = {
            "ffmpeg": {
                "available": False,
                "path": str(ffmpeg) if ffmpeg else None,
                "reason": str(error),
                "requested_encoder": state.encoder,
                "selected_encoder": None,
                "warning": None,
                "probe_failures": {},
            }
        }
    return state.runtime_cache


def _activate_video(state: StudioState, video: Path) -> None:
    library = state.library or video.parent
    timeline = _ensure_working_timeline(library, video)
    with state.project_lock:
        state.video = video
        state.proxy = _proxy_for(video, library)
        state.rallies = timeline
        default_output = _default_output(library, video)
        state.output = (state.output_directory / default_output.name) if state.output_directory else default_output
        state.render_status = {"state": "idle"}


def _library_payload(state: StudioState) -> dict[str, Any]:
    library = state.library or state.video.parent
    videos = _discover_videos(library)
    return {
        "path": str(library),
        "videos": [
            {
                "id": _video_id(library, video),
                "relative_path": _video_id(library, video),
                "project_name": _project_root(library, video).name,
                "name": video.name,
                "size_bytes": video.stat().st_size,
                "proxy_available": _proxy_for(video, library) is not None,
                "timeline_available": _timeline_has_segments(_working_timeline_path(library, video)),
                "latest_auto_cut_available": _timeline_has_segments(_latest_auto_cut_path(library, video)),
                "ground_truth_available": _ground_truth_path(library, video).exists(),
                "completed_marker": _project_has_completed_marker(library, video),
                "active": video.resolve() == state.video.resolve(),
            }
            for video in videos
        ],
    }


def _analysis_directory(library: Path, video: Path) -> Path:
    return ProjectLayout.for_video(library, video).analysis.root


def _vision_features_path(analysis_root: Path) -> Path:
    return analysis_root / "vision-features.csv"


def _visual_evidence_features_path(analysis_root: Path) -> Path:
    vision_features = _vision_features_path(analysis_root)
    return vision_features if vision_features.exists() else analysis_root / "smart-features.csv"


def _preserve_state_features(state_features: Path, vision_features: Path) -> None:
    """Seed a new project once without replacing an existing state-model input."""
    if not state_features.exists() and vision_features.exists():
        shutil.copy2(vision_features, state_features)


def _shuttle_annotations_path(library: Path, video: Path) -> Path:
    return ProjectLayout.for_video(library, video).analysis.shuttle_annotations


def _score_corrections_path(library: Path, video: Path) -> Path:
    return ProjectLayout.for_video(library, video).score_corrections


def _segmentation_adapter_path(library: Path, video: Path) -> Path:
    return ProjectLayout.for_video(library, video).segmentation_adapter


def _trajectory_needs_refinement(
    features_path: Path,
    trajectory_path: Path,
    annotations_path: Path,
) -> bool:
    if not trajectory_path.exists():
        return False
    trajectory_mtime = trajectory_path.stat().st_mtime
    return any(
        path.exists() and path.stat().st_mtime >= trajectory_mtime
        for path in (features_path, annotations_path)
    )


def _default_calibration_path(state: StudioState) -> Path:
    library = state.library or state.video.parent
    return library / "Calibration" / "court-config.json"


def _polygon_area(points: list[list[float]]) -> float:
    return abs(
        sum(
            points[index][0] * points[(index + 1) % len(points)][1]
            - points[(index + 1) % len(points)][0] * points[index][1]
            for index in range(len(points))
        )
        / 2.0
    )


def _validated_polygon(value: Any, label: str, required: bool = False) -> list[list[float]]:
    if value in (None, []) and not required:
        return []
    if not isinstance(value, list) or len(value) < 3 or len(value) > 64:
        raise ValueError(f"{label} requires between 3 and 64 points")
    points = []
    for point in value:
        if not isinstance(point, list | tuple) or len(point) != 2:
            raise ValueError(f"{label} contains an invalid point")
        x, y = float(point[0]), float(point[1])
        if not math.isfinite(x) or not math.isfinite(y) or not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
            raise ValueError(f"{label} points must be normalized between 0 and 1")
        points.append([round(x, 6), round(y, 6)])
    if _polygon_area(points) < 0.0001:
        raise ValueError(f"{label} has almost no area")
    return points


def _validated_line(value: Any, label: str) -> list[list[float]]:
    if value in (None, []):
        return []
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{label} requires exactly 2 points")
    points = []
    for point in value:
        if not isinstance(point, list | tuple) or len(point) != 2:
            raise ValueError(f"{label} contains an invalid point")
        x, y = float(point[0]), float(point[1])
        if not math.isfinite(x) or not math.isfinite(y) or not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
            raise ValueError(f"{label} points must be normalized between 0 and 1")
        points.append([round(x, 6), round(y, 6)])
    if math.dist(points[0], points[1]) < 0.05 or points[1][1] - points[0][1] < 0.05:
        raise ValueError(f"{label} must run from a higher point to a lower point")
    return points


def _validated_court_corners(value: Any, label: str) -> list[list[float]]:
    if value in (None, []):
        return []
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{label} requires exactly 4 ordered points")
    points = []
    for point in value:
        if not isinstance(point, list | tuple) or len(point) != 2:
            raise ValueError(f"{label} contains an invalid point")
        x, y = float(point[0]), float(point[1])
        if not math.isfinite(x) or not math.isfinite(y) or not 0 <= x <= 1 or not 0 <= y <= 1:
            raise ValueError(f"{label} points must be normalized between 0 and 1")
        points.append([round(x, 6), round(y, 6)])
    contour = pd.DataFrame(points).to_numpy(dtype="float32")
    if not cv2.isContourConvex(contour.astype("int32") if max(map(max, points)) > 2 else (contour * 10000).astype("int32")):
        raise ValueError(f"{label} must form one convex quadrilateral in the documented order")
    if _polygon_area(points) < 0.001:
        raise ValueError(f"{label} has almost no area")
    return points


def _load_calibration_data(state: StudioState) -> dict[str, Any]:
    if state.config and state.config.exists():
        return json.loads(state.config.read_text(encoding="utf-8"))
    return {}


def _calibration_regions(data: dict[str, Any]) -> list[dict[str, Any]]:
    masks = data.get("masks_normalized", {})
    regions = [
        {"id": key, **definition, "minimum_points": 3, "maximum_points": 64, "points": masks.get(key, [])}
        for key, definition in CALIBRATION_REGION_DEFINITIONS.items()
    ]
    regions.append(
        {
            "id": "court_corners",
            **COURT_CORNERS_DEFINITION,
            "points": data.get("calibration", {}).get("court_corners_normalized", []),
        }
    )
    regions.append(
        {
            "id": "shuttle_perspective_axis",
            **PERSPECTIVE_AXIS_DEFINITION,
            "points": data.get("calibration", {}).get("shuttle_perspective_axis_normalized", []),
        }
    )
    dynamic_definitions = {
        "background_court_polygons": ("背景排除区", "#909a94"),
        "static_false_positive_polygons": ("静态误检区", "#ff9a55"),
    }
    for region_type, (label, color) in dynamic_definitions.items():
        for index, points in enumerate(masks.get(region_type, []), 1):
            regions.append(
                {
                    "id": f"{region_type}:{index}",
                    "type": region_type,
                    "label": f"{label} {index}",
                    "color": color,
                    "required": False,
                    "points": points,
                }
            )
    return regions


def _calibration_ready_data(data: dict[str, Any]) -> bool:
    masks = data.get("masks_normalized", {})
    return all(
        len(masks.get(key, [])) >= 3
        for key, definition in CALIBRATION_REGION_DEFINITIONS.items()
        if definition.get("required")
    )


def _calibration_ready(state: StudioState) -> bool:
    try:
        return bool(state.config and state.config.exists() and _calibration_ready_data(_load_calibration_data(state)))
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _calibration_payload(state: StudioState) -> dict[str, Any]:
    data = _load_calibration_data(state)
    metadata = _video_metadata(state.video)
    regions = _calibration_regions(data)
    completed = sum(bool(region["points"]) for region in regions if region.get("required"))
    homography: dict[str, Any] = {"available": False, "reason": "four ordered court corners are not calibrated"}
    if state.config and state.config.exists():
        geometry = CourtGeometry.from_json(state.config)
        homography = geometry.homography_quality(int(metadata["width"]), int(metadata["height"]))
    return {
        "available": True,
        "path": str(state.config or _default_calibration_path(state)),
        "exists": bool(state.config and state.config.exists()),
        "ready": _calibration_ready_data(data),
        "required_completed": completed,
        "required_total": len(CALIBRATION_REGION_DEFINITIONS),
        "reference_frame": {
            "width": int(metadata["width"]),
            "height": int(metadata["height"]),
        },
        "regions": regions,
        "homography": homography,
    }


def _save_calibration(state: StudioState, payload: dict[str, Any]) -> tuple[Path, Path | None]:
    regions = payload.get("regions")
    if not isinstance(regions, list):
        raise TypeError("Calibration regions must be a list")
    by_id = {str(region.get("id")): region for region in regions if isinstance(region, dict)}
    masks: dict[str, Any] = {}
    for key, definition in CALIBRATION_REGION_DEFINITIONS.items():
        region = by_id.get(key)
        masks[key] = _validated_polygon(
            region.get("points") if region else None,
            str(definition["label"]),
            required=True,
        )
    masks["court_ground_polygon"] = masks["active_court_polygon"]
    for region_type, label in (
        ("background_court_polygons", "背景排除区"),
        ("static_false_positive_polygons", "静态误检区"),
    ):
        candidates = [
            region
            for region in regions
            if isinstance(region, dict)
            and (region.get("type") == region_type or str(region.get("id", "")).startswith(f"{region_type}:"))
        ]
        masks[region_type] = [
            _validated_polygon(region.get("points"), f"{label} {index}")
            for index, region in enumerate(candidates, 1)
            if region.get("points")
        ]
    existing = _load_calibration_data(state)
    perspective_axis = _validated_line(
        by_id.get("shuttle_perspective_axis", {}).get("points"),
        str(PERSPECTIVE_AXIS_DEFINITION["label"]),
    )
    court_corners = _validated_court_corners(
        by_id.get("court_corners", {}).get("points"),
        str(COURT_CORNERS_DEFINITION["label"]),
    )
    metadata = _video_metadata(state.video)
    config_data = {
        **existing,
        "name": existing.get("name", f"Studio calibration for {state.video.stem}"),
        "reference_frame": {
            "width": int(metadata["width"]),
            "height": int(metadata["height"]),
            "source_time_seconds": round(float(payload.get("source_time_seconds", 0.0)), 3),
        },
        "calibration": {
            **existing.get("calibration", {}),
            "mode": "studio interactive normalized polygons",
            "court_corners_normalized": court_corners or None,
            "shuttle_perspective_axis_normalized": perspective_axis,
            "shuttle_vanishing_extension": float(
                existing.get("calibration", {}).get("shuttle_vanishing_extension", 1.5)
            ),
            "note": "Created or edited interactively in Smart Badminton Studio.",
        },
        "masks_normalized": masks,
        "analysis": existing.get(
            "analysis",
            {
                "sample_fps": 10.0,
                "pose_fps": 10.0,
                "pose_confidence": 0.2,
                "audio_event_window_seconds": 0.12,
                "motion_threshold": 7,
                "serve_preroll_seconds": 0.35,
                "end_postroll_seconds": 0.55,
                "end_pending_seconds": 0.55,
                "maximum_internal_gap_seconds": 1.2,
                "suppress_handoffs": True,
            },
        ),
    }
    path = state.config or _default_calibration_path(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if path.exists():
        backup = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
            json.dump(config_data, destination, ensure_ascii=False, indent=2)
            destination.write("\n")
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    state.config = path
    return path, backup


def _build_analytics_payload(state: StudioState) -> dict[str, Any]:
    library = state.library or state.video.parent
    analysis_root = _analysis_directory(library, state.video)
    features = _visual_evidence_features_path(analysis_root)
    if not features.exists() or not state.rallies.exists():
        return {"available": False, "reason": "Analyze the video and create a rally timeline first."}
    output = analysis_root / "rally-actions.csv"
    summary_path = analysis_root / "action-summary.json"
    audio_events = analysis_root / "audio-events.csv"
    trajectory = analysis_root / "shuttle-track.csv"
    contacts = analysis_root / "rally-contacts.csv"
    contact_summary = analysis_root / "contact-summary.json"
    events = analysis_root / "rally-events.csv"
    event_summary = analysis_root / "event-summary.json"
    try:
        inputs = [features.stat().st_mtime, state.rallies.stat().st_mtime]
        if audio_events.exists():
            inputs.append(audio_events.stat().st_mtime)
        if state.config is not None and state.config.exists():
            inputs.append(state.config.stat().st_mtime)
        if trajectory.exists():
            contact_inputs = [features.stat().st_mtime, state.rallies.stat().st_mtime, trajectory.stat().st_mtime]
            if not contacts.exists() or contacts.stat().st_mtime < max(contact_inputs):
                analyze_contacts(features, state.rallies, contacts, contact_summary)
            if state.config is not None and state.config.exists():
                event_inputs = [
                    features.stat().st_mtime,
                    state.rallies.stat().st_mtime,
                    trajectory.stat().st_mtime,
                    contacts.stat().st_mtime,
                    state.config.stat().st_mtime,
                ]
                probabilities = analysis_root / "rally-probabilities.csv"
                if probabilities.exists():
                    event_inputs.append(probabilities.stat().st_mtime)
                if not events.exists() or events.stat().st_mtime < max(event_inputs):
                    analyze_terminal_events(
                        state.video,
                        state.config,
                        state.rallies,
                        trajectory,
                        events,
                        contacts,
                        event_summary,
                        probabilities,
                        features,
                    )
            inputs.append(contacts.stat().st_mtime)
            if events.exists():
                inputs.append(events.stat().st_mtime)
        newest_input = max(inputs)
        cached_summary = None
        if summary_path.exists() and summary_path.stat().st_mtime >= newest_input:
            cached_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if cached_summary is None or cached_summary.get("schema_version") != ANALYTICS_SCHEMA_VERSION:
            return analyze_rally_actions(
                features,
                state.rallies,
                output,
                summary_path,
                audio_events,
                contacts if contacts.exists() else None,
                events if events.exists() else None,
                trajectory if trajectory.exists() else None,
                state.config if state.config is not None and state.config.exists() else None,
            )
        return cached_summary
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        return {"available": False, "reason": str(error)}


def _analytics_payload(state: StudioState) -> dict[str, Any]:
    library = state.library or state.video.parent
    analysis_root = _analysis_directory(library, state.video)
    paths = (
        state.rallies,
        state.config,
        analysis_root / "smart-features.csv",
        _vision_features_path(analysis_root),
        analysis_root / "rally-probabilities.csv",
        analysis_root / "audio-events.csv",
        analysis_root / "shuttle-track.csv",
        analysis_root / "rally-contacts.csv",
        analysis_root / "rally-events.csv",
        analysis_root / "action-summary.json",
    )
    return _cached_payload(state, "analytics", paths, lambda: _build_analytics_payload(state))


def _serve_observations(state: StudioState, evidence: dict[str, Any]) -> list[dict[str, Any]]:
    if not evidence.get("available"):
        return []
    rallies = load_rallies(state.rallies)
    analysis_root = _analysis_directory(state.library or state.video.parent, state.video)
    features_path = _visual_evidence_features_path(analysis_root)
    trajectory_path = analysis_root / "shuttle-track.csv"
    if not features_path.exists() or not trajectory_path.exists():
        return []
    try:
        features = pd.read_csv(features_path)
        trajectory = pd.read_csv(trajectory_path)
        return infer_serve_observations(
            rallies,
            features,
            trajectory,
            evidence.get("serves") or [],
            evidence.get("contacts") or [],
        )
    except (OSError, ValueError, KeyError):
        return []


def _score_paths(state: StudioState) -> tuple[Path, Path, Path, Path]:
    library = state.library or state.video.parent
    layout = ProjectLayout.for_video(library, state.video)
    return (
        layout.score_corrections,
        layout.analysis.score_events,
        layout.analysis.score_state,
        layout.analysis.score_summary,
    )


def _score_label_paths(state: StudioState) -> tuple[Path, Path, Path]:
    library = state.library or state.video.parent
    layout = ProjectLayout.for_video(library, state.video)
    root = layout.analysis.score_labeling
    legacy_root = library / "Analysis" / "Score_Labeling"
    if not layout.is_match_project and legacy_root.exists() and not root.exists():
        root = legacy_root
    return root, root / "manifest.json", root / "machine-score-labels.json"


def _model_registry(state: StudioState) -> ModelRegistry:
    library = state.library or state.video.parent
    labeling = score_labeling_config()
    return ModelRegistry(
        rally_state=state.model,
        pose=state.pose_model,
        shuttle_yolo=state.shuttle_model,
        tracknet=state.tracknet_model,
        inpaint=state.inpaint_model,
        score_evidence=default_score_model_path(library),
        vision_models=tuple(labeling["models"]),
    )


def _score_labeling_payload(state: StudioState) -> dict[str, Any]:
    root, manifest_path, labels_path = _score_label_paths(state)
    config = score_labeling_config()
    labels = load_machine_labels(labels_path)
    status_counts = {status: 0 for status in ("consensus", "review", "accepted", "rejected")}
    suggestions = []
    for row in labels:
        status = str(row.get("status", "review"))
        if status in status_counts:
            status_counts[status] += 1
        suggestions.append(
            {
                "rally": int(row.get("rally", 0)),
                "status": status,
                "suggestion": row.get("suggestion", {}),
                "model": row.get("model"),
            }
        )
    return {
        "configured": bool(config["configured"]),
        "model": config["model"],
        "prepared": manifest_path.exists(),
        "directory": str(root),
        "total": len(labels),
        **status_counts,
        "suggestions": suggestions,
    }


def _score_payload(state: StudioState) -> dict[str, Any]:
    if not state.rallies.exists():
        return {"available": False, "generated": False, "stale": False, "reason": "请先完成剪辑时间表"}
    corrections, events, output, summary = _score_paths(state)
    if not summary.exists() or not output.exists():
        return {"available": False, "generated": False, "stale": False, "reason": "尚未计算比分"}
    try:
        analysis_root = _analysis_directory(state.library or state.video.parent, state.video)
        inputs = [state.rallies.stat().st_mtime]
        for path in (
            corrections,
            events,
            default_score_model_path(state.library or state.video.parent),
            _score_label_paths(state)[2],
            analysis_root / "smart-features.csv",
            analysis_root / "rally-probabilities.csv",
            analysis_root / "shuttle-track.csv",
        ):
            if path.exists():
                inputs.append(path.stat().st_mtime)
        if summary.stat().st_mtime < max(inputs):
            return {"available": False, "generated": True, "stale": True, "reason": "时间表或分析数据已更新"}
        result = json.loads(summary.read_text(encoding="utf-8"))
        result.update({"available": True, "generated": True, "stale": False})
        return result
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        return {"available": False, "generated": True, "stale": True, "reason": str(error)}


def _calculate_score_payload(state: StudioState, evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    if not state.rallies.exists():
        return {"available": False, "generated": False, "stale": False, "reason": "请先完成剪辑时间表"}
    corrections, events, output, summary = _score_paths(state)
    try:
        library = state.library or state.video.parent
        evidence_model_path = default_score_model_path(library)
        fit_score_evidence_model(library, evidence_model_path)
        project_root = _project_root(library, state.video)
        evidence_project = (
            str(project_root.relative_to(library)).replace("\\", "/") if project_root != library else state.video.stem
        )
        evidence_payload = evidence if evidence is not None else _evidence_payload(state)
        serve_observations = _serve_observations(state, evidence_payload)
        result = analyze_score(
            state.rallies,
            output,
            events if events.exists() else None,
            corrections if corrections.exists() else None,
            summary,
            serve_observations=serve_observations,
            evidence_model_path=evidence_model_path,
            evidence_project=evidence_project,
            machine_labels_path=_score_label_paths(state)[2],
        )
        result["serve_observations"] = serve_observations
        result.update({"available": True, "generated": True, "stale": False, "updated_at": time.time()})
        summary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result
    except (OSError, TypeError, ValueError, KeyError) as error:
        return {"available": False, "generated": False, "stale": False, "reason": str(error)}


def _boolean_spans(times: list[float], values: Any) -> list[list[float]]:
    if not times:
        return []
    step = float(pd.Series(times).diff().dropna().median()) if len(times) > 1 else 0.1
    step = step if math.isfinite(step) and step > 0 else 0.1
    spans: list[list[float]] = []
    start: float | None = None
    for index, active in enumerate(values):
        if bool(active) and start is None:
            start = times[index]
        if start is not None and (not bool(active) or index == len(times) - 1):
            end = times[index] + step if bool(active) and index == len(times) - 1 else times[index]
            spans.append([round(start, 3), round(end, 3)])
            start = None
    return spans


def _build_evidence_payload(state: StudioState) -> dict[str, Any]:
    analysis_root = _analysis_directory(state.library or state.video.parent, state.video)
    features_path = _visual_evidence_features_path(analysis_root)
    probabilities_path = analysis_root / "rally-probabilities.csv"
    trajectory_path = analysis_root / "shuttle-track.csv"
    contacts_path = analysis_root / "rally-contacts.csv"
    events_path = analysis_root / "rally-events.csv"
    if not features_path.exists() or not probabilities_path.exists():
        return {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "available": False,
            "reason": "Run automatic analysis to populate frame evidence.",
        }
    try:
        features = pd.read_csv(features_path)
        probabilities = pd.read_csv(probabilities_path)
        data = features.merge(
            probabilities[["time_seconds", "rally_probability"]],
            on="time_seconds",
            how="inner",
        )
        if data.empty:
            raise ValueError("Feature and probability timelines do not overlap")
        times = pd.to_numeric(data["time_seconds"], errors="coerce").fillna(0.0).tolist()
        probability = pd.to_numeric(data["rally_probability"], errors="coerce").fillna(0.0).to_numpy()
        evidence = build_rally_evidence(data, trajectory_path if trajectory_path.exists() else None)
        signals = {
            "model_keep": _boolean_spans(times, probability >= 0.30),
            "model_active": _boolean_spans(times, probability >= 0.56),
            "activity_high": _boolean_spans(times, evidence.activity >= 0.70),
            "trajectory_visible": _boolean_spans(times, evidence.trajectory_visible),
            "trajectory_descending": _boolean_spans(times, evidence.trajectory_descending),
            "trajectory_occluded": _boolean_spans(times, evidence.trajectory_occluded),
            "landing_candidate": _boolean_spans(times, evidence.landing_candidate),
            "near_ready": _boolean_spans(times, evidence.near_ready),
            "far_ready": _boolean_spans(times, evidence.far_ready),
            "between_points": _boolean_spans(times, evidence.between_points),
            "handoff": _boolean_spans(times, evidence.handoff_candidate),
        }
        serves = []
        for index, active in enumerate(evidence.formal_serve):
            if not bool(active):
                continue
            server = (
                "near"
                if bool(evidence.near_serving[index])
                else "far"
                if bool(evidence.far_serving[index])
                else "unknown"
            )
            serves.append(
                {
                    "time": round(times[index], 3),
                    "server": server,
                    "confidence": round(float(evidence.serve_confidence[index]), 3),
                }
            )
        contacts = []
        if contacts_path.exists():
            contacts = [
                {
                    "time": round(float(row["time_seconds"]), 3),
                    "player": row.get("player", "unknown"),
                    "confidence": round(float(row.get("confidence", 0.0)), 3),
                }
                for row in read_rows(contacts_path)
            ]
        terminal_events = []
        if events_path.exists():
            terminal_events = [
                {
                    "time": round(float(row["event_time_seconds"]), 3),
                    "event": row.get("event", "unknown"),
                    "confidence": round(float(row.get("confidence", 0.0)), 3),
                }
                for row in read_rows(events_path)
                if row.get("event", "unknown") != "unknown"
            ]
        return {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "available": True,
            "signals": signals,
            "serves": serves,
            "contacts": contacts,
            "terminal_events": terminal_events,
        }
    except (OSError, ValueError, KeyError) as error:
        return {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "available": False,
            "reason": str(error),
        }


def _evidence_payload(state: StudioState) -> dict[str, Any]:
    analysis_root = _analysis_directory(state.library or state.video.parent, state.video)
    paths = (
        analysis_root / "smart-features.csv",
        _vision_features_path(analysis_root),
        analysis_root / "rally-probabilities.csv",
        analysis_root / "shuttle-track.csv",
        analysis_root / "rally-contacts.csv",
        analysis_root / "rally-events.csv",
    )
    return _cached_payload(state, "evidence", paths, lambda: _build_evidence_payload(state))


def _shuttle_annotation_payload(state: StudioState) -> dict[str, Any]:
    library = state.library or state.video.parent
    analysis_root = _analysis_directory(library, state.video)
    trajectory_path = analysis_root / "shuttle-track.csv"
    annotation_path = _shuttle_annotations_path(library, state.video)
    shuttle_status = _shuttle_status_payload(state)
    metadata = _video_metadata(state.video)
    detections = []
    if trajectory_path.exists():
        for row in read_rows(trajectory_path):
            if row.get("status") not in VISIBLE_SHUTTLE_STATUSES:
                continue
            coordinate_width = float(row.get("source_width") or metadata["width"])
            coordinate_height = float(row.get("source_height") or metadata["height"])
            detections.append(
                {
                    "time": round(float(row["time_seconds"]), 4),
                    "frame": int(row["frame"]),
                    "x": round(float(row["center_x"]) / coordinate_width, 7),
                    "y": round(float(row["center_y"]) / coordinate_height, 7),
                    "confidence": round(float(row.get("confidence", 0.0)), 3),
                    "status": row.get("status", "unknown"),
                    "source": row.get("source", "unknown"),
                    "detection_status": row.get("detection_status", "detected"),
                    "evidence_weight": round(float(row.get("evidence_weight") or 1.0), 3),
                    "flight_id": str(row.get("flight_id", "")).strip() or None,
                }
            )
    annotations = [
        {
            "id": row["id"],
            "time_seconds": row["time_seconds"],
            "x_normalized": row["x_normalized"],
            "y_normalized": row["y_normalized"],
            "action": row["action"],
            "note": row["note"],
        }
        for row in load_shuttle_annotations(annotation_path)
    ]
    return {
        "project_id": _video_id(library, state.video),
        "available": shuttle_status["generated"],
        **shuttle_status,
        "path": str(annotation_path),
        "detections": detections,
        "annotations": annotations,
    }


def _available_shuttle_modes(state: StudioState) -> list[str]:
    return _model_registry(state).available_shuttle_modes()


def _effective_shuttle_mode(state: StudioState) -> str | None:
    available = _available_shuttle_modes(state)
    if state.shuttle_mode in available:
        return state.shuttle_mode
    if "hybrid" in available:
        return "hybrid"
    return available[0] if available else None


def _shuttle_detection_paths(analysis_root: Path) -> dict[str, Path]:
    return {
        "yolo": analysis_root / "shuttle-yolo-raw.csv",
        "tracknet": analysis_root / "shuttle-tracknet-raw.csv",
        "hybrid": analysis_root / "shuttle-raw.csv",
    }


def _shuttle_detection_metadata_path(analysis_root: Path) -> Path:
    return analysis_root / "shuttle-detection.json"


def _run_shuttle_detection(
    state: StudioState,
    video: Path,
    analysis_root: Path,
    force: bool,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    mode = _effective_shuttle_mode(state)
    if mode is None or state.config is None:
        raise RuntimeError("Configure a shuttle detector and court calibration first")
    paths = _shuttle_detection_paths(analysis_root)
    final_path = paths["hybrid"]

    def progress(value: float, label: str) -> None:
        if progress_callback is not None:
            progress_callback(max(0.0, min(1.0, value)), label)

    results: dict[str, Any] = {"mode": mode, "output": str(final_path)}
    if mode in {"yolo", "hybrid"}:
        if state.shuttle_model is None:
            raise RuntimeError("YOLO shuttle model is not configured")
        yolo_output = paths["yolo"] if mode == "hybrid" else final_path
        if force or not yolo_output.exists():
            results["yolo"] = detect_shuttle(
                video,
                state.config,
                state.shuttle_model,
                state.packages,
                yolo_output,
                sample_fps=15.0,
                confidence=0.04,
                image_size=1280,
                progress_callback=lambda value: progress(value * (0.38 if mode == "hybrid" else 0.92), "YOLO"),
            )
    if mode in {"tracknet", "hybrid"}:
        if state.tracknet_model is None:
            raise RuntimeError("TrackNet model is not configured")
        tracknet_output = paths["tracknet"] if mode == "hybrid" else final_path
        if force or not tracknet_output.exists():
            start = 0.40 if mode == "hybrid" else 0.0
            scale = 0.52 if mode == "hybrid" else 0.92
            results["tracknet"] = detect_tracknet(
                video,
                state.config,
                state.tracknet_model,
                tracknet_output,
                state.inpaint_model if state.inpaint_model and state.inpaint_model.exists() else None,
                state.tracknet_packages,
                TrackNetRuntimeConfig(),
                progress_callback=lambda value: progress(start + value * scale, "TrackNet"),
            )
    if mode == "hybrid":
        progress(0.94, "融合")
        results["hybrid"] = fuse_shuttle_detections(
            paths["yolo"],
            paths["tracknet"],
            state.config,
            final_path,
        )
    metadata = {
        "mode": mode,
        "video": str(video),
        "config": str(state.config),
        "yolo_model": str(state.shuttle_model) if state.shuttle_model else None,
        "tracknet_model": str(state.tracknet_model) if state.tracknet_model else None,
        "inpaint_model": str(state.inpaint_model) if state.inpaint_model else None,
        "yolo_model_mtime_ns": state.shuttle_model.stat().st_mtime_ns if state.shuttle_model else None,
        "tracknet_model_mtime_ns": state.tracknet_model.stat().st_mtime_ns if state.tracknet_model else None,
        "inpaint_model_mtime_ns": state.inpaint_model.stat().st_mtime_ns if state.inpaint_model else None,
        "completed_at": time.time(),
    }
    _shuttle_detection_metadata_path(analysis_root).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    progress(1.0, "完成")
    return results


def _build_shuttle_status_payload(state: StudioState) -> dict[str, Any]:
    library = state.library or state.video.parent
    analysis_root = _analysis_directory(library, state.video)
    raw_path = analysis_root / "shuttle-raw.csv"
    trajectory_path = analysis_root / "shuttle-track.csv"
    features_path = _visual_evidence_features_path(analysis_root)
    annotations_path = _shuttle_annotations_path(library, state.video)
    detection_metadata_path = _shuttle_detection_metadata_path(analysis_root)
    accepted_rows = []
    if trajectory_path.exists():
        accepted_rows = [row for row in read_rows(trajectory_path) if row.get("status") in VISIBLE_SHUTTLE_STATUSES]
    flight_ids = {str(row.get("flight_id", "")).strip() for row in accepted_rows}
    flight_ids.discard("")
    mode = _effective_shuttle_mode(state)
    configured = bool(state.config and state.config.exists() and mode)
    stale_sources: list[str] = []
    if trajectory_path.exists():
        trajectory_mtime = trajectory_path.stat().st_mtime_ns
        freshness_inputs = (
            (state.config, "球场校准已更改"),
            (raw_path, "原始检测已更新"),
            (features_path, "动作特征已更新"),
            (annotations_path, "人工羽球标注已更新"),
        )
        stale_sources = [
            reason
            for path, reason in freshness_inputs
            if path is not None and path.exists() and path.stat().st_mtime_ns > trajectory_mtime
        ]
    detection_metadata: dict[str, Any] = {}
    if detection_metadata_path.exists():
        try:
            detection_metadata = json.loads(detection_metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            detection_metadata = {}
    if raw_path.exists() and detection_metadata.get("mode") != mode:
        stale_sources.append("检测模式已更改")
    if raw_path.exists() and not detection_metadata_path.exists():
        stale_sources.append("检测缓存需要更新")
    model_inputs = (
        (state.shuttle_model if mode in {"yolo", "hybrid"} else None, "yolo_model_mtime_ns", "YOLO 模型已更新"),
        (
            state.tracknet_model if mode in {"tracknet", "hybrid"} else None,
            "tracknet_model_mtime_ns",
            "TrackNet 模型已更新",
        ),
        (
            state.inpaint_model if mode in {"tracknet", "hybrid"} else None,
            "inpaint_model_mtime_ns",
            "Inpaint 模型已更新",
        ),
    )
    for model_path, metadata_key, reason in model_inputs:
        if model_path is not None and detection_metadata.get(metadata_key) not in {None, model_path.stat().st_mtime_ns}:
            stale_sources.append(reason)
    stale = bool(stale_sources)
    return {
        "configured": configured,
        "generated": trajectory_path.exists(),
        "stale": stale,
        "current": bool(trajectory_path.exists() and not stale),
        "stale_reason": "；".join(stale_sources) if stale_sources else None,
        "raw_generated": raw_path.exists(),
        "point_count": len(accepted_rows),
        "flight_count": len(flight_ids),
        "track_path": str(trajectory_path),
        "mode": mode,
        "requested_mode": state.shuttle_mode,
        "available_modes": _available_shuttle_modes(state),
        "tracknet_configured": bool(state.tracknet_model and state.tracknet_model.exists()),
        "inpaint_configured": bool(state.inpaint_model and state.inpaint_model.exists()),
    }


def _shuttle_status_payload(state: StudioState) -> dict[str, Any]:
    library = state.library or state.video.parent
    analysis_root = _analysis_directory(library, state.video)
    paths = (
        state.config,
        state.shuttle_model,
        state.tracknet_model,
        state.inpaint_model,
        analysis_root / "shuttle-raw.csv",
        analysis_root / "shuttle-track.csv",
        analysis_root / "smart-features.csv",
        _vision_features_path(analysis_root),
        _shuttle_annotations_path(library, state.video),
        _shuttle_detection_metadata_path(analysis_root),
    )
    return _cached_payload(
        state,
        f"shuttle-status:{state.shuttle_mode}",
        paths,
        lambda: _build_shuttle_status_payload(state),
    )


def _set_analysis_status(studio_state: StudioState, **values: Any) -> None:
    studio_state.analysis_status = {
        **studio_state.analysis_status,
        **values,
        "updated_at": time.time(),
    }
    _persist_analysis_status(studio_state)


def _analysis_status_path(studio_state: StudioState) -> Path:
    root = studio_state.library or studio_state.video.parent
    return LibraryLayout(root).analysis_status


def _persist_analysis_status(studio_state: StudioState, force: bool = False) -> None:
    now = time.monotonic()
    last_write = float(studio_state.runtime_cache.get("analysis_status_write", 0.0))
    terminal = studio_state.analysis_status.get("state") in {"complete", "error"}
    if not force and not terminal and now - last_write < 0.5:
        return
    try:
        path = _analysis_status_path(studio_state)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(studio_state.analysis_status, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)
        studio_state.runtime_cache["analysis_status_write"] = now
    except OSError:
        return


def _begin_analysis_status(studio_state: StudioState, **values: Any) -> dict[str, Any]:
    now = time.time()
    studio_state.analysis_status = {
        "job_id": uuid.uuid4().hex[:12],
        "state": "running",
        "background": True,
        "started_at": now,
        "updated_at": now,
        **values,
    }
    _persist_analysis_status(studio_state, force=True)
    return studio_state.analysis_status


def _restore_analysis_status(studio_state: StudioState) -> None:
    path = _analysis_status_path(studio_state)
    if not path.exists():
        return
    try:
        restored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if not isinstance(restored, dict):
        return
    if restored.get("state") == "running":
        restored = {
            **restored,
            "state": "error",
            "label": "上次后台任务因 Studio 服务停止而中断",
            "message": "重新打开对应视频并再次启动分析；已完成的缓存步骤会被复用。",
            "updated_at": time.time(),
        }
    studio_state.analysis_status = restored


def _make_proxy(state: StudioState, library: Path, video: Path) -> Path:
    existing = _proxy_for(video, library)
    if existing is not None:
        return existing
    project_root = _project_root(library, video)
    proxy_root = (project_root if project_root != library else library / "Analysis" / "Auto" / video.stem) / "Proxy"
    proxy = proxy_root / f"{video.stem}_proxy_720p.mp4"
    proxy.parent.mkdir(parents=True, exist_ok=True)

    def command(encoder: str) -> list[str]:
        result = [
            str(resolve_ffmpeg(state.ffmpeg)),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-vf",
            "scale=-2:720:flags=lanczos,fps=30",
            *h264_encoding_arguments(encoder, 28, "proxy"),
        ]
        result += ["-g", "30", "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(proxy)]
        return result

    ffmpeg_path = resolve_ffmpeg(state.ffmpeg)
    selected_encoder = run_ffmpeg_with_encoder_fallback(ffmpeg_path, state.encoder, command, proxy)
    runtime = _runtime_payload(state)["ffmpeg"]
    if runtime.get("selected_encoder") != selected_encoder:
        runtime["warning"] = f"{runtime.get('selected_encoder')} could not complete the proxy; using {selected_encoder}"
        runtime["selected_encoder"] = selected_encoder
    return proxy


def _analyze_video(
    state: StudioState,
    library: Path,
    video: Path,
    index: int,
    total: int,
) -> dict[str, Any]:
    if state.config is None or state.model is None:
        raise ValueError("Automatic analysis requires both a court config and a trained model")
    analysis_root = _analysis_directory(library, video)
    analysis_root.mkdir(parents=True, exist_ok=True)
    audio_csv = analysis_root / "audio-events.csv"
    features_csv = analysis_root / "smart-features.csv"
    vision_features_csv = _vision_features_path(analysis_root)
    probabilities_csv = analysis_root / "rally-probabilities.csv"
    automatic_csv = analysis_root / "rallies-auto.csv"
    shuttle_raw_csv = analysis_root / "shuttle-raw.csv"
    shuttle_track_csv = analysis_root / "shuttle-track.csv"
    shuttle_annotations_csv = _shuttle_annotations_path(library, video)
    timeline = _working_timeline_path(library, video)

    def stage(name: str, label: str, progress: float) -> None:
        _set_analysis_status(
            state,
            state="running",
            stage=name,
            label=label,
            progress=((index - 1) + progress) / total,
            current=_video_id(library, video),
            current_name=video.name,
            completed=index - 1,
            total=total,
        )

    stage("proxy", "正在建立流畅预览代理", 0.05)
    proxy = _make_proxy(state, library, video)
    stage("audio", "正在分析击球声音", 0.18)
    if not audio_csv.exists():
        analyze_audio(video, audio_csv, ffmpeg=state.ffmpeg)
    if _effective_shuttle_mode(state) is not None:
        stage("shuttle", "正在检测并连接本场羽球轨迹", 0.24)
        if not shuttle_raw_csv.exists():
            _run_shuttle_detection(
                state,
                video,
                analysis_root,
                False,
                lambda value, detector: stage(
                    "shuttle", f"正在运行 {detector} 羽球检测", 0.24 + value * 0.05
                ),
            )
        if not shuttle_track_csv.exists():
            _run_shuttle_trajectory(
                shuttle_raw_csv,
                shuttle_track_csv,
                _visual_evidence_features_path(analysis_root)
                if _visual_evidence_features_path(analysis_root).exists()
                else None,
                video,
                shuttle_annotations_csv,
                state.config,
            )
    stage("features", "正在理解球场、球员动作与运动轨迹", 0.30)
    feature_columns = set()
    if vision_features_csv.exists():
        with vision_features_csv.open(encoding="utf-8-sig") as existing_features:
            feature_columns = set(next(csv.reader(existing_features), []))
    if not vision_features_csv.exists() or not POSE_ASSOCIATION_FIELDS.issubset(feature_columns):
        extract_features(
            video,
            state.config,
            vision_features_csv,
            audio_csv,
            shuttle_track_csv if shuttle_track_csv.exists() else None,
            state.pose_model,
            state.packages,
            None,
            0.0,
            None,
            lambda progress: stage(
                "features",
                "正在理解球场、球员动作与运动轨迹",
                0.30 + max(0.0, min(1.0, progress)) * 0.52,
            ),
        )
    if (
        _effective_shuttle_mode(state) is not None
        and shuttle_raw_csv.exists()
        and _trajectory_needs_refinement(vision_features_csv, shuttle_track_csv, shuttle_annotations_csv)
    ):
        stage("shuttle-contact", "正在用球拍接触证据复核竞争轨迹", 0.83)
        _run_shuttle_trajectory(
            shuttle_raw_csv,
            shuttle_track_csv,
            vision_features_csv,
            video,
            shuttle_annotations_csv,
            state.config,
        )
    _preserve_state_features(features_csv, vision_features_csv)
    stage("predict", "正在用标准答案模型判断每一球", 0.84)
    predict_model(features_csv, state.model, probabilities_csv)
    stage("segment", "正在生成保守、不漏球的时间表", 0.94)
    adapter_path = _segmentation_adapter_path(library, video)
    model_adapter_path = state.model.with_suffix(".adapter.json")
    selected_adapter = adapter_path if adapter_path.exists() else model_adapter_path
    trajectory_policy = "integrated"
    if model_adapter_path.exists():
        model_profile = json.loads(model_adapter_path.read_text(encoding="utf-8"))
        trajectory_policy = str(model_profile.get("trajectory_policy", trajectory_policy))
    intervals = segment_rallies(
        features_csv,
        probabilities_csv,
        automatic_csv,
        start_threshold=0.56,
        keep_threshold=0.30,
        preroll=float(state.analysis_options["preroll"]),
        postroll=float(state.analysis_options["postroll"]),
        end_pending=float(state.analysis_options["end_pending"]),
        maximum_internal_gap=float(state.analysis_options["maximum_internal_gap"]),
        suppress_handoffs=bool(state.analysis_options["suppress_handoffs"]),
        shuttle_trajectory_csv=shuttle_track_csv if shuttle_track_csv.exists() else None,
        adapter=selected_adapter if selected_adapter.exists() else None,
        trajectory_policy=trajectory_policy,
    )
    metadata = _video_metadata(video)
    public_rows = _public_segments(automatic_csv)
    rows = _validated_segments(public_rows, float(metadata["duration"]))
    _save_segments(timeline, rows)
    with state.project_lock:
        if state.video.resolve() == video.resolve():
            state.rallies = timeline
            state.proxy = proxy
    return {
        "video": _video_id(library, video),
        "timeline": str(timeline),
        "automatic_timeline": str(automatic_csv),
        "rallies": len(intervals),
        "proxy": str(proxy),
    }


@lru_cache(maxsize=32)
def _read_video_metadata(video_path: str, mtime_ns: int, size: int) -> tuple[float | int | str, ...]:
    del mtime_ns, size
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    return Path(video_path).name, frame_count / fps, fps, frame_count, width, height


def _video_metadata(video: Path) -> dict[str, float | int | str]:
    resolved = video.resolve()
    stat = resolved.stat()
    name, duration, fps, frame_count, width, height = _read_video_metadata(
        str(resolved), stat.st_mtime_ns, stat.st_size
    )
    return {
        "name": name,
        "duration": duration,
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
    }


def _public_segments(path: Path) -> list[dict[str, Any]]:
    segments = []
    for index, row in enumerate(read_rows(path), 1):
        segments.append(
            {
                "id": str(row.get("rally", index)),
                "start": float(row["start_seconds"]),
                "end": float(row["end_seconds"]),
                "confidence": row.get("confidence", ""),
                "review_required": row.get("review_required", "no"),
                "boundary_reason": row.get("boundary_reason", row.get("notes", "")),
            }
        )
    return segments


def _validated_segments(payload: Any, duration: float) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        raise TypeError("segments must be a list")
    result = []
    for index, item in enumerate(payload, 1):
        if not isinstance(item, dict):
            raise TypeError(f"segment {index} is invalid")
        start, end = float(item["start"]), float(item["end"])
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError(f"segment {index} has a non-finite boundary")
        if start < 0 or end > duration + 0.05 or end - start < 0.05:
            raise ValueError(f"segment {index} is outside the video or too short")
        result.append(
            {
                "rally": index,
                "start_seconds": f"{start:.3f}",
                "end_seconds": f"{min(end, duration):.3f}",
                "confidence": str(item.get("confidence", "manual")),
                "review_required": str(item.get("review_required", "no")),
                "boundary_reason": str(item.get("boundary_reason", "edited in studio")),
            }
        )
    result.sort(key=lambda row: float(row["start_seconds"]))
    for previous, current in pairwise(result):
        if float(previous["end_seconds"]) > float(current["start_seconds"]) + 0.001:
            raise ValueError("segments may touch but may not overlap")
    for index, row in enumerate(result, 1):
        row["rally"] = index
    return result


def _save_segments(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = path.with_suffix(path.suffix + ".bak")
    if path.exists():
        shutil.copy2(path, backup)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=".csv", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(
                output,
                fieldnames=[
                    "rally",
                    "start_seconds",
                    "end_seconds",
                    "confidence",
                    "review_required",
                    "boundary_reason",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return backup


def _pose_overlay_path(state: StudioState, rally_number: int) -> tuple[Path, dict[str, Any]]:
    segments = _public_segments(state.rallies)
    if rally_number < 1 or rally_number > len(segments):
        raise ValueError(f"Rally must be between 1 and {len(segments)}")
    segment = segments[rally_number - 1]
    start, end = float(segment["start"]), float(segment["end"])
    analysis_root = _analysis_directory(state.library or state.video.parent, state.video)
    output = analysis_root / "Pose_Overlays" / (
        f"pose_v3_rally_{rally_number:03d}_{round(start * 1000):09d}_{round(end * 1000):09d}.mp4"
    )
    return output, segment


def _full_pose_overlay_path(state: StudioState) -> Path:
    analysis_root = _analysis_directory(state.library or state.video.parent, state.video)
    return analysis_root / "Pose_Overlays" / "pose_v4_full_video.mp4"


def _pose_status_payload(state: StudioState) -> dict[str, Any]:
    output = _full_pose_overlay_path(state)
    configured = bool(
        _calibration_ready(state)
        and state.pose_model
        and state.pose_model.exists()
        and _runtime_payload(state)["ffmpeg"]["available"]
    )
    stale_sources: list[str] = []
    if output.exists():
        overlay_mtime = output.stat().st_mtime_ns
        freshness_inputs = (
            (state.video, "原视频已更新"),
            (state.config, "球场校准已更改"),
            (state.pose_model, "姿态模型已更新"),
        )
        stale_sources = [
            reason
            for path, reason in freshness_inputs
            if path is not None and path.exists() and path.stat().st_mtime_ns > overlay_mtime
        ]
    stale = bool(stale_sources)
    return {
        "configured": configured,
        "generated": output.exists(),
        "stale": stale,
        "current": bool(output.exists() and not stale),
        "stale_reason": "；".join(stale_sources) if stale_sources else None,
        "path": str(output),
        "url": f"/media/pose-overlay/full?v={output.stat().st_mtime_ns}" if output.exists() else None,
    }


def create_studio_app(state: StudioState):
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import FileResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as error:
        raise RuntimeError('Studio dependencies are missing. Install with: pip install -e ".[studio]"') from error

    static_root = Path(str(files("smart_badminton").joinpath("studio_static")))
    _restore_analysis_status(state)
    app = FastAPI(title="Smart Badminton Studio", docs_url=None, redoc_url=None)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:3000", "http://localhost:3000"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.mount("/static", StaticFiles(directory=static_root), name="static")

    @app.get("/")
    def index():
        return FileResponse(static_root / "index.html")

    @app.get("/media/video")
    def video():
        return FileResponse(state.proxy or state.video, media_type="video/mp4")

    @app.get("/api/health")
    def health():
        return {"ok": True, "api_schema_version": API_SCHEMA_VERSION, "runtime": _runtime_payload(state)}

    @app.get("/api/tutorial")
    def tutorial():
        tutorial_path = static_root / "studio-tutorial.zh-CN.md"
        if not tutorial_path.exists():
            # Editable source checkouts can run the API before the frontend has
            # copied documentation into the packaged static directory.
            source_tutorial = Path(__file__).resolve().parent.parent / "docs" / "studio-tutorial.zh-CN.md"
            if source_tutorial.exists():
                tutorial_path = source_tutorial
        if not tutorial_path.exists():
            raise HTTPException(status_code=404, detail="Studio tutorial is not packaged")
        return {"markdown": tutorial_path.read_text(encoding="utf-8")}

    def project_payload() -> dict[str, Any]:
        metadata = _video_metadata(state.video)
        preview_video = state.proxy or state.video
        library_root = state.library or state.video.parent
        analytics = _analytics_payload(state)
        evidence = _evidence_payload(state)
        runtime = _runtime_payload(state)
        shuttle_analysis = _shuttle_status_payload(state)
        pose_analysis = _pose_status_payload(state)
        ffmpeg_available = bool(runtime["ffmpeg"]["available"])
        return {
            "id": _video_id(library_root, state.video),
            "video": metadata,
            "preview": _video_metadata(preview_video),
            "segments": _public_segments(state.rallies),
            "timeline_path": str(state.rallies),
            "output_path": str(state.output),
            "output": _output_payload(state),
            "runtime": runtime,
            "models": _model_registry(state).public_payload(),
            "shuttle_analysis": shuttle_analysis,
            "pose_analysis": pose_analysis,
            "api_schema_version": API_SCHEMA_VERSION,
            "automatic_analysis": {
                "configured": bool(_calibration_ready(state) and state.model and ffmpeg_available),
                "pose_overlay_configured": bool(_calibration_ready(state) and state.pose_model and ffmpeg_available),
                "configuration_issue": runtime["ffmpeg"]["reason"] if not ffmpeg_available else None,
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
            "calibration": {
                "ready": _calibration_ready(state),
                "path": str(state.config or _default_calibration_path(state)),
            },
            "analytics": analytics,
            "score": _score_payload(state),
            "score_labeling": _score_labeling_payload(state),
            "evidence": evidence,
        }

    @app.get("/api/project")
    def project():
        return project_payload()

    @app.get("/api/library")
    def library():
        return _library_payload(state)

    @app.get("/api/filesystem/directories")
    def browse_directories(path: str | None = None):
        try:
            return _directory_browser_payload(path, state.library or state.video.parent)
        except (OSError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/api/library")
    async def change_library(payload: dict[str, Any]):
        if state.analysis_status.get("state") == "running":
            raise HTTPException(status_code=409, detail="Wait for automatic analysis to finish before changing folders")
        try:
            candidate = Path(str(payload["path"])).expanduser().resolve()
            videos = _discover_videos(candidate)
            if not videos:
                raise ValueError("No supported video files found in this folder")
            state.library = candidate
            return _library_payload(state)
        except (KeyError, OSError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/api/project/open")
    async def open_project(payload: dict[str, Any]):
        if state.render_status.get("state") == "running":
            raise HTTPException(status_code=409, detail="Wait for the current render to finish before switching videos")
        if state.analysis_status.get("state") == "running":
            raise HTTPException(status_code=409, detail="Wait for automatic analysis to finish before switching videos")
        library_root = state.library or state.video.parent
        requested = str(payload.get("id", ""))
        match = next(
            (
                video
                for video in _discover_videos(library_root)
                if _video_id(library_root, video) == requested or video.stem == requested
            ),
            None,
        )
        if match is None:
            raise HTTPException(status_code=404, detail="Video not found in the selected folder")
        _activate_video(state, match)
        return project_payload()

    @app.put("/api/output")
    async def change_output(payload: dict[str, Any]):
        library_root = state.library or state.video.parent
        if str(payload.get("project_id", "")) != _video_id(library_root, state.video):
            raise HTTPException(status_code=409, detail="The open project changed; reload before changing output")
        if state.render_status.get("state") == "running":
            raise HTTPException(status_code=409, detail="Wait for the current render to finish before changing output")
        try:
            directory = Path(str(payload["directory"])).expanduser().resolve()
            if not directory.is_dir():
                raise ValueError(f"Output folder does not exist: {directory}")
            if not os.access(directory, os.W_OK):
                raise ValueError(f"Output folder is not writable: {directory}")
            filename = str(payload.get("filename") or state.output.name).strip()
            if not filename or Path(filename).name != filename:
                raise ValueError("Output filename must not contain a folder path")
            if Path(filename).suffix.lower() != ".mp4":
                raise ValueError("Output filename must end with .mp4")
            state.output_directory = directory
            state.output = directory / filename
            state.render_status = {"state": "idle"}
            return {"ok": True, **_output_payload(state)}
        except (KeyError, OSError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.put("/api/timeline")
    async def save_timeline(payload: dict[str, Any]):
        try:
            library_root = state.library or state.video.parent
            active_project = _video_id(library_root, state.video)
            if str(payload.get("project_id", "")) != active_project:
                raise ValueError("The open project changed; reload before saving this timeline")
            metadata = _video_metadata(state.video)
            rows = _validated_segments(payload.get("segments"), float(metadata["duration"]))
            backup = _save_segments(state.rallies, rows)
            analysis_root = _analysis_directory(library_root, state.video)
            phase_examples = capture_phase_examples(
                active_project,
                rows,
                analysis_root / "rallies-auto.csv",
                analysis_root / "Boundary_Corrections" / "user-phase-examples.csv",
                analysis_root / "smart-features.csv",
                analysis_root / "rally-probabilities.csv",
                analysis_root / "shuttle-track.csv",
            )
            return {
                "ok": True,
                "segments": _public_segments(state.rallies),
                "backup": backup.name,
                "analytics": _analytics_payload(state),
                "score": _score_payload(state),
                "phase_examples": phase_examples,
            }
        except (KeyError, TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    def render_worker(include_trajectory: bool, winner_filter: str, include_score: bool) -> None:
        try:
            with state.project_lock:
                video_path, timeline_path, output_path = state.video, state.rallies, state.output
                library = state.library or state.video.parent
                trajectory_path = _analysis_directory(library, state.video) / "shuttle-track.csv"
                score_path = _score_paths(state)[2]
            used_encoder = render_rallies(
                video_path,
                timeline_path,
                output_path,
                state.ffmpeg,
                state.encoder,
                state.quality,
                trajectory_csv=trajectory_path if include_trajectory else None,
                score_csv=score_path if include_score or winner_filter != "all" else None,
                winner_filter=winner_filter,
                include_score=include_score,
            )
            runtime = _runtime_payload(state)["ffmpeg"]
            if runtime.get("selected_encoder") != used_encoder:
                runtime["warning"] = f"{runtime.get('selected_encoder')} could not complete the render; using {used_encoder}"
                runtime["selected_encoder"] = used_encoder
            state.render_status = {
                "state": "complete",
                "output": str(output_path),
                "encoder": used_encoder,
                "include_trajectory": include_trajectory,
                "winner_filter": winner_filter,
                "include_score": include_score,
            }
        except Exception as error:  # noqa: BLE001 - every renderer failure must reach the local UI
            state.render_status = {"state": "error", "message": str(error)}

    @app.post("/api/render")
    def start_render(payload: dict[str, Any] | None = None):
        payload = payload or {}
        include_trajectory = payload.get("include_trajectory", False)
        winner_filter = payload.get("winner_filter", "all")
        include_score = payload.get("include_score", False)
        if not isinstance(include_trajectory, bool):
            raise HTTPException(status_code=400, detail="include_trajectory must be a boolean")
        if winner_filter not in {"all", "near", "far"}:
            raise HTTPException(status_code=400, detail="winner_filter must be all, near, or far")
        if not isinstance(include_score, bool):
            raise HTTPException(status_code=400, detail="include_score must be a boolean")
        if state.analysis_status.get("state") == "running":
            raise HTTPException(status_code=409, detail="Wait for automatic analysis to finish before rendering")
        if not _timeline_has_segments(state.rallies):
            raise HTTPException(status_code=400, detail="No rally clips are available to render")
        runtime = _runtime_payload(state)["ffmpeg"]
        if not runtime["available"]:
            error = runtime["reason"] or "FFmpeg H.264 encoder is unavailable"
            raise HTTPException(status_code=503, detail=str(error))
        if include_trajectory:
            library = state.library or state.video.parent
            trajectory_path = _analysis_directory(library, state.video) / "shuttle-track.csv"
            if not trajectory_path.exists():
                raise HTTPException(status_code=409, detail="请先分析当前视频球路")
        if include_score or winner_filter != "all":
            score = _score_payload(state)
            if not score.get("available"):
                raise HTTPException(status_code=409, detail="请先计算比分")
        with state.render_lock:
            if state.render_status.get("state") == "running":
                return state.render_status
            state.render_status = {
                "state": "running",
                "output": str(state.output),
                "include_trajectory": include_trajectory,
                "winner_filter": winner_filter,
                "include_score": include_score,
            }
            threading.Thread(
                target=render_worker,
                args=(include_trajectory, winner_filter, include_score),
                daemon=True,
            ).start()
        return state.render_status

    @app.get("/api/render")
    def render_status():
        return state.render_status

    @app.get("/api/analytics")
    def analytics():
        return _analytics_payload(state)

    @app.get("/api/score")
    def score():
        return _score_payload(state)

    @app.post("/api/score/analyze")
    async def calculate_score(payload: dict[str, Any]):
        library = state.library or state.video.parent
        active_project = _video_id(library, state.video)
        if str(payload.get("project_id", "")) != active_project:
            raise HTTPException(status_code=409, detail="The open project changed; reload before calculating score")
        if state.analysis_status.get("state") == "running":
            raise HTTPException(status_code=409, detail="Wait for automatic analysis to finish before calculating score")
        try:
            _analytics_payload(state)
            return _calculate_score_payload(state)
        except (OSError, TypeError, ValueError, KeyError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.put("/api/score")
    async def update_score(payload: dict[str, Any]):
        library = state.library or state.video.parent
        active_project = _video_id(library, state.video)
        if str(payload.get("project_id", "")) != active_project:
            raise HTTPException(status_code=409, detail="The open project changed; reload before saving score")
        try:
            rally_count = len(load_rallies(state.rallies))
            rows = validate_score_corrections(payload.get("corrections"), rally_count)
            backup = save_score_corrections(_score_corrections_path(library, state.video), rows)
            result = _calculate_score_payload(state)
            result["backup"] = str(backup) if backup else None
            return result
        except (OSError, TypeError, ValueError, KeyError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    def score_labeling_worker(video: Path, rally_ids: list[int]) -> None:
        root, manifest_path, labels_path = _score_label_paths(state)
        config = score_labeling_config()
        total = len(rally_ids)
        try:
            prepare_score_evidence(
                video,
                state.rallies,
                root,
                rally_ids,
                trajectory_csv=root.parent / "shuttle-track.csv",
                progress_callback=lambda completed, count: _set_analysis_status(
                    state,
                    state="running",
                    stage="frames",
                    label="正在提取终局画面",
                    progress=0.05 + 0.35 * completed / max(1, count),
                    completed=completed,
                    total=count,
                ),
            )
            if not config["configured"]:
                _set_analysis_status(
                    state,
                    state="complete",
                    mode="score-labels",
                    stage="complete",
                    label="待标注素材已生成",
                    progress=1.0,
                    completed=total,
                    total=total,
                    results=[{"directory": str(root), "rallies": rally_ids, "labeled": 0}],
                )
                return
            result = label_score_evidence(
                manifest_path,
                labels_path,
                list(config["models"]),
                str(config["endpoint"]),
                str(config["api_key"]),
                progress_callback=lambda completed, count: _set_analysis_status(
                    state,
                    state="running",
                    stage="label",
                    label="正在判断终局",
                    progress=0.40 + 0.60 * completed / max(1, count),
                    completed=completed,
                    total=count,
                ),
            )
            library = state.library or video.parent
            model_payload = fit_score_evidence_model(library, default_score_model_path(library))
            _calculate_score_payload(state)
            target_rallies = set(rally_ids)
            consensus = sum(
                row.get("status") == "consensus" and int(row.get("rally", 0)) in target_rallies
                for row in result["labels"]
            )
            _set_analysis_status(
                state,
                state="complete",
                mode="score-labels",
                stage="complete",
                label="终局建议已生成",
                progress=1.0,
                completed=total,
                total=total,
                results=[
                    {
                        "directory": str(root),
                        "rallies": rally_ids,
                        "labeled": total,
                        "consensus": consensus,
                        "pseudo_examples": model_payload.get("pseudo_examples", 0),
                    }
                ],
            )
        except Exception as error:  # noqa: BLE001 - background failures must reach Studio
            _set_analysis_status(
                state,
                state="error",
                mode="score-labels",
                label="终局标注失败",
                message=str(error),
            )
        finally:
            state.analysis_lock.release()

    @app.post("/api/score-labels/analyze")
    def start_score_labeling(payload: dict[str, Any] | None = None):
        library = state.library or state.video.parent
        requested_project = str((payload or {}).get("project_id", "")).strip()
        active_project = _video_id(library, state.video)
        if requested_project and requested_project != active_project:
            raise HTTPException(status_code=409, detail="The active video changed; refresh Studio before labeling")
        if state.render_status.get("state") == "running":
            raise HTTPException(status_code=409, detail="Wait for rendering to finish before labeling")
        if not state.rallies.exists() or not load_rallies(state.rallies):
            raise HTTPException(status_code=400, detail="请先完成剪辑时间表")
        score = _score_payload(state)
        if not score.get("available"):
            score = _calculate_score_payload(state)
        _, _, labels_path = _score_label_paths(state)
        review_rallies = {
            int(row.get("rally", 0)) for row in load_machine_labels(labels_path) if row.get("status") == "review"
        }
        rally_ids = sorted(
            review_rallies
            | {
                int(row["rally"])
                for row in score.get("rallies", [])
                if row.get("winner_source") == "unresolved"
            }
        )
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
        if not state.analysis_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="Another analysis job is already running")
        with state.project_lock:
            source = state.video
        _begin_analysis_status(
            state,
            mode="score-labels",
            stage="queued",
            label="终局标注已排队",
            progress=0.0,
            completed=0,
            total=len(rally_ids),
            project_id=active_project,
            project_name=source.name,
        )
        threading.Thread(target=score_labeling_worker, args=(source, rally_ids), daemon=True).start()
        return state.analysis_status

    @app.put("/api/score-labels/review")
    async def review_score_label(payload: dict[str, Any]):
        library = state.library or state.video.parent
        if str(payload.get("project_id", "")) != _video_id(library, state.video):
            raise HTTPException(status_code=409, detail="The active video changed; refresh Studio before reviewing")
        try:
            rally = int(payload["rally"])
            decision = str(payload["decision"])
            _, _, labels_path = _score_label_paths(state)
            label = next((row for row in load_machine_labels(labels_path) if int(row.get("rally", 0)) == rally), None)
            if label is None:
                raise ValueError(f"No machine score label for rally {rally}")
            if decision == "accepted":
                suggestion = label.get("suggestion", {})
                corrections_path = _score_corrections_path(library, state.video)
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
                replacement = {**current}
                for field in ("winner", "last_hitter", "terminal_event", "landing_side", "post_rally_event"):
                    value = str(suggestion.get(field, "unknown"))
                    if value != "unknown":
                        replacement[field] = value
                replacement["note"] = "multimodal review"
                rows = [row for row in corrections if int(row["rally"]) != rally] + [replacement]
                rows = validate_score_corrections(rows, len(load_rallies(state.rallies)))
                save_score_corrections(corrections_path, rows)
            save_machine_label_review(labels_path, rally, decision)
            score_result = _calculate_score_payload(state)
            return {"score": score_result, "score_labeling": _score_labeling_payload(state)}
        except (KeyError, OSError, TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/api/shuttle-annotations")
    def shuttle_annotations():
        return _shuttle_annotation_payload(state)

    @app.put("/api/shuttle-annotations")
    async def update_shuttle_annotations(payload: dict[str, Any]):
        library = state.library or state.video.parent
        if str(payload.get("project_id", "")) != _video_id(library, state.video):
            raise HTTPException(status_code=409, detail="The open project changed; reload before saving annotations")
        try:
            metadata = _video_metadata(state.video)
            rows = validate_shuttle_annotations(
                payload.get("annotations"),
                float(metadata["duration"]),
                float(metadata["fps"]),
                int(metadata["width"]),
                int(metadata["height"]),
            )
            path = _shuttle_annotations_path(library, state.video)
            backup = save_shuttle_annotations(path, rows)
            result = _shuttle_annotation_payload(state)
            result["backup"] = str(backup) if backup else None
            return result
        except (OSError, TypeError, ValueError, KeyError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/api/calibration")
    def calibration():
        try:
            return _calibration_payload(state)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.put("/api/calibration")
    async def save_calibration(payload: dict[str, Any]):
        try:
            library_root = state.library or state.video.parent
            active_project = _video_id(library_root, state.video)
            if str(payload.get("project_id", "")) != active_project:
                raise ValueError("The open project changed; reload before saving calibration")
            path, backup = _save_calibration(state, payload)
            result = _calibration_payload(state)
            ffmpeg_available = bool(_runtime_payload(state)["ffmpeg"]["available"])
            return {
                **result,
                "ok": True,
                "path": str(path),
                "backup": str(backup) if backup else None,
                "automatic_analysis_configured": bool(_calibration_ready(state) and state.model and ffmpeg_available),
                "pose_overlay_configured": bool(_calibration_ready(state) and state.pose_model and ffmpeg_available),
                "shuttle_analysis": _shuttle_status_payload(state),
            }
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/api/pose-overlay")
    async def pose_overlay(payload: dict[str, Any]):
        if not state.config or not state.config.exists() or not state.pose_model or not state.pose_model.exists():
            raise HTTPException(status_code=400, detail="YOLO pose model and court config are required")
        runtime = _runtime_payload(state)["ffmpeg"]
        if not runtime["available"]:
            error = runtime["reason"] or "FFmpeg H.264 encoder is unavailable"
            raise HTTPException(status_code=503, detail=str(error))
        try:
            library_root = state.library or state.video.parent
            active_project = _video_id(library_root, state.video)
            if str(payload.get("project_id", "")) != active_project:
                raise ValueError("The open project changed; reload before generating a pose preview")
            rally_number = int(payload["rally"])
            output, segment = _pose_overlay_path(state, rally_number)
            with state.pose_lock:
                newest_input = max(
                    state.video.stat().st_mtime,
                    state.rallies.stat().st_mtime,
                    state.config.stat().st_mtime,
                    state.pose_model.stat().st_mtime,
                )
                if not output.exists() or output.stat().st_mtime < newest_input:
                    render_pose_overlay(
                        state.video,
                        float(segment["start"]),
                        float(segment["end"]),
                        output,
                        state.pose_model,
                        state.config,
                        state.packages,
                        state.ffmpeg,
                        str(runtime["selected_encoder"]),
                    )
            return {
                "ok": True,
                "rally": rally_number,
                "url": f"/media/pose-overlay?rally={rally_number}&v={output.stat().st_mtime_ns}",
                "output": str(output),
            }
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/media/pose-overlay")
    def pose_overlay_media(rally: int):
        try:
            output, _segment = _pose_overlay_path(state, rally)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        if not output.exists():
            raise HTTPException(status_code=404, detail="Generate the pose preview first")
        return FileResponse(output, media_type="video/mp4")

    @app.get("/media/pose-overlay/full")
    def full_pose_overlay_media():
        output = _full_pose_overlay_path(state)
        if not output.exists():
            raise HTTPException(status_code=404, detail="Run full-video visual analysis first")
        return FileResponse(output, media_type="video/mp4")

    def shuttle_analysis_worker(video: Path, force: bool) -> None:
        library_root = state.library or video.parent
        analysis_root = _analysis_directory(library_root, video)
        analysis_root.mkdir(parents=True, exist_ok=True)
        raw_path = analysis_root / "shuttle-raw.csv"
        trajectory_path = analysis_root / "shuttle-track.csv"
        features_path = _visual_evidence_features_path(analysis_root)
        annotations_path = _shuttle_annotations_path(library_root, video)
        try:
            detection_result: dict[str, Any] = {"reused": True, "output": str(raw_path)}
            if force or not raw_path.exists():
                _set_analysis_status(
                    state,
                    state="running",
                    mode="shuttle",
                    stage="shuttle-detect",
                    label="正在逐帧检测本场羽球；剪辑时间表不会改变",
                    progress=0.12,
                )

                def shuttle_detection_progress(value: float, detector: str) -> None:
                    _set_analysis_status(
                        state,
                        state="running",
                        mode="shuttle",
                        stage="shuttle-detect",
                        label=f"正在运行 {detector} 羽球检测",
                        progress=0.12 + value * 0.68,
                    )

                detection_result = _run_shuttle_detection(
                    state,
                    video,
                    analysis_root,
                    force,
                    shuttle_detection_progress,
                )
            _set_analysis_status(
                state,
                state="running",
                mode="shuttle",
                stage="shuttle-track",
                label="正在过滤隔壁场目标并连接连续飞行轨迹",
                progress=0.82,
            )
            _run_shuttle_trajectory(
                raw_path,
                trajectory_path,
                features_path if features_path.exists() else None,
                video,
                annotations_path,
                state.config,
            )
            shuttle_status = _shuttle_status_payload(state)
            _set_analysis_status(
                state,
                state="complete",
                mode="shuttle",
                stage="complete",
                label="羽球轨迹分析完成；剪辑时间表未改变",
                progress=1.0,
                completed=1,
                total=1,
                results=[
                    {
                        "video": _video_id(library_root, video),
                        "detector": detection_result.get("mode"),
                        "points": shuttle_status["point_count"],
                        "flights": shuttle_status["flight_count"],
                        "track": shuttle_status["track_path"],
                        "timeline_unchanged": str(state.rallies),
                    }
                ],
            )
        except Exception as error:  # noqa: BLE001 - every detector failure must reach the local UI
            _set_analysis_status(
                state,
                state="error",
                mode="shuttle",
                label="羽球轨迹分析失败",
                message=str(error),
            )
        finally:
            state.analysis_lock.release()

    @app.post("/api/analyze/shuttle")
    def start_shuttle_analysis(payload: dict[str, Any] | None = None):
        requested_project = str((payload or {}).get("project_id", "")).strip()
        active_project = _video_id(state.library or state.video.parent, state.video)
        if requested_project and requested_project != active_project:
            raise HTTPException(status_code=409, detail="The active video changed; refresh Studio before starting analysis")
        if not _calibration_ready(state):
            raise HTTPException(status_code=400, detail="Complete the five required court calibration regions first")
        if _effective_shuttle_mode(state) is None:
            raise HTTPException(status_code=400, detail="Configure YOLO or TrackNet first")
        if state.render_status.get("state") == "running":
            raise HTTPException(status_code=409, detail="Wait for rendering to finish before shuttle analysis")
        if state.analysis_status.get("state") == "running":
            raise HTTPException(status_code=409, detail="Another analysis job is already running")
        force = bool((payload or {}).get("force", False))
        current_status = _shuttle_status_payload(state)
        if current_status["current"] and not force:
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
                        "video": _video_id(state.library or state.video.parent, state.video),
                        "points": current_status["point_count"],
                        "flights": current_status["flight_count"],
                        "track": current_status["track_path"],
                        "timeline_unchanged": str(state.rallies),
                    }
                ],
            }
            return state.analysis_status
        if current_status["stale"] and any(
            reason in str(current_status.get("stale_reason") or "")
            for reason in ("球场校准已更改", "检测模式已更改", "检测缓存需要更新")
        ):
            force = True
        if not state.analysis_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="Another analysis job is already running")
        with state.project_lock:
            source = state.video
        _begin_analysis_status(
            state,
            mode="shuttle",
            stage="queued",
            label="羽球轨迹分析已排队；剪辑时间表不会改变",
            progress=0.0,
            completed=0,
            total=1,
            project_id=active_project,
            project_name=source.name,
        )
        threading.Thread(target=shuttle_analysis_worker, args=(source, force), daemon=True).start()
        return state.analysis_status

    @app.put("/api/settings/shuttle-mode")
    def update_shuttle_mode(payload: dict[str, Any]):
        mode = str(payload.get("mode", "")).strip().lower()
        if mode not in SHUTTLE_MODES:
            raise HTTPException(status_code=400, detail="Mode must be yolo, tracknet, or hybrid")
        available = _available_shuttle_modes(state)
        if mode not in available:
            raise HTTPException(status_code=400, detail=f"{mode} is not configured")
        state.shuttle_mode = mode
        return _shuttle_status_payload(state)

    def visual_analysis_worker(
        video: Path,
        run_pose: bool,
        run_shuttle: bool,
        rerun_shuttle_raw: bool,
    ) -> None:
        library_root = state.library or video.parent
        analysis_root = _analysis_directory(library_root, video)
        analysis_root.mkdir(parents=True, exist_ok=True)
        results: dict[str, Any] = {"video": _video_id(library_root, video)}
        try:
            if run_pose:
                output = _full_pose_overlay_path(state)

                def pose_progress(value: float) -> None:
                    _set_analysis_status(
                        state,
                        state="running",
                        mode="visual",
                        stage="pose",
                        label="正在分析整段人物姿态并生成可开关覆盖层",
                        progress=0.05 + value * (0.52 if run_shuttle else 0.9),
                    )

                render_pose_overlay(
                    video,
                    0.0,
                    float(_video_metadata(video)["duration"]),
                    output,
                    state.pose_model,
                    state.config,
                    state.packages,
                    state.ffmpeg,
                    str(_runtime_payload(state)["ffmpeg"]["selected_encoder"]),
                    progress_callback=pose_progress,
                )
                results["pose"] = _pose_status_payload(state)

            if run_shuttle:
                raw_path = analysis_root / "shuttle-raw.csv"
                trajectory_path = analysis_root / "shuttle-track.csv"
                state_features_path = analysis_root / "smart-features.csv"
                features_path = _vision_features_path(analysis_root)
                audio_path = analysis_root / "audio-events.csv"
                annotations_path = _shuttle_annotations_path(library_root, video)
                detection_result: dict[str, Any] = {"reused": True, "output": str(raw_path)}
                if rerun_shuttle_raw or not raw_path.exists():
                    _set_analysis_status(
                        state,
                        state="running",
                        mode="visual",
                        stage="shuttle-detect",
                        label="正在分析整段羽球候选；高远球与隔壁场会交给轨迹器复核",
                        progress=0.6 if run_pose else 0.12,
                    )

                    def visual_shuttle_progress(value: float, detector: str) -> None:
                        start = 0.60 if run_pose else 0.12
                        end = 0.74
                        _set_analysis_status(
                            state,
                            state="running",
                            mode="visual",
                            stage="shuttle-detect",
                            label=f"正在运行 {detector} 羽球检测",
                            progress=start + value * (end - start),
                        )

                    detection_result = _run_shuttle_detection(
                        state,
                        video,
                        analysis_root,
                        rerun_shuttle_raw,
                        visual_shuttle_progress,
                    )
                feature_columns: set[str] = set()
                if features_path.exists():
                    with features_path.open(encoding="utf-8-sig") as existing_features:
                        feature_columns = set(next(csv.reader(existing_features), []))
                needs_pose_association = bool(
                    state.pose_model
                    and not POSE_ASSOCIATION_FIELDS.issubset(feature_columns)
                )
                if needs_pose_association:
                    _set_analysis_status(
                        state,
                        state="running",
                        mode="visual",
                        stage="shuttle-track-initial",
                        label="正在先排除长期静止杂物并建立初始羽球轨迹",
                        progress=0.76,
                    )
                    _run_shuttle_trajectory(raw_path, trajectory_path, None, video, annotations_path, state.config)
                    if not audio_path.exists():
                        analyze_audio(video, audio_path, ffmpeg=state.ffmpeg)

                    def association_progress(value: float) -> None:
                        _set_analysis_status(
                            state,
                            state="running",
                            mode="visual",
                            stage="player-association",
                            label="正在提取手腕、捡球与持球证据",
                            progress=0.79 + value * 0.16,
                        )

                    extract_features(
                        video,
                        state.config,
                        features_path,
                        audio_path,
                        trajectory_path,
                        state.pose_model,
                        state.packages,
                        None,
                        0.0,
                        None,
                        association_progress,
                    )
                    _preserve_state_features(state_features_path, features_path)
                _set_analysis_status(
                    state,
                    state="running",
                    mode="visual",
                    stage="shuttle-track",
                    label="正在结合静态杂物、飞行连续性和球员交互复核羽球轨迹",
                    progress=0.96 if needs_pose_association else 0.93,
                )
                _run_shuttle_trajectory(
                    raw_path,
                    trajectory_path,
                    features_path if features_path.exists() else None,
                    video,
                    annotations_path,
                    state.config,
                )
                results["shuttle"] = {
                    **_shuttle_status_payload(state),
                    "detector": detection_result.get("mode"),
                }

            _set_analysis_status(
                state,
                state="complete",
                mode="visual",
                stage="complete",
                label="整段视觉分析完成；播放器可以随时开关姿态和球路",
                progress=1.0,
                completed=int(run_pose) + int(run_shuttle),
                total=int(run_pose) + int(run_shuttle),
                results=[results],
            )
        except Exception as error:  # noqa: BLE001 - every analysis failure must reach the local UI
            _set_analysis_status(
                state,
                state="error",
                mode="visual",
                label="整段视觉分析失败",
                message=str(error),
            )
        finally:
            state.analysis_lock.release()

    @app.post("/api/analyze/visual")
    def start_visual_analysis(payload: dict[str, Any] | None = None):
        requested_project = str((payload or {}).get("project_id", "")).strip()
        active_project = _video_id(state.library or state.video.parent, state.video)
        if requested_project and requested_project != active_project:
            raise HTTPException(status_code=409, detail="The active video changed; refresh Studio before starting analysis")
        if not _calibration_ready(state):
            raise HTTPException(status_code=400, detail="Complete court calibration before visual analysis")
        if state.render_status.get("state") == "running":
            raise HTTPException(status_code=409, detail="Wait for rendering to finish before visual analysis")
        if state.analysis_status.get("state") == "running":
            raise HTTPException(status_code=409, detail="Another analysis job is already running")
        pose_status = _pose_status_payload(state)
        shuttle_status = _shuttle_status_payload(state)
        if not pose_status["configured"] and not shuttle_status["configured"]:
            raise HTTPException(status_code=400, detail="Configure a pose model or shuttle model first")
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
                "results": [{"video": active_project, "pose": pose_status, "shuttle": shuttle_status}],
            }
            return state.analysis_status
        if not state.analysis_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="Another analysis job is already running")
        _begin_analysis_status(
            state,
            mode="visual",
            stage="queued",
            label="整段视觉分析已排队；不会修改剪辑时间表",
            progress=0.0,
            completed=0,
            total=int(run_pose) + int(run_shuttle),
            project_id=active_project,
            project_name=state.video.name,
        )
        rerun_shuttle_raw = bool(force or shuttle_status["stale"] or not shuttle_status["raw_generated"])
        with state.project_lock:
            source = state.video
        threading.Thread(
            target=visual_analysis_worker,
            args=(source, run_pose, run_shuttle, rerun_shuttle_raw),
            daemon=True,
        ).start()
        return state.analysis_status

    def analysis_worker(videos: list[Path], mode: str) -> None:
        library_root = state.library or state.video.parent
        results: list[dict[str, Any]] = []
        try:
            for index, source in enumerate(videos, 1):
                results.append(_analyze_video(state, library_root, source, index, len(videos)))
            _set_analysis_status(
                state,
                state="complete",
                mode=mode,
                label="自动分析完成，时间表已可校对",
                progress=1.0,
                completed=len(results),
                total=len(videos),
                results=results,
            )
        except Exception as error:  # noqa: BLE001 - every analysis failure must reach the local UI
            _set_analysis_status(state, state="error", label="自动分析失败", message=str(error))
        finally:
            state.analysis_lock.release()

    def launch_analysis(videos: list[Path], mode: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
        if not _calibration_ready(state):
            raise HTTPException(status_code=400, detail="Complete the five required court calibration regions first")
        if not state.model or not state.model.exists():
            raise HTTPException(status_code=400, detail="Trained rally model is not configured or does not exist")
        runtime = _runtime_payload(state)["ffmpeg"]
        if not runtime["available"]:
            error = runtime["reason"] or "FFmpeg H.264 encoder is unavailable"
            raise HTTPException(status_code=503, detail=str(error))
        if state.render_status.get("state") == "running":
            raise HTTPException(status_code=409, detail="Wait for rendering to finish before automatic analysis")
        if not state.analysis_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="Automatic analysis is already running")
        try:
            requested = options or {}
            preroll = float(requested.get("preroll", state.analysis_options["preroll"]))
            postroll = float(requested.get("postroll", state.analysis_options["postroll"]))
            if not 0.0 <= preroll <= 2.0 or not 0.0 <= postroll <= 2.0:
                raise ValueError("Pre-roll and post-roll must be between 0 and 2 seconds")
            state.analysis_options = {
                **state.analysis_options,
                "preroll": preroll,
                "postroll": postroll,
                "suppress_handoffs": bool(requested.get("suppress_handoffs", True)),
            }
        except (TypeError, ValueError) as error:
            state.analysis_lock.release()
            raise HTTPException(status_code=400, detail=str(error)) from error
        _begin_analysis_status(
            state,
            mode=mode,
            stage="queued",
            label="自动分析已排队",
            progress=0.0,
            completed=0,
            total=len(videos),
            project_id=_video_id(state.library or state.video.parent, videos[0]) if len(videos) == 1 else "batch",
            project_name=videos[0].name if len(videos) == 1 else f"{len(videos)} 个比赛",
        )
        threading.Thread(target=analysis_worker, args=(videos, mode), daemon=True).start()
        return state.analysis_status

    @app.post("/api/analyze")
    def start_analysis(payload: dict[str, Any] | None = None):
        with state.project_lock:
            source = state.video
        return launch_analysis([source], "single", payload)

    @app.post("/api/analyze/batch")
    def start_batch_analysis(payload: dict[str, Any] | None = None):
        library_root = state.library or state.video.parent
        pending = [
            video
            for video in _discover_videos(library_root)
            if not _timeline_has_segments(_working_timeline_path(library_root, video))
            and not _project_has_completed_marker(library_root, video)
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
        raise RuntimeError('Studio dependencies are missing. Install with: pip install -e ".[studio]"') from error
    if video is None:
        if library is None:
            raise ValueError("Pass a video or a library folder")
        videos = _discover_videos(library)
        if not videos:
            raise ValueError(f"No supported video files found in: {library}")
        video = videos[0]
    library = (library or video.parent).resolve()
    model = resolve_model(model, "rally", library)
    pose_model = resolve_model(pose_model, "pose", library)
    shuttle_model = resolve_model(shuttle_model, "shuttle", library)
    tracknet_model = resolve_model(tracknet_model, "tracknet", library)
    inpaint_model = resolve_model(inpaint_model, "inpaint", library)
    rallies = rallies or _ensure_working_timeline(library, video)
    output_directory = output.expanduser().resolve().parent if output is not None else None
    output = output.expanduser().resolve() if output is not None else _default_output(library, video)
    proxy = proxy or _proxy_for(video, library)
    state = StudioState(
        video=video,
        rallies=rallies,
        output=output,
        proxy=proxy,
        ffmpeg=ffmpeg,
        encoder=encoder,
        quality=quality,
        library=library,
        output_directory=output_directory,
        config=config,
        model=model,
        pose_model=pose_model,
        shuttle_model=shuttle_model,
        tracknet_model=tracknet_model,
        inpaint_model=inpaint_model,
        shuttle_mode=shuttle_mode,
        packages=packages,
        tracknet_packages=tracknet_packages,
    )
    app = create_studio_app(state)
    url = f"http://{host}:{port}"
    print(f"Smart Badminton Studio: {url}")
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=host, port=port, log_level="info")
