from __future__ import annotations

import json
import math
import os
from collections.abc import Container
import shutil
import tempfile
from pathlib import Path
from typing import Any

from ..geometry import CourtGeometry
from .project import video_metadata
from .state import StudioState

REGION_DEFINITIONS = {
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
EXCLUSION_REGIONS = {
    "background_court_polygons": ("背景排除区", "#909a94"),
    "static_false_positive_polygons": ("静态误检区", "#ff9a55"),
}
DEFAULT_ANALYSIS_SETTINGS = {
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
}


def _area(points: list[list[float]]) -> float:
    return abs(sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1]))) / 2


def _is_convex(points: list[list[float]]) -> bool:
    turns = [
        (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        for a, b, c in zip(points, points[1:] + points[:1], points[2:] + points[:2])
    ]
    return all(turn > 0 for turn in turns) or all(turn < 0 for turn in turns)


def _points(value: Any, label: str, counts: Container[int]) -> list[list[float]]:
    """Normalize one region, rejecting geometry that would silently corrupt analysis."""
    points = [[round(float(x), 6), round(float(y), 6)] for x, y in value or []]
    if len(points) not in counts:
        raise ValueError(f"{label}: {len(points)} points is not a valid shape")
    if not all(0 <= coordinate <= 1 for point in points for coordinate in point):  # NaN fails too
        raise ValueError(f"{label}: every point must lie inside the video frame")
    if len(points) == 2 and math.dist(*points) < 0.05:
        raise ValueError(f"{label}: the two points are too close together")
    if len(points) >= 3 and _area(points) < 1e-4:
        raise ValueError(f"{label}: the points are coincident or collinear")
    if len(points) == 4 and label == COURT_CORNERS_DEFINITION["label"] and not _is_convex(points):
        raise ValueError(f"{label}: the four corners must form a convex quadrilateral in order")
    return points


def default_calibration_path(state: StudioState) -> Path:
    return state.library_root / "Calibration" / "court-config.json"


def load_calibration_data(state: StudioState) -> dict[str, Any]:
    if not state.config_ready():
        return {}
    try:
        return json.loads(state.config.read_text(encoding="utf-8"))
    except ValueError:
        return {}  # An unreadable config shows as uncalibrated; saving keeps the old file as .bak.


def calibration_ready(state: StudioState) -> bool:
    masks = load_calibration_data(state).get("masks_normalized", {})
    return state.config_ready() and all(len(masks.get(key, [])) >= 3 for key in REGION_DEFINITIONS)


def _regions(data: dict[str, Any]) -> list[dict[str, Any]]:
    masks = data.get("masks_normalized", {})
    calibration = data.get("calibration", {})
    regions = [
        {"id": key, **definition, "minimum_points": 3, "maximum_points": 64, "points": masks.get(key, [])}
        for key, definition in REGION_DEFINITIONS.items()
    ]
    regions.append(
        {"id": "court_corners", **COURT_CORNERS_DEFINITION, "points": calibration.get("court_corners_normalized") or []}
    )
    regions.append(
        {
            "id": "shuttle_perspective_axis",
            **PERSPECTIVE_AXIS_DEFINITION,
            "points": calibration.get("shuttle_perspective_axis_normalized", []),
        }
    )
    for region_type, (label, color) in EXCLUSION_REGIONS.items():
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


def calibration_payload(state: StudioState) -> dict[str, Any]:
    data = load_calibration_data(state)
    metadata = video_metadata(state.video)
    width, height = int(metadata["width"]), int(metadata["height"])
    regions = _regions(data)
    homography: dict[str, Any] = {"available": False, "reason": "four ordered court corners are not calibrated"}
    if state.config_ready():
        homography = CourtGeometry.from_json(state.config).homography_quality(width, height)
    return {
        "available": True,
        "path": str(state.config or default_calibration_path(state)),
        "exists": state.config_ready(),
        "ready": calibration_ready(state),
        "required_completed": sum(bool(region["points"]) for region in regions if region.get("required")),
        "required_total": len(REGION_DEFINITIONS),
        "reference_frame": {"width": width, "height": height},
        "regions": regions,
        "homography": homography,
    }


def save_calibration(state: StudioState, payload: dict[str, Any]) -> tuple[Path, Path | None]:
    regions = [region for region in payload["regions"] if isinstance(region, dict)]
    by_id = {str(region.get("id")): region for region in regions}
    polygon = range(3, 65)
    masks: dict[str, Any] = {
        key: _points(by_id.get(key, {}).get("points"), str(definition["label"]), polygon)
        for key, definition in REGION_DEFINITIONS.items()
    }
    masks["court_ground_polygon"] = masks["active_court_polygon"]
    for region_type in EXCLUSION_REGIONS:
        masks[region_type] = [
            _points(region["points"], EXCLUSION_REGIONS[region_type][0], polygon)
            for region in regions
            if region.get("points")
            and (region.get("type") == region_type or str(region.get("id", "")).startswith(f"{region_type}:"))
        ]
    existing = load_calibration_data(state)
    existing_calibration = existing.get("calibration", {})
    metadata = video_metadata(state.video)
    config_data = {
        **existing,
        "name": existing.get("name", f"Studio calibration for {state.video.stem}"),
        "reference_frame": {
            "width": int(metadata["width"]),
            "height": int(metadata["height"]),
            "source_time_seconds": round(float(payload.get("source_time_seconds", 0.0)), 3),
        },
        "calibration": {
            **existing_calibration,
            "mode": "studio interactive normalized polygons",
            "court_corners_normalized": _points(
                by_id.get("court_corners", {}).get("points"), str(COURT_CORNERS_DEFINITION["label"]), (0, 4)
            )
            or None,
            "shuttle_perspective_axis_normalized": _points(
                by_id.get("shuttle_perspective_axis", {}).get("points"), str(PERSPECTIVE_AXIS_DEFINITION["label"]), (0, 2)
            ),
            "shuttle_vanishing_extension": float(existing_calibration.get("shuttle_vanishing_extension", 1.5)),
            "note": "Created or edited interactively in Smart Badminton Studio.",
        },
        "masks_normalized": masks,
        "analysis": existing.get("analysis", DEFAULT_ANALYSIS_SETTINGS),
    }
    path = state.config or default_calibration_path(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if path.exists():
        backup = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=".json", dir=path.parent)
    with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
        json.dump(config_data, destination, ensure_ascii=False, indent=2)
        destination.write("\n")
    os.replace(temporary_name, path)
    state.config = path
    return path, backup
