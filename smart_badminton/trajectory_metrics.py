from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import CourtGeometry

ACCEPTED_STATUSES = {"tracked", "recovered", "manual"}
MAX_CONTINUOUS_GAP_SECONDS = 0.18
MAX_STEP_DISTANCE = 0.20


@dataclass(frozen=True)
class TrajectoryPoint:
    time: float
    x: float
    y: float
    flight_id: str
    quality: float


def _quality(row: dict[str, str]) -> float:
    manual = 2.0 if row.get("status") == "manual" else 0.0
    ownership = float(row.get("ownership_confidence") or 0.0)
    evidence = float(row.get("evidence_weight") or 0.0)
    confidence = float(row.get("confidence") or 0.0)
    return manual + ownership + evidence * 0.2 + confidence * 0.1


def load_owned_trajectory(path: Path | None) -> list[TrajectoryPoint]:
    if path is None or not path.exists():
        return []
    selected: dict[float, TrajectoryPoint] = {}
    with path.open(newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            if row.get("status") not in ACCEPTED_STATUSES or row.get("detection_status") == "inpainted":
                continue
            ownership_evidence = str(row.get("ownership_evidence") or "").strip().lower()
            ownership_confidence = float(row.get("ownership_confidence") or 0.0)
            if ownership_evidence and (ownership_evidence == "unknown" or ownership_confidence < 0.45):
                continue
            width = float(row.get("source_width") or 0.0)
            height = float(row.get("source_height") or 0.0)
            if width <= 0 or height <= 0:
                continue
            time = round(float(row["time_seconds"]), 6)
            point = TrajectoryPoint(
                time=time,
                x=float(row["center_x"]) / width,
                y=float(row["center_y"]) / height,
                flight_id=str(row.get("flight_id") or row.get("track_id") or ""),
                quality=_quality(row),
            )
            previous = selected.get(time)
            if previous is None or point.quality > previous.quality:
                selected[time] = point
    return sorted(selected.values(), key=lambda point: point.time)


def _components(points: list[TrajectoryPoint]) -> list[list[TrajectoryPoint]]:
    components: list[list[TrajectoryPoint]] = []
    for point in points:
        previous = components[-1][-1] if components else None
        changes_flight = bool(previous and point.flight_id and previous.flight_id and point.flight_id != previous.flight_id)
        if previous is None or point.time - previous.time > MAX_CONTINUOUS_GAP_SECONDS or changes_flight:
            components.append([point])
        else:
            components[-1].append(point)
    return components


def _net_centerline(geometry: CourtGeometry | None) -> tuple[np.ndarray, np.ndarray] | None:
    if geometry is None or len(geometry.net_band) < 4:
        return None
    points = sorted((np.asarray(point, dtype=float) for point in geometry.net_band), key=lambda point: point[0])
    left = np.mean(points[:2], axis=0)
    right = np.mean(points[-2:], axis=0)
    if float(np.linalg.norm(right - left)) < 1e-6:
        return None
    return left, right


def _net_zone_transits(components: list[list[TrajectoryPoint]], geometry: CourtGeometry | None) -> int:
    line = _net_centerline(geometry)
    if line is None:
        return 0
    left, right = line
    direction = right - left
    length = float(np.linalg.norm(direction))
    minimum_x, maximum_x = min(left[0], right[0]) - 0.06, max(left[0], right[0]) + 0.06
    transits: list[float] = []
    for component in components:
        stable_side: int | None = None
        for point in component:
            if not minimum_x <= point.x <= maximum_x:
                continue
            offset = np.asarray([point.x, point.y], dtype=float) - left
            signed_distance = float(direction[0] * offset[1] - direction[1] * offset[0]) / length
            side = -1 if signed_distance < -0.018 else 1 if signed_distance > 0.018 else 0
            if side == 0:
                continue
            if (
                stable_side is not None
                and side != stable_side
                and (not transits or point.time - transits[-1] >= 0.45)
            ):
                transits.append(point.time)
            stable_side = side
    return len(transits)


def summarize_trajectory(
    points: list[TrajectoryPoint],
    start: float,
    end: float,
    geometry: CourtGeometry | None,
) -> dict[str, Any]:
    rally_points = [point for point in points if start <= point.time <= end]
    components = [component for component in _components(rally_points) if len(component) >= 2]
    duration = max(end - start, 0.001)
    visible_seconds = sum(component[-1].time - component[0].time for component in components)
    longest = max((component[-1].time - component[0].time for component in components), default=0.0)
    step_distances: list[float] = []
    step_speeds: list[float] = []
    for component in components:
        for previous, current in pairwise(component):
            delta_time = current.time - previous.time
            distance = math.hypot(current.x - previous.x, current.y - previous.y) / math.sqrt(2.0)
            if 0 < delta_time <= MAX_CONTINUOUS_GAP_SECONDS and distance <= MAX_STEP_DISTANCE:
                step_distances.append(distance)
                step_speeds.append(distance / delta_time)
    return {
        "trajectory_available": bool(components),
        "trajectory_points": len(rally_points),
        "trajectory_visible_seconds": round(visible_seconds, 2),
        "trajectory_coverage_percent": round(min(100.0, visible_seconds / duration * 100.0), 1),
        "longest_continuous_track_seconds": round(longest, 2),
        "trajectory_distance_frames": round(sum(step_distances), 2),
        "peak_visual_speed_frames_per_second": round(float(np.percentile(step_speeds, 95)), 2)
        if step_speeds
        else 0.0,
        "net_zone_transits": _net_zone_transits(components, geometry),
    }
