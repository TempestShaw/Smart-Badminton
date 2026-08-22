from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import CourtGeometry


def _trajectory_points(path: Path | None) -> list[dict[str, float]]:
    if path is None or not path.exists():
        return []
    points = []
    with path.open(newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            if row.get("status") not in {"tracked", "recovered", "manual"}:
                continue
            if row.get("detection_status") == "inpainted":
                continue
            width = float(row.get("source_width") or 0.0)
            height = float(row.get("source_height") or 0.0)
            if width <= 0 or height <= 0:
                continue
            points.append(
                {
                    "time": float(row["time_seconds"]),
                    "x": float(row["center_x"]) / width,
                    "y": float(row["center_y"]) / height,
                }
            )
    return sorted(points, key=lambda point: point["time"])


def _local_velocity(points: list[dict[str, float]], start: float, end: float) -> tuple[float, float] | None:
    selected = [point for point in points if start <= point["time"] <= end]
    if len(selected) < 2:
        return None
    first, last = selected[0], selected[-1]
    delta = last["time"] - first["time"]
    if delta <= 0:
        return None
    return (last["x"] - first["x"]) / delta, (last["y"] - first["y"]) / delta


def classify_shots(
    contacts: list[dict[str, str]] | None,
    trajectory_csv: Path | None,
    geometry: CourtGeometry | None = None,
) -> list[dict[str, Any]]:
    points = _trajectory_points(trajectory_csv)
    results = []
    for contact in contacts or []:
        time_seconds = float(contact["time_seconds"])
        nearby = [point for point in points if abs(point["time"] - time_seconds) <= 0.22]
        anchor = min(nearby, key=lambda point: abs(point["time"] - time_seconds), default=None)
        after = _local_velocity(points, time_seconds - 0.03, time_seconds + 0.34)
        before = _local_velocity(points, time_seconds - 0.34, time_seconds + 0.03)
        shot_type, confidence, reason = "unknown", 0.15, "insufficient post-contact shuttle trajectory"
        if anchor is not None and after is not None:
            vx, vy = after
            speed = math.hypot(vx, vy)
            near_net = bool(geometry and geometry.contains_normalized("net_band", anchor["x"], anchor["y"]))
            direction_change = 0.0
            if before is not None:
                before_norm, after_norm = math.hypot(*before), math.hypot(*after)
                if before_norm > 1e-6 and after_norm > 1e-6:
                    cosine = np.clip((before[0] * vx + before[1] * vy) / (before_norm * after_norm), -1.0, 1.0)
                    direction_change = float((1.0 - cosine) / 2.0)
            if near_net and speed <= 0.75:
                shot_type, confidence, reason = "net_candidate", 0.58, "contact near net with a slow outgoing path"
            elif vy >= 0.45 and speed >= 0.75:
                shot_type, confidence, reason = "smash_candidate", 0.62, "fast downward image trajectory after contact"
            elif vy <= -0.25:
                shot_type, confidence, reason = "clear_or_lift_candidate", 0.52, "upward image trajectory after contact"
            elif abs(vx) >= 0.55 and abs(vy) <= 0.45:
                shot_type, confidence, reason = "drive_candidate", 0.48, "fast, comparatively flat outgoing trajectory"
            elif direction_change >= 0.45:
                shot_type, confidence, reason = "placement_candidate", 0.42, "large direction change at racket contact"
        results.append(
            {
                "time_seconds": round(time_seconds, 3),
                "player": contact.get("player", "unknown"),
                "shot_type": shot_type,
                "confidence": round(confidence, 3),
                "reason": reason,
            }
        )
    return results


def shot_type_counts(shots: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(shot["shot_type"]) for shot in shots if shot["shot_type"] != "unknown")
    return dict(sorted(counts.items()))
