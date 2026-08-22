from __future__ import annotations

import csv
import math
import tempfile
from bisect import bisect_left, bisect_right
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import DBSCAN

from .encoding import h264_encoding_arguments, run_ffmpeg_with_encoder_fallback
from .geometry import CourtGeometry
from .io import resolve_ffmpeg
from .shuttle_annotations import load_shuttle_annotations

# Inpainted points may extend a trail but cannot establish track ownership.
_LINKABLE_DETECTION_STATUSES = {"detected", "fused", "recovered", "manual"}


def _evidence_weight(row: dict[str, float | int | str]) -> float:
    value = row.get("evidence_weight")
    return 1.0 if value is None or value == "" else float(value)


def _is_linkable_detection(row: dict[str, float | int | str]) -> bool:
    detection_status = str(row.get("detection_status") or "detected").lower()
    if detection_status not in _LINKABLE_DETECTION_STATUSES:
        return False
    return _evidence_weight(row) > 0.0


def _geometry_context_score(
    row: dict[str, float | int | str], geometry: CourtGeometry | None
) -> float:
    """Return a soft active-court prior in [-1, 1].

    The projected airspace is deliberately a score rather than a hard gate:
    a calibrated mask should down-rank a high clear outside its exact outline,
    while explicit background/static exclusion regions remain strong penalties.
    """
    if geometry is None:
        return 0.0
    width = float(row.get("source_width") or 0.0)
    height = float(row.get("source_height") or 0.0)
    if width <= 0 or height <= 0:
        return 0.0
    x = float(row["center_x"]) / width
    y = float(row["center_y"]) / height
    if geometry.excluded_normalized(x, y):
        return -1.0
    pixel = (float(row["center_x"]), float(row["center_y"]))
    projected = geometry.projected_shuttle_volume_polygon(round(width), round(height))
    margin = float(cv2.pointPolygonTest(projected.astype(np.float32), pixel, True))
    volume_score = float(np.clip(margin / max(width * 0.18, 1.0), -1.0, 1.0))
    active_score = 0.0
    if geometry.active_court_polygon:
        active = geometry.denormalize(geometry.active_court_polygon, round(width), round(height))
        active_margin = float(cv2.pointPolygonTest(active.astype(np.float32), pixel, True))
        active_score = float(np.clip(active_margin / max(width * 0.12, 1.0), -1.0, 1.0))
    return float(np.clip(0.72 * volume_score + 0.28 * active_score, -1.0, 1.0))


def _axis_proximity_score(
    row: dict[str, float | int | str], geometry: CourtGeometry | None
) -> float:
    if geometry is None:
        return 0.0
    width = float(row.get("source_width") or 0.0)
    height = float(row.get("source_height") or 0.0)
    if width <= 0 or height <= 0:
        return 0.0
    point = np.asarray([float(row["center_x"]) / width, float(row["center_y"]) / height])
    if geometry.shuttle_perspective_axis and len(geometry.shuttle_perspective_axis) == 2:
        first = np.asarray(geometry.shuttle_perspective_axis[0], dtype=float)
        second = np.asarray(geometry.shuttle_perspective_axis[1], dtype=float)
        direction = second - first
        length = float(np.linalg.norm(direction))
        if length > 1e-6:
            distance = abs(float(np.cross(direction, point - first))) / length
            return float(np.clip(1.0 - distance / 0.32, -1.0, 1.0))
    return float(np.clip(1.0 - abs(float(point[0]) - 0.5) / 0.36, -1.0, 1.0))


def _track_net_support(
    rows: list[dict[str, float | int | str]],
    indices: list[int],
    geometry: CourtGeometry | None,
) -> float:
    if geometry is None or len(geometry.net_band) < 3:
        return 0.0
    polygon = np.asarray(geometry.net_band, dtype=np.float32)
    points: list[np.ndarray] = []
    for index in sorted(indices, key=lambda item: float(rows[item]["time_seconds"])):
        row = rows[index]
        width = float(row.get("source_width") or 0.0)
        height = float(row.get("source_height") or 0.0)
        if width <= 0 or height <= 0:
            continue
        point = np.asarray([float(row["center_x"]) / width, float(row["center_y"]) / height])
        points.append(point)
        if cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0:
            return 1.0
    for first, second in pairwise(points):
        for fraction in np.linspace(0.1, 0.9, 9):
            point = first + (second - first) * fraction
            if cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0:
                return 0.85
    return 0.0


def _track_continuity_score(
    rows: list[dict[str, float | int | str]], indices: list[int]
) -> float:
    """Score temporal continuity without rejecting valid high-speed clears."""
    ordered = sorted(indices, key=lambda index: float(rows[index]["time_seconds"]))
    if len(ordered) < 2:
        return 0.0
    velocities: list[np.ndarray] = []
    penalties: list[float] = []
    for first_index, second_index in pairwise(ordered):
        first, second = rows[first_index], rows[second_index]
        delta = float(second["time_seconds"]) - float(first["time_seconds"])
        if delta <= 1e-6:
            penalties.append(1.0)
            continue
        vector = np.asarray(
            [float(second["center_x"]) - float(first["center_x"]), float(second["center_y"]) - float(first["center_y"])],
            dtype=float,
        )
        velocity = vector / delta
        velocities.append(velocity)
        speed = float(np.linalg.norm(velocity))
        penalties.append(float(np.clip((speed - 3600.0) / 3600.0, 0.0, 1.0)))
    if not velocities:
        return -1.0
    direction_penalty = 0.0
    for first, second in pairwise(velocities):
        first_norm = float(np.linalg.norm(first))
        second_norm = float(np.linalg.norm(second))
        if first_norm <= 1e-6 or second_norm <= 1e-6:
            continue
        cosine = float(np.dot(first, second) / (first_norm * second_norm))
        direction_penalty += max(0.0, -cosine) * 0.35
    return float(np.clip(0.85 - float(np.mean(penalties)) - direction_penalty / max(1, len(velocities)), -1.0, 1.0))


def _read_contact_hints(path: Path | None) -> list[dict[str, float | str]]:
    if path is None or not path.exists():
        return []
    hints: list[dict[str, float | str]] = []
    with path.open(newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            time_seconds = float(row["time_seconds"])
            audio = float(row.get("audio_hit_score") or 0.0)
            for side in ("near", "far"):
                if float(row.get(f"{side}_active_wrist_visible") or 0.0) <= 0.5:
                    continue
                swing = float(row.get(f"{side}_swing_score") or 0.0)
                wrist_speed = float(row.get(f"{side}_active_wrist_speed_normalized") or 0.0)
                support = max(swing, audio * 0.85)
                radius = 0.12
                kind = "contact"
                if support < 0.20:
                    # A shuttle being picked up or held may have no hit sound
                    # and no violent swing. Keep a weaker, tighter possession
                    # hint so it can break a tie without overriding flight
                    # continuity on its own.
                    support = min(0.32, 0.12 + wrist_speed * 0.25)
                    radius = 0.045
                    kind = "possession"
                hints.append(
                    {
                        "time": time_seconds,
                        "x": float(row.get(f"{side}_active_wrist_x_normalized") or 0.0),
                        "y": float(row.get(f"{side}_active_wrist_y_normalized") or 0.0),
                        "support": min(1.0, support),
                        "radius": radius,
                        "kind": kind,
                        "side": side,
                    }
                )
    return hints


def _track_contact_support(
    rows: list[dict[str, float | int | str]],
    indices: list[int],
    contact_hints: list[dict[str, float | str]],
) -> float:
    matches: list[float] = []
    for index in indices:
        row = rows[index]
        coordinate_width = float(row.get("source_width") or 0.0)
        coordinate_height = float(row.get("source_height") or 0.0)
        if coordinate_width <= 0 or coordinate_height <= 0:
            continue
        time_seconds = float(row["time_seconds"])
        x = float(row["center_x"]) / coordinate_width
        y = float(row["center_y"]) / coordinate_height
        for hint in contact_hints:
            delta = abs(time_seconds - float(hint["time"]))
            if delta > 0.22:
                continue
            distance = math.hypot(x - float(hint["x"]), y - float(hint["y"]))
            radius = float(hint.get("radius") or 0.12)
            if distance > radius:
                continue
            spatial = 1.0 - distance / radius
            temporal = 1.0 - delta / 0.22
            matches.append(float(hint["support"]) * spatial * temporal)
    if not matches:
        return 0.0
    matches.sort(reverse=True)
    return float(matches[0] + sum(matches[1:3]) * 0.25)


def _read_detections(path: Path) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    with path.open(newline="", encoding="utf-8") as source:
        for raw in csv.DictReader(source):
            rows.append(
                {
                    "time_seconds": float(raw["time_seconds"]),
                    "frame": int(raw["frame"]),
                    "source_width": int(raw["source_width"]) if raw.get("source_width") else "",
                    "source_height": int(raw["source_height"]) if raw.get("source_height") else "",
                    "confidence": float(raw["confidence"]),
                    "center_x": float(raw["center_x"]),
                    "center_y": float(raw["center_y"]),
                    "width": float(raw["width"]),
                    "height": float(raw["height"]),
                    "source": raw.get("source") or "yolo",
                    "detection_status": raw.get("detection_status") or "detected",
                    "evidence_weight": float(raw["evidence_weight"]) if raw.get("evidence_weight") not in (None, "") else 1.0,
                }
            )
    return rows


def _apply_user_rejections(
    rows: list[dict[str, float | int | str]], annotations: list[dict[str, object]]
) -> tuple[list[dict[str, float | int | str]], int]:
    rejected = [annotation for annotation in annotations if annotation.get("action") == "reject"]
    if not rejected:
        return rows, 0
    kept = []
    removed = 0
    for row in rows:
        coordinate_width = float(row.get("source_width") or 0.0)
        coordinate_height = float(row.get("source_height") or 0.0)
        if coordinate_width <= 0 or coordinate_height <= 0:
            kept.append(row)
            continue
        x = float(row["center_x"]) / coordinate_width
        y = float(row["center_y"]) / coordinate_height
        matches = any(
            abs(float(row["time_seconds"]) - float(annotation["time_seconds"])) <= 0.085
            and math.hypot(x - float(annotation["x_normalized"]), y - float(annotation["y_normalized"])) <= 0.03
            for annotation in rejected
        )
        if matches:
            removed += 1
        else:
            kept.append(row)
    return kept, removed


def _manual_annotation_rows(
    annotations: list[dict[str, object]],
    reference_rows: list[dict[str, float | int | str]] | None = None,
) -> list[dict[str, float | int | str]]:
    rows = [
        {
            "time_seconds": float(annotation["time_seconds"]),
            "frame": int(annotation["frame"]),
            "source_width": int(annotation["source_width"]),
            "source_height": int(annotation["source_height"]),
            "confidence": 1.0,
            "center_x": float(annotation["x_normalized"]) * int(annotation["source_width"]),
            "center_y": float(annotation["y_normalized"]) * int(annotation["source_height"]),
            "width": 8.0,
            "height": 8.0,
            "status": "manual",
            "track_id": "",
            "track_length": "",
            "flight_id": "",
            "source": "manual",
            "detection_status": "manual",
            "evidence_weight": 1.0,
        }
        for annotation in annotations
        if annotation.get("action") == "add"
    ]
    rows.sort(key=lambda row: (float(row["time_seconds"]), int(row["frame"])))
    if not rows:
        return rows

    groups: list[list[dict[str, float | int | str]]] = []
    for row in rows:
        if not groups or float(row["time_seconds"]) - float(groups[-1][-1]["time_seconds"]) > 1.25:
            groups.append([])
        groups[-1].append(row)

    references = [
        row
        for row in reference_rows or []
        if row.get("status") in {"tracked", "recovered"}
        and row.get("detection_status") != "inpainted"
        and str(row.get("flight_id") or "").strip()
    ]
    for group_index, group in enumerate(groups, start=1):
        best_match: tuple[float, str] | None = None
        for manual in group:
            manual_time = float(manual["time_seconds"])
            manual_x = float(manual["center_x"])
            manual_y = float(manual["center_y"])
            diagonal = max(
                1.0,
                math.hypot(float(manual["source_width"]), float(manual["source_height"])),
            )
            for reference in references:
                delta = abs(manual_time - float(reference["time_seconds"]))
                if delta > 1.5:
                    continue
                distance = math.hypot(
                    manual_x - float(reference["center_x"]),
                    manual_y - float(reference["center_y"]),
                )
                if distance > max(diagonal * 0.035, 140.0 + 1700.0 * delta):
                    continue
                score = delta + distance / diagonal
                candidate = (score, str(reference["flight_id"]))
                if best_match is None or candidate < best_match:
                    best_match = candidate
        flight_id = best_match[1] if best_match is not None else f"manual-{group_index}"
        for row in group:
            row["track_id"] = flight_id
            row["track_length"] = len(group)
            row["flight_id"] = flight_id
    return rows


def _stationary_labels(rows: list[dict[str, float | int | str]]) -> tuple[np.ndarray, set[int]]:
    if not rows:
        return np.empty(0, dtype=int), set()
    points = np.asarray([(float(row["center_x"]), float(row["center_y"])) for row in rows])
    labels = DBSCAN(eps=12.0, min_samples=5).fit_predict(points)
    frames = sorted({int(row["frame"]) for row in rows})
    frame_span = max(1, frames[-1] - frames[0])
    estimated_samples = max(1, len(frames))
    minimum_count = max(10, round(estimated_samples * 0.03))
    stationary: set[int] = set()
    for label in {int(value) for value in labels if value >= 0}:
        indices = np.flatnonzero(labels == label)
        cluster = points[indices]
        cluster_frames = [int(rows[index]["frame"]) for index in indices]
        persistence = (max(cluster_frames) - min(cluster_frames)) / frame_span
        if len(indices) >= minimum_count and persistence >= 0.30 and float(np.max(np.std(cluster, axis=0))) <= 2.5:
            stationary.add(label)

    # A tiny feather fragment or line reflection can be detected throughout a
    # video while its YOLO box jitters by several pixels. DBSCAN may chain that
    # dense core to nearby transient detections, making the whole cluster too
    # wide for the strict standard-deviation rule above. Build a second,
    # scale-aware occupancy map: a real shuttle visits one image cell briefly,
    # while venue clutter repeatedly occupies it across a large part of the
    # recording. Two offset grids avoid missing a site on a cell boundary.
    dimensions = [
        math.hypot(float(row.get("source_width") or 0.0), float(row.get("source_height") or 0.0))
        for row in rows
        if float(row.get("source_width") or 0.0) > 0 and float(row.get("source_height") or 0.0) > 0
    ]
    if dimensions:
        image_diagonal = float(np.median(dimensions))
    else:
        image_diagonal = math.hypot(float(np.max(points[:, 0])), float(np.max(points[:, 1])))
    cell_size = float(np.clip(image_diagonal * 0.008, 8.0, 24.0))
    static_sites: list[np.ndarray] = []
    for offset in (0.0, cell_size / 2.0):
        cells: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index, (x, y) in enumerate(points):
            cells[(math.floor((float(x) + offset) / cell_size), math.floor((float(y) + offset) / cell_size))].append(index)
        for indices in cells.values():
            unique_frames = sorted({int(rows[index]["frame"]) for index in indices})
            if len(unique_frames) < minimum_count:
                continue
            persistence = (unique_frames[-1] - unique_frames[0]) / frame_span
            if persistence < 0.30:
                continue
            static_sites.append(np.median(points[indices], axis=0))

    if static_sites:
        next_label = max((int(value) for value in labels if value >= 0), default=-1) + 1
        site_radius = cell_size * 1.4
        merged_sites: list[np.ndarray] = []
        for site in static_sites:
            if not any(float(np.linalg.norm(site - existing)) <= cell_size for existing in merged_sites):
                merged_sites.append(site)
        for site in merged_sites:
            site_label = next_label
            next_label += 1
            nearby = np.linalg.norm(points - site, axis=1) <= site_radius
            labels[nearby] = site_label
            stationary.add(site_label)
    return labels, stationary


def _predict_track_point(
    rows: list[dict[str, float | int | str]], indices: list[int], target_time: float
) -> tuple[float, float]:
    last = rows[indices[-1]]
    last_time = float(last["time_seconds"])
    delta = max(0.0, target_time - last_time)
    predicted_x, predicted_y = float(last["center_x"]), float(last["center_y"])
    if len(indices) < 2 or delta <= 0:
        return predicted_x, predicted_y
    previous = rows[indices[-2]]
    history_delta = last_time - float(previous["time_seconds"])
    if history_delta <= 1e-6:
        return predicted_x, predicted_y
    velocity_x = (predicted_x - float(previous["center_x"])) / history_delta
    velocity_y = (predicted_y - float(previous["center_y"])) / history_delta
    acceleration_x = 0.0
    acceleration_y = 0.0
    if len(indices) >= 3:
        earlier = rows[indices[-3]]
        earlier_delta = float(previous["time_seconds"]) - float(earlier["time_seconds"])
        if earlier_delta > 1e-6:
            earlier_velocity_x = (float(previous["center_x"]) - float(earlier["center_x"])) / earlier_delta
            earlier_velocity_y = (float(previous["center_y"]) - float(earlier["center_y"])) / earlier_delta
            velocity_delta = max((history_delta + earlier_delta) / 2.0, 1e-3)
            acceleration_x = float(np.clip((velocity_x - earlier_velocity_x) / velocity_delta, -4500.0, 4500.0))
            # Positive image-y acceleration is the weak-perspective analogue
            # of gravity. Negative values remain possible around a racket hit.
            acceleration_y = float(np.clip((velocity_y - earlier_velocity_y) / velocity_delta, -6000.0, 6000.0))
    predicted_x += velocity_x * delta + 0.5 * acceleration_x * delta * delta
    predicted_y += velocity_y * delta + 0.5 * acceleration_y * delta * delta
    return predicted_x, predicted_y


def _link_dynamic_rows(
    rows: list[dict[str, float | int | str]],
    dynamic_indices: list[int],
    maximum_gap_seconds: float = 0.28,
    maximum_speed_pixels: float = 2600.0,
) -> dict[int, int]:
    by_frame: dict[int, list[int]] = defaultdict(list)
    for index in dynamic_indices:
        if _is_linkable_detection(rows[index]):
            by_frame[int(rows[index]["frame"])].append(index)
    tracks: dict[int, list[int]] = {}
    next_track = 1
    for frame in sorted(by_frame):
        candidates = sorted(
            by_frame[frame],
            key=lambda index: (
                str(rows[index].get("detection_status") or "detected") == "inpainted",
                str(rows[index].get("source") or ""),
                -float(rows[index]["confidence"]),
            ),
        )
        pairs: list[tuple[float, int, int]] = []
        for track_id, indices in tracks.items():
            last = rows[indices[-1]]
            delta = float(rows[candidates[0]]["time_seconds"]) - float(last["time_seconds"])
            if delta <= 0 or delta > maximum_gap_seconds:
                continue
            predicted_x, predicted_y = _predict_track_point(
                rows, indices, float(rows[candidates[0]]["time_seconds"])
            )
            gate = 22.0 + maximum_speed_pixels * delta
            for index in candidates:
                row = rows[index]
                distance = math.hypot(float(row["center_x"]) - predicted_x, float(row["center_y"]) - predicted_y)
                if distance <= gate:
                    confidence_bonus = float(row["confidence"]) * 0.12
                    source_bonus = 0.08 if str(row.get("detection_status") or "") == "fused" else 0.0
                    manual_bonus = 0.30 if str(row.get("detection_status") or "") == "manual" else 0.0
                    pairs.append((distance / gate - confidence_bonus - source_bonus - manual_bonus, track_id, index))
        assigned_tracks: set[int] = set()
        assigned_rows: set[int] = set()
        for _cost, track_id, index in sorted(pairs):
            if track_id in assigned_tracks or index in assigned_rows:
                continue
            tracks[track_id].append(index)
            assigned_tracks.add(track_id)
            assigned_rows.add(index)
        for index in candidates:
            if index not in assigned_rows:
                tracks[next_track] = [index]
                next_track += 1

    accepted: dict[int, int] = {}
    final_track = 1
    for indices in tracks.values():
        if len(indices) < 3:
            continue
        points = np.asarray([(float(rows[index]["center_x"]), float(rows[index]["center_y"])) for index in indices])
        displacement = float(np.max(np.linalg.norm(points - points[0], axis=1)))
        if displacement < 12.0:
            continue
        for index in indices:
            accepted[index] = final_track
        final_track += 1
    return accepted


def _select_primary_track_indices(
    rows: list[dict[str, float | int | str]],
    proposed: dict[int, int],
    contact_hints: list[dict[str, float | str]] | None = None,
    geometry: CourtGeometry | None = None,
) -> set[int]:
    """Assign ownership at detection time rather than for a whole track.

    A detector track can continue after the shuttle lands while a new flight
    starts, so whole-interval selection would incorrectly discard the new
    flight or keep a landed tail. Local ownership lets the stronger candidate
    win only during the overlap and preserves reliable points on both sides.
    """
    candidates = [index for index in proposed if _is_linkable_detection(rows[index])]
    if not candidates:
        return set()
    hints = contact_hints or []
    grouped_candidates: dict[int, list[int]] = defaultdict(list)
    for index in candidates:
        grouped_candidates[proposed[index]].append(index)
    track_continuity = {
        track_id: _track_continuity_score(rows, indices)
        for track_id, indices in grouped_candidates.items()
    }

    hint_bucket_size = 0.22
    hints_by_bucket: dict[int, list[dict[str, float | str]]] = defaultdict(list)
    for hint in hints:
        hints_by_bucket[math.floor(float(hint["time"]) / hint_bucket_size)].append(hint)

    def point_hint_support(index: int, kinds: set[str]) -> float:
        if not hints:
            return 0.0
        row = rows[index]
        width = float(row.get("source_width") or 0.0)
        height = float(row.get("source_height") or 0.0)
        if width <= 0 or height <= 0:
            return 0.0
        time_seconds = float(row["time_seconds"])
        x = float(row["center_x"]) / width
        y = float(row["center_y"]) / height
        bucket = math.floor(time_seconds / hint_bucket_size)
        matches: list[float] = []
        for hint_bucket in (bucket - 1, bucket, bucket + 1):
            for hint in hints_by_bucket.get(hint_bucket, []):
                if str(hint.get("kind") or "contact") not in kinds:
                    continue
                delta = abs(time_seconds - float(hint["time"]))
                if delta > 0.22:
                    continue
                radius = float(hint.get("radius") or 0.12)
                distance = math.hypot(x - float(hint["x"]), y - float(hint["y"]))
                if distance <= radius:
                    matches.append(float(hint["support"]) * (1.0 - distance / radius) * (1.0 - delta / 0.22))
        if not matches:
            return 0.0
        matches.sort(reverse=True)
        return float(matches[0] + sum(matches[1:3]) * 0.25)

    contexts = {index: _geometry_context_score(rows[index], geometry) for index in candidates}
    axis_scores = {index: _axis_proximity_score(rows[index], geometry) for index in candidates}
    contacts = {index: point_hint_support(index, {"contact"}) for index in candidates}
    manual_anchors = {index: point_hint_support(index, {"manual_anchor"}) for index in candidates}
    ownership_hints = [hint for hint in hints if str(hint.get("kind") or "contact") == "contact"]
    track_contacts = {
        track_id: _track_contact_support(rows, indices, ownership_hints)
        for track_id, indices in grouped_candidates.items()
    }
    track_net_support = {
        track_id: _track_net_support(rows, indices, geometry)
        for track_id, indices in grouped_candidates.items()
    }
    credentialed_tracks = {
        track_id
        for track_id, indices in grouped_candidates.items()
        if any(str(rows[index].get("detection_status") or "") == "manual" for index in indices)
        or track_contacts[track_id] >= 0.08
        or track_net_support[track_id] > 0.0
    }
    background_suppressed_tracks = {
        track_id
        for track_id, indices in grouped_candidates.items()
        if track_id not in credentialed_tracks
        and sum(contexts[index] <= -0.70 for index in indices) / max(1, len(indices)) >= 0.50
    }

    def local_score(index: int) -> float:
        row = rows[index]
        evidence = float(row.get("confidence") or 0.0) * _evidence_weight(row)
        manual_bonus = 0.90 if str(row.get("detection_status") or "") == "manual" else 0.0
        track_id = proposed[index]
        ownership_bonus = min(0.75, track_contacts[track_id] * 1.5 + track_net_support[track_id] * 0.35)
        return (
            evidence
            + contexts[index] * 1.35
            + axis_scores[index] * 0.22
            + contacts[index] * 4.20
            + manual_anchors[index] * 5.50
            + track_continuity.get(track_id, 0.0) * 0.12
            + ownership_bonus
            + manual_bonus
        )

    scores = {index: local_score(index) for index in candidates}
    ordered = sorted(candidates, key=lambda index: float(rows[index]["time_seconds"]))
    ownership_window = 0.12
    bucket_size = 0.04
    bucket_tracks: dict[int, dict[int, int]] = defaultdict(dict)
    for index in ordered:
        bucket = math.floor(float(rows[index]["time_seconds"]) / bucket_size)
        track_id = proposed[index]
        previous = bucket_tracks[bucket].get(track_id)
        if previous is None or scores[index] > scores[previous]:
            bucket_tracks[bucket][track_id] = index
    bucket_winners: dict[int, list[int]] = {
        bucket: sorted(track_rows.values(), key=lambda index: scores[index], reverse=True)[:8]
        for bucket, track_rows in bucket_tracks.items()
    }
    accepted: set[int] = set()
    for index in ordered:
        if proposed[index] in background_suppressed_tracks:
            continue
        time_seconds = float(rows[index]["time_seconds"])
        if (
            contexts[index] <= -0.70
            and proposed[index] not in credentialed_tracks
            and str(rows[index].get("detection_status") or "") != "manual"
        ):
            continue
        bucket = math.floor(time_seconds / bucket_size)
        competitors = []
        for candidate_bucket in range(bucket - 3, bucket + 4):
            competitors.extend(
                other
                for other in bucket_winners.get(candidate_bucket, [])
                if proposed[other] != proposed[index]
                and abs(float(rows[other]["time_seconds"]) - time_seconds) <= ownership_window
            )
        if not competitors:
            accepted.add(index)
            continue
        winner = max(
            [index, *competitors],
            key=lambda candidate: (
                scores[candidate],
                contacts[candidate],
                contexts[candidate],
                float(rows[candidate].get("confidence") or 0.0),
            ),
        )
        if winner == index:
            accepted.add(index)
    for index in candidates:
        track_id = proposed[index]
        if str(rows[index].get("detection_status") or "") == "manual":
            ownership_evidence = "manual"
            ownership_confidence = 1.0
        elif manual_anchors[index] >= 0.04:
            ownership_evidence = "manual_anchor"
            ownership_confidence = min(0.99, 0.88 + manual_anchors[index] * 0.08)
        elif contacts[index] >= 0.04 or track_contacts[track_id] >= 0.08:
            ownership_evidence = "player_contact"
            ownership_confidence = min(0.98, 0.78 + max(contacts[index], track_contacts[track_id]) * 0.30)
        elif track_net_support[track_id] > 0:
            ownership_evidence = "net_crossing"
            ownership_confidence = min(0.92, 0.72 + track_net_support[track_id] * 0.18)
        elif (
            contexts[index] >= -0.15
            and axis_scores[index] >= 0.18
            and track_continuity.get(track_id, 0.0) >= 0.15
        ) or (
            geometry is None
            and track_continuity.get(track_id, 0.0) >= 0.15
        ):
            ownership_evidence = "court_continuity"
            ownership_confidence = float(
                np.clip(0.52 + contexts[index] * 0.16 + track_continuity[track_id] * 0.20, 0.45, 0.82)
            )
        else:
            ownership_evidence = "unknown"
            ownership_confidence = float(np.clip(0.20 + scores[index] * 0.08, 0.05, 0.44))
        if index not in accepted:
            ownership_confidence = min(ownership_confidence, 0.35)
        rows[index]["ownership_confidence"] = ownership_confidence
        rows[index]["ownership_evidence"] = ownership_evidence
    return {index for index in accepted if rows[index]["ownership_evidence"] != "unknown"}


def _drop_weak_partial_tracks(
    rows: list[dict[str, float | int | str]],
    proposed: dict[int, int],
    accepted_indices: set[int],
) -> set[int]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, track_id in proposed.items():
        grouped[track_id].append(index)
    kept = set(accepted_indices)
    for indices in grouped.values():
        accepted = sorted(
            (index for index in indices if index in accepted_indices),
            key=lambda index: float(rows[index]["time_seconds"]),
        )
        if not 0 < len(accepted) <= 2 or len(indices) < 8:
            continue
        if len(accepted) / len(indices) > 0.20:
            continue
        if any(str(rows[index].get("ownership_evidence")) != "court_continuity" for index in accepted):
            continue
        span = float(rows[accepted[-1]]["time_seconds"]) - float(rows[accepted[0]]["time_seconds"])
        if span <= 0.12:
            kept.difference_update(accepted)
    return kept


def _stitch_primary_flights(
    rows: list[dict[str, float | int | str]], accepted: dict[int, int]
) -> dict[int, int]:
    """Group short accepted tracklets into one flight across brief occlusions."""
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, track_id in accepted.items():
        grouped[track_id].append(index)
    tracklets = []
    for track_id, indices in grouped.items():
        ordered = sorted(indices, key=lambda index: float(rows[index]["time_seconds"]))
        tracklets.append(
            {
                "track_id": track_id,
                "start": float(rows[ordered[0]]["time_seconds"]),
                "end": float(rows[ordered[-1]]["time_seconds"]),
                "start_x": float(rows[ordered[0]]["center_x"]),
                "start_y": float(rows[ordered[0]]["center_y"]),
                "end_x": float(rows[ordered[-1]]["center_x"]),
                "end_y": float(rows[ordered[-1]]["center_y"]),
            }
        )
    tracklets.sort(key=lambda item: float(item["start"]))
    mapping: dict[int, int] = {}
    flight_id = 0
    previous = None
    for tracklet in tracklets:
        same_flight = False
        if previous is not None:
            gap = float(tracklet["start"]) - float(previous["end"])
            distance = math.hypot(
                float(tracklet["start_x"]) - float(previous["end_x"]),
                float(tracklet["start_y"]) - float(previous["end_y"]),
            )
            # The detector commonly disappears for several seconds through a
            # player, motion blur or a quiet net exchange. The primary-chain
            # competition has already removed simultaneous neighbouring balls,
            # so a physically reachable gap inside the same 3.5 s component is
            # one flight sequence rather than a new shuttle.
            same_flight = 0.0 <= gap <= 3.5 and distance <= 140.0 + 1700.0 * gap
        if not same_flight:
            flight_id += 1
        mapping[int(tracklet["track_id"])] = flight_id
        previous = tracklet
    return mapping


def _recover_short_optical_flow_gaps(
    video: Path | None,
    rows: list[dict[str, float | int | str]],
    accepted: dict[int, int],
    flight_ids: dict[int, int],
    maximum_attempts: int = 80,
) -> list[dict[str, float | int | str]]:
    """Recover only short, endpoint-validated gaps in an accepted flight.

    Lucas-Kanade can drift badly on a tiny blurred shuttle. A recovered chain
    is kept only when it arrives close to the next accepted YOLO point; a
    failed chain contributes no evidence at all.
    """
    if video is None or not video.exists() or not accepted:
        return []
    ordered = sorted(accepted, key=lambda index: float(rows[index]["time_seconds"]))
    gaps = []
    for first_index, second_index in pairwise(ordered):
        first_track, second_track = accepted[first_index], accepted[second_index]
        if flight_ids.get(first_track) != flight_ids.get(second_track):
            continue
        delta = float(rows[second_index]["time_seconds"]) - float(rows[first_index]["time_seconds"])
        if 0.14 <= delta <= 0.45:
            gaps.append((first_index, second_index, first_track, flight_ids.get(first_track, 0)))
    if not gaps:
        return []

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        return []
    video_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    export_stride = max(1, round(source_fps / 20.0))
    recovered: list[dict[str, float | int | str]] = []
    try:
        for first_index, second_index, track_id, flight_id in gaps[:maximum_attempts]:
            first, second = rows[first_index], rows[second_index]
            coordinate_width = float(first.get("source_width") or 0.0)
            coordinate_height = float(first.get("source_height") or 0.0)
            if coordinate_width <= 0 or coordinate_height <= 0:
                continue
            first_frame, second_frame = int(first["frame"]), int(second["frame"])
            if second_frame - first_frame < 3:
                continue
            capture.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
            ok, frame = capture.read()
            if not ok:
                continue
            previous_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            point = np.asarray(
                [[[float(first["center_x"]) * video_width / coordinate_width,
                   float(first["center_y"]) * video_height / coordinate_height]]],
                dtype=np.float32,
            )
            chain = []
            valid = True
            for frame_index in range(first_frame + 1, second_frame + 1):
                ok, frame = capture.read()
                if not ok:
                    valid = False
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                next_point, status, error = cv2.calcOpticalFlowPyrLK(
                    previous_gray,
                    gray,
                    point,
                    None,
                    winSize=(21, 21),
                    maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
                )
                if (
                    next_point is None
                    or status is None
                    or int(status.ravel()[0]) != 1
                    or error is None
                    or float(error.ravel()[0]) > 45.0
                ):
                    valid = False
                    break
                point = next_point
                previous_gray = gray
                if frame_index < second_frame and (frame_index - first_frame) % export_stride == 0:
                    chain.append((frame_index, float(point[0, 0, 0]), float(point[0, 0, 1])))
            if not valid:
                continue
            expected = np.asarray(
                [
                    float(second["center_x"]) * video_width / coordinate_width,
                    float(second["center_y"]) * video_height / coordinate_height,
                ]
            )
            endpoint_error = float(np.linalg.norm(point.reshape(2) - expected))
            if endpoint_error > max(10.0, math.hypot(video_width, video_height) * 0.009):
                continue
            confidence = min(float(first["confidence"]), float(second["confidence"])) * 0.45
            for frame_index, x, y in chain:
                fraction = (frame_index - first_frame) / (second_frame - first_frame)
                recovered.append(
                    {
                        "time_seconds": frame_index / source_fps,
                        "frame": frame_index,
                        "source_width": round(coordinate_width),
                        "source_height": round(coordinate_height),
                        "confidence": confidence,
                        "center_x": x * coordinate_width / video_width,
                        "center_y": y * coordinate_height / video_height,
                        "width": float(first["width"]) * (1.0 - fraction) + float(second["width"]) * fraction,
                        "height": float(first["height"]) * (1.0 - fraction) + float(second["height"]) * fraction,
                        "status": "recovered",
                        "track_id": track_id,
                        "track_length": "",
                        "flight_id": flight_id,
                        "source": "optical_flow",
                        "detection_status": "recovered",
                        "evidence_weight": 0.35,
                    }
                )
    finally:
        capture.release()
    return recovered


def analyze_shuttle_trajectory(
    detections_csv: Path,
    output_csv: Path,
    contact_features_csv: Path | None = None,
    video: Path | None = None,
    annotations_csv: Path | None = None,
    config: Path | None = None,
) -> dict[str, int | str]:
    rows = _read_detections(detections_csv)
    annotations = load_shuttle_annotations(annotations_csv)
    rows, user_rejected_points = _apply_user_rejections(rows, annotations)
    contact_hints = _read_contact_hints(contact_features_csv)
    contact_hints.extend(
        {
            "time": float(annotation["time_seconds"]),
            "x": float(annotation["x_normalized"]),
            "y": float(annotation["y_normalized"]),
            "support": 1.8,
            "radius": 0.055,
            "kind": "manual_anchor",
        }
        for annotation in annotations
        if annotation.get("action") == "add"
    )
    labels, stationary_labels = _stationary_labels(rows)
    dynamic_indices = [
        index
        for index, label in enumerate(labels)
        if int(label) not in stationary_labels and _is_linkable_detection(rows[index])
    ]
    proposed = _link_dynamic_rows(rows, dynamic_indices)
    geometry = CourtGeometry.from_json(config) if config is not None and config.exists() else None
    accepted_indices = _select_primary_track_indices(rows, proposed, contact_hints, geometry)
    accepted_indices = _drop_weak_partial_tracks(rows, proposed, accepted_indices)
    accepted = {index: proposed[index] for index in accepted_indices}
    primary_track_ids = set(accepted.values())
    flight_ids = _stitch_primary_flights(rows, accepted)
    recovered_rows = _recover_short_optical_flow_gaps(video, rows, accepted, flight_ids)
    track_lengths: dict[int, int] = defaultdict(int)
    proposed_lengths: dict[int, int] = defaultdict(int)
    for track_id in proposed.values():
        proposed_lengths[track_id] += 1
    for track_id in accepted.values():
        track_lengths[track_id] += 1
    grouped_proposals: dict[int, list[int]] = defaultdict(list)
    for index, track_id in proposed.items():
        grouped_proposals[track_id].append(index)
    contact_supported_tracks = sum(
        _track_contact_support(rows, indices, contact_hints) > 0
        for track_id, indices in grouped_proposals.items()
        if track_id in primary_track_ids
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "time_seconds",
        "frame",
        "source_width",
        "source_height",
        "confidence",
        "center_x",
        "center_y",
        "width",
        "height",
        "status",
        "track_id",
        "track_length",
        "flight_id",
        "source",
        "detection_status",
        "evidence_weight",
        "ownership_confidence",
        "ownership_evidence",
    ]
    output_rows = []
    for index, row in enumerate(rows):
        label = int(labels[index]) if len(labels) else -1
        track_id = proposed.get(index)
        status = (
            "tracked"
            if index in accepted
            else "competing"
            if track_id is not None
            else "stationary"
            if label in stationary_labels
            else "unlinked"
        )
        output_rows.append(
            {
                **row,
                "status": status,
                "track_id": track_id or "",
                "track_length": proposed_lengths.get(track_id, "") if track_id is not None else "",
                "flight_id": flight_ids.get(track_id, "") if status == "tracked" else "",
            }
        )
    output_rows.extend(recovered_rows)
    manual_rows = _manual_annotation_rows(annotations, output_rows)
    accepted_by_track: dict[int, list[dict[str, float | int | str]]] = defaultdict(list)
    for index, track_id in accepted.items():
        accepted_by_track[track_id].append(rows[index])
    for row in recovered_rows:
        track_id = int(row["track_id"])
        references = accepted_by_track.get(track_id, [])
        row["ownership_confidence"] = max(
            0.45,
            min((float(item.get("ownership_confidence") or 0.45) for item in references), default=0.45) * 0.75,
        )
        row["ownership_evidence"] = "owned_continuity"
    for row in manual_rows:
        row["ownership_confidence"] = 1.0
        row["ownership_evidence"] = "manual"
    output_rows.extend(manual_rows)
    output_rows.sort(key=lambda row: (float(row["time_seconds"]), int(row["frame"]), str(row["status"])))
    with output_csv.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in output_rows:
            writer.writerow(
                {
                    **row,
                    "time_seconds": f"{float(row['time_seconds']):.6f}",
                    "confidence": f"{float(row['confidence']):.6f}",
                    "center_x": f"{float(row['center_x']):.3f}",
                    "center_y": f"{float(row['center_y']):.3f}",
                    "width": f"{float(row['width']):.3f}",
                    "height": f"{float(row['height']):.3f}",
                    "ownership_confidence": f"{float(row.get('ownership_confidence') or 0.0):.6f}",
                    "ownership_evidence": str(row.get("ownership_evidence") or "unknown"),
                }
            )
    return {
        "raw_candidates": len(rows),
        "stationary_candidates": sum(int(label) in stationary_labels for label in labels),
        "tracked_points": len(accepted),
        "tracks": len(track_lengths),
        "flights": len(set(flight_ids.values())),
        "competing_points": len(proposed) - len(accepted),
        "contact_hints": len(contact_hints),
        "contact_supported_tracks": contact_supported_tracks,
        "recovered_points": len(recovered_rows),
        "manual_points": len(manual_rows),
        "user_rejected_points": user_rejected_points,
        "output": str(output_csv),
    }


def render_shuttle_trajectory(
    video: Path,
    trajectory_csv: Path,
    config: Path,
    output: Path,
    ffmpeg: Path | None = None,
    encoder: str = "auto",
    output_fps: float = 30.0,
    style: str = "debug",
) -> Path:
    if style not in {"debug", "trail"}:
        raise ValueError("Trajectory render style must be debug or trail")
    rows = _read_trajectory(trajectory_csv)
    if not rows:
        raise ValueError("Trajectory CSV has no detections")
    geometry = CourtGeometry.from_json(config)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first_frame = min(int(row["frame"]) for row in rows)
    last_frame = max(int(row["frame"]) for row in rows)
    stride = max(1, round(source_fps / output_fps))
    actual_fps = source_fps / stride
    output_width = min(1280, source_width)
    output_height = round(source_height * output_width / source_width)
    output_width -= output_width % 2
    output_height -= output_height % 2
    scale_x, scale_y = output_width / source_width, output_height / source_height
    accepted_statuses = {"tracked", "recovered", "manual"}
    by_frame: dict[int, list[dict[str, float | int | str]]] = defaultdict(list)
    tracks: dict[str, list[dict[str, float | int | str]]] = defaultdict(list)
    for row in rows:
        by_frame[int(row["frame"])].append(row)
        if row["status"] in accepted_statuses:
            tracks[_trajectory_track_key(row)].append(row)
    for track_rows in tracks.values():
        track_rows.sort(key=lambda row: (float(row["time_seconds"]), int(row["frame"])))
    track_times = {track_key: [float(row["time_seconds"]) for row in track_rows] for track_key, track_rows in tracks.items()}
    track_flights = {
        track_key: _trajectory_flight_key(track_rows[0])
        for track_key, track_rows in tracks.items()
        if track_rows
    }
    accepted_timeline = sorted(
        (row for row in rows if row["status"] in accepted_statuses),
        key=lambda row: (float(row["time_seconds"]), int(row["frame"])),
    )
    accepted_times = [float(row["time_seconds"]) for row in accepted_timeline]
    latest_timeline_index = 0
    capture.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="badminton-shuttle-") as temporary_directory:
        intermediate = Path(temporary_directory) / "trajectory.avi"
        writer = cv2.VideoWriter(
            str(intermediate), cv2.VideoWriter_fourcc(*"MJPG"), actual_fps, (output_width, output_height)
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError("Could not create temporary trajectory video")
        frame_index = first_frame
        written = 0
        try:
            while frame_index <= last_frame:
                ok, frame = capture.read()
                if not ok:
                    break
                if (frame_index - first_frame) % stride:
                    frame_index += 1
                    continue
                annotated = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_AREA)
                current_time = frame_index / source_fps
                while (
                    latest_timeline_index < len(accepted_timeline)
                    and accepted_times[latest_timeline_index] <= current_time + 0.02
                ):
                    latest_timeline_index += 1
                active_flight = (
                    _trajectory_flight_key(accepted_timeline[latest_timeline_index - 1])
                    if latest_timeline_index
                    else None
                )
                if style == "debug":
                    active = geometry.polygon("active_court_polygon", output_width, output_height)
                    airspace = geometry.projected_shuttle_volume_polygon(output_width, output_height)
                    cv2.polylines(annotated, [active], True, (40, 70, 245), 2, cv2.LINE_AA)
                    cv2.polylines(annotated, [airspace], True, (40, 210, 210), 1, cv2.LINE_AA)
                for track_key, track_rows in tracks.items():
                    if style == "trail" and active_flight and track_flights.get(track_key) != active_flight:
                        continue
                    times = track_times[track_key]
                    left = bisect_left(times, current_time - 0.75)
                    right = bisect_right(times, current_time + 0.02)
                    for trail in _trajectory_trail_segments(
                        track_rows[left:right], scale_x, scale_y, source_fps, source_width
                    ):
                        if len(trail) >= 2:
                            color = (98, 157, 255) if style == "trail" else _track_color(track_key)
                            cv2.polylines(
                                annotated,
                                [np.asarray(trail, dtype=np.int32)],
                                False,
                                color,
                                4 if style == "trail" else 3,
                                cv2.LINE_AA,
                            )
                if style == "trail":
                    writer.write(annotated)
                    written += 1
                    frame_index += 1
                    continue
                current_rows = by_frame.get(frame_index, [])
                for row in current_rows:
                    point = (round(float(row["center_x"]) * scale_x), round(float(row["center_y"]) * scale_y))
                    if row["status"] in {"tracked", "recovered"}:
                        track_key = _trajectory_track_key(row)
                        color = _track_color(track_key)
                        cv2.circle(annotated, point, 9, (15, 15, 15), 3, cv2.LINE_AA)
                        cv2.circle(annotated, point, 7, color, 2, cv2.LINE_AA)
                        cv2.putText(
                            annotated,
                            f"{'OF' if row['status'] == 'recovered' else 'T'}{track_key} {float(row['confidence']):.2f}",
                            (point[0] + 10, point[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.42,
                            color,
                            1,
                            cv2.LINE_AA,
                        )
                    elif row["status"] == "stationary":
                        cv2.drawMarker(annotated, point, (110, 110, 110), cv2.MARKER_TILTED_CROSS, 7, 1)
                    elif row["status"] == "competing":
                        cv2.drawMarker(annotated, point, (70, 130, 255), cv2.MARKER_TILTED_CROSS, 11, 2)
                    elif row["status"] == "manual":
                        cv2.circle(annotated, point, 9, (235, 90, 220), 2, cv2.LINE_AA)
                        cv2.putText(
                            annotated,
                            "USER",
                            (point[0] + 10, point[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.42,
                            (235, 90, 220),
                            1,
                            cv2.LINE_AA,
                        )
                cv2.rectangle(annotated, (12, 12), (460, 72), (18, 25, 22), -1)
                cv2.putText(
                    annotated,
                    "SHUTTLE TRACK  yellow=projected volume  red=active court",
                    (24, 37),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (225, 235, 230),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    f"source {current_time:08.3f}s  orange crosses lose single-shuttle competition",
                    (24, 59),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (170, 190, 178),
                    1,
                    cv2.LINE_AA,
                )
                writer.write(annotated)
                written += 1
                frame_index += 1
        finally:
            capture.release()
            writer.release()
        if written == 0:
            raise RuntimeError("No frames were available for the trajectory preview")
        _encode_preview(
            intermediate,
            video,
            output,
            first_frame / source_fps,
            (last_frame - first_frame + 1) / source_fps,
            ffmpeg,
            encoder,
        )
    return output


def _read_trajectory(path: Path) -> list[dict[str, float | int | str]]:
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    return [
        {
            **row,
            "time_seconds": float(row["time_seconds"]),
            "frame": int(row["frame"]),
            "confidence": float(row["confidence"]),
            "center_x": float(row["center_x"]),
            "center_y": float(row["center_y"]),
            "track_id": int(row["track_id"]) if str(row["track_id"] or "").lstrip("-").isdigit() else str(row["track_id"] or ""),
        }
        for row in rows
    ]


def _trajectory_track_key(row: dict[str, float | int | str]) -> str:
    track_id = str(row.get("track_id") or "").strip()
    if track_id:
        return track_id
    flight_id = str(row.get("flight_id") or "").strip()
    if flight_id:
        return flight_id
    return f"frame-{int(row['frame'])}"


def _trajectory_flight_key(row: dict[str, float | int | str]) -> str:
    return str(row.get("flight_id") or row.get("track_id") or "").strip()


def _trajectory_trail_segments(
    rows: list[dict[str, float | int | str]],
    scale_x: float,
    scale_y: float,
    source_fps: float,
    source_width: int,
) -> list[list[tuple[int, int]]]:
    if not rows:
        return []
    max_gap = max(0.20, 5.0 / max(source_fps, 1.0))
    max_speed = max(3600.0, float(source_width) * 2.6)
    segments: list[list[tuple[int, int]]] = []
    segment: list[tuple[int, int]] = []
    previous = None
    for row in rows:
        current_time = float(row["time_seconds"])
        current = (float(row["center_x"]), float(row["center_y"]))
        if previous is not None:
            previous_time, previous_point = previous
            delta = current_time - previous_time
            distance = math.hypot(current[0] - previous_point[0], current[1] - previous_point[1])
            if delta <= 1e-6 or delta > max_gap or distance / max(delta, 1e-6) > max_speed:
                if len(segment) >= 2:
                    segments.append(segment)
                segment = []
        segment.append((round(current[0] * scale_x), round(current[1] * scale_y)))
        previous = (current_time, current)
    if len(segment) >= 2:
        segments.append(segment)
    return segments


def _track_color(track_id: int | str) -> tuple[int, int, int]:
    colors = ((40, 235, 255), (255, 180, 60), (80, 240, 120), (230, 90, 220))
    try:
        numeric_id = int(track_id)
    except (TypeError, ValueError):
        numeric_id = sum(ord(character) for character in str(track_id))
    return colors[(numeric_id - 1) % len(colors)]


def _encode_preview(
    intermediate: Path,
    source_video: Path,
    output: Path,
    start_seconds: float,
    duration: float,
    ffmpeg: Path | None,
    encoder: str,
) -> None:
    ffmpeg_path = resolve_ffmpeg(ffmpeg)

    def command_for(video_encoder: str) -> list[str]:
        return [
            str(ffmpeg_path),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(intermediate),
            "-ss",
            f"{start_seconds:.6f}",
            "-t",
            f"{duration:.6f}",
            "-i",
            str(source_video),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            *h264_encoding_arguments(video_encoder, 23, "preview"),
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-shortest",
            "-movflags",
            "+faststart",
            str(output),
        ]

    run_ffmpeg_with_encoder_fallback(ffmpeg_path, encoder, command_for, output)
