from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from .geometry import CourtGeometry
from .io import load_rallies, write_rows

EVENT_FIELDS = [
    "rally",
    "event",
    "confidence",
    "event_time_seconds",
    "last_hitter",
    "landing_side",
    "post_rally_probability",
    "terminal_x_normalized",
    "terminal_y_normalized",
    "court_x_meters",
    "court_y_meters",
    "homography_uncertainty_meters",
    "apparent_net_crossing",
    "net_crossing_time_seconds",
    "net_crossing_x_meters",
    "net_crossing_confidence",
    "height_ambiguous",
    "score_usable",
    "reason",
]


def _read_primary_trajectory(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as source:
        return [
            {
                "time": float(row["time_seconds"]),
                "x": float(row["center_x"]),
                "y": float(row["center_y"]),
                "source_width": float(row["source_width"]) if row.get("source_width") else 0.0,
                "source_height": float(row["source_height"]) if row.get("source_height") else 0.0,
                "flight_id": row.get("flight_id") or row.get("track_id") or "0",
                "status": row.get("status") or "tracked",
            }
            for row in csv.DictReader(source)
            if row.get("status") in {"tracked", "recovered", "manual"}
            and row.get("detection_status") != "inpainted"
        ]


def _last_hitters(path: Path | None) -> dict[int, str]:
    if path is None or not path.exists():
        return {}
    grouped: dict[int, list[tuple[float, str]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            grouped[int(row["rally"])].append((float(row["time_seconds"]), str(row["player"])))
    return {rally: max(events)[1] for rally, events in grouped.items() if events}


def _terminal_velocity(points: list[dict[str, Any]]) -> tuple[float, float]:
    if len(points) < 2:
        return 0.0, 0.0
    recent = points[-min(4, len(points)) :]
    velocities = []
    for previous, current in pairwise(recent):
        delta = float(current["time"]) - float(previous["time"])
        if 0 < delta <= 0.5:
            velocities.append(((float(current["x"]) - float(previous["x"])) / delta, (float(current["y"]) - float(previous["y"])) / delta))
    if not velocities:
        return 0.0, 0.0
    return float(np.median([value[0] for value in velocities])), float(np.median([value[1] for value in velocities]))


def _infer_pickup_landing(
    features: pd.DataFrame | None,
    rally_end: float,
    geometry: CourtGeometry,
) -> dict[str, Any] | None:
    if features is None or features.empty:
        return None
    required = {"time_seconds"}
    for side in ("near", "far"):
        required.update(
            {
                f"{side}_person_visible",
                f"{side}_active_wrist_visible",
                f"{side}_active_wrist_x_normalized",
                f"{side}_active_wrist_y_normalized",
                f"{side}_foot_x_normalized",
                f"{side}_foot_y_normalized",
            }
        )
    if not required.issubset(features.columns):
        return None

    before = features[
        (features["time_seconds"] >= rally_end - 0.55)
        & (features["time_seconds"] <= rally_end + 0.05)
    ]
    after = features[
        (features["time_seconds"] >= rally_end - 0.05)
        & (features["time_seconds"] <= rally_end + 1.6)
    ]
    if before.empty or after.empty:
        return None

    candidates = []
    for side in ("near", "far"):
        wrist_x = f"{side}_active_wrist_x_normalized"
        wrist_y = f"{side}_active_wrist_y_normalized"
        foot_x = f"{side}_foot_x_normalized"
        foot_y = f"{side}_foot_y_normalized"
        baseline_values = pd.to_numeric(before[wrist_y], errors="coerce").dropna()
        if baseline_values.empty:
            continue
        baseline_y = float(baseline_values.median())
        visible = after[
            (pd.to_numeric(after[f"{side}_person_visible"], errors="coerce") >= 0.5)
            & (pd.to_numeric(after[f"{side}_active_wrist_visible"], errors="coerce") >= 0.5)
        ].copy()
        if visible.empty:
            continue
        for column in (wrist_x, wrist_y, foot_x, foot_y):
            visible[column] = pd.to_numeric(visible[column], errors="coerce")
        visible = visible.dropna(subset=[wrist_x, wrist_y, foot_x, foot_y])
        if visible.empty:
            continue
        visible["pickup_distance"] = np.hypot(
            visible[wrist_x] - visible[foot_x],
            visible[wrist_y] - visible[foot_y],
        )
        visible["wrist_descent"] = visible[wrist_y] - baseline_y
        visible = visible[
            (visible["pickup_distance"] <= 0.14)
            & (visible["wrist_descent"] >= 0.08)
            & (visible[wrist_y] >= visible[foot_y] - 0.16)
        ]
        visible = visible[
            [geometry.contains_active_court(float(row[wrist_x]), float(row[wrist_y])) for _, row in visible.iterrows()]
        ]
        if visible.empty:
            continue
        visible["pickup_score"] = visible["wrist_descent"] + (0.14 - visible["pickup_distance"])
        row = visible.sort_values("pickup_score", ascending=False).iloc[0]
        candidates.append(
            {
                "side": side,
                "time": float(row["time_seconds"]),
                "x": float(row[wrist_x]),
                "y": float(row[wrist_y]),
                "score": float(row["pickup_score"]),
            }
        )
    if not candidates:
        return None
    candidates.sort(key=lambda row: row["score"], reverse=True)
    if len(candidates) > 1 and candidates[0]["score"] - candidates[1]["score"] < 0.05:
        return None
    return candidates[0]


def _apparent_net_crossing(
    points: list[dict[str, Any]],
    geometry: CourtGeometry,
    width: int,
    height: int,
) -> dict[str, Any]:
    mapped = []
    for point in points:
        coordinate_width = round(float(point.get("source_width") or width))
        coordinate_height = round(float(point.get("source_height") or height))
        court = geometry.image_to_court((float(point["x"]), float(point["y"])), coordinate_width, coordinate_height)
        if court is not None:
            mapped.append((float(point["time"]), court[0], court[1]))
    candidates = []
    for previous, current in pairwise(mapped):
        delta_time = current[0] - previous[0]
        if not 0 < delta_time <= 0.55:
            continue
        first_side, second_side = previous[2] - 6.7, current[2] - 6.7
        if first_side == 0 or second_side == 0 or first_side * second_side < 0:
            denominator = abs(first_side) + abs(second_side)
            ratio = abs(first_side) / denominator if denominator > 1e-9 else 0.5
            crossing_time = previous[0] + delta_time * ratio
            crossing_x = previous[1] + (current[1] - previous[1]) * ratio
            confidence = min(0.55, 0.30 + min(abs(current[2] - previous[2]) / 3.0, 0.25))
            candidates.append((confidence, crossing_time, crossing_x))
    if not candidates:
        return {
            "apparent_net_crossing": "no",
            "net_crossing_time_seconds": "",
            "net_crossing_x_meters": "",
            "net_crossing_confidence": 0.0,
            "height_ambiguous": "yes",
        }
    confidence, crossing_time, crossing_x = max(candidates)
    return {
        "apparent_net_crossing": "candidate",
        "net_crossing_time_seconds": crossing_time,
        "net_crossing_x_meters": crossing_x,
        "net_crossing_confidence": confidence,
        "height_ambiguous": "yes",
    }


def _infer_terminal_event(
    points: list[dict[str, Any]],
    rally_end: float,
    geometry: CourtGeometry,
    width: int,
    height: int,
    last_hitter: str,
    post_rally_probability: float | None = None,
) -> dict[str, Any]:
    if not points:
        return {
            "event": "unknown",
            "confidence": 0.0,
            "event_time_seconds": rally_end,
            "last_hitter": last_hitter,
            "landing_side": "unknown",
            "post_rally_probability": "" if post_rally_probability is None else post_rally_probability,
            "terminal_x_normalized": "",
            "terminal_y_normalized": "",
            "court_x_meters": "",
            "court_y_meters": "",
            "homography_uncertainty_meters": "",
            **_apparent_net_crossing([], geometry, width, height),
            "score_usable": "no",
            "reason": "no accepted shuttle trajectory near rally end",
        }

    terminal = points[-1]
    coordinate_width = float(terminal.get("source_width") or width)
    coordinate_height = float(terminal.get("source_height") or height)
    x, y = float(terminal["x"]) / coordinate_width, float(terminal["y"]) / coordinate_height
    terminal_flight = [point for point in points if point.get("flight_id") == terminal.get("flight_id")]
    velocity_x, velocity_y = _terminal_velocity(terminal_flight)
    speed = math.hypot(velocity_x, velocity_y)
    prior_terminal_points = [
        point
        for point in terminal_flight[:-1]
        if 0.05 <= float(terminal["time"]) - float(point["time"]) <= 0.5
    ]
    descent_span = (
        float(terminal["y"]) - float(np.median([float(point["y"]) for point in prior_terminal_points]))
        if prior_terminal_points
        else 0.0
    )
    strongly_descending = descent_span >= coordinate_height * 0.04
    descending = velocity_y >= 12.0 or strongly_descending
    terminal_gap = rally_end - float(terminal["time"])
    near_end = -0.5 <= terminal_gap <= 1.5
    near_net = geometry.contains_normalized("net_band", x, y)
    inside_ground = geometry.contains_active_court(x, y)
    in_near = geometry.contains_normalized("near_player_zone", x, y)
    in_far = geometry.contains_normalized("far_player_zone", x, y)
    landing_side = "near" if in_near and not in_far else "far" if in_far and not in_near else "unknown"
    ground_top = min((point[1] for point in geometry.active_court_polygon or []), default=0.55)
    event, confidence, reason = "unknown", 0.20, "single-camera terminal geometry is ambiguous"
    if near_end and near_net and (descending or speed <= 160.0):
        event, confidence, reason = "net_candidate", 0.55, "track ended in calibrated net band"
    elif near_end and descending and inside_ground:
        confidence = 0.82 if strongly_descending or terminal.get("status") == "manual" else 0.72
        event, reason = "landing_in_candidate", "descending track ended inside active-court ground polygon"
    elif near_end and descending and y >= ground_top and geometry.contains_shuttle_volume(x, y):
        event, confidence, reason = "landing_out_candidate", 0.50, "descending track ended outside active court but inside projected flight volume"
    elif terminal_gap > 0.8:
        event, confidence, reason = "flight_lost", 0.25, "accepted trajectory disappeared before the reviewed rally end"

    court_point = geometry.image_to_court(
        (float(terminal["x"]), float(terminal["y"])),
        round(coordinate_width),
        round(coordinate_height),
    )
    court_x = f"{court_point[0]:.3f}" if court_point is not None else ""
    court_y = f"{court_point[1]:.3f}" if court_point is not None else ""
    homography = geometry.homography_quality(width, height)
    uncertainty = homography.get("p95_uncertainty_meters", "") if homography.get("available") else ""
    net_crossing = _apparent_net_crossing(points, geometry, width, height)
    landing_score_usable = bool(
        event == "landing_in_candidate"
        and landing_side in {"near", "far"}
        and confidence >= 0.78
        and (post_rally_probability is None or post_rally_probability <= 0.45)
    )
    projected_score_usable = bool(
        court_point is not None
        and homography.get("score_safe") is True
        and last_hitter in {"near", "far"}
        and event in {"landing_out_candidate", "net_candidate"}
        and confidence >= 0.75
    )
    return {
        "event": event,
        "confidence": confidence,
        "event_time_seconds": float(terminal["time"]),
        "last_hitter": last_hitter,
        "landing_side": landing_side,
        "post_rally_probability": "" if post_rally_probability is None else post_rally_probability,
        "terminal_x_normalized": x,
        "terminal_y_normalized": y,
        "court_x_meters": court_x,
        "court_y_meters": court_y,
        "homography_uncertainty_meters": uncertainty,
        **net_crossing,
        "score_usable": "yes" if landing_score_usable or projected_score_usable else "no",
        "reason": reason,
    }


def analyze_terminal_events(
    video: Path,
    config: Path,
    rallies_csv: Path,
    trajectory_csv: Path,
    output_csv: Path,
    contacts_csv: Path | None = None,
    summary_json: Path | None = None,
    probabilities_csv: Path | None = None,
    features_csv: Path | None = None,
) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    geometry = CourtGeometry.from_json(config)
    trajectory = _read_primary_trajectory(trajectory_csv)
    last_hitters = _last_hitters(contacts_csv)
    probability_frame = None
    if probabilities_csv is not None and probabilities_csv.exists():
        probability_frame = pd.read_csv(probabilities_csv)
    feature_frame = None
    if features_csv is not None and features_csv.exists():
        feature_frame = pd.read_csv(features_csv)
    results = []
    for rally, (start, end) in enumerate(load_rallies(rallies_csv), 1):
        points = [point for point in trajectory if start - 0.2 <= float(point["time"]) <= end + 0.8]
        post_probability = None
        if probability_frame is not None:
            post = probability_frame[
                (probability_frame["time_seconds"] >= end - 0.05)
                & (probability_frame["time_seconds"] <= end + 0.6)
            ]
            if not post.empty:
                post_probability = float(post["rally_probability"].max())
        event = _infer_terminal_event(
            points,
            end,
            geometry,
            width,
            height,
            last_hitters.get(rally, "unknown"),
            post_probability,
        )
        pickup = _infer_pickup_landing(feature_frame, end, geometry)
        if (
            pickup is not None
            and event["event"] in {"unknown", "flight_lost"}
            and (post_probability is None or post_probability <= 0.45)
        ):
            court_point = geometry.image_to_court(
                (pickup["x"] * width, pickup["y"] * height),
                width,
                height,
            )
            event.update(
                {
                    "event": "landing_in_candidate",
                    "confidence": 0.80,
                    "event_time_seconds": pickup["time"],
                    "landing_side": pickup["side"],
                    "terminal_x_normalized": pickup["x"],
                    "terminal_y_normalized": pickup["y"],
                    "court_x_meters": f"{court_point[0]:.3f}" if court_point is not None else "",
                    "court_y_meters": f"{court_point[1]:.3f}" if court_point is not None else "",
                    "score_usable": "yes",
                    "reason": f"{pickup['side']} player picked up the shuttle inside the active court",
                }
            )
        results.append({"rally": rally, **event})
    formatted = [
        {
            **row,
            "confidence": f"{float(row['confidence']):.3f}",
            "event_time_seconds": f"{float(row['event_time_seconds']):.3f}",
            "terminal_x_normalized": f"{float(row['terminal_x_normalized']):.6f}"
            if row["terminal_x_normalized"] != ""
            else "",
            "terminal_y_normalized": f"{float(row['terminal_y_normalized']):.6f}"
            if row["terminal_y_normalized"] != ""
            else "",
            "post_rally_probability": f"{float(row['post_rally_probability']):.3f}"
            if row["post_rally_probability"] != ""
            else "",
            "net_crossing_time_seconds": f"{float(row['net_crossing_time_seconds']):.3f}"
            if row["net_crossing_time_seconds"] != ""
            else "",
            "net_crossing_x_meters": f"{float(row['net_crossing_x_meters']):.3f}"
            if row["net_crossing_x_meters"] != ""
            else "",
            "net_crossing_confidence": f"{float(row['net_crossing_confidence']):.3f}",
        }
        for row in results
    ]
    write_rows(output_csv, EVENT_FIELDS, formatted)
    summary = {
        "available": bool(trajectory),
        "rallies": results,
        "event_counts": {
            event: sum(row["event"] == event for row in results)
            for event in sorted({str(row["event"]) for row in results})
        },
        "score_usable_rallies": sum(row["score_usable"] == "yes" for row in results),
        "disclaimer": (
            "Terminal events are conservative candidates. A single camera cannot resolve shuttle height, so unknown "
            "is preferred and no event changes score or editing automatically."
        ),
    }
    if summary_json is not None:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
