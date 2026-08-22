from __future__ import annotations

import bisect
import csv
import math
from collections import defaultdict
from pathlib import Path

from .geometry import CourtGeometry

DETECTION_FIELDS = [
    "time_seconds",
    "frame",
    "source_width",
    "source_height",
    "confidence",
    "center_x",
    "center_y",
    "width",
    "height",
    "source",
    "detection_status",
    "evidence_weight",
]


def _read(path: Path | None, default_source: str) -> list[dict[str, float | int | str]]:
    if path is None or not path.exists():
        return []
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as source:
        for raw in csv.DictReader(source):
            rows.append(
                {
                    "time_seconds": float(raw["time_seconds"]),
                    "frame": int(raw["frame"]),
                    "source_width": int(raw["source_width"]),
                    "source_height": int(raw["source_height"]),
                    "confidence": float(raw["confidence"]),
                    "center_x": float(raw["center_x"]),
                    "center_y": float(raw["center_y"]),
                    "width": float(raw["width"]),
                    "height": float(raw["height"]),
                    "source": raw.get("source") or default_source,
                    "detection_status": raw.get("detection_status") or "detected",
                    "evidence_weight": float(raw.get("evidence_weight") or 1.0),
                }
            )
    rows.sort(key=lambda row: (float(row["time_seconds"]), -float(row["confidence"])))
    return rows


def _normalized_distance(first: dict, second: dict) -> float:
    first_x = float(first["center_x"]) / max(1.0, float(first["source_width"]))
    first_y = float(first["center_y"]) / max(1.0, float(first["source_height"]))
    second_x = float(second["center_x"]) / max(1.0, float(second["source_width"]))
    second_y = float(second["center_y"]) / max(1.0, float(second["source_height"]))
    return math.hypot(first_x - second_x, first_y - second_y)


def _fused_row(tracknet: dict, yolo: dict) -> dict[str, float | int | str]:
    target_width = int(tracknet["source_width"])
    target_height = int(tracknet["source_height"])
    tx = float(tracknet["center_x"]) / target_width
    ty = float(tracknet["center_y"]) / target_height
    yx = float(yolo["center_x"]) / float(yolo["source_width"])
    yy = float(yolo["center_y"]) / float(yolo["source_height"])
    tracknet_weight = max(0.25, float(tracknet["confidence"]))
    yolo_weight = max(0.20, float(yolo["confidence"]))
    total = tracknet_weight + yolo_weight
    confidence = min(1.0, 0.45 + 0.35 * float(tracknet["confidence"]) + 0.20 * float(yolo["confidence"]))
    return {
        "time_seconds": float(tracknet["time_seconds"]),
        "frame": int(tracknet["frame"]),
        "source_width": target_width,
        "source_height": target_height,
        "confidence": confidence,
        "center_x": (tx * tracknet_weight + yx * yolo_weight) / total * target_width,
        "center_y": (ty * tracknet_weight + yy * yolo_weight) / total * target_height,
        "width": max(float(tracknet["width"]), float(yolo["width"]) * target_width / float(yolo["source_width"])),
        "height": max(
            float(tracknet["height"]), float(yolo["height"]) * target_height / float(yolo["source_height"])
        ),
        "source": "hybrid",
        "detection_status": "fused",
        "evidence_weight": 1.0,
    }


def fuse_shuttle_detections(
    yolo_csv: Path | None,
    tracknet_csv: Path | None,
    config: Path,
    output_csv: Path,
    maximum_time_delta: float = 0.075,
    agreement_radius: float = 0.035,
) -> dict[str, int | str]:
    geometry = CourtGeometry.from_json(config)
    yolo = _read(yolo_csv, "yolo")
    tracknet = _read(tracknet_csv, "tracknet")
    yolo_times = [float(row["time_seconds"]) for row in yolo]
    used_yolo: set[int] = set()
    output: list[dict[str, float | int | str]] = []
    counts: defaultdict[str, int] = defaultdict(int)

    for tracknet_row in tracknet:
        time_seconds = float(tracknet_row["time_seconds"])
        left = bisect.bisect_left(yolo_times, time_seconds - maximum_time_delta)
        right = bisect.bisect_right(yolo_times, time_seconds + maximum_time_delta)
        choices = [
            (index, _normalized_distance(tracknet_row, yolo[index]))
            for index in range(left, right)
            if index not in used_yolo
        ]
        matched = min(choices, key=lambda item: item[1]) if choices else None
        if matched is not None and matched[1] <= agreement_radius and tracknet_row["detection_status"] == "detected":
            used_yolo.add(matched[0])
            output.append(_fused_row(tracknet_row, yolo[matched[0]]))
            counts["fused"] += 1
            continue

        status = str(tracknet_row["detection_status"])
        row = {**tracknet_row}
        if status == "inpainted":
            row["evidence_weight"] = min(0.15, float(row["evidence_weight"]))
            row["confidence"] = min(0.28, float(row["confidence"]))
        else:
            row["evidence_weight"] = min(0.90, float(row["evidence_weight"]))
        output.append(row)
        counts[status] += 1

    for index, row in enumerate(yolo):
        if index in used_yolo:
            continue
        nx = float(row["center_x"]) / float(row["source_width"])
        ny = float(row["center_y"]) / float(row["source_height"])
        if not geometry.contains_shuttle_volume(nx, ny) or geometry.excluded_normalized(nx, ny):
            continue
        output.append({**row, "source": "yolo", "detection_status": "detected", "evidence_weight": 0.65})
        counts["yolo_only"] += 1

    output.sort(key=lambda row: (float(row["time_seconds"]), -float(row["confidence"]), str(row["source"])))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=DETECTION_FIELDS)
        writer.writeheader()
        writer.writerows(output)
    return {"output": str(output_csv), "points": len(output), **counts}
