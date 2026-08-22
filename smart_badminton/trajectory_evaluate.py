from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from .geometry import CourtGeometry
from .shuttle_annotations import load_shuttle_annotations


def _read_points(path: Path) -> list[dict[str, float | str]]:
    if not path.exists():
        return []
    points = []
    with path.open(newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            width = float(row.get("source_width") or 0.0)
            height = float(row.get("source_height") or 0.0)
            if width <= 0 or height <= 0:
                continue
            points.append(
                {
                    "time": float(row["time_seconds"]),
                    "x": float(row["center_x"]) / width,
                    "y": float(row["center_y"]) / height,
                    "status": str(row.get("status") or ""),
                    "source": str(row.get("source") or ""),
                    "ownership_evidence": str(row.get("ownership_evidence") or ""),
                }
            )
    return points


def _matched(annotation: dict, points: list[dict[str, float | str]]) -> bool:
    return any(
        abs(float(point["time"]) - float(annotation["time_seconds"])) <= 0.16
        and math.hypot(
            float(point["x"]) - float(annotation["x_normalized"]),
            float(point["y"]) - float(annotation["y_normalized"]),
        )
        <= 0.035
        for point in points
    )


def evaluate_shuttle_annotations(
    detections_csv: Path,
    trajectory_csv: Path,
    annotations_csv: Path,
    output_json: Path | None = None,
    config: Path | None = None,
) -> dict:
    annotations = load_shuttle_annotations(annotations_csv)
    positives = [row for row in annotations if row.get("action") == "add"]
    negatives = [row for row in annotations if row.get("action") == "reject"]
    detections = _read_points(detections_csv)
    trajectory = _read_points(trajectory_csv)
    owned = [row for row in trajectory if row["status"] in {"tracked", "recovered", "manual"}]
    owned_detector = [row for row in owned if row["source"] != "manual"]
    geometry = CourtGeometry.from_json(config) if config is not None and config.exists() else None
    excluded_owned = 0
    if geometry is not None:
        excluded_owned = sum(
            geometry.excluded_normalized(float(point["x"]), float(point["y"])) for point in owned
        )
    report = {
        "positive_annotations": len(positives),
        "negative_annotations": len(negatives),
        "raw_detector_positive_recall": (
            sum(_matched(row, detections) for row in positives) / len(positives) if positives else None
        ),
        "owned_detector_positive_recall": (
            sum(_matched(row, owned_detector) for row in positives) / len(positives) if positives else None
        ),
        "final_positive_recall": sum(_matched(row, owned) for row in positives) / len(positives) if positives else None,
        "rejected_point_leakage": sum(_matched(row, owned) for row in negatives),
        "excluded_owned_points": excluded_owned,
        "unknown_owned_points": sum(point["ownership_evidence"] == "unknown" for point in owned),
        "owned_points": len(owned),
    }
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
