from __future__ import annotations

import csv
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

ANNOTATION_FIELDS = [
    "id",
    "time_seconds",
    "frame",
    "source_width",
    "source_height",
    "x_normalized",
    "y_normalized",
    "action",
    "note",
]


def load_shuttle_annotations(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as source:
        return [
            {
                "id": row.get("id", ""),
                "time_seconds": float(row["time_seconds"]),
                "frame": int(row["frame"]),
                "source_width": int(row["source_width"]),
                "source_height": int(row["source_height"]),
                "x_normalized": float(row["x_normalized"]),
                "y_normalized": float(row["y_normalized"]),
                "action": row["action"],
                "note": row.get("note", ""),
            }
            for row in csv.DictReader(source)
        ]


def validate_shuttle_annotations(
    payload: Any,
    duration: float,
    fps: float,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        raise TypeError("annotations must be a list")
    result = []
    for index, item in enumerate(payload, 1):
        if not isinstance(item, dict):
            raise TypeError(f"annotation {index} is invalid")
        time_seconds = float(item["time_seconds"])
        x = float(item["x_normalized"])
        y = float(item["y_normalized"])
        action = str(item["action"])
        if not math.isfinite(time_seconds) or not 0 <= time_seconds <= duration + 0.05:
            raise ValueError(f"annotation {index} time is outside the video")
        if not math.isfinite(x) or not math.isfinite(y) or not (0 <= x <= 1 and 0 <= y <= 1):
            raise ValueError(f"annotation {index} point must be normalized to [0, 1]")
        if action not in {"add", "reject"}:
            raise ValueError(f"annotation {index} action must be add or reject")
        result.append(
            {
                "id": str(item.get("id") or f"annotation-{index}"),
                "time_seconds": f"{time_seconds:.6f}",
                "frame": round(time_seconds * fps),
                "source_width": width,
                "source_height": height,
                "x_normalized": f"{x:.7f}",
                "y_normalized": f"{y:.7f}",
                "action": action,
                "note": str(item.get("note", ""))[:240],
            }
        )
    result.sort(key=lambda row: (float(row["time_seconds"]), str(row["action"]), str(row["id"])))
    return result


def save_shuttle_annotations(path: Path, rows: list[dict[str, Any]]) -> Path | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = path.with_suffix(path.suffix + ".bak") if path.exists() else None
    if backup is not None:
        shutil.copy2(path, backup)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=".csv", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=ANNOTATION_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return backup
