from __future__ import annotations

import csv
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .io import read_rows
from .rally_evidence import build_rally_evidence

PHASE_EXAMPLE_FIELDS = [
    "project_id",
    "rally",
    "boundary",
    "proposed_time_seconds",
    "corrected_time_seconds",
    "delta_seconds",
    "rally_probability",
    "near_ready",
    "far_ready",
    "formal_serve",
    "serving_side",
    "handoff_candidate",
    "trajectory_visible",
    "trajectory_descending",
    "trajectory_occluded",
]


def _interval_iou(first: tuple[float, float], second: tuple[float, float]) -> float:
    overlap = max(0.0, min(first[1], second[1]) - max(first[0], second[0]))
    union = max(first[1], second[1]) - min(first[0], second[0])
    return overlap / union if union > 0 else 0.0


def _atomic_write(path: Path, rows: list[dict[str, Any]]) -> Path | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = path.with_suffix(path.suffix + ".bak") if path.exists() else None
    if backup is not None:
        shutil.copy2(path, backup)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=".csv", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=PHASE_EXAMPLE_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return backup


def capture_phase_examples(
    project_id: str,
    corrected_rows: list[dict[str, Any]],
    proposed_csv: Path | None,
    output_csv: Path,
    features_csv: Path | None = None,
    probabilities_csv: Path | None = None,
    trajectory_csv: Path | None = None,
) -> dict[str, Any]:
    proposed = []
    if proposed_csv is not None and proposed_csv.exists():
        proposed = [
            (float(row["start_seconds"]), float(row["end_seconds"]))
            for row in read_rows(proposed_csv)
            if float(row["end_seconds"]) > float(row["start_seconds"])
        ]

    evidence_frame: pd.DataFrame | None = None
    evidence = None
    if (
        features_csv is not None
        and probabilities_csv is not None
        and features_csv.exists()
        and probabilities_csv.exists()
    ):
        features = pd.read_csv(features_csv)
        probabilities = pd.read_csv(probabilities_csv)
        evidence_frame = features.merge(
            probabilities[["time_seconds", "rally_probability"]],
            on="time_seconds",
            how="inner",
        ).sort_values("time_seconds")
        if not evidence_frame.empty:
            evidence = build_rally_evidence(
                evidence_frame,
                trajectory_csv if trajectory_csv is not None and trajectory_csv.exists() else None,
            )

    rows = []
    for rally_number, corrected in enumerate(corrected_rows, 1):
        interval = (float(corrected["start_seconds"]), float(corrected["end_seconds"]))
        match = max(proposed, key=lambda candidate: _interval_iou(interval, candidate), default=None)
        if match is not None and _interval_iou(interval, match) < 0.05:
            match = None
        for boundary, corrected_time, proposed_time in (
            ("start", interval[0], match[0] if match else None),
            ("end", interval[1], match[1] if match else None),
        ):
            sample: dict[str, Any] = {
                "project_id": project_id,
                "rally": rally_number,
                "boundary": boundary,
                "proposed_time_seconds": f"{proposed_time:.6f}" if proposed_time is not None else "",
                "corrected_time_seconds": f"{corrected_time:.6f}",
                "delta_seconds": f"{corrected_time - proposed_time:.6f}" if proposed_time is not None else "",
                "rally_probability": "",
                "near_ready": "",
                "far_ready": "",
                "formal_serve": "",
                "serving_side": "unknown",
                "handoff_candidate": "",
                "trajectory_visible": "",
                "trajectory_descending": "",
                "trajectory_occluded": "",
            }
            if evidence_frame is not None and evidence is not None and not evidence_frame.empty:
                times = evidence_frame["time_seconds"].to_numpy(dtype=float)
                index = int(np.argmin(np.abs(times - corrected_time)))
                sample.update(
                    {
                        "rally_probability": f"{float(evidence_frame.iloc[index]['rally_probability']):.6f}",
                        "near_ready": int(bool(evidence.near_ready[index])),
                        "far_ready": int(bool(evidence.far_ready[index])),
                        "formal_serve": int(bool(evidence.formal_serve[index])),
                        "serving_side": (
                            "near"
                            if bool(evidence.near_serving[index])
                            else "far"
                            if bool(evidence.far_serving[index])
                            else "unknown"
                        ),
                        "handoff_candidate": int(bool(evidence.handoff_candidate[index])),
                        "trajectory_visible": int(bool(evidence.trajectory_visible[index])),
                        "trajectory_descending": int(bool(evidence.trajectory_descending[index])),
                        "trajectory_occluded": int(bool(evidence.trajectory_occluded[index])),
                    }
                )
            rows.append(sample)

    backup = _atomic_write(output_csv, rows)
    changed = sum(row["delta_seconds"] not in {"", "0.000000", "-0.000000"} for row in rows)
    return {
        "path": str(output_csv),
        "backup": str(backup) if backup else None,
        "examples": len(rows),
        "changed_boundaries": changed,
    }
