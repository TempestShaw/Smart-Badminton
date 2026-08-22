from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .io import load_rallies, write_rows

CONTACT_FIELDS = [
    "rally",
    "time_seconds",
    "player",
    "confidence",
    "shuttle_x_normalized",
    "shuttle_y_normalized",
    "wrist_x_normalized",
    "wrist_y_normalized",
    "distance_normalized",
    "direction_change_score",
    "swing_score",
    "audio_score",
]

REQUIRED_FEATURES = {
    "shuttle_visible",
    "shuttle_x_normalized",
    "shuttle_y_normalized",
    "near_active_wrist_x_normalized",
    "near_active_wrist_y_normalized",
    "near_active_wrist_visible",
    "far_active_wrist_x_normalized",
    "far_active_wrist_y_normalized",
    "far_active_wrist_visible",
}


def _values(frame: pd.DataFrame, name: str) -> np.ndarray:
    if name not in frame:
        return np.zeros(len(frame), dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").fillna(0.0).to_numpy(dtype=float)


def _direction_change(frame: pd.DataFrame) -> np.ndarray:
    times = _values(frame, "time_seconds")
    x = _values(frame, "shuttle_x_normalized")
    y = _values(frame, "shuttle_y_normalized")
    visible = _values(frame, "shuttle_visible") > 0.5
    score = np.zeros(len(frame), dtype=float)
    for index in range(1, len(frame) - 1):
        if not (visible[index - 1] and visible[index] and visible[index + 1]):
            continue
        before_delta = times[index] - times[index - 1]
        after_delta = times[index + 1] - times[index]
        if before_delta <= 0 or after_delta <= 0 or before_delta > 0.25 or after_delta > 0.25:
            continue
        before = np.asarray([(x[index] - x[index - 1]) / before_delta, (y[index] - y[index - 1]) / before_delta])
        after = np.asarray([(x[index + 1] - x[index]) / after_delta, (y[index + 1] - y[index]) / after_delta])
        denominator = float(np.linalg.norm(before) * np.linalg.norm(after))
        if denominator <= 1e-6:
            continue
        cosine = float(np.clip(np.dot(before, after) / denominator, -1.0, 1.0))
        score[index] = (1.0 - cosine) / 2.0
    return score


def infer_contact_events(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if not REQUIRED_FEATURES.issubset(frame.columns) or frame.empty:
        return []
    times = _values(frame, "time_seconds")
    shuttle_x = _values(frame, "shuttle_x_normalized")
    shuttle_y = _values(frame, "shuttle_y_normalized")
    shuttle_visible = _values(frame, "shuttle_visible") > 0.5
    direction_change = _direction_change(frame)
    audio = _values(frame, "audio_hit_score")
    candidates: list[dict[str, Any]] = []
    for side in ("near", "far"):
        wrist_visible = _values(frame, f"{side}_active_wrist_visible") > 0.5
        wrist_x = _values(frame, f"{side}_active_wrist_x_normalized")
        wrist_y = _values(frame, f"{side}_active_wrist_y_normalized")
        swing = _values(frame, f"{side}_swing_score")
        stance = _values(frame, f"{side}_stance_width_normalized")
        for index in np.flatnonzero(shuttle_visible & wrist_visible):
            distance = math.hypot(shuttle_x[index] - wrist_x[index], shuttle_y[index] - wrist_y[index])
            proximity_radius = float(np.clip(stance[index] * 1.8, 0.025, 0.085))
            proximity = float(np.clip(1.0 - distance / proximity_radius, 0.0, 1.0))
            swing_support = float(np.clip(swing[index] / 0.65, 0.0, 1.0))
            audio_support = float(np.clip(audio[index], 0.0, 1.0))
            direction_support = float(direction_change[index])
            confidence = 0.55 * proximity + 0.20 * swing_support + 0.15 * direction_support + 0.10 * audio_support
            if proximity <= 0.05 or max(swing_support, direction_support, audio_support) < 0.18:
                continue
            candidates.append(
                {
                    "rally": 0,
                    "time_seconds": float(times[index]),
                    "player": side,
                    "confidence": float(confidence),
                    "shuttle_x_normalized": float(shuttle_x[index]),
                    "shuttle_y_normalized": float(shuttle_y[index]),
                    "wrist_x_normalized": float(wrist_x[index]),
                    "wrist_y_normalized": float(wrist_y[index]),
                    "distance_normalized": float(distance),
                    "direction_change_score": direction_support,
                    "swing_score": float(swing[index]),
                    "audio_score": audio_support,
                }
            )

    clustered: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda row: float(row["time_seconds"])):
        if clustered and float(candidate["time_seconds"]) - float(clustered[-1]["time_seconds"]) <= 0.28:
            if float(candidate["confidence"]) > float(clustered[-1]["confidence"]):
                clustered[-1] = candidate
        else:
            clustered.append(candidate)
    return clustered


def analyze_contacts(
    features_csv: Path,
    rallies_csv: Path,
    output_csv: Path,
    summary_json: Path | None = None,
) -> dict[str, Any]:
    features = pd.read_csv(features_csv)
    available = REQUIRED_FEATURES.issubset(features.columns)
    events = infer_contact_events(features) if available else []
    rallies = load_rallies(rallies_csv)
    assigned = []
    for event in events:
        time_seconds = float(event["time_seconds"])
        rally_number = next(
            (number for number, (start, end) in enumerate(rallies, 1) if start - 0.15 <= time_seconds <= end + 0.15),
            0,
        )
        if rally_number:
            assigned.append({**event, "rally": rally_number})
    formatted = [
        {
            **event,
            "time_seconds": f"{float(event['time_seconds']):.3f}",
            "confidence": f"{float(event['confidence']):.3f}",
            "shuttle_x_normalized": f"{float(event['shuttle_x_normalized']):.6f}",
            "shuttle_y_normalized": f"{float(event['shuttle_y_normalized']):.6f}",
            "wrist_x_normalized": f"{float(event['wrist_x_normalized']):.6f}",
            "wrist_y_normalized": f"{float(event['wrist_y_normalized']):.6f}",
            "distance_normalized": f"{float(event['distance_normalized']):.6f}",
            "direction_change_score": f"{float(event['direction_change_score']):.3f}",
            "swing_score": f"{float(event['swing_score']):.3f}",
            "audio_score": f"{float(event['audio_score']):.3f}",
        }
        for event in assigned
    ]
    write_rows(output_csv, CONTACT_FIELDS, formatted)
    by_rally = {
        number: [event for event in assigned if int(event["rally"]) == number]
        for number in range(1, len(rallies) + 1)
    }
    summary = {
        "available": available,
        "events": len(assigned),
        "rallies_with_contacts": sum(bool(events) for events in by_rally.values()),
        "last_hitter_by_rally": {
            str(number): events[-1]["player"] for number, events in by_rally.items() if events
        },
        "disclaimer": (
            "Contact candidates require shuttle proximity to the active wrist plus swing, direction-change or audio "
            "support. They are experimental and never change score or cuts by themselves."
        ),
    }
    if summary_json is not None:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
