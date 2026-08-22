from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .rally_evidence import RallyEvidence

START_QUALITY_FEATURES = (
    "rally_probability",
    "probability_ahead_max",
    "activity",
    "player_engagement",
    "players_ready",
    "ready_recent",
    "formal_serve",
    "trajectory_serve_start",
    "trajectory_contact_start",
    "trajectory_flight_start",
    "handoff_recent",
    "between_points",
    "audio_hit_score",
    "near_swing_score",
    "far_swing_score",
    "previous_candidate_gap",
    "next_candidate_gap",
)


def _rolling_max(values: np.ndarray, samples: int, center: bool = False) -> np.ndarray:
    return (
        pd.Series(values)
        .rolling(max(1, samples), center=center, min_periods=1)
        .max()
        .to_numpy(dtype=float)
    )


def start_candidate_frame(
    data: pd.DataFrame,
    evidence: RallyEvidence,
) -> pd.DataFrame:
    times = pd.to_numeric(data["time_seconds"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    fps = 1.0 / max(float(np.median(np.diff(times))), 1e-3) if len(times) > 1 else 10.0

    def column(name: str) -> np.ndarray:
        value = data[name] if name in data else pd.Series(0.0, index=data.index)
        return pd.to_numeric(value, errors="coerce").fillna(0.0).to_numpy(dtype=float)

    candidate_mask = (
        evidence.formal_serve
        | evidence.trajectory_serve_start
        | evidence.trajectory_contact_start
        | evidence.trajectory_flight_start
    )
    indices = np.flatnonzero(candidate_mask)
    if not len(indices):
        return pd.DataFrame(columns=("candidate_index", "time_seconds", *START_QUALITY_FEATURES))

    probability = column("rally_probability")
    audio = column("audio_hit_score")
    near_swing = column("near_swing_score")
    far_swing = column("far_swing_score")
    probability_ahead = _rolling_max(probability[::-1], max(1, round(0.8 * fps)))[::-1]
    ready_recent = _rolling_max(evidence.players_ready.astype(float), max(1, round(2.0 * fps)))
    handoff_recent = _rolling_max(evidence.handoff_candidate.astype(float), max(1, round(1.2 * fps)))
    candidate_times = times[indices]
    previous_gap = np.r_[30.0, np.diff(candidate_times)]
    next_gap = np.r_[np.diff(candidate_times), 30.0]
    rows = []
    for position, index in enumerate(indices):
        rows.append(
            {
                "candidate_index": int(index),
                "time_seconds": float(times[index]),
                "rally_probability": float(probability[index]),
                "probability_ahead_max": float(probability_ahead[index]),
                "activity": float(evidence.activity[index]),
                "player_engagement": float(evidence.player_engagement[index]),
                "players_ready": float(evidence.players_ready[index]),
                "ready_recent": float(ready_recent[index]),
                "formal_serve": float(evidence.formal_serve[index]),
                "trajectory_serve_start": float(evidence.trajectory_serve_start[index]),
                "trajectory_contact_start": float(evidence.trajectory_contact_start[index]),
                "trajectory_flight_start": float(evidence.trajectory_flight_start[index]),
                "handoff_recent": float(handoff_recent[index]),
                "between_points": float(evidence.between_points[index]),
                "audio_hit_score": float(audio[index]),
                "near_swing_score": float(near_swing[index]),
                "far_swing_score": float(far_swing[index]),
                "previous_candidate_gap": min(30.0, float(previous_gap[position])),
                "next_candidate_gap": min(30.0, float(next_gap[position])),
            }
        )
    return pd.DataFrame(rows, columns=("candidate_index", "time_seconds", *START_QUALITY_FEATURES))


def start_quality_probabilities(frame: pd.DataFrame, model: dict[str, Any] | None) -> np.ndarray:
    if not model or frame.empty:
        return np.zeros(len(frame), dtype=float)
    names = list(model.get("feature_names") or START_QUALITY_FEATURES)
    matrix = frame.reindex(columns=names, fill_value=0.0).to_numpy(dtype=float)
    children_left = np.asarray(model["children_left"], dtype=int)
    children_right = np.asarray(model["children_right"], dtype=int)
    features = np.asarray(model["split_features"], dtype=int)
    thresholds = np.asarray(model["thresholds"], dtype=float)
    values = np.asarray(model["positive_probability"], dtype=float)
    result = []
    for row in matrix:
        node = 0
        while children_left[node] != children_right[node]:
            node = children_left[node] if row[features[node]] <= thresholds[node] else children_right[node]
        result.append(values[node])
    return np.asarray(result, dtype=float)


def _tree_values(frame: pd.DataFrame, model: dict[str, Any]) -> np.ndarray:
    names = list(model["feature_names"])
    matrix = frame.reindex(columns=names, fill_value=0.0).to_numpy(dtype=float)
    children_left = np.asarray(model["children_left"], dtype=int)
    children_right = np.asarray(model["children_right"], dtype=int)
    features = np.asarray(model["split_features"], dtype=int)
    thresholds = np.asarray(model["thresholds"], dtype=float)
    values = np.asarray(model["values"], dtype=float)
    result = []
    for row in matrix:
        node = 0
        while children_left[node] != children_right[node]:
            node = children_left[node] if row[features[node]] <= thresholds[node] else children_right[node]
        result.append(values[node])
    return np.asarray(result, dtype=float)


def selected_start_candidates(
    frame: pd.DataFrame, model: dict[str, Any] | None
) -> list[tuple[float, float]]:
    if not model or frame.empty:
        return []
    scores = start_quality_probabilities(frame, model)
    threshold = float(model.get("threshold", 1.0))
    if model.get("lead_model"):
        leads = np.maximum(0.0, _tree_values(frame, model["lead_model"]))
    else:
        leads = np.full(len(frame), float(model.get("lead_seconds", 0.35)), dtype=float)
    candidates = [
        (float(time_seconds), float(score), float(lead))
        for time_seconds, score, lead in zip(frame["time_seconds"], scores, leads)
        if score >= threshold
    ]
    dedupe = float(model.get("dedupe_seconds", 1.0))
    selected: list[tuple[float, float, float]] = []
    for candidate in candidates:
        if selected and candidate[0] - selected[-1][0] <= dedupe:
            if candidate[1] > selected[-1][1]:
                selected[-1] = candidate
            continue
        selected.append(candidate)
    return [
        (time_seconds, lead)
        for time_seconds, _score, lead in selected
        if math.isfinite(time_seconds) and math.isfinite(lead)
    ]


def selected_start_times(frame: pd.DataFrame, model: dict[str, Any] | None) -> list[float]:
    return [time_seconds for time_seconds, _lead in selected_start_candidates(frame, model)]
