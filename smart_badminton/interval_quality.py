from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .rally_evidence import RallyEvidence, build_rally_evidence

INTERVAL_QUALITY_FEATURES = (
    "duration",
    "maximum_probability",
    "top_probability_mean",
    "audio_events",
    "player_engagement_max",
    "player_engagement_mean",
    "credible_start",
    "handoff_start_fraction",
    "protected_flight_fraction",
    "between_points_fraction",
    "players_ready_fraction",
    "previous_gap",
    "next_gap",
)


def _audio_events(times: np.ndarray, scores: np.ndarray, start: float, end: float) -> int:
    mask = (times >= start) & (times <= end) & (scores >= 0.12)
    indices = np.flatnonzero(mask & np.r_[True, ~mask[:-1]])
    events: list[float] = []
    for time_seconds in times[indices]:
        if not events or float(time_seconds) - events[-1] > 0.22:
            events.append(float(time_seconds))
    return len(events)


def interval_quality_frame(
    intervals: list[Any],
    times: np.ndarray,
    probability: np.ndarray,
    audio: np.ndarray,
    evidence: RallyEvidence,
) -> pd.DataFrame:
    rows = []
    for index, interval in enumerate(intervals):
        mask = (times >= interval.start - 1e-6) & (times <= interval.end + 1e-6)
        start_mask = mask & (times <= interval.start + 1.8)
        values = probability[mask]
        top_count = max(1, len(values) // 3)
        previous_gap = interval.start - intervals[index - 1].end if index else 30.0
        next_gap = intervals[index + 1].start - interval.end if index + 1 < len(intervals) else 30.0
        credible_start = bool(
            np.any(start_mask & evidence.formal_serve)
            or np.any(start_mask & evidence.trajectory_serve_start & evidence.trajectory_contact_start)
        )
        rows.append(
            {
                "duration": interval.end - interval.start,
                "maximum_probability": float(np.max(values)) if len(values) else 0.0,
                "top_probability_mean": float(np.mean(np.sort(values)[-top_count:])) if len(values) else 0.0,
                "audio_events": _audio_events(times, audio, interval.start, interval.end),
                "player_engagement_max": float(np.max(evidence.player_engagement[mask])) if np.any(mask) else 0.0,
                "player_engagement_mean": float(np.mean(evidence.player_engagement[mask])) if np.any(mask) else 0.0,
                "credible_start": float(credible_start),
                "handoff_start_fraction": (
                    float(np.mean(evidence.handoff_candidate[start_mask])) if np.any(start_mask) else 0.0
                ),
                "protected_flight_fraction": float(np.mean(evidence.protected_flight[mask])) if np.any(mask) else 0.0,
                "between_points_fraction": float(np.mean(evidence.between_points[mask])) if np.any(mask) else 0.0,
                "players_ready_fraction": float(np.mean(evidence.players_ready[mask])) if np.any(mask) else 0.0,
                "previous_gap": min(30.0, max(0.0, previous_gap)),
                "next_gap": min(30.0, max(0.0, next_gap)),
            }
        )
    return pd.DataFrame(rows, columns=INTERVAL_QUALITY_FEATURES)


def load_interval_quality_frame(
    intervals: list[Any],
    features_csv: Path,
    probabilities_csv: Path,
    trajectory_csv: Path | None,
) -> pd.DataFrame:
    features = pd.read_csv(features_csv)
    probabilities = pd.read_csv(probabilities_csv)
    data = features.merge(probabilities[["time_seconds", "rally_probability"]], on="time_seconds", how="inner")
    times = pd.to_numeric(data["time_seconds"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    probability = pd.to_numeric(data["rally_probability"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    audio = pd.to_numeric(data.get("audio_hit_score", 0.0), errors="coerce")
    if not isinstance(audio, pd.Series):
        audio = pd.Series(0.0, index=data.index)
    evidence = build_rally_evidence(data, trajectory_csv)
    return interval_quality_frame(intervals, times, probability, audio.to_numpy(dtype=float), evidence)


def interval_quality_probabilities(frame: pd.DataFrame, model: dict[str, Any] | None) -> np.ndarray:
    if not model or frame.empty:
        return np.ones(len(frame), dtype=float)
    names = list(model.get("feature_names") or INTERVAL_QUALITY_FEATURES)
    matrix = frame.reindex(columns=names, fill_value=0.0).to_numpy(dtype=float)
    if model.get("type") == "decision_tree":
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
    means = np.asarray(model["means"], dtype=float)
    scales = np.asarray(model["scales"], dtype=float)
    weights = np.asarray(model["weights"], dtype=float)
    logits = ((matrix - means) / np.maximum(scales, 1e-6)) @ weights + float(model["intercept"])
    return np.asarray([1.0 / (1.0 + math.exp(-float(np.clip(value, -30.0, 30.0)))) for value in logits])
