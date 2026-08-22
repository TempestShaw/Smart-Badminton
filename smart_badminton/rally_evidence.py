from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class RallyEvidence:
    """Signals used after the frozen classifier, aligned to the feature timeline."""

    activity: np.ndarray
    player_engagement: np.ndarray
    trajectory_visible: np.ndarray
    trajectory_descending: np.ndarray
    trajectory_occluded: np.ndarray
    trajectory_flight_start: np.ndarray
    trajectory_flight_end: np.ndarray
    trajectory_serve_start: np.ndarray
    trajectory_contact_start: np.ndarray
    landing_candidate: np.ndarray
    protected_flight: np.ndarray
    near_ready: np.ndarray
    far_ready: np.ndarray
    players_ready: np.ndarray
    between_points: np.ndarray
    handoff_candidate: np.ndarray
    formal_serve: np.ndarray
    near_serving: np.ndarray
    far_serving: np.ndarray
    serve_confidence: np.ndarray


def _column(data: pd.DataFrame, name: str) -> np.ndarray:
    if name not in data:
        return np.zeros(len(data), dtype=float)
    return pd.to_numeric(data[name], errors="coerce").fillna(0.0).to_numpy(dtype=float)


def _rolling_max(values: np.ndarray, samples: int, center: bool = True) -> np.ndarray:
    return (
        pd.Series(values)
        .rolling(max(1, samples), center=center, min_periods=1)
        .max()
        .to_numpy(dtype=float)
    )


def _rolling_mean(values: np.ndarray, samples: int, center: bool = True) -> np.ndarray:
    return (
        pd.Series(values)
        .rolling(max(1, samples), center=center, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )


def _sample_fps(times: np.ndarray) -> float:
    return 1.0 / max(float(np.median(np.diff(times))), 1e-3) if len(times) > 1 else 10.0


def _mark_nearest(times: np.ndarray, values: np.ndarray, event_time: float, value: bool = True) -> None:
    if not len(times):
        return
    index = int(np.searchsorted(times, event_time))
    candidates = [candidate for candidate in (index - 1, index) if 0 <= candidate < len(times)]
    if not candidates:
        return
    selected = min(candidates, key=lambda candidate: abs(float(times[candidate]) - event_time))
    values[selected] = value


def _adaptive_threshold(
    values: np.ndarray,
    visible: np.ndarray,
    percentile: float,
    scale: float,
    minimum: float,
    maximum: float,
    fallback: float,
) -> float:
    samples = values[visible & np.isfinite(values) & (values > 1e-6)]
    if len(samples) < 8:
        return fallback
    return float(np.clip(np.percentile(samples, percentile) * scale, minimum, maximum))


def _trajectory_signals(
    times: np.ndarray, trajectory_csv: Path | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple[float, float]]]:
    visible = np.zeros(len(times), dtype=bool)
    descending = np.zeros(len(times), dtype=bool)
    descending_end = np.zeros(len(times), dtype=bool)
    flight_start = np.zeros(len(times), dtype=bool)
    flight_end = np.zeros(len(times), dtype=bool)
    components: list[tuple[float, float]] = []
    if trajectory_csv is None or not trajectory_csv.exists() or not len(times):
        return visible, descending, descending_end, flight_start, flight_end, components

    tracks: dict[str, list[tuple[float, float, float, float, float, bool]]] = defaultdict(list)
    with trajectory_csv.open(newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            if row.get("status") not in {"tracked", "recovered", "manual"}:
                continue
            ownership_evidence = str(row.get("ownership_evidence") or "").strip().lower()
            if ownership_evidence:
                ownership_confidence = float(row.get("ownership_confidence") or 0.0)
                if ownership_evidence == "unknown" or ownership_confidence < 0.45:
                    continue
            flight_or_track = row.get("flight_id") or row.get("track_id")
            if not flight_or_track:
                continue
            tracks[str(flight_or_track)].append(
                (
                    float(row["time_seconds"]),
                    float(row.get("center_x") or 0.0),
                    float(row["center_y"]),
                    float(row.get("source_width") or 0.0),
                    float(row.get("source_height") or 0.0),
                    row.get("detection_status") != "inpainted",
                )
            )

    sample_fps = _sample_fps(times)
    for points in tracks.values():
        ordered = sorted(points)
        component_start = 0
        split_points = [
            index
            for index in range(1, len(ordered))
            if ordered[index][0] - ordered[index - 1][0] > 0.9
        ]
        for component_end in [*split_points, len(ordered)]:
            component = ordered[component_start:component_end]
            component_start = component_end
            observed = [point for point in component if point[5]]
            if len(observed) < 3 or observed[-1][0] - observed[0][0] < 0.05:
                continue
            unique_observed = [observed[0]]
            for point in observed[1:]:
                if point[0] > unique_observed[-1][0] + 1e-6:
                    unique_observed.append(point)
            if len(unique_observed) < 3:
                continue
            track_times = np.asarray([point[0] for point in unique_observed], dtype=float)
            track_x = np.asarray([point[1] for point in unique_observed], dtype=float)
            track_y = np.asarray([point[2] for point in unique_observed], dtype=float)
            widths = np.asarray([point[3] for point in unique_observed], dtype=float)
            heights = np.asarray([point[4] for point in unique_observed], dtype=float)
            if np.all(widths > 0) and np.all(heights > 0):
                coordinates = np.column_stack([track_x / widths, track_y / heights])
                path_length = float(np.linalg.norm(np.diff(coordinates, axis=0), axis=1).sum())
                moving = path_length >= 0.004
            else:
                coordinates = np.column_stack([track_x, track_y])
                path_length = float(np.linalg.norm(np.diff(coordinates, axis=0), axis=1).sum())
                moving = path_length >= 8.0
            if not moving:
                continue
            vertical_speed = np.gradient(track_y, track_times)
            for time_seconds, speed in zip(track_times, vertical_speed):
                _mark_nearest(times, visible, float(time_seconds))
                if float(speed) >= 12.0:
                    _mark_nearest(times, descending, float(time_seconds))
            _mark_nearest(times, flight_start, float(track_times[0]))
            _mark_nearest(times, flight_end, float(track_times[-1]))
            components.append((float(track_times[0]), float(track_times[-1])))
            recent_indices = np.arange(max(0, len(track_times) - 3), len(track_times))
            recent_speed = float(np.median(vertical_speed[recent_indices]))
            if recent_speed >= 12.0:
                _mark_nearest(times, descending_end, float(track_times[-1]))

    visible = _rolling_max(visible.astype(float), max(1, round(0.16 * sample_fps))) > 0
    descending = _rolling_max(descending.astype(float), max(1, round(0.22 * sample_fps))) > 0
    return visible, descending, descending_end, flight_start, flight_end, components


def build_rally_evidence(data: pd.DataFrame, trajectory_csv: Path | None = None) -> RallyEvidence:
    times = _column(data, "time_seconds")
    fps = _sample_fps(times)
    near_swing = _column(data, "near_swing_score")
    far_swing = _column(data, "far_swing_score")
    near_lunge = _column(data, "near_lunge_score")
    far_lunge = _column(data, "far_lunge_score")
    near_flow = _column(data, "near_flow_mean")
    far_flow = _column(data, "far_flow_mean")
    near_motion = _column(data, "near_motion_fraction")
    far_motion = _column(data, "far_motion_fraction")
    audio = _column(data, "audio_hit_score")
    probability = _column(data, "rally_probability")
    shuttle = _column(data, "shuttle_visible") + np.minimum(_column(data, "shuttle_speed_normalized"), 1.0)

    combined_swing = near_swing + far_swing
    combined_lunge = near_lunge + far_lunge
    combined_flow = near_flow + far_flow
    combined_motion = near_motion + far_motion
    player_engagement = np.maximum.reduce(
        [
            np.clip(combined_swing, 0.0, 1.0),
            np.clip(combined_lunge * 0.65, 0.0, 1.0),
            np.clip(combined_flow / 5.0, 0.0, 1.0),
            np.clip(combined_motion / 0.18, 0.0, 1.0),
            np.clip(audio * 0.8, 0.0, 1.0),
        ]
    )
    activity = np.maximum.reduce(
        [
            _rolling_max(combined_swing, max(1, round(0.8 * fps))) * 0.95,
            _rolling_max(combined_lunge, max(1, round(0.8 * fps))) * 0.65,
            np.clip(_rolling_max(combined_flow, max(1, round(0.6 * fps))) / 5.0, 0.0, 1.0) * 0.65,
            np.clip(_rolling_max(combined_motion, max(1, round(0.6 * fps))) / 0.18, 0.0, 1.0) * 0.55,
            _rolling_max(audio, max(1, round(0.5 * fps))) * 0.8,
            _rolling_max(shuttle, max(1, round(0.8 * fps))) * 0.75,
        ]
    )

    (
        raw_trajectory_visible,
        raw_trajectory_descending,
        raw_descending_end,
        _raw_flight_start,
        _raw_flight_end,
        trajectory_components,
    ) = _trajectory_signals(times, trajectory_csv)
    trajectory_visible = np.zeros(len(times), dtype=bool)
    trajectory_descending = np.zeros(len(times), dtype=bool)
    descending_end = np.zeros(len(times), dtype=bool)
    trajectory_flight_start = np.zeros(len(times), dtype=bool)
    trajectory_flight_end = np.zeros(len(times), dtype=bool)
    raw_trajectory_contact_start = np.zeros(len(times), dtype=bool)
    contact_signal = np.maximum(np.clip(combined_swing, 0.0, 1.0), audio)
    for component_start, component_end in trajectory_components:
        support = (times >= component_start - 0.25) & (times <= component_end + 0.25)
        if not np.any(support):
            continue
        contact_supported = float(np.max(contact_signal[support])) >= 0.14
        contact_start_supported = (
            contact_supported
            and float(np.max(probability[support])) >= 0.08
            and float(np.max(player_engagement[support])) >= 0.16
        )
        model_supported = (
            float(np.max(probability[support])) >= 0.30
            and float(np.max(player_engagement[support])) >= 0.16
        )
        if not contact_supported and not model_supported:
            continue
        component = (times >= component_start - 0.18) & (times <= component_end + 0.18)
        trajectory_visible |= raw_trajectory_visible & component
        trajectory_descending |= raw_trajectory_descending & component
        descending_end |= raw_descending_end & component
        _mark_nearest(times, trajectory_flight_start, component_start)
        _mark_nearest(times, trajectory_flight_end, component_end)
        if contact_start_supported:
            _mark_nearest(times, raw_trajectory_contact_start, component_start)
    recent_contact = _rolling_max(
        np.maximum(np.clip(combined_swing, 0.0, 1.0), audio),
        max(1, round(1.1 * fps)),
        center=False,
    )
    gameplay_context = (recent_contact >= 0.18) | (activity >= 0.26) | (probability >= 0.08)

    last_visible_time = -np.inf
    occluded = np.zeros(len(times), dtype=bool)
    for index, time_seconds in enumerate(times):
        if trajectory_visible[index]:
            last_visible_time = float(time_seconds)
            continue
        occlusion_age = float(time_seconds) - last_visible_time
        strong_occlusion_context = probability[index] >= 0.18 or player_engagement[index] >= 0.28
        if (
            occlusion_age <= 0.9
            and gameplay_context[index]
            and (occlusion_age <= 0.25 or strong_occlusion_context)
        ):
            occluded[index] = True

    quiet = _rolling_mean((player_engagement < 0.16).astype(float), max(1, round(0.45 * fps))) >= 0.72
    landing = np.zeros(len(times), dtype=bool)
    for index in np.flatnonzero(descending_end):
        stop = min(len(times), index + max(2, round(0.65 * fps)))
        candidates = np.flatnonzero(quiet[index:stop] & ~trajectory_visible[index:stop])
        if len(candidates):
            landing[index + int(candidates[0])] = True
    readiness_columns = {
        "near_foot_speed_normalized",
        "far_foot_speed_normalized",
        "near_stance_width_normalized",
        "far_stance_width_normalized",
    }
    players_ready = np.zeros(len(times), dtype=bool)
    near_ready = np.zeros(len(times), dtype=bool)
    far_ready = np.zeros(len(times), dtype=bool)
    handoff_candidate = np.zeros(len(times), dtype=bool)
    formal_serve = np.zeros(len(times), dtype=bool)
    near_serving = np.zeros(len(times), dtype=bool)
    far_serving = np.zeros(len(times), dtype=bool)
    serve_confidence = np.zeros(len(times), dtype=float)
    trajectory_serve_start = np.zeros(len(times), dtype=bool)
    if readiness_columns.issubset(data.columns):
        near_visible = _column(data, "near_person_visible") > 0.5
        far_visible = _column(data, "far_person_visible") > 0.5
        both_visible = near_visible & far_visible
        near_speed = _column(data, "near_foot_speed_normalized")
        far_speed = _column(data, "far_foot_speed_normalized")
        near_stance = _column(data, "near_stance_width_normalized")
        far_stance = _column(data, "far_stance_width_normalized")
        near_speed_limit = _adaptive_threshold(near_speed, near_visible, 45, 1.7, 0.045, 0.13, 0.10)
        far_speed_limit = _adaptive_threshold(far_speed, far_visible, 45, 1.7, 0.045, 0.13, 0.10)
        near_stance_floor = _adaptive_threshold(near_stance, near_visible, 20, 0.62, 0.008, 0.030, 0.012)
        far_stance_floor = _adaptive_threshold(far_stance, far_visible, 20, 0.62, 0.004, 0.018, 0.006)
        instant_near_ready = (
            near_visible
            & (near_speed <= near_speed_limit)
            & (near_stance >= near_stance_floor)
            & (near_swing <= 0.18)
            & (near_flow <= 1.45)
        )
        instant_far_ready = (
            far_visible
            & (far_speed <= far_speed_limit)
            & (far_stance >= far_stance_floor)
            & (far_swing <= 0.18)
            & (far_flow <= 1.45)
        )
        ready_samples = max(2, round(0.55 * fps))
        near_ready = _rolling_mean(
            instant_near_ready.astype(float), ready_samples, center=False
        ) >= 0.72
        far_ready = _rolling_mean(
            instant_far_ready.astype(float), ready_samples, center=False
        ) >= 0.78
        players_ready = near_ready & far_ready

        dominant_swing = np.maximum(near_swing, far_swing)
        other_swing = np.minimum(near_swing, far_swing)
        raw_handoff = (
            both_visible
            & (dominant_swing >= 0.20)
            & (other_swing <= 0.14)
            & (combined_lunge <= 0.55)
            & (np.minimum(near_flow, far_flow) <= 1.3)
            & ~players_ready
        )
        ready_ahead = _rolling_max(
            players_ready[::-1].astype(float), max(2, round(3.0 * fps)), center=False
        )[::-1] > 0
        handoff_candidate = raw_handoff & ready_ahead

        serve_support = (audio >= 0.22) | trajectory_flight_start
        near_serve_event = (near_swing >= 0.22) & (near_swing >= far_swing * 1.18) & (
            serve_support | (near_swing >= 0.34)
        )
        far_serve_event = (far_swing >= 0.22) & (far_swing >= near_swing * 1.18) & (
            serve_support | (far_swing >= 0.34)
        )
        receiver_window = max(2, round(2.0 * fps))
        near_ready_recent = _rolling_max(near_ready.astype(float), receiver_window, center=False) > 0
        far_ready_recent = _rolling_max(far_ready.astype(float), receiver_window, center=False) > 0
        ready_recent = _rolling_max(players_ready.astype(float), receiver_window, center=False) > 0
        trajectory_serve_start = trajectory_flight_start & ready_recent
        armed = False
        was_ready = False
        for index in range(len(times)):
            if players_ready[index] and not was_ready:
                armed = True
            if armed and near_serve_event[index] and far_ready_recent[index]:
                formal_serve[index] = True
                near_serving[index] = True
                serve_confidence[index] = min(
                    0.96,
                    0.62
                    + (0.12 if audio[index] >= 0.22 else 0.0)
                    + (0.12 if trajectory_flight_start[index] else 0.0)
                    + 0.10 * min(1.0, max(0.0, (near_swing[index] - 0.22) / 0.30)),
                )
                armed = False
            elif armed and far_serve_event[index] and near_ready_recent[index]:
                formal_serve[index] = True
                far_serving[index] = True
                serve_confidence[index] = min(
                    0.96,
                    0.62
                    + (0.12 if audio[index] >= 0.22 else 0.0)
                    + (0.12 if trajectory_flight_start[index] else 0.0)
                    + 0.10 * min(1.0, max(0.0, (far_swing[index] - 0.22) / 0.30)),
                )
                armed = False
            was_ready = bool(players_ready[index])

    continuation_samples = max(2, round(1.2 * fps))
    for landing_index in np.flatnonzero(landing):
        stop = min(len(times), landing_index + continuation_samples + 1)
        starts = np.flatnonzero(trajectory_flight_start[landing_index + 1 : stop])
        if not len(starts):
            continue
        next_start = landing_index + 1 + int(starts[0])
        continuation = slice(landing_index + 1, next_start + 1)
        if np.any(players_ready[continuation]) or np.any(handoff_candidate[continuation]):
            continue
        landing[landing_index] = False
        occluded[landing_index:next_start] = True

    between_points = np.zeros(len(times), dtype=bool)
    awaiting_formal_serve = False
    for index in range(len(times)):
        if landing[index]:
            awaiting_formal_serve = True
        if formal_serve[index] or trajectory_serve_start[index]:
            awaiting_formal_serve = False
        between_points[index] = awaiting_formal_serve

    handoff_candidate |= raw_trajectory_contact_start & between_points
    landing_window = _rolling_max(landing.astype(float), max(1, round(0.25 * fps)), center=False) > 0
    protected_flight = (trajectory_visible | trajectory_descending | occluded) & ~landing_window
    handoff_near_start = _rolling_max(
        handoff_candidate.astype(float), max(2, round(0.5 * fps))
    ) > 0
    trajectory_contact_start = raw_trajectory_contact_start & ~handoff_near_start & ~between_points

    return RallyEvidence(
        activity=activity,
        player_engagement=player_engagement,
        trajectory_visible=trajectory_visible,
        trajectory_descending=trajectory_descending,
        trajectory_occluded=occluded,
        trajectory_flight_start=trajectory_flight_start,
        trajectory_flight_end=trajectory_flight_end,
        trajectory_serve_start=trajectory_serve_start,
        trajectory_contact_start=trajectory_contact_start,
        landing_candidate=landing,
        protected_flight=protected_flight,
        near_ready=near_ready,
        far_ready=far_ready,
        players_ready=players_ready,
        between_points=between_points,
        handoff_candidate=handoff_candidate,
        formal_serve=formal_serve,
        near_serving=near_serving,
        far_serving=far_serving,
        serve_confidence=serve_confidence,
    )
