from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _numeric(frame: pd.DataFrame, name: str, default: float = 0.0) -> np.ndarray:
    series = frame[name] if name in frame else pd.Series(default, index=frame.index)
    return pd.to_numeric(series, errors="coerce").fillna(default).to_numpy(dtype=float)


def _nearest_indices(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    if not len(reference):
        return np.zeros(len(values), dtype=int)
    right = np.searchsorted(reference, values, side="left")
    right = np.clip(right, 0, len(reference) - 1)
    left = np.maximum(0, right - 1)
    choose_left = np.abs(reference[left] - values) <= np.abs(reference[right] - values)
    return np.where(choose_left, left, right)


def _wrist_distances(points: pd.DataFrame, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    feature_times = _numeric(features, "time_seconds")
    point_times = _numeric(points, "time_seconds")
    indices = _nearest_indices(feature_times, point_times)
    width = np.maximum(_numeric(points, "source_width", 1.0), 1.0)
    height = np.maximum(_numeric(points, "source_height", 1.0), 1.0)
    x = _numeric(points, "center_x") / width
    y = _numeric(points, "center_y") / height
    distances = []
    for side in ("near", "far"):
        visible = _numeric(features, f"{side}_active_wrist_visible")[indices] > 0.5
        wrist_x = _numeric(features, f"{side}_active_wrist_x_normalized")[indices]
        wrist_y = _numeric(features, f"{side}_active_wrist_y_normalized")[indices]
        distance = np.hypot(x - wrist_x, y - wrist_y)
        distances.append(np.where(visible, distance, np.inf))
    return distances[0], distances[1]


def _side_from_distances(near: float, far: float, maximum: float, minimum_margin: float) -> tuple[str, float]:
    closest = min(near, far)
    margin = abs(near - far)
    if not np.isfinite(closest) or closest > maximum or margin < minimum_margin:
        return "unknown", 0.0
    side = "near" if near < far else "far"
    confidence = 0.76 + min(0.12, margin * 0.8) + min(0.06, (maximum - closest) * 0.5)
    return side, min(0.93, confidence)


def _trajectory_server(
    trajectory: pd.DataFrame,
    features: pd.DataFrame,
    start: float,
    end: float,
) -> tuple[str, float, float] | None:
    if trajectory.empty or "flight_id" not in trajectory:
        return None
    times = pd.to_numeric(trajectory.get("time_seconds"), errors="coerce")
    window = trajectory.loc[
        times.between(start - 0.15, min(end, start + 6.0))
        & trajectory["flight_id"].fillna("").astype(str).ne("")
    ].copy()
    if "ownership_evidence" in window:
        window = window.loc[window["ownership_evidence"].fillna("unknown").ne("unknown")]
    if "ownership_confidence" in window:
        confidence = pd.to_numeric(window["ownership_confidence"], errors="coerce").fillna(0.0)
        window = window.loc[confidence >= 0.45]
    if window.empty:
        return None
    window["time_seconds"] = pd.to_numeric(window["time_seconds"], errors="coerce")
    ordered_flights = sorted(
        window.groupby("flight_id", sort=False),
        key=lambda item: float(item[1]["time_seconds"].min()),
    )
    for _flight_id, flight in ordered_flights:
        flight = flight.sort_values("time_seconds")
        flight_start = float(flight.iloc[0]["time_seconds"])
        early = flight.loc[flight["time_seconds"] <= flight_start + 0.8]
        near_distance, far_distance = _wrist_distances(early, features)
        if not len(near_distance):
            continue
        side, confidence = _side_from_distances(
            float(near_distance[0]),
            float(far_distance[0]),
            maximum=0.10,
            minimum_margin=0.025,
        )
        if side == "unknown":
            side, confidence = _side_from_distances(
                float(np.min(near_distance)),
                float(np.min(far_distance)),
                maximum=0.06,
                minimum_margin=0.025,
            )
            confidence = max(0.0, confidence - 0.02)
        if side != "unknown":
            return side, confidence, flight_start
    return None


def infer_serve_observations(
    rallies: list[tuple[float, float]],
    features: pd.DataFrame,
    trajectory: pd.DataFrame,
    formal_serves: list[dict[str, Any]] | None = None,
    contacts: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Infer the server conservatively from owned shuttle-player proximity.

    Pose-only serve candidates remain a low-confidence fallback. They are
    useful for review but cannot award the previous point on their own.
    """
    formal_serves = formal_serves or []
    contacts = contacts or []
    observations = []
    for rally, (start, end) in enumerate(rallies, 1):
        trajectory_result = _trajectory_server(trajectory, features, start, end)
        if trajectory_result is not None:
            server, confidence, time_seconds = trajectory_result
            nearby_contacts = [
                row
                for row in contacts
                if row.get("player") in {"near", "far"}
                and abs(float(row.get("time", row.get("time_seconds", -10.0))) - time_seconds) <= 0.55
            ]
            if nearby_contacts:
                contact = max(nearby_contacts, key=lambda row: float(row.get("confidence", 0.0)))
                if str(contact["player"]) == server:
                    confidence = min(0.96, confidence + 0.04)
            agrees_with_pose = any(
                row.get("server") == server
                and start - 0.25 <= float(row.get("time", -10.0)) <= min(end, start + 6.0)
                for row in formal_serves
            )
            if agrees_with_pose:
                confidence = min(0.97, confidence + 0.03)
            observations.append(
                {
                    "rally": rally,
                    "time": round(time_seconds, 3),
                    "server": server,
                    "confidence": round(confidence, 3),
                    "source": "owned-trajectory-wrist",
                }
            )
            continue

        contact_candidates = [
            row
            for row in contacts
            if row.get("player") in {"near", "far"}
            and start - 0.15 <= float(row.get("time", row.get("time_seconds", -10.0))) <= min(end, start + 6.0)
        ]
        if contact_candidates:
            contact = min(
                contact_candidates,
                key=lambda row: float(row.get("time", row.get("time_seconds", 1e9))),
            )
            observations.append(
                {
                    "rally": rally,
                    "time": round(float(contact.get("time", contact.get("time_seconds"))), 3),
                    "server": str(contact["player"]),
                    "confidence": 0.68,
                    "source": "contact-only-review",
                }
            )
            continue

        pose_candidates = [
            row
            for row in formal_serves
            if row.get("server") in {"near", "far"}
            and start - 0.25 <= float(row.get("time", -10.0)) <= min(end, start + 6.0)
        ]
        if pose_candidates:
            strongest = max(pose_candidates, key=lambda row: float(row.get("confidence", 0.0)))
            observations.append(
                {
                    "rally": rally,
                    "time": round(float(strongest["time"]), 3),
                    "server": str(strongest["server"]),
                    "confidence": 0.6,
                    "source": "pose-only-review",
                }
            )
    return observations
