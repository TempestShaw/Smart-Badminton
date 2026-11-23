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
    "player_id",
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


# A wrist near the shuttle is common (players stand under the flight, carry the shuttle, or overlap it in 2D); a hit
# also needs evidence of impact. These floors reject candidates that only have proximity.
MINIMUM_PROXIMITY = 0.2
MINIMUM_DIRECTION_CHANGE = 0.3
MINIMUM_AUDIO = 0.3
MINIMUM_SWING_SUPPORT = 0.4
# A stroke must cross the net before the other side can reply; even a smash return takes ~0.2 s.
MINIMUM_REPLY_SECONDS = 0.22
# The same side can only hit again after a full exchange (out and back), which takes at least ~0.6 s. Closer
# same-side candidates are one swing seen twice, or a false contact.
MINIMUM_SAME_SIDE_SECONDS = 0.6
# A same-side repeat beyond that implies the opponent's hit was not seen; it costs this much.
MISSED_REPLY_PENALTY = 0.08
# Beyond this pause, the exchange has ended and any side may hit next.
EXCHANGE_BREAK_SECONDS = 2.5
# Each accepted hit earns its confidence minus this; weaker candidates survive only when they complete an exchange.
ACCEPT_CONFIDENCE = 0.3
SIDE_PEAK_WINDOW_SECONDS = 0.25


def _swing_reference(swing: np.ndarray, visible: np.ndarray) -> float:
    """A side's own fast-swing level. The far player looks about half as tall, so their wrist speed in frame units is
    about half the near player's; judging each side against its own 90th percentile removes that bias."""
    values = swing[visible & (swing > 0)]
    reference = float(np.percentile(values, 90)) if len(values) >= 20 else 0.65
    return float(np.clip(reference, 0.2, 0.65))


def contact_candidates(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Every sample where a side's racket wrist meets the shuttle with evidence of impact."""
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
        swing_reference = _swing_reference(swing, wrist_visible)
        for index in np.flatnonzero(shuttle_visible & wrist_visible):
            distance = math.hypot(shuttle_x[index] - wrist_x[index], shuttle_y[index] - wrist_y[index])
            proximity_radius = float(np.clip(stance[index] * 1.8, 0.025, 0.085))
            proximity = float(np.clip(1.0 - distance / proximity_radius, 0.0, 1.0))
            swing_support = float(np.clip(swing[index] / swing_reference, 0.0, 1.0))
            audio_support = float(np.clip(audio[index], 0.0, 1.0))
            direction_support = float(direction_change[index])
            impact = (
                direction_support >= MINIMUM_DIRECTION_CHANGE
                or audio_support >= MINIMUM_AUDIO
                or swing_support >= MINIMUM_SWING_SUPPORT
            )
            if proximity < MINIMUM_PROXIMITY or not impact:
                continue
            confidence = 0.55 * proximity + 0.20 * swing_support + 0.15 * direction_support + 0.10 * audio_support
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
    return candidates


def _side_peaks(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One candidate per swing: within each side, neighbours closer than the peak window collapse to the strongest."""
    peaks: list[dict[str, Any]] = []
    for side in ("near", "far"):
        group_start: float | None = None
        for candidate in sorted((item for item in candidates if item["player"] == side), key=lambda item: item["time_seconds"]):
            if group_start is not None and candidate["time_seconds"] - group_start <= SIDE_PEAK_WINDOW_SECONDS:
                if candidate["confidence"] > peaks[-1]["confidence"]:
                    peaks[-1] = candidate
                continue
            group_start = candidate["time_seconds"]
            peaks.append(candidate)
    return sorted(peaks, key=lambda item: item["time_seconds"])


def _link_cost(earlier: dict[str, Any], later: dict[str, Any]) -> float | None:
    """Cost of ``later`` directly following ``earlier`` as the next hit, or None if the rules forbid it."""
    gap = float(later["time_seconds"]) - float(earlier["time_seconds"])
    if gap > EXCHANGE_BREAK_SECONDS:
        return 0.0
    if earlier["player"] != later["player"]:
        return 0.0 if gap >= MINIMUM_REPLY_SECONDS else None
    return MISSED_REPLY_PENALTY if gap >= MINIMUM_SAME_SIDE_SECONDS else None


def decode_hit_sequence(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The most plausible hit sequence under the rules of play.

    A reply needs ``MINIMUM_REPLY_SECONDS``; the same side (in doubles, either partner) can only hit again after a full
    exchange, so same-side candidates closer than ``MINIMUM_SAME_SIDE_SECONDS`` are one swing seen twice. Each chosen
    hit earns ``confidence - ACCEPT_CONFIDENCE``, so weak candidates without impact evidence drop out. Exact dynamic
    programming over candidates sorted by time.
    """
    peaks = _side_peaks(candidates)
    if not peaks:
        return []
    times = [float(item["time_seconds"]) for item in peaks]
    gains = [float(item["confidence"]) - ACCEPT_CONFIDENCE for item in peaks]
    best = [0.0] * len(peaks)
    previous: list[int | None] = [None] * len(peaks)
    # Prefix maximum of best[] over candidates far enough back that any hit may follow them.
    prefix_best, prefix_index, pointer = 0.0, None, 0
    for index, item in enumerate(peaks):
        while pointer < index and times[index] - times[pointer] > EXCHANGE_BREAK_SECONDS:
            if best[pointer] > prefix_best:
                prefix_best, prefix_index = best[pointer], pointer
            pointer += 1
        value, link = prefix_best, prefix_index
        for other in range(pointer, index):
            cost = _link_cost(peaks[other], item)
            if cost is not None and best[other] - cost > value:
                value, link = best[other] - cost, other
        best[index] = gains[index] + max(0.0, value)
        previous[index] = link if value > 0.0 else None
    end = max(range(len(peaks)), key=lambda index: best[index])
    if best[end] <= 0.0:
        return []
    chosen = []
    cursor: int | None = end
    while cursor is not None:
        chosen.append(peaks[cursor])
        cursor = previous[cursor]
    return list(reversed(chosen))


def infer_contact_events(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return decode_hit_sequence(contact_candidates(frame))


def attribute_players(events: list[dict[str, Any]], player_tracks: pd.DataFrame | None) -> None:
    """Name the hitter of each contact: on the hitting side, the tracked player whose wrist is nearest the shuttle."""
    for event in events:
        event["player_id"] = ""
    if player_tracks is None or player_tracks.empty:
        return
    tracks = player_tracks.loc[player_tracks["status"].eq("detected")].copy()
    for name in ("time_seconds", "wrist_x_normalized", "wrist_y_normalized", "foot_x_normalized", "foot_y_normalized"):
        tracks[name] = pd.to_numeric(tracks[name], errors="coerce")
    for event in events:
        nearby = tracks.loc[
            tracks["side"].eq(event["player"]) & (tracks["time_seconds"] - float(event["time_seconds"])).abs().le(0.06)
        ]
        if nearby.empty:
            continue
        wrist_x = nearby["wrist_x_normalized"].fillna(nearby["foot_x_normalized"])
        wrist_y = nearby["wrist_y_normalized"].fillna(nearby["foot_y_normalized"])
        distance = np.hypot(wrist_x - float(event["shuttle_x_normalized"]), wrist_y - float(event["shuttle_y_normalized"]))
        event["player_id"] = str(nearby.loc[distance.idxmin(), "player_id"])


def analyze_contacts(
    features_csv: Path,
    rallies_csv: Path,
    output_csv: Path,
    summary_json: Path | None = None,
    player_tracks_csv: Path | None = None,
) -> dict[str, Any]:
    features = pd.read_csv(features_csv)
    available = REQUIRED_FEATURES.issubset(features.columns)
    events = infer_contact_events(features) if available else []
    tracks = pd.read_csv(player_tracks_csv) if player_tracks_csv is not None and player_tracks_csv.exists() else None
    attribute_players(events, tracks)
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
        "hits_by_player": {
            player_id: sum(1 for event in assigned if event["player_id"] == player_id)
            for player_id in sorted({str(event["player_id"]) for event in assigned} - {""})
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


def evaluate_contacts(predicted_csv: Path, truth_csv: Path, tolerance_seconds: float = 0.15) -> dict[str, Any]:
    """Precision and recall of hit times against hand-marked hits (``time_seconds`` and ``player`` near/far).

    A prediction matches the nearest unmatched true hit of the same side within the tolerance. Also reports the
    physically impossible same-side repeats, which flag over-counting without any labels.
    """
    predicted = pd.read_csv(predicted_csv)
    truth = pd.read_csv(truth_csv)
    truth_rows = [(float(row.time_seconds), str(row.player)) for row in truth.itertuples()]
    used: set[int] = set()
    matched = 0
    errors = []
    for row in predicted.sort_values("time_seconds").itertuples():
        options = [
            (abs(float(row.time_seconds) - time_seconds), index)
            for index, (time_seconds, side) in enumerate(truth_rows)
            if index not in used and side == str(row.player) and abs(float(row.time_seconds) - time_seconds) <= tolerance_seconds
        ]
        if options:
            error, index = min(options)
            used.add(index)
            matched += 1
            errors.append(error)
    ordered = predicted.sort_values("time_seconds")
    sides, times = ordered["player"].astype(str).tolist(), ordered["time_seconds"].astype(float).tolist()
    impossible = sum(
        1
        for index in range(1, len(sides))
        if sides[index] == sides[index - 1] and times[index] - times[index - 1] < MINIMUM_SAME_SIDE_SECONDS
    )
    precision = matched / len(predicted) if len(predicted) else 0.0
    recall = matched / len(truth_rows) if truth_rows else 0.0
    return {
        "predicted": int(len(predicted)),
        "truth": len(truth_rows),
        "matched": matched,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(2 * precision * recall / (precision + recall), 3) if precision + recall else 0.0,
        "mean_time_error_seconds": round(float(np.mean(errors)), 3) if errors else None,
        "impossible_same_side_repeats": impossible,
        "tolerance_seconds": tolerance_seconds,
    }
