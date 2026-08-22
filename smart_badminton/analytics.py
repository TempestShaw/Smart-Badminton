from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .io import load_rallies, write_rows
from .shot_analysis import classify_shots, shot_type_counts

ANALYTICS_SCHEMA_VERSION = 3

ACTION_FIELDS = [
    "rally",
    "start_seconds",
    "end_seconds",
    "duration_seconds",
    "estimated_hits",
    "contact_hit_candidates",
    "hit_estimation_source",
    "audio_hit_candidates",
    "pose_only_hit_candidates",
    "pace_hits_per_minute",
    "smash_candidates",
    "near_lunges",
    "far_lunges",
    "near_movement_score",
    "far_movement_score",
    "near_attack_score",
    "far_attack_score",
    "highlight_score",
    "last_hitter",
    "terminal_event",
    "terminal_event_confidence",
    "shot_type_counts",
    "highlight_reasons",
    "tags",
]


def _peaks(times: np.ndarray, scores: np.ndarray, threshold: float, minimum_gap: float) -> list[tuple[float, float]]:
    candidates: list[tuple[float, float]] = []
    for index, score in enumerate(scores):
        if score < threshold:
            continue
        previous = scores[index - 1] if index else -np.inf
        following = scores[index + 1] if index + 1 < len(scores) else -np.inf
        if score < previous or score < following:
            continue
        event = (float(times[index]), float(score))
        if candidates and event[0] - candidates[-1][0] < minimum_gap:
            if event[1] > candidates[-1][1]:
                candidates[-1] = event
        else:
            candidates.append(event)
    return candidates


def _cluster_times(times: list[float], maximum_gap: float) -> list[float]:
    clusters: list[list[float]] = []
    for time_seconds in sorted(times):
        if clusters and time_seconds - clusters[-1][-1] <= maximum_gap:
            clusters[-1].append(time_seconds)
        else:
            clusters.append([time_seconds])
    return [float(np.mean(cluster)) for cluster in clusters]


def _movement_score(frame: pd.DataFrame, side: str) -> float:
    flow = pd.to_numeric(frame[f"{side}_flow_mean"], errors="coerce").fillna(0.0)
    motion = pd.to_numeric(frame[f"{side}_motion_fraction"], errors="coerce").fillna(0.0)
    score = np.mean(np.clip(flow / 5.0, 0.0, 1.0) * 0.55 + np.clip(motion / 0.18, 0.0, 1.0) * 0.45)
    return float(np.clip(score * 100.0, 0.0, 100.0))


def _rally_actions(
    number: int,
    start: float,
    end: float,
    frame: pd.DataFrame,
    raw_audio_times: list[float] | None,
    contact_rows: list[dict[str, str]] | None,
    terminal_event: dict[str, str] | None,
    shot_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    duration = end - start
    times = frame["time_seconds"].to_numpy(dtype=float)

    def values(name: str) -> np.ndarray:
        if name not in frame:
            return np.zeros(len(frame), dtype=float)
        return pd.to_numeric(frame[name], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    audio = values("audio_hit_score")
    near_swing, far_swing = values("near_swing_score"), values("far_swing_score")
    near_lunge, far_lunge = values("near_lunge_score"), values("far_lunge_score")
    audio_peaks = _peaks(times, audio, 0.52, 0.20)
    near_swing_peaks = _peaks(times, near_swing, 0.75, 0.42)
    far_swing_peaks = _peaks(times, far_swing, 0.75, 0.42)
    if raw_audio_times is None:
        audio_hit_times = _cluster_times([event[0] for event in audio_peaks], maximum_gap=0.18)
    else:
        audio_hit_times = _cluster_times(
            [time_seconds for time_seconds in raw_audio_times if start <= time_seconds <= end],
            maximum_gap=0.18,
        )
    strong_pose_times = _cluster_times(
        [event[0] for event in near_swing_peaks + far_swing_peaks],
        maximum_gap=0.42,
    )
    pose_only_times = [
        event_time
        for event_time in strong_pose_times
        if not audio_hit_times or min(abs(event_time - audio_time) for audio_time in audio_hit_times) > 0.35
    ]
    fallback_hits = len(audio_hit_times) + len(pose_only_times)
    contact_hit_candidates = len(contact_rows or [])
    # A strict contact detector has high precision but lower recall when the
    # tiny shuttle is blurred or hidden by a player. Never let one surviving
    # strict contact erase several independently supported hit candidates.
    estimated_hits = max(contact_hit_candidates, fallback_hits)
    hit_estimation_source = (
        "racket-contact"
        if contact_hit_candidates and contact_hit_candidates >= fallback_hits
        else "contact+audio-pose-fusion"
        if contact_hit_candidates
        else "audio-pose-fallback"
    )

    smash_events = []
    for event_time, score in near_swing_peaks + far_swing_peaks:
        if score < 0.72:
            continue
        has_audio_anchor = any(abs(event_time - audio_time) <= 0.22 for audio_time in audio_hit_times)
        if has_audio_anchor:
            smash_events.append(event_time)
    smash_candidates = len(_cluster_times(smash_events, maximum_gap=0.42))
    near_lunges = len(_peaks(times, near_lunge, 0.58, 0.48))
    far_lunges = len(_peaks(times, far_lunge, 0.58, 0.48))
    total_lunges = near_lunges + far_lunges
    pace = estimated_hits / max(duration, 1.0) * 60.0
    near_attack = float(np.percentile(near_swing, 95) * 100.0) if len(near_swing) else 0.0
    far_attack = float(np.percentile(far_swing, 95) * 100.0) if len(far_swing) else 0.0

    tags = []
    if estimated_hits >= 12 or duration >= 18.0:
        tags.append("多拍拉锯")
    if smash_candidates >= 2:
        tags.append("火力全开")
    if total_lunges >= 3:
        tags.append("极限救球")
    if pace >= 85 and estimated_hits >= 6:
        tags.append("高速对攻")
    if duration <= 5.0 and estimated_hits <= 3:
        tags.append("闪电回合")
    if estimated_hits >= 5 and len(audio_hit_times) <= max(1, estimated_hits // 3):
        tags.append("疑似细腻网前")
    if not tags:
        tags.append("稳健相持")

    highlight = (
        min(duration / 22.0, 1.0) * 22.0
        + min(estimated_hits / 14.0, 1.0) * 28.0
        + min(smash_candidates / 3.0, 1.0) * 20.0
        + min(total_lunges / 5.0, 1.0) * 12.0
        + min(pace / 100.0, 1.0) * 10.0
        + min(max(near_attack, far_attack) / 100.0, 1.0) * 8.0
    )
    shot_counts = shot_type_counts(shot_rows or [])
    highlight_reasons = []
    if estimated_hits >= 8:
        highlight_reasons.append(f"{estimated_hits} 拍连续回合")
    if smash_candidates:
        highlight_reasons.append(f"{smash_candidates} 次强攻候选")
    if total_lunges:
        highlight_reasons.append(f"{total_lunges} 次救球步伐")
    if pace >= 75:
        highlight_reasons.append(f"节奏 {pace:.0f} 拍/分钟")
    if shot_counts.get("net_candidate"):
        highlight_reasons.append("包含细腻网前候选")
    if not highlight_reasons:
        highlight_reasons.append("动作与时长综合评分")
    return {
        "rally": number,
        "start_seconds": round(start, 3),
        "end_seconds": round(end, 3),
        "duration_seconds": round(duration, 3),
        "estimated_hits": estimated_hits,
        "contact_hit_candidates": contact_hit_candidates,
        "hit_estimation_source": hit_estimation_source,
        "audio_hit_candidates": len(audio_hit_times),
        "pose_only_hit_candidates": len(pose_only_times),
        "pace_hits_per_minute": round(pace, 1),
        "smash_candidates": smash_candidates,
        "near_lunges": near_lunges,
        "far_lunges": far_lunges,
        "near_movement_score": round(_movement_score(frame, "near"), 1),
        "far_movement_score": round(_movement_score(frame, "far"), 1),
        "near_attack_score": round(near_attack, 1),
        "far_attack_score": round(far_attack, 1),
        "highlight_score": round(float(np.clip(highlight, 0.0, 100.0)), 1),
        "last_hitter": (
            str(terminal_event.get("last_hitter", "unknown"))
            if terminal_event is not None
            else str(contact_rows[-1]["player"])
            if contact_rows
            else "unknown"
        ),
        "terminal_event": str(terminal_event.get("event", "unknown")) if terminal_event else "unknown",
        "terminal_event_confidence": float(terminal_event.get("confidence", 0.0)) if terminal_event else 0.0,
        "shot_type_counts": json.dumps(shot_counts, ensure_ascii=False, sort_keys=True),
        "highlight_reasons": " | ".join(highlight_reasons),
        "tags": " | ".join(tags),
    }


def analyze_rally_actions(
    features_csv: Path,
    rallies_csv: Path,
    output_csv: Path | None = None,
    summary_json: Path | None = None,
    audio_events_csv: Path | None = None,
    contacts_csv: Path | None = None,
    events_csv: Path | None = None,
    trajectory_csv: Path | None = None,
    config: Path | None = None,
) -> dict[str, Any]:
    features = pd.read_csv(features_csv)
    features["time_seconds"] = pd.to_numeric(features["time_seconds"], errors="coerce")
    rallies = load_rallies(rallies_csv)
    raw_audio_times: list[float] | None = None
    if audio_events_csv is not None and audio_events_csv.exists():
        with audio_events_csv.open(newline="", encoding="utf-8-sig") as source:
            raw_audio_times = [float(row["time_seconds"]) for row in csv.DictReader(source)]
    contacts_by_rally: dict[int, list[dict[str, str]]] = {}
    if contacts_csv is not None and contacts_csv.exists():
        with contacts_csv.open(newline="", encoding="utf-8-sig") as source:
            for row in csv.DictReader(source):
                contacts_by_rally.setdefault(int(row["rally"]), []).append(row)
    events_by_rally: dict[int, dict[str, str]] = {}
    if events_csv is not None and events_csv.exists():
        with events_csv.open(newline="", encoding="utf-8-sig") as source:
            events_by_rally = {int(row["rally"]): row for row in csv.DictReader(source)}
    geometry = None
    if config is not None and config.exists():
        from .geometry import CourtGeometry

        geometry = CourtGeometry.from_json(config)
    shots_by_rally: dict[int, list[dict[str, Any]]] = {}
    if trajectory_csv is not None and trajectory_csv.exists():
        for number, contacts in contacts_by_rally.items():
            shots_by_rally[number] = classify_shots(contacts, trajectory_csv, geometry)
    results = []
    for number, (start, end) in enumerate(rallies, 1):
        frame = features[(features.time_seconds >= start) & (features.time_seconds <= end)]
        if frame.empty:
            continue
        results.append(
            _rally_actions(
                number,
                start,
                end,
                frame,
                raw_audio_times,
                contacts_by_rally.get(number),
                events_by_rally.get(number),
                shots_by_rally.get(number),
            )
        )
    if output_csv is not None:
        write_rows(output_csv, ACTION_FIELDS, results)

    ranked = sorted(results, key=lambda row: (-float(row["highlight_score"]), int(row["rally"])))
    summary: dict[str, Any] = {
        "schema_version": ANALYTICS_SCHEMA_VERSION,
        "available": True,
        "disclaimer": (
            "Strict racket-contact candidates are fused with isolated audio and pose evidence because shuttle "
            "occlusion can make contact recall incomplete. Terminal events and hit counts are not official statistics."
        ),
        "rallies": results,
        "match": {
            "rallies": len(results),
            "estimated_hits": sum(int(row["estimated_hits"]) for row in results),
            "smash_candidates": sum(int(row["smash_candidates"]) for row in results),
            "average_pace": round(float(np.mean([row["pace_hits_per_minute"] for row in results])), 1)
            if results
            else 0.0,
            "top_highlights": [int(row["rally"]) for row in ranked[:5]],
            "shot_type_counts": dict(
                sum((Counter(json.loads(row["shot_type_counts"])) for row in results), Counter())
            ),
            "court_heatmap": [
                {
                    "rally": rally,
                    "x_meters": float(event["court_x_meters"]),
                    "y_meters": float(event["court_y_meters"]),
                    "event": event.get("event", "unknown"),
                    "confidence": float(event.get("confidence", 0.0)),
                }
                for rally, event in sorted(events_by_rally.items())
                if event.get("court_x_meters") not in {None, ""}
                and event.get("court_y_meters") not in {None, ""}
            ],
        },
    }
    if summary_json is not None:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
