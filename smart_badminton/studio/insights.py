"""Derived per-video views: action analytics, the evidence lane, and score state."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import pandas as pd

from ..analytics import ANALYTICS_SCHEMA_VERSION, analyze_rally_actions
from ..contacts import analyze_contacts
from ..court_events import analyze_terminal_events
from ..io import load_rallies, read_rows
from ..rally_evidence import build_rally_evidence
from ..score_labeling import load_machine_labels, score_labeling_config
from ..score_learning import default_score_model_path, fit_score_evidence_model
from ..scoring import analyze_score
from ..serve_inference import infer_serve_observations
from .shuttle import features_for_trajectory
from .state import StudioState, cached_payload

EVIDENCE_SCHEMA_VERSION = 1


def _is_stale(output: Path, inputs: list[Path | None]) -> bool:
    if not output.exists():
        return True
    newest = max((path.stat().st_mtime for path in inputs if path is not None and path.exists()), default=0.0)
    return output.stat().st_mtime < newest


def _build_analytics_payload(state: StudioState) -> dict[str, Any]:
    root = state.layout().analysis.root
    features = features_for_trajectory(root)
    if features is None or not state.rallies.exists():
        return {"available": False, "reason": "Analyze the video and create a rally timeline first."}
    summary_path = root / "action-summary.json"
    audio_events = root / "audio-events.csv"
    trajectory = root / "shuttle-track.csv"
    contacts = root / "rally-contacts.csv"
    events = root / "rally-events.csv"
    config = state.config if state.config_ready() else None
    if trajectory.exists():
        if _is_stale(contacts, [features, state.rallies, trajectory]):
            analyze_contacts(features, state.rallies, contacts, root / "contact-summary.json")
        probabilities = root / "rally-probabilities.csv"
        if config is not None and _is_stale(events, [features, state.rallies, trajectory, contacts, config, probabilities]):
            analyze_terminal_events(
                state.video,
                config,
                state.rallies,
                trajectory,
                events,
                contacts,
                root / "event-summary.json",
                probabilities,
                features,
            )
    tracked = trajectory.exists()
    inputs = [features, state.rallies, audio_events, config, contacts if tracked else None, events if tracked else None]
    if not _is_stale(summary_path, inputs):
        cached = json.loads(summary_path.read_text(encoding="utf-8"))
        if cached.get("schema_version") == ANALYTICS_SCHEMA_VERSION:
            return cached
    return analyze_rally_actions(
        features,
        state.rallies,
        root / "rally-actions.csv",
        summary_path,
        audio_events,
        contacts if contacts.exists() else None,
        events if events.exists() else None,
        trajectory if trajectory.exists() else None,
        config,
    )


def analytics_payload(state: StudioState) -> dict[str, Any]:
    root = state.layout().analysis.root
    paths = (
        state.rallies,
        state.config,
        *(root / name for name in ("smart-features.csv", "vision-features.csv", "rally-probabilities.csv")),
        *(root / name for name in ("audio-events.csv", "shuttle-track.csv", "rally-contacts.csv")),
        root / "rally-events.csv",
        root / "action-summary.json",
    )
    return cached_payload(state, "analytics", paths, lambda: _build_analytics_payload(state))


def boolean_spans(times: list[float], values: Any) -> list[list[float]]:
    if not times:
        return []
    step = float(pd.Series(times).diff().dropna().median()) if len(times) > 1 else 0.1
    step = step if math.isfinite(step) and step > 0 else 0.1
    spans: list[list[float]] = []
    start: float | None = None
    for index, active in enumerate(values):
        if bool(active) and start is None:
            start = times[index]
        if start is not None and (not bool(active) or index == len(times) - 1):
            end = times[index] + step if bool(active) and index == len(times) - 1 else times[index]
            spans.append([round(start, 3), round(end, 3)])
            start = None
    return spans


def _build_evidence_payload(state: StudioState) -> dict[str, Any]:
    root = state.layout().analysis.root
    features_path = features_for_trajectory(root)
    probabilities_path = root / "rally-probabilities.csv"
    trajectory_path = root / "shuttle-track.csv"
    if features_path is None or not probabilities_path.exists():
        return {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "available": False,
            "reason": "Run automatic analysis to populate frame evidence.",
        }
    probabilities = pd.read_csv(probabilities_path)[["time_seconds", "rally_probability"]]
    data = pd.read_csv(features_path).merge(probabilities, on="time_seconds", how="inner")
    if data.empty:
        raise ValueError("Feature and probability timelines do not overlap; rerun automatic analysis")
    times = pd.to_numeric(data["time_seconds"], errors="coerce").fillna(0.0).tolist()
    probability = pd.to_numeric(data["rally_probability"], errors="coerce").fillna(0.0).to_numpy()
    evidence = build_rally_evidence(data, trajectory_path if trajectory_path.exists() else None)
    signals = {
        "model_keep": boolean_spans(times, probability >= 0.30),
        "model_active": boolean_spans(times, probability >= 0.56),
        "activity_high": boolean_spans(times, evidence.activity >= 0.70),
        **{
            name: boolean_spans(times, getattr(evidence, name))
            for name in (
                "trajectory_visible",
                "trajectory_descending",
                "trajectory_occluded",
                "landing_candidate",
                "near_ready",
                "far_ready",
                "between_points",
            )
        },
        "handoff": boolean_spans(times, evidence.handoff_candidate),
    }
    serves = [
        {
            "time": round(times[index], 3),
            "server": "near" if evidence.near_serving[index] else "far" if evidence.far_serving[index] else "unknown",
            "confidence": round(float(evidence.serve_confidence[index]), 3),
        }
        for index, active in enumerate(evidence.formal_serve)
        if active
    ]
    contacts_path, events_path = root / "rally-contacts.csv", root / "rally-events.csv"
    contacts = [
        {
            "time": round(float(row["time_seconds"]), 3),
            "player": row.get("player", "unknown"),
            "confidence": round(float(row.get("confidence", 0.0)), 3),
        }
        for row in (read_rows(contacts_path) if contacts_path.exists() else [])
    ]
    terminal_events = [
        {
            "time": round(float(row["event_time_seconds"]), 3),
            "event": row["event"],
            "confidence": round(float(row.get("confidence", 0.0)), 3),
        }
        for row in (read_rows(events_path) if events_path.exists() else [])
        if row.get("event", "unknown") != "unknown"
    ]
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "available": True,
        "signals": signals,
        "serves": serves,
        "contacts": contacts,
        "terminal_events": terminal_events,
    }


def evidence_payload(state: StudioState) -> dict[str, Any]:
    root = state.layout().analysis.root
    paths = tuple(
        root / name
        for name in (
            "smart-features.csv",
            "vision-features.csv",
            "rally-probabilities.csv",
            "shuttle-track.csv",
            "rally-contacts.csv",
            "rally-events.csv",
        )
    )
    payload = cached_payload(state, "evidence", paths, lambda: _build_evidence_payload(state))
    return {"schema_version": EVIDENCE_SCHEMA_VERSION, **payload}


def serve_observations(state: StudioState, evidence: dict[str, Any]) -> list[dict[str, Any]]:
    root = state.layout().analysis.root
    features_path = features_for_trajectory(root)
    trajectory_path = root / "shuttle-track.csv"
    if not evidence.get("available") or features_path is None or not trajectory_path.exists():
        return []
    return infer_serve_observations(
        load_rallies(state.rallies),
        pd.read_csv(features_path),
        pd.read_csv(trajectory_path),
        evidence.get("serves") or [],
        evidence.get("contacts") or [],
    )


def score_label_paths(state: StudioState) -> tuple[Path, Path, Path]:
    layout = state.layout()
    root = layout.analysis.score_labeling
    legacy_root = state.library_root / "Analysis" / "Score_Labeling"
    if not layout.is_match_project and legacy_root.exists() and not root.exists():
        root = legacy_root
    return root, root / "manifest.json", root / "machine-score-labels.json"


def score_labeling_payload(state: StudioState) -> dict[str, Any]:
    root, manifest_path, labels_path = score_label_paths(state)
    config = score_labeling_config()
    labels = load_machine_labels(labels_path)
    counts = {status: 0 for status in ("consensus", "review", "accepted", "rejected")}
    for row in labels:
        status = str(row.get("status", "review"))
        if status in counts:
            counts[status] += 1
    return {
        "configured": bool(config["configured"]),
        "model": config["model"],
        "prepared": manifest_path.exists(),
        "directory": str(root),
        "total": len(labels),
        **counts,
        "suggestions": [
            {
                "rally": int(row.get("rally", 0)),
                "status": str(row.get("status", "review")),
                "suggestion": row.get("suggestion", {}),
                "model": row.get("model"),
            }
            for row in labels
        ],
    }


def score_payload(state: StudioState) -> dict[str, Any]:
    if not state.rallies.exists():
        return {"available": False, "generated": False, "stale": False, "reason": "请先完成剪辑时间表"}
    layout = state.layout()
    summary, output = layout.analysis.score_summary, layout.analysis.score_state
    if not summary.exists() or not output.exists():
        return {"available": False, "generated": False, "stale": False, "reason": "尚未计算比分"}
    inputs = [
        state.rallies,
        layout.score_corrections,
        layout.analysis.score_events,
        default_score_model_path(state.library_root),
        score_label_paths(state)[2],
        layout.analysis.features,
        layout.analysis.probabilities,
        layout.analysis.shuttle_track,
    ]
    if _is_stale(summary, inputs):
        return {"available": False, "generated": True, "stale": True, "reason": "时间表或分析数据已更新"}
    try:
        cached = json.loads(summary.read_text(encoding="utf-8"))
    except ValueError:
        return {"available": False, "generated": True, "stale": True, "reason": "比分缓存已损坏，请重新计算比分"}
    return {**cached, "available": True, "generated": True, "stale": False}


def calculate_score_payload(state: StudioState) -> dict[str, Any]:
    if not state.rallies.exists():
        return {"available": False, "generated": False, "stale": False, "reason": "请先完成剪辑时间表"}
    layout = state.layout()
    library = state.library_root
    evidence_model_path = default_score_model_path(library)
    fit_score_evidence_model(library, evidence_model_path)
    evidence_project = (
        layout.root.relative_to(library).as_posix() if layout.is_match_project else state.video.stem
    )
    observations = serve_observations(state, evidence_payload(state))
    corrections, events = layout.score_corrections, layout.analysis.score_events
    result = analyze_score(
        state.rallies,
        layout.analysis.score_state,
        events if events.exists() else None,
        corrections if corrections.exists() else None,
        layout.analysis.score_summary,
        serve_observations=observations,
        evidence_model_path=evidence_model_path,
        evidence_project=evidence_project,
        machine_labels_path=score_label_paths(state)[2],
    )
    result.update(
        {
            "serve_observations": observations,
            "available": True,
            "generated": True,
            "stale": False,
            "updated_at": time.time(),
        }
    )
    temporary = layout.analysis.score_summary.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(layout.analysis.score_summary)  # atomic, so an interrupted write cannot corrupt the cache
    return result
