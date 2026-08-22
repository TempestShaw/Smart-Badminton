from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import CourtGeometry
from .io import load_rallies, write_rows
from .trajectory_metrics import load_owned_trajectory, summarize_trajectory

ANALYTICS_SCHEMA_VERSION = 4

ACTION_FIELDS = [
    "rally",
    "start_seconds",
    "end_seconds",
    "duration_seconds",
    "trajectory_available",
    "trajectory_points",
    "trajectory_visible_seconds",
    "trajectory_coverage_percent",
    "longest_continuous_track_seconds",
    "trajectory_distance_frames",
    "peak_visual_speed_frames_per_second",
    "net_zone_transits",
    "rally_style",
    "highlight_score",
    "last_hitter",
    "terminal_event",
    "terminal_event_confidence",
    "terminal_landing_side",
    "highlight_reasons",
    "tags",
]


def _event_rows(path: Path | None) -> dict[int, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8-sig") as source:
        return {int(row["rally"]): row for row in csv.DictReader(source)}


def _rally_style(duration: float, metrics: dict[str, Any], event: dict[str, str] | None) -> str:
    terminal = str((event or {}).get("event") or "unknown")
    if terminal == "net_candidate":
        return "网前终结"
    if terminal == "landing_out_candidate":
        return "界外终结"
    if duration <= 3.5:
        return "快速终结"
    if duration >= 12.0 or float(metrics["trajectory_distance_frames"]) >= 3.0:
        return "长回合"
    if int(metrics["net_zone_transits"]) >= 3:
        return "多次攻防"
    if float(metrics["trajectory_coverage_percent"]) >= 72.0:
        return "连续相持"
    return "相持回合"


def _highlight(duration: float, metrics: dict[str, Any]) -> float:
    if not metrics["trajectory_available"]:
        return 0.0
    score = (
        min(duration / 18.0, 1.0) * 25.0
        + min(float(metrics["trajectory_coverage_percent"]) / 85.0, 1.0) * 25.0
        + min(float(metrics["trajectory_distance_frames"]) / 4.0, 1.0) * 30.0
        + min(int(metrics["net_zone_transits"]) / 3.0, 1.0) * 20.0
    )
    return round(float(np.clip(score, 0.0, 100.0)), 1)


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
    del features_csv, audio_events_csv, contacts_csv
    rallies = load_rallies(rallies_csv)
    events_by_rally = _event_rows(events_csv)
    geometry = CourtGeometry.from_json(config) if config is not None and config.exists() else None
    trajectory = load_owned_trajectory(trajectory_csv)
    results: list[dict[str, Any]] = []
    for number, (start, end) in enumerate(rallies, 1):
        duration = end - start
        metrics = summarize_trajectory(trajectory, start, end, geometry)
        event = events_by_rally.get(number)
        style = _rally_style(duration, metrics, event)
        reasons = []
        if metrics["trajectory_coverage_percent"] >= 75:
            reasons.append("球路连续")
        if metrics["trajectory_distance_frames"] >= 3:
            reasons.append("球路距离长")
        if metrics["net_zone_transits"]:
            reasons.append("经过网区")
        results.append(
            {
                "rally": number,
                "start_seconds": round(start, 3),
                "end_seconds": round(end, 3),
                "duration_seconds": round(duration, 3),
                **metrics,
                "rally_style": style,
                "highlight_score": _highlight(duration, metrics),
                "last_hitter": str((event or {}).get("last_hitter") or "unknown"),
                "terminal_event": str((event or {}).get("event") or "unknown"),
                "terminal_event_confidence": float((event or {}).get("confidence") or 0.0),
                "terminal_landing_side": str((event or {}).get("landing_side") or "unknown"),
                "highlight_reasons": " | ".join(reasons),
                "tags": style,
            }
        )
    if output_csv is not None:
        write_rows(output_csv, ACTION_FIELDS, results)

    ranked = sorted(results, key=lambda row: (-float(row["highlight_score"]), int(row["rally"])))
    trajectory_results = [row for row in results if row["trajectory_available"]]
    summary: dict[str, Any] = {
        "schema_version": ANALYTICS_SCHEMA_VERSION,
        "available": True,
        "rallies": results,
        "match": {
            "rallies": len(results),
            "trajectory_rallies": len(trajectory_results),
            "trajectory_visible_seconds": round(
                sum(float(row["trajectory_visible_seconds"]) for row in trajectory_results), 1
            ),
            "average_coverage_percent": round(
                float(np.mean([row["trajectory_coverage_percent"] for row in trajectory_results])), 1
            )
            if trajectory_results
            else 0.0,
            "net_zone_transits": sum(int(row["net_zone_transits"]) for row in trajectory_results),
            "top_highlights": [int(row["rally"]) for row in ranked[:5]],
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
