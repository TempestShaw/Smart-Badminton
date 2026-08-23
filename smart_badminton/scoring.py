from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .io import load_rallies, read_rows, write_rows
from .score_learning import evidence_rule, load_score_evidence_model, raw_terminal_prediction

CORRECTION_FIELDS = [
    "rally",
    "winner",
    "server_override",
    "server",
    "last_hitter",
    "terminal_event",
    "landing_side",
    "post_rally_event",
    "note",
]
SCORE_FIELDS = [
    "rally",
    "winner",
    "winner_source",
    "near_score",
    "far_score",
    "near_games",
    "far_games",
    "server_next",
    "game_finished",
    "confidence",
    "note",
    "score_complete",
]
NEXT_SERVE_CONFIDENCE_THRESHOLD = 0.72
MULTIMODAL_CONFIDENCE_THRESHOLD = 0.82
SIDE_VALUES = {"near", "far", "unknown"}
TERMINAL_VALUES = {"landing_in", "landing_out", "net", "unreturned", "unknown"}
POST_RALLY_VALUES = {"handoff", "none", "unknown"}


def _next_serve_calibration(
    corrections: list[dict[str, Any]], serve_observations: list[dict[str, Any]]
) -> dict[str, Any]:
    serve_map = {
        int(row["rally"]): str(row.get("server", "unknown"))
        for row in serve_observations
        if float(row.get("confidence", 0.0)) >= NEXT_SERVE_CONFIDENCE_THRESHOLD
    }
    checks = []
    for correction in corrections:
        winner = str(correction.get("winner", "auto"))
        observed = serve_map.get(int(correction["rally"]) + 1, "unknown")
        if winner in {"near", "far"} and observed in {"near", "far"}:
            checks.append(winner == observed)
    accuracy = sum(checks) / len(checks) if checks else None
    return {
        "samples": len(checks),
        "accuracy": accuracy,
        "enabled": len(checks) < 3 or accuracy >= 0.70,
    }


def load_score_corrections(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    return [
        {
            "rally": int(row["rally"]),
            "winner": str(row.get("winner", "auto")),
            "server_override": str(row.get("server_override", "unknown")),
            "server": str(row.get("server", "unknown")),
            "last_hitter": str(row.get("last_hitter", "unknown")),
            "terminal_event": str(row.get("terminal_event", "unknown")),
            "landing_side": str(row.get("landing_side", "unknown")),
            "post_rally_event": str(row.get("post_rally_event", "unknown")),
            "note": str(row.get("note", "")),
        }
        for row in read_rows(path)
    ]


def validate_score_corrections(payload: Any, rally_count: int) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        raise TypeError("score corrections must be a list")
    result = []
    seen = set()
    for index, item in enumerate(payload, 1):
        if not isinstance(item, dict):
            raise TypeError(f"score correction {index} is invalid")
        rally = int(item["rally"])
        winner = str(item.get("winner", "auto"))
        server = str(item.get("server_override", "unknown"))
        rally_server = str(item.get("server", "unknown"))
        last_hitter = str(item.get("last_hitter", "unknown"))
        terminal_event = str(item.get("terminal_event", "unknown"))
        landing_side = str(item.get("landing_side", "unknown"))
        post_rally_event = str(item.get("post_rally_event", "unknown"))
        if not 1 <= rally <= rally_count:
            raise ValueError(f"score correction {index} rally is outside the timeline")
        if rally in seen:
            raise ValueError(f"score correction for rally {rally} is duplicated")
        if winner not in {"near", "far", "no_point", "auto"}:
            raise ValueError(f"score correction {index} winner is invalid")
        if server not in SIDE_VALUES:
            raise ValueError(f"score correction {index} server override is invalid")
        if rally_server not in SIDE_VALUES:
            raise ValueError(f"score correction {index} server is invalid")
        if last_hitter not in SIDE_VALUES:
            raise ValueError(f"score correction {index} last hitter is invalid")
        if terminal_event not in TERMINAL_VALUES:
            raise ValueError(f"score correction {index} terminal event is invalid")
        if landing_side not in SIDE_VALUES:
            raise ValueError(f"score correction {index} landing side is invalid")
        if post_rally_event not in POST_RALLY_VALUES:
            raise ValueError(f"score correction {index} post-rally event is invalid")
        seen.add(rally)
        result.append(
            {
                "rally": rally,
                "winner": winner,
                "server_override": server,
                "server": rally_server,
                "last_hitter": last_hitter,
                "terminal_event": terminal_event,
                "landing_side": landing_side,
                "post_rally_event": post_rally_event,
                "note": str(item.get("note", ""))[:240],
            }
        )
    return sorted(result, key=lambda row: int(row["rally"]))


def save_score_corrections(path: Path, rows: list[dict[str, Any]]) -> Path | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = path.with_suffix(path.suffix + ".bak") if path.exists() else None
    if backup is not None:
        shutil.copy2(path, backup)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=".csv", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=CORRECTION_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return backup


def load_machine_score_suggestions(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    suggestions = []
    for row in payload.get("labels", []) if isinstance(payload, dict) else []:
        suggestion = row.get("suggestion", {}) if isinstance(row, dict) else {}
        if row.get("status") != "consensus" or not isinstance(suggestion, dict):
            continue
        winner = str(suggestion.get("winner", "unknown"))
        confidence = float(suggestion.get("confidence", 0.0))
        if winner in {"near", "far"} and confidence >= MULTIMODAL_CONFIDENCE_THRESHOLD:
            suggestions.append(
                {
                    "rally": int(row["rally"]),
                    "winner": winner,
                    "confidence": confidence,
                    "terminal_event": str(suggestion.get("terminal_event", "unknown")),
                }
            )
    return suggestions


def _automatic_winner(
    event: dict[str, str] | None,
    evidence_model: dict[str, Any] | None = None,
    evidence_project: str | None = None,
) -> tuple[str, float, str]:
    predicted, rule_key = raw_terminal_prediction(event)
    if event is None or predicted == "unknown" or rule_key is None:
        return "unknown", 0.0, "terminal event does not determine a winner"
    confidence = float(event.get("confidence", 0.0))
    rule = evidence_rule(evidence_model, rule_key, evidence_project)
    if rule and rule.get("status") == "disabled":
        return "unknown", confidence, f"manual evidence disabled {rule_key}"
    default_safe = event.get("score_usable") == "yes" and confidence >= 0.75
    learned_safe = bool(
        rule
        and rule.get("status") == "trusted"
        and confidence >= float(rule.get("confidence_floor", 0.75))
    )
    if not default_safe and not learned_safe:
        return "unknown", confidence, "terminal event is not score-safe"
    event_name, subject = rule_key.split(":", 1)
    if event_name == "landing_in":
        note = f"shuttle landed in {subject} court"
    else:
        note = f"{subject} player made a {event_name} fault"
    if learned_safe and not default_safe:
        note = f"learned evidence: {note}"
    return predicted, confidence, note


def _game_won(score: int, opponent: int) -> bool:
    return score >= 21 and (score - opponent >= 2 or score >= 30)


def calculate_score_state(
    rally_count: int,
    events: list[dict[str, str]] | None = None,
    corrections: list[dict[str, Any]] | None = None,
    initial_server: str = "unknown",
    serve_observations: list[dict[str, Any]] | None = None,
    allow_next_serve: bool = True,
    evidence_model: dict[str, Any] | None = None,
    evidence_project: str | None = None,
    machine_suggestions: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    event_map = {int(row["rally"]): row for row in events or []}
    correction_map = {int(row["rally"]): row for row in corrections or []}
    serve_map = {int(row["rally"]): row for row in serve_observations or []}
    machine_map = {int(row["rally"]): row for row in machine_suggestions or []}
    near_score = far_score = near_games = far_games = 0
    first_observed_server = serve_map.get(1, {}).get("server", "unknown")
    server = initial_server if initial_server in {"near", "far"} else str(first_observed_server)
    if server not in {"near", "far"}:
        server = "unknown"
    unresolved_before = 0
    results = []
    for rally in range(1, rally_count + 1):
        winner, confidence, note = _automatic_winner(event_map.get(rally), evidence_model, evidence_project)
        source = "automatic-terminal" if winner in {"near", "far"} else "unresolved"
        machine = machine_map.get(rally)
        if winner == "unknown" and machine:
            winner = str(machine["winner"])
            confidence = float(machine["confidence"])
            source = "automatic-multimodal"
            note = f"multimodal consensus: {machine.get('terminal_event', 'unknown')}"
        next_serve = serve_map.get(rally + 1)
        if winner == "unknown" and next_serve and allow_next_serve:
            next_server = str(next_serve.get("server", "unknown"))
            next_confidence = float(next_serve.get("confidence", 0.0))
            if next_server in {"near", "far"} and next_confidence >= NEXT_SERVE_CONFIDENCE_THRESHOLD:
                winner = next_server
                confidence = next_confidence
                source = "automatic-next-serve"
                note = f"next rally has a formal {next_server} serve ({next_confidence:.2f})"
        correction = correction_map.get(rally)
        if correction and correction.get("winner") != "auto":
            requested = str(correction["winner"])
            winner = "unknown" if requested == "no_point" else requested
            source = "manual-no-point" if requested == "no_point" else "manual"
            confidence = 1.0
            note = str(correction.get("note") or "manual Studio correction")
        if winner == "near":
            near_score += 1
            server = "near"
        elif winner == "far":
            far_score += 1
            server = "far"
        elif source == "unresolved":
            unresolved_before += 1
        if correction and correction.get("server_override") in {"near", "far"}:
            server = str(correction["server_override"])
            note = f"{note}; manual server override"
        game_finished = ""
        if _game_won(near_score, far_score):
            near_games += 1
            game_finished = "near"
        elif _game_won(far_score, near_score):
            far_games += 1
            game_finished = "far"
        results.append(
            {
                "rally": rally,
                "winner": winner,
                "winner_source": source,
                "near_score": near_score,
                "far_score": far_score,
                "near_games": near_games,
                "far_games": far_games,
                "server_next": server,
                "game_finished": game_finished,
                "confidence": round(confidence, 3),
                "note": note,
                "score_complete": unresolved_before == 0,
            }
        )
        if game_finished:
            near_score = far_score = 0
    return results


def analyze_score(
    rallies_csv: Path,
    output_csv: Path,
    events_csv: Path | None = None,
    corrections_csv: Path | None = None,
    summary_json: Path | None = None,
    initial_server: str = "unknown",
    serve_observations: list[dict[str, Any]] | None = None,
    evidence_model_path: Path | None = None,
    evidence_project: str | None = None,
    machine_labels_path: Path | None = None,
) -> dict[str, Any]:
    rallies = load_rallies(rallies_csv)
    events = read_rows(events_csv) if events_csv is not None and events_csv.exists() else []
    corrections = load_score_corrections(corrections_csv)
    evidence_model = load_score_evidence_model(evidence_model_path)
    machine_suggestions = load_machine_score_suggestions(machine_labels_path)
    next_serve_calibration = _next_serve_calibration(corrections, serve_observations or [])
    rows = calculate_score_state(
        len(rallies),
        events,
        corrections,
        initial_server,
        serve_observations,
        bool(next_serve_calibration["enabled"]),
        evidence_model,
        evidence_project,
        machine_suggestions,
    )
    write_rows(output_csv, SCORE_FIELDS, rows)
    summary = {
        "available": True,
        "rallies": rows,
        "corrections": corrections,
        "resolved": sum(row["winner"] in {"near", "far"} for row in rows),
        "unresolved": sum(row["winner"] == "unknown" and row["winner_source"] == "unresolved" for row in rows),
        "manual": sum(str(row["winner_source"]).startswith("manual") for row in rows),
        "automatic": sum(str(row["winner_source"]).startswith("automatic") for row in rows),
        "complete": all(bool(row["score_complete"]) for row in rows),
        "next_serve_calibration": next_serve_calibration,
        "evidence_model": {
            "available": evidence_model is not None,
            "version": evidence_model.get("version") if evidence_model else None,
            "manual_examples": evidence_model.get("manual_examples", 0) if evidence_model else 0,
            "trusted_rules": evidence_model.get("trusted_rules", 0) if evidence_model else 0,
            "disabled_rules": evidence_model.get("disabled_rules", 0) if evidence_model else 0,
            "local_disabled_rules": evidence_model.get("local_disabled_rules", 0) if evidence_model else 0,
            "project": evidence_project,
        },
        "method": "terminal event, multimodal consensus, or next rally's visually detected formal server",
        "disclaimer": (
            "Automatic points require independent visual evidence. Unresolved rallies are not counted, so a partial "
            "score is never presented as official. Manual review remains independent of editing."
        ),
    }
    if summary_json is not None:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def evaluate_score(predicted_csv: Path, truth_csv: Path, output_json: Path | None = None) -> dict[str, Any]:
    predicted = {int(row["rally"]): row.get("winner", "unknown") for row in read_rows(predicted_csv)}
    truth = {int(row["rally"]): row.get("winner", "unknown") for row in read_rows(truth_csv)}
    comparable = [rally for rally, winner in truth.items() if winner in {"near", "far"}]
    covered = [rally for rally in comparable if predicted.get(rally) in {"near", "far"}]
    correct = sum(predicted[rally] == truth[rally] for rally in covered)
    result = {
        "truth_rallies": len(comparable),
        "covered_rallies": len(covered),
        "coverage": len(covered) / len(comparable) if comparable else 0.0,
        "accuracy_on_covered": correct / len(covered) if covered else 0.0,
        "unresolved_rallies": [rally for rally in comparable if rally not in covered],
        "editing_metrics_included": False,
    }
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
