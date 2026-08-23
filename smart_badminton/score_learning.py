from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MODEL_VERSION = 2
TRAINING_FIELDS = [
    "project",
    "rally",
    "winner",
    "server",
    "last_hitter_truth",
    "terminal_event_truth",
    "landing_side_truth",
    "post_rally_event",
    "detected_event",
    "detected_last_hitter",
    "detected_landing_side",
    "detected_confidence",
    "detected_score_usable",
    "predicted",
    "rule",
    "correct",
]
MIN_DISABLE_SAMPLES = 3
MIN_TRUST_SAMPLES = 4
DISABLE_ACCURACY = 0.70
TRUST_ACCURACY = 0.90
LEARNED_CONFIDENCE_FLOOR = 0.55


def raw_terminal_prediction(event: dict[str, Any] | None) -> tuple[str, str | None]:
    if not event:
        return "unknown", None
    event_name = str(event.get("event", "unknown")).replace("_candidate", "")
    landing_side = str(event.get("landing_side", "unknown"))
    last_hitter = str(event.get("last_hitter", "unknown"))
    if event_name == "landing_in" and landing_side in {"near", "far"}:
        return ("far" if landing_side == "near" else "near"), f"landing_in:{landing_side}"
    if event_name in {"landing_out", "net"} and last_hitter in {"near", "far"}:
        winner = "far" if last_hitter == "near" else "near"
        return winner, f"{event_name}:{last_hitter}"
    return "unknown", None


def load_score_evidence_model(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("version", 0)) != MODEL_VERSION or not isinstance(payload.get("rules"), dict):
        raise ValueError(f"Unsupported score evidence model: {path}")
    return payload


def evidence_rule(
    model: dict[str, Any] | None,
    key: str | None,
    project: str | None = None,
) -> dict[str, Any] | None:
    if model is None or key is None:
        return None
    if project:
        project_rule = model.get("project_rules", {}).get(project, {}).get(key)
        if isinstance(project_rule, dict):
            return project_rule
    rule = model.get("rules", {}).get(key)
    return rule if isinstance(rule, dict) else None


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def collect_score_examples(library: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    examples: list[dict[str, Any]] = []
    inputs: list[Path] = []
    for corrections_path in sorted(library.rglob("score-corrections.csv")):
        project_root = corrections_path.parent.parent
        events_path = project_root / "Analysis" / "rally-events.csv"
        if not events_path.exists():
            continue
        corrections = _read_csv(corrections_path)
        events = {int(row["rally"]): row for row in _read_csv(events_path)}
        project = str(project_root.relative_to(library)).replace("\\", "/")
        inputs.extend([corrections_path, events_path])
        for correction in corrections:
            winner = str(correction.get("winner", "auto"))
            if winner not in {"near", "far"}:
                continue
            rally = int(correction["rally"])
            event = events.get(rally, {})
            predicted, rule_key = raw_terminal_prediction(event)
            examples.append(
                {
                    "project": project,
                    "rally": rally,
                    "winner": winner,
                    "server": str(correction.get("server", "unknown")),
                    "last_hitter_truth": str(correction.get("last_hitter", "unknown")),
                    "terminal_event_truth": str(correction.get("terminal_event", "unknown")),
                    "landing_side_truth": str(correction.get("landing_side", "unknown")),
                    "post_rally_event": str(correction.get("post_rally_event", "unknown")),
                    "detected_event": str(event.get("event", "unknown")),
                    "detected_last_hitter": str(event.get("last_hitter", "unknown")),
                    "detected_landing_side": str(event.get("landing_side", "unknown")),
                    "detected_confidence": str(event.get("confidence", "0")),
                    "detected_score_usable": str(event.get("score_usable", "no")),
                    "predicted": predicted,
                    "rule": rule_key or "",
                    "correct": predicted == winner if rule_key is not None else "",
                }
            )
    return examples, inputs


def _dataset_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        digest.update(str(path).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def fit_score_evidence_model(library: Path, output: Path) -> dict[str, Any]:
    examples, inputs = collect_score_examples(library)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for example in examples:
        if example["rule"]:
            grouped.setdefault(str(example["rule"]), []).append(example)

    def summarize_rule(rows: list[dict[str, Any]], require_multiple_projects: bool) -> dict[str, Any]:
        samples = len(rows)
        correct = sum(bool(row["correct"]) for row in rows)
        accuracy = correct / samples
        projects = sorted({str(row["project"]) for row in rows})
        status = "default"
        cross_project_ready = not require_multiple_projects or len(projects) >= 2
        if cross_project_ready and samples >= MIN_DISABLE_SAMPLES and accuracy < DISABLE_ACCURACY:
            status = "disabled"
        elif cross_project_ready and samples >= MIN_TRUST_SAMPLES and accuracy >= TRUST_ACCURACY:
            status = "trusted"
        elif require_multiple_projects and not cross_project_ready:
            status = "candidate"
        return {
            "samples": samples,
            "correct": correct,
            "accuracy": round(accuracy, 4),
            "projects": projects,
            "status": status,
            "confidence_floor": LEARNED_CONFIDENCE_FLOOR if status == "trusted" else 0.75,
        }

    rules = {key: summarize_rule(rows, True) for key, rows in sorted(grouped.items())}
    project_rules: dict[str, dict[str, Any]] = {}
    for project in sorted({str(row["project"]) for row in examples}):
        project_rows = [row for row in examples if row["project"] == project and row["rule"]]
        project_grouped: dict[str, list[dict[str, Any]]] = {}
        for row in project_rows:
            project_grouped.setdefault(str(row["rule"]), []).append(row)
        project_rules[project] = {
            key: summarize_rule(rows, False) for key, rows in sorted(project_grouped.items())
        }

    dataset_sha256 = _dataset_hash(inputs) if inputs else hashlib.sha256(b"").hexdigest()
    training_csv = output.with_name(f"{output.stem}-training.csv")
    if output.exists() and training_csv.exists():
        current = load_score_evidence_model(output)
        if current and current.get("dataset_sha256") == dataset_sha256:
            return current
    payload = {
        "version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "library": str(library.resolve()),
        "dataset_sha256": dataset_sha256,
        "manual_examples": len(examples),
        "comparable_examples": sum(bool(row["rule"]) for row in examples),
        "projects": sorted({str(row["project"]) for row in examples}),
        "rules": rules,
        "project_rules": project_rules,
        "trusted_rules": sum(row["status"] == "trusted" for row in rules.values()),
        "disabled_rules": sum(row["status"] == "disabled" for row in rules.values()),
        "local_disabled_rules": sum(
            row["status"] == "disabled" for project in project_rules.values() for row in project.values()
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with training_csv.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=TRAINING_FIELDS)
        writer.writeheader()
        writer.writerows(examples)
    return payload


def default_score_model_path(library: Path) -> Path:
    return library / ".smart-badminton" / "models" / "score-evidence-v2.json"
