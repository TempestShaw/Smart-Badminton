from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import evaluate_rallies
from .model import (
    MODEL_FAMILIES,
    _classifier,
    _load_multi_dataset,
    _rally_gap_profile,
    predict_model,
    train_multi_model,
)
from .segmenter import segment_rallies


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_truth_locks(dataset_path: Path) -> list[dict[str, str]]:
    manifest = json.loads(dataset_path.read_text(encoding="utf-8"))
    checks = []
    for source in manifest.get("sources", []):
        truth = Path(source["rallies"])
        truth = truth.resolve() if truth.is_absolute() else (dataset_path.parent / truth).resolve()
        expected = str(source.get("truth_sha256") or "").lower()
        if not expected:
            raise ValueError(f"Missing truth_sha256 for {source['id']}")
        actual = sha256_file(truth)
        if actual != expected:
            raise ValueError(f"Truth lock mismatch for {source['id']}: {truth}")
        check = {"id": str(source["id"]), "path": str(truth), "sha256": actual}
        trajectory_hash = str(source.get("trajectory_sha256") or "").lower()
        if trajectory_hash:
            trajectory = _trajectory_path(dataset_path, source)
            if trajectory is None or sha256_file(trajectory) != trajectory_hash:
                raise ValueError(f"Trajectory lock mismatch for {source['id']}")
            check["trajectory_sha256"] = trajectory_hash
        checks.append(check)
    if len(checks) < 2:
        raise ValueError("Regression gate requires at least two truth-locked matches")
    return checks


def _trajectory_path(dataset_path: Path, source: dict) -> Path | None:
    value = source.get("trajectory")
    if not value:
        return None
    candidate = Path(str(value))
    resolved = candidate.resolve() if candidate.is_absolute() else (dataset_path.parent / candidate).resolve()
    return resolved if resolved.exists() else None


def _coverage_rate(report: dict) -> float:
    return report["truth_rallies_with_at_least_98_percent_coverage"] / max(1, report["truth_rallies"])


def regression_gate(
    dataset_path: Path,
    baseline_model: Path,
    report_path: Path | None = None,
    model_family: str = "hist_gradient_boosting",
    recall_tolerance: float = 0.005,
) -> dict:
    if model_family not in MODEL_FAMILIES:
        raise ValueError(f"Unknown model family: {model_family}")
    truth_locks = _verify_truth_locks(dataset_path)
    manifest = json.loads(dataset_path.read_text(encoding="utf-8"))
    source_config = {str(source["id"]): source for source in manifest["sources"]}
    loaded, feature_names, matrices, labels = _load_multi_dataset(dataset_path)
    folds = []

    with tempfile.TemporaryDirectory(prefix="smart-badminton-regression-") as temporary_directory:
        root = Path(temporary_directory)
        for held_out_index, held_out in enumerate(loaded):
            training_matrix = pd.concat(
                [matrix for index, matrix in enumerate(matrices) if index != held_out_index], ignore_index=True
            )
            training_labels = np.concatenate(
                [label for index, label in enumerate(labels) if index != held_out_index]
            )
            candidate = _classifier(family=model_family)
            candidate.fit(training_matrix, training_labels)
            probability = candidate.predict_proba(matrices[held_out_index][feature_names])[:, 1]
            gap_profile = _rally_gap_profile(
                [source["rallies"] for index, source in enumerate(loaded) if index != held_out_index]
            )
            candidate_probability = root / f"{held_out_index}-candidate-probabilities.csv"
            candidate_timeline = root / f"{held_out_index}-candidate-rallies.csv"
            baseline_probability = root / f"{held_out_index}-baseline-probabilities.csv"
            baseline_timeline = root / f"{held_out_index}-baseline-rallies.csv"
            pd.DataFrame(
                {
                    "time_seconds": held_out["frame"]["time_seconds"],
                    "frame_index": held_out["frame"]["frame_index"],
                    "rally_probability": probability,
                    "gap_soft_min_seconds": gap_profile["soft_min_seconds"],
                }
            ).to_csv(candidate_probability, index=False, float_format="%.6f")
            trajectory = _trajectory_path(dataset_path, source_config[held_out["id"]])
            segment_rallies(
                held_out["features"],
                candidate_probability,
                candidate_timeline,
                shuttle_trajectory_csv=trajectory,
            )
            predict_model(held_out["features"], baseline_model, baseline_probability)
            segment_rallies(
                held_out["features"],
                baseline_probability,
                baseline_timeline,
                shuttle_trajectory_csv=trajectory,
            )
            candidate_report = evaluate_rallies(candidate_timeline, held_out["rallies"])
            baseline_report = evaluate_rallies(baseline_timeline, held_out["rallies"])
            failures = []
            if candidate_report["active_time_recall"] + recall_tolerance < baseline_report["active_time_recall"]:
                failures.append("active_time_recall")
            if (
                candidate_report["truth_rallies_with_at_least_98_percent_coverage"]
                < baseline_report["truth_rallies_with_at_least_98_percent_coverage"]
            ):
                failures.append("whole_rally_coverage")
            if (
                candidate_report["editing_quality"]["premature_cut_seconds"]
                > baseline_report["editing_quality"]["premature_cut_seconds"] + 0.20
            ):
                failures.append("premature_cut")
            folds.append(
                {
                    "held_out": held_out["id"],
                    "training_sources": [
                        source["id"] for index, source in enumerate(loaded) if index != held_out_index
                    ],
                    "baseline": baseline_report,
                    "candidate": candidate_report,
                    "passed": not failures,
                    "failures": failures,
                }
            )

    baseline_loss = float(np.mean([row["baseline"]["editing_quality"]["editing_quality_loss"] for row in folds]))
    candidate_loss = float(np.mean([row["candidate"]["editing_quality"]["editing_quality_loss"] for row in folds]))
    per_match_passed = all(row["passed"] for row in folds)
    result = {
        "schema_version": 1,
        "dataset": str(dataset_path),
        "baseline_model": str(baseline_model),
        "model_family": model_family,
        "truth_locks": truth_locks,
        "folds": folds,
        "baseline_mean_editing_quality_loss": baseline_loss,
        "candidate_mean_editing_quality_loss": candidate_loss,
        "passed": per_match_passed and candidate_loss < baseline_loss,
        "promotion_blockers": [
            *([] if per_match_passed else ["per_match_regression"]),
            *([] if candidate_loss < baseline_loss else ["editing_quality_not_improved"]),
        ],
    }
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def train_guarded_candidate(
    dataset_path: Path,
    baseline_model: Path,
    candidate_model: Path,
    gate_report: Path,
    training_report: Path | None = None,
    model_family: str = "hist_gradient_boosting",
) -> dict:
    if candidate_model.resolve() == baseline_model.resolve():
        raise ValueError("Candidate output must not overwrite the baseline model")
    gate = regression_gate(dataset_path, baseline_model, gate_report, model_family)
    if not gate["passed"]:
        return {"trained": False, "gate": gate}
    training = train_multi_model(dataset_path, candidate_model, training_report, model_family)
    return {"trained": True, "model": str(candidate_model), "gate": gate, "training": training}
