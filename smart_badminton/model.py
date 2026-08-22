from __future__ import annotations

import json
import tempfile
from itertools import pairwise
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .evaluate import evaluate_rallies
from .io import load_rallies
from .segmenter import segment_rallies

BASE_FEATURES = [
    "court_motion_mean",
    "court_motion_fraction",
    "near_motion_mean",
    "near_motion_fraction",
    "far_motion_mean",
    "far_motion_fraction",
    "near_flow_mean",
    "far_flow_mean",
    "near_person_visible",
    "far_person_visible",
    "near_swing_score",
    "far_swing_score",
    "near_lunge_score",
    "far_lunge_score",
    "audio_hit_score",
    "seconds_since_audio_hit",
    "shuttle_dynamic_count",
    "shuttle_best_confidence",
    "shuttle_visible",
    "shuttle_speed_normalized",
]

MODEL_FAMILIES = (
    "hist_gradient_boosting",
    "extra_trees",
    "random_forest",
    "logistic_regression",
)


def build_feature_matrix(frame: pd.DataFrame, sample_fps: float | None = None) -> pd.DataFrame:
    def numeric_source(name: str) -> pd.Series:
        source = frame[name] if name in frame else pd.Series(0.0, index=frame.index)
        return pd.to_numeric(source, errors="coerce").fillna(0.0)

    columns: dict[str, pd.Series] = {}
    for name in BASE_FEATURES:
        columns[name] = numeric_source(name)
    data = pd.DataFrame(columns, index=frame.index)
    if sample_fps is None:
        times = pd.to_numeric(frame["time_seconds"], errors="coerce").to_numpy()
        sample_fps = 1.0 / max(float(np.median(np.diff(times))), 1e-3) if len(times) > 1 else 10.0
    engineered: dict[str, pd.Series] = {}
    for seconds in (0.5, 1.0, 2.0, 4.0):
        window = max(1, round(seconds * sample_fps))
        for name in (
            "court_motion_fraction",
            "near_motion_fraction",
            "far_motion_fraction",
            "near_flow_mean",
            "far_flow_mean",
            "near_swing_score",
            "far_swing_score",
            "near_lunge_score",
            "far_lunge_score",
            "audio_hit_score",
            "shuttle_visible",
            "shuttle_speed_normalized",
        ):
            centered = data[name].rolling(window, center=True, min_periods=1)
            engineered[f"{name}_mean_{seconds:g}s"] = centered.mean()
            engineered[f"{name}_max_{seconds:g}s"] = centered.max()
    engineered["players_visible"] = data["near_person_visible"] + data["far_person_visible"]
    engineered["combined_swing"] = data["near_swing_score"] + data["far_swing_score"]
    engineered["combined_lunge"] = data["near_lunge_score"] + data["far_lunge_score"]
    engineered["combined_flow"] = data["near_flow_mean"] + data["far_flow_mean"]
    shuttle_visible = data["shuttle_visible"].clip(0.0, 1.0)
    shuttle_x = numeric_source("shuttle_x_normalized")
    shuttle_y = numeric_source("shuttle_y_normalized")
    contact_support = []
    for side in ("near", "far"):
        wrist_visible = numeric_source(f"{side}_active_wrist_visible").clip(0.0, 1.0)
        wrist_x = numeric_source(f"{side}_active_wrist_x_normalized")
        wrist_y = numeric_source(f"{side}_active_wrist_y_normalized")
        distance = np.hypot(shuttle_x - wrist_x, shuttle_y - wrist_y)
        proximity = shuttle_visible * wrist_visible * (1.0 - distance / 0.22).clip(0.0, 1.0)
        support = proximity * (0.35 + data[f"{side}_swing_score"].clip(0.0, 1.0) * 0.65)
        engineered[f"shuttle_{side}_contact_support"] = support
        contact_support.append(support)
    engineered["shuttle_player_contact_support"] = np.maximum(contact_support[0], contact_support[1])
    engineered["shuttle_motion_support"] = shuttle_visible * data["shuttle_speed_normalized"].clip(0.0, 4.0)
    engineered["shuttle_audio_support"] = shuttle_visible * data["audio_hit_score"].clip(0.0, 1.0)
    for seconds in (0.5, 1.0, 2.0):
        window = max(1, round(seconds * sample_fps))
        for name in (
            "shuttle_player_contact_support",
            "shuttle_motion_support",
            "shuttle_audio_support",
        ):
            values = pd.Series(engineered[name], index=frame.index)
            engineered[f"{name}_max_{seconds:g}s"] = values.rolling(window, center=True, min_periods=1).max()
            engineered[f"{name}_mean_{seconds:g}s"] = values.rolling(window, center=True, min_periods=1).mean()
    data = pd.concat([data, pd.DataFrame(engineered, index=frame.index)], axis=1)
    return data.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def _labels(times: np.ndarray, rallies_path: Path) -> np.ndarray:
    rallies = load_rallies(rallies_path)
    labels = np.zeros(len(times), dtype=np.uint8)
    for start, end in rallies:
        labels[(times >= start) & (times <= end)] = 1
    return labels


def _rally_gap_profile(rallies_paths: list[Path]) -> dict[str, float | int]:
    gaps: list[float] = []
    for rallies_path in rallies_paths:
        rallies = load_rallies(rallies_path)
        gaps.extend(
            start - previous_end
            for (_previous_start, previous_end), (start, _end) in pairwise(rallies)
            if start - previous_end >= 0.5
        )
    if not gaps:
        return {
            "sample_count": 0,
            "hard_min_seconds": 0.0,
            "soft_min_seconds": 0.0,
            "median_seconds": 0.0,
        }
    values = np.asarray(gaps, dtype=float)
    return {
        "sample_count": len(gaps),
        "hard_min_seconds": round(float(np.quantile(values, 0.01)), 3),
        "soft_min_seconds": round(float(np.quantile(values, 0.15)), 3),
        "median_seconds": round(float(np.median(values)), 3),
    }


def _classifier(final: bool = False, family: str = "hist_gradient_boosting"):
    if family == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=240 if final else 180,
            max_leaf_nodes=24 if final else 20,
            min_samples_leaf=15 if final else 18,
            l2_regularization=0.5 if final else 0.4,
            class_weight="balanced",
            random_state=19,
        )
    if family == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=360 if final else 220,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=19,
        )
    if family == "random_forest":
        return RandomForestClassifier(
            n_estimators=320 if final else 180,
            min_samples_leaf=6,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=19,
        )
    if family == "logistic_regression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.7,
                class_weight="balanced",
                max_iter=1200,
                random_state=19,
            ),
        )
    raise ValueError(f"Unknown model family: {family}. Choose from {', '.join(MODEL_FAMILIES)}")


def train_model(
    features_csv: Path,
    rallies_csv: Path,
    model_path: Path,
    report_path: Path | None = None,
    model_family: str = "hist_gradient_boosting",
) -> dict:
    frame = pd.read_csv(features_csv)
    times = frame["time_seconds"].to_numpy(dtype=float)
    sample_fps = 1.0 / max(float(np.median(np.diff(times))), 1e-3)
    matrix = build_feature_matrix(frame, sample_fps)
    labels = _labels(times, rallies_csv)
    groups = np.floor(times / 60.0).astype(int)
    scores = []
    unique_groups = np.unique(groups)
    if len(unique_groups) >= 3:
        splitter = GroupKFold(n_splits=min(5, len(unique_groups)))
        for train_index, test_index in splitter.split(matrix, labels, groups):
            candidate = _classifier(family=model_family)
            candidate.fit(matrix.iloc[train_index], labels[train_index])
            prediction = candidate.predict(matrix.iloc[test_index])
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels[test_index], prediction, average="binary", zero_division=0
            )
            scores.append({"precision": float(precision), "recall": float(recall), "f1": float(f1)})
    model = _classifier(final=True, family=model_family)
    model.fit(matrix, labels)
    gap_profile = _rally_gap_profile([rallies_csv])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_names": list(matrix.columns),
            "sample_fps": sample_fps,
            "model_family": model_family,
            "rally_gap_profile": gap_profile,
        },
        model_path,
    )
    fitted = model.predict(matrix)
    report = {
        "training_samples": len(labels),
        "model_family": model_family,
        "positive_fraction": float(labels.mean()),
        "sample_fps": sample_fps,
        "rally_gap_profile": gap_profile,
        "cross_validation": scores,
        "cross_validation_mean": {
            key: float(np.mean([row[key] for row in scores])) if scores else None
            for key in ("precision", "recall", "f1")
        },
        "training_classification_report": classification_report(labels, fitted, output_dict=True, zero_division=0),
        "note": "Grouped cross-validation splits the timeline into minute blocks; final model is fitted on all corrected labels.",
    }
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _resolve_manifest_path(base: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()


def _load_multi_dataset(dataset_path: Path) -> tuple[list[dict], list[str], list[pd.DataFrame], list[np.ndarray]]:
    manifest = json.loads(dataset_path.read_text(encoding="utf-8"))
    sources = manifest.get("sources")
    if not isinstance(sources, list) or len(sources) < 2:
        raise ValueError("Multi-video training requires at least two sources")

    loaded: list[dict] = []
    feature_names: list[str] = []
    seen_names: set[str] = set()
    for source in sources:
        source_id = str(source["id"])
        features_path = _resolve_manifest_path(dataset_path.parent, str(source["features"]))
        rallies_path = _resolve_manifest_path(dataset_path.parent, str(source["rallies"]))
        frame = pd.read_csv(features_path)
        times = frame["time_seconds"].to_numpy(dtype=float)
        sample_fps = 1.0 / max(float(np.median(np.diff(times))), 1e-3)
        matrix = build_feature_matrix(frame, sample_fps)
        for name in matrix.columns:
            if name not in seen_names:
                feature_names.append(name)
                seen_names.add(name)
        labels = _labels(times, rallies_path)
        loaded.append(
            {
                "id": source_id,
                "features": features_path,
                "rallies": rallies_path,
                "frame": frame,
                "matrix": matrix,
                "labels": labels,
                "sample_fps": sample_fps,
            }
        )

    matrices = [source["matrix"].reindex(columns=feature_names, fill_value=0.0) for source in loaded]
    labels = [source["labels"] for source in loaded]
    return loaded, feature_names, matrices, labels


def _cross_validate_family(
    loaded: list[dict],
    matrices: list[pd.DataFrame],
    labels: list[np.ndarray],
    family: str,
    evaluate_segments: bool = False,
) -> dict:
    leave_one_video_out = []
    out_of_fold_labels: list[np.ndarray] = []
    out_of_fold_predictions: list[np.ndarray] = []
    with tempfile.TemporaryDirectory(prefix="smart-badminton-model-benchmark-") as temporary_directory:
        temporary_root = Path(temporary_directory)
        for held_out_index, held_out in enumerate(loaded):
            train_matrix = pd.concat(
                [matrix for index, matrix in enumerate(matrices) if index != held_out_index], ignore_index=True
            )
            train_labels = np.concatenate([label for index, label in enumerate(labels) if index != held_out_index])
            candidate = _classifier(family=family)
            candidate.fit(train_matrix, train_labels)
            gap_profile = _rally_gap_profile(
                [source["rallies"] for index, source in enumerate(loaded) if index != held_out_index]
            )
            probability = candidate.predict_proba(matrices[held_out_index])[:, 1]
            prediction = probability >= 0.5
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels[held_out_index], prediction, average="binary", zero_division=0
            )
            row = {
                "held_out": held_out["id"],
                "training_sources": [
                    source["id"] for index, source in enumerate(loaded) if index != held_out_index
                ],
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "samples": len(prediction),
                "positive_fraction": float(labels[held_out_index].mean()),
            }
            if evaluate_segments:
                probability_path = temporary_root / f"{family}-{held_out_index}-probabilities.csv"
                rallies_path = temporary_root / f"{family}-{held_out_index}-rallies.csv"
                frame = held_out["frame"]
                pd.DataFrame(
                    {
                        "time_seconds": frame["time_seconds"],
                        "frame_index": frame["frame_index"],
                        "rally_probability": probability,
                        "gap_hard_min_seconds": gap_profile["hard_min_seconds"],
                        "gap_soft_min_seconds": gap_profile["soft_min_seconds"],
                    }
                ).to_csv(probability_path, index=False, float_format="%.6f")
                segment_rallies(held_out["features"], probability_path, rallies_path)
                segment_report = evaluate_rallies(rallies_path, held_out["rallies"])
                row["segmentation"] = {
                    key: segment_report[key]
                    for key in (
                        "predicted_rallies",
                        "truth_rallies",
                        "active_time_precision",
                        "active_time_recall",
                        "active_time_f1",
                        "truth_rallies_with_at_least_98_percent_coverage",
                        "all_truth_rallies_preserved",
                    )
                }
            leave_one_video_out.append(row)
            out_of_fold_labels.append(labels[held_out_index])
            out_of_fold_predictions.append(prediction)

    means = {
        key: float(np.mean([row[key] for row in leave_one_video_out])) for key in ("precision", "recall", "f1")
    }
    result = {
        "model_family": family,
        "leave_one_video_out": leave_one_video_out,
        "leave_one_video_out_mean": means,
        "leave_one_video_out_classification_report": classification_report(
            np.concatenate(out_of_fold_labels),
            np.concatenate(out_of_fold_predictions),
            output_dict=True,
            zero_division=0,
        ),
    }
    if evaluate_segments:
        segment_rows = [row["segmentation"] for row in leave_one_video_out]
        coverage_rates = [
            row["truth_rallies_with_at_least_98_percent_coverage"] / max(1, row["truth_rallies"])
            for row in segment_rows
        ]
        result["segmentation_mean"] = {
            key: float(np.mean([row[key] for row in segment_rows]))
            for key in ("active_time_precision", "active_time_recall", "active_time_f1")
        }
        result["segmentation_mean"]["rally_98_percent_coverage_rate"] = float(np.mean(coverage_rates))
        result["selection_score"] = float(
            0.45 * result["segmentation_mean"]["active_time_f1"]
            + 0.25 * result["segmentation_mean"]["active_time_recall"]
            + 0.20 * result["segmentation_mean"]["rally_98_percent_coverage_rate"]
            + 0.10 * means["f1"]
        )
    return result


def benchmark_multi_models(
    dataset_path: Path,
    report_path: Path | None = None,
    families: list[str] | None = None,
) -> dict:
    selected_families = families or list(MODEL_FAMILIES)
    unknown = [family for family in selected_families if family not in MODEL_FAMILIES]
    if unknown:
        raise ValueError(f"Unknown model families: {', '.join(unknown)}")
    loaded, _feature_names, matrices, labels = _load_multi_dataset(dataset_path)
    results = [
        _cross_validate_family(loaded, matrices, labels, family, evaluate_segments=True)
        for family in selected_families
    ]
    winner = max(results, key=lambda result: result["selection_score"])
    report = {
        "dataset": str(dataset_path),
        "sources": [source["id"] for source in loaded],
        "families": results,
        "selected_family": winner["model_family"],
        "selection_score": winner["selection_score"],
        "selection_policy": (
            "45% segmented active-time F1 + 25% segmented recall + 20% whole-rally 98%-coverage rate "
            "+ 10% frame F1, with each fold holding out one complete video."
        ),
    }
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def train_multi_model(
    dataset_path: Path,
    model_path: Path,
    report_path: Path | None = None,
    model_family: str = "hist_gradient_boosting",
) -> dict:
    loaded, feature_names, matrices, labels = _load_multi_dataset(dataset_path)
    validation = _cross_validate_family(loaded, matrices, labels, model_family)

    combined_matrix = pd.concat(matrices, ignore_index=True)
    combined_labels = np.concatenate(labels)
    model = _classifier(final=True, family=model_family)
    model.fit(combined_matrix, combined_labels)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    median_fps = float(np.median([source["sample_fps"] for source in loaded]))
    gap_profile = _rally_gap_profile([source["rallies"] for source in loaded])
    joblib.dump(
        {
            "model": model,
            "feature_names": feature_names,
            "sample_fps": median_fps,
            "model_family": model_family,
            "training_sources": [source["id"] for source in loaded],
            "dataset": str(dataset_path),
            "rally_gap_profile": gap_profile,
        },
        model_path,
    )
    report = {
        "dataset": str(dataset_path),
        "model_family": model_family,
        "training_samples": len(combined_labels),
        "positive_fraction": float(combined_labels.mean()),
        "sample_fps": median_fps,
        "rally_gap_profile": gap_profile,
        "sources": [
            {
                "id": source["id"],
                "features": str(source["features"]),
                "rallies": str(source["rallies"]),
                "samples": len(source["labels"]),
                "positive_fraction": float(source["labels"].mean()),
                "sample_fps": float(source["sample_fps"]),
            }
            for source in loaded
        ],
        "leave_one_video_out": validation["leave_one_video_out"],
        "leave_one_video_out_mean": validation["leave_one_video_out_mean"],
        "leave_one_video_out_classification_report": validation["leave_one_video_out_classification_report"],
        "note": "Each validation fold holds out an entire video; the final model is fitted on every validated source.",
    }
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def predict_model(features_csv: Path, model_path: Path, output_csv: Path) -> None:
    frame = pd.read_csv(features_csv)
    bundle = joblib.load(model_path)
    matrix = build_feature_matrix(frame, float(bundle["sample_fps"]))
    for name in bundle["feature_names"]:
        if name not in matrix:
            matrix[name] = 0.0
    probability = bundle["model"].predict_proba(matrix[bundle["feature_names"]])[:, 1]
    output = frame[["time_seconds", "frame_index"]].copy()
    output["rally_probability"] = probability
    gap_profile = bundle.get("rally_gap_profile") or {}
    output["gap_hard_min_seconds"] = float(gap_profile.get("hard_min_seconds", 0.0))
    output["gap_soft_min_seconds"] = float(gap_profile.get("soft_min_seconds", 0.0))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False, float_format="%.6f")
