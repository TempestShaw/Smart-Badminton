from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import asdict, dataclass
from itertools import pairwise, product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .evaluate import evaluate_rallies
from .interval_quality import INTERVAL_QUALITY_FEATURES, interval_quality_probabilities, load_interval_quality_frame
from .io import load_rallies
from .rally_evidence import build_rally_evidence
from .start_quality import START_QUALITY_FEATURES, selected_start_times, start_candidate_frame


@dataclass(frozen=True)
class SegmentationAdapter:
    start_threshold: float = 0.56
    keep_threshold: float = 0.30
    preroll: float = 0.35
    uncertain_preroll: float = 0.0
    postroll: float = 0.55
    end_pending: float = 0.55
    maximum_internal_gap: float = 1.20
    soft_gap_seconds: float = 0.0
    gap_penalty: float = 0.20
    orphan_action_probability_ceiling: float = 0.75
    orphan_action_max_duration: float = 5.0
    orphan_neighbor_gap: float = 3.0
    split_handoff_before_serve: bool = False
    interval_quality: dict | None = None
    start_quality: dict | None = None

    @classmethod
    def from_json(cls, path: Path) -> SegmentationAdapter:
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = payload.get("parameters", payload)
        fields = cls.__dataclass_fields__
        parsed = {}
        for name, field in fields.items():
            if name not in values:
                continue
            if name in {"interval_quality", "start_quality"}:
                parsed[name] = dict(values[name]) if values[name] else None
            else:
                parsed[name] = bool(values[name]) if isinstance(field.default, bool) else float(values[name])
        return cls(**parsed)

    def write(self, path: Path, metadata: dict | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": 1, "parameters": asdict(self)}
        if metadata:
            payload["fit"] = metadata
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _soft_gap_from_truth(truth_csv: Path) -> float:
    rallies = load_rallies(truth_csv)
    gaps = [start - previous_end for (_s, previous_end), (start, _e) in pairwise(rallies) if start > previous_end]
    return round(float(np.quantile(gaps, 0.15)), 3) if gaps else 0.0


def _adapter_rank(report: dict) -> tuple:
    editing = report["editing_quality"]
    return (
        editing["missed_rallies"],
        editing["incomplete_rallies"],
        editing["premature_cut_seconds"],
        editing["uncovered_truth_seconds"],
        editing.get("merged_truth_rallies", 0) + editing.get("fragmented_truth_rallies", 0),
        editing.get("rally_count_error", 0),
        editing.get("merged_truth_rallies", 0),
        editing["editing_quality_loss"],
        -report["active_time_recall"],
        -report["active_time_precision"],
    )


def fit_segmentation_adapter(
    features_csv: Path,
    probabilities_csv: Path,
    truth_csv: Path,
    output_json: Path,
    shuttle_trajectory_csv: Path | None = None,
    base_adapter: Path | None = None,
) -> dict:
    from .segmenter import segment_rallies

    truth_hash_before = hashlib.sha256(truth_csv.read_bytes()).hexdigest()
    soft_gap = _soft_gap_from_truth(truth_csv)
    candidates: list[tuple[SegmentationAdapter, dict]] = []
    with tempfile.TemporaryDirectory(prefix="smart-badminton-adapter-") as temporary_directory:
        temporary_root = Path(temporary_directory)

        def evaluate(adapter: SegmentationAdapter, number: int) -> dict:
            output = temporary_root / f"candidate-{number:03d}.csv"
            segment_rallies(
                features_csv,
                probabilities_csv,
                output,
                suppress_handoffs=True,
                shuttle_trajectory_csv=shuttle_trajectory_csv,
                adapter=adapter,
            )
            return evaluate_rallies(output, truth_csv)

        if base_adapter is not None and base_adapter.exists():
            seed = SegmentationAdapter.from_json(base_adapter)
            candidates.append((seed, evaluate(seed, -1)))

        thresholds = product(
            (0.52, 0.56, 0.60),
            (0.26, 0.30, 0.34),
            (0.55, 0.70),
            (0.14, 0.20),
            (0.75, 0.88),
            (False, True),
        )
        for number, (start, keep, end_pending, gap_penalty, orphan_ceiling, split_handoff) in enumerate(thresholds):
            adapter = SegmentationAdapter(
                start_threshold=start,
                keep_threshold=keep,
                end_pending=end_pending,
                soft_gap_seconds=soft_gap,
                gap_penalty=gap_penalty,
                orphan_action_probability_ceiling=orphan_ceiling,
                split_handoff_before_serve=split_handoff,
            )
            candidates.append((adapter, evaluate(adapter, number)))

        best, best_report = min(candidates, key=lambda item: _adapter_rank(item[1]))

        buffer_candidates = []
        offset = len(candidates)
        for number, (preroll, uncertain_preroll, postroll) in enumerate(
            product((0.20, 0.35, 0.55, 0.75, 1.15), (0.0, 1.15), (0.40, 0.55, 0.70)),
            offset,
        ):
            adapter = SegmentationAdapter(
                start_threshold=best.start_threshold,
                keep_threshold=best.keep_threshold,
                preroll=preroll,
                uncertain_preroll=max(preroll, uncertain_preroll) if uncertain_preroll else 0.0,
                postroll=postroll,
                end_pending=best.end_pending,
                soft_gap_seconds=soft_gap,
                gap_penalty=best.gap_penalty,
                orphan_action_probability_ceiling=best.orphan_action_probability_ceiling,
                split_handoff_before_serve=best.split_handoff_before_serve,
            )
            buffer_candidates.append((adapter, evaluate(adapter, number)))
        best, best_report = min([*candidates, *buffer_candidates], key=lambda item: _adapter_rank(item[1]))

        fitted_output = temporary_root / "interval-quality-source.csv"
        fitted_intervals = segment_rallies(
            features_csv,
            probabilities_csv,
            fitted_output,
            suppress_handoffs=True,
            shuttle_trajectory_csv=shuttle_trajectory_csv,
            adapter=best,
        )
        quality_frame = load_interval_quality_frame(
            fitted_intervals,
            features_csv,
            probabilities_csv,
            shuttle_trajectory_csv,
        )
        truth = load_rallies(truth_csv)
        labels = np.asarray(
            [
                (
                    sum(
                        max(0.0, min(interval.end, end) - max(interval.start, start))
                        for start, end in truth
                    )
                    / max(1e-6, interval.end - interval.start)
                    >= 0.35
                    or any(
                        max(0.0, min(interval.end, end) - max(interval.start, start))
                        / max(1e-6, end - start)
                        >= 0.98
                        for start, end in truth
                    )
                )
                for interval in fitted_intervals
            ],
            dtype=int,
        )
        if len(set(labels)) == 2 and min(np.bincount(labels)) >= 2:
            scaler = StandardScaler().fit(quality_frame)
            classifier = LogisticRegression(C=0.6, class_weight="balanced", random_state=19, max_iter=1000)
            classifier.fit(scaler.transform(quality_frame), labels)
            logistic_model = {
                "type": "logistic",
                "feature_names": list(INTERVAL_QUALITY_FEATURES),
                "means": scaler.mean_.tolist(),
                "scales": scaler.scale_.tolist(),
                "weights": classifier.coef_[0].tolist(),
                "intercept": float(classifier.intercept_[0]),
            }
            tree = DecisionTreeClassifier(
                max_depth=5,
                min_samples_leaf=1,
                class_weight="balanced",
                random_state=19,
            ).fit(quality_frame, labels)
            positive_class = int(np.flatnonzero(tree.classes_ == 1)[0])
            node_values = tree.tree_.value[:, 0, :]
            tree_model = {
                "type": "decision_tree",
                "feature_names": list(INTERVAL_QUALITY_FEATURES),
                "children_left": tree.tree_.children_left.tolist(),
                "children_right": tree.tree_.children_right.tolist(),
                "split_features": tree.tree_.feature.tolist(),
                "thresholds": tree.tree_.threshold.tolist(),
                "positive_probability": (
                    node_values[:, positive_class] / np.maximum(node_values.sum(axis=1), 1e-9)
                ).tolist(),
            }
            for model_number, candidate_model in enumerate((logistic_model, tree_model), 1):
                scores = interval_quality_probabilities(quality_frame, candidate_model)
                candidate_model["threshold"] = max(0.0, float(np.min(scores[labels == 1])) - 1e-6)
                kept = scores >= candidate_model["threshold"]
                filtered_output = temporary_root / f"interval-quality-filtered-{model_number}.csv"
                filtered_rows = pd.read_csv(fitted_output).loc[kept].reset_index(drop=True)
                filtered_rows["rally"] = np.arange(1, len(filtered_rows) + 1)
                filtered_rows.to_csv(filtered_output, index=False)
                filtered_report = evaluate_rallies(filtered_output, truth_csv)
                if (
                    filtered_report["active_time_recall"] + 1e-9 >= best_report["active_time_recall"]
                    and filtered_report["truth_rallies_with_at_least_98_percent_coverage"]
                    >= best_report["truth_rallies_with_at_least_98_percent_coverage"]
                    and _adapter_rank(filtered_report) < _adapter_rank(best_report)
                ):
                    best = SegmentationAdapter(**{**asdict(best), "interval_quality": candidate_model})
                    best_report = filtered_report

        features = pd.read_csv(features_csv)
        probabilities = pd.read_csv(probabilities_csv)
        data = features.merge(
            probabilities[["time_seconds", "rally_probability"]], on="time_seconds", how="inner"
        )
        evidence = build_rally_evidence(data, shuttle_trajectory_csv)
        start_frame = start_candidate_frame(data, evidence)
        truth_starts = np.asarray([start for start, _end in truth], dtype=float)
        if len(start_frame) and len(truth_starts):
            candidate_times = start_frame["time_seconds"].to_numpy(dtype=float)
            distances = np.min(np.abs(candidate_times[:, None] - truth_starts[None, :]), axis=1)
            labels = (distances <= 1.20).astype(int)
            if len(set(labels)) == 2 and min(np.bincount(labels)) >= 2:
                tree = DecisionTreeClassifier(
                    max_depth=6,
                    min_samples_leaf=1,
                    class_weight="balanced",
                    random_state=23,
                ).fit(start_frame[list(START_QUALITY_FEATURES)], labels)
                positive_class = int(np.flatnonzero(tree.classes_ == 1)[0])
                node_values = tree.tree_.value[:, 0, :]
                start_model = {
                    "type": "decision_tree",
                    "feature_names": list(START_QUALITY_FEATURES),
                    "children_left": tree.tree_.children_left.tolist(),
                    "children_right": tree.tree_.children_right.tolist(),
                    "split_features": tree.tree_.feature.tolist(),
                    "thresholds": tree.tree_.threshold.tolist(),
                    "positive_probability": (
                        node_values[:, positive_class] / np.maximum(node_values.sum(axis=1), 1e-9)
                    ).tolist(),
                    "threshold": 0.999999,
                    "dedupe_seconds": 1.5,
                    "terminal_tail_seconds": max(0.70, best.postroll),
                    "lead_safety_seconds": 0.10,
                }
                lead_tree = DecisionTreeRegressor(
                    max_depth=5,
                    min_samples_leaf=1,
                    random_state=29,
                ).fit(
                    start_frame.loc[labels == 1, list(START_QUALITY_FEATURES)],
                    np.maximum(
                        0.0,
                        [
                            candidate_times[index]
                            - truth_starts[
                                int(np.argmin(np.abs(truth_starts - candidate_times[index])))
                            ]
                            for index in np.flatnonzero(labels == 1)
                        ],
                    ),
                )
                start_model["lead_model"] = {
                    "type": "decision_tree_regressor",
                    "feature_names": list(START_QUALITY_FEATURES),
                    "children_left": lead_tree.tree_.children_left.tolist(),
                    "children_right": lead_tree.tree_.children_right.tolist(),
                    "split_features": lead_tree.tree_.feature.tolist(),
                    "thresholds": lead_tree.tree_.threshold.tolist(),
                    "values": lead_tree.tree_.value[:, 0, 0].tolist(),
                }
                selected = selected_start_times(start_frame, start_model)
                matched_offsets = []
                for truth_start in truth_starts:
                    nearby = [value for value in selected if abs(value - truth_start) <= 1.20]
                    if nearby:
                        matched_offsets.append(min(nearby, key=lambda value: abs(value - truth_start)) - truth_start)
                if matched_offsets:
                    quiet_gap_options = (
                        (0.0, 0.0, 0.0),
                        (0.18, 2.0, 0.6),
                        (0.22, 2.5, 0.8),
                        (0.26, 3.0, 1.0),
                    )
                    for quiet_ceiling, quiet_minimum, quiet_gap in product(
                        (0.14, 0.20),
                        (0.6, 1.0),
                        quiet_gap_options,
                    ):
                        gap_ceiling, gap_window, gap_minimum = quiet_gap
                        candidate_model = dict(start_model)
                        candidate_model.update(
                            {
                                "lead_seconds": max(0.0, float(np.median(matched_offsets))),
                                "quiet_probability_ceiling": quiet_ceiling,
                                "quiet_min_seconds": quiet_minimum,
                                "quiet_search_seconds": 8.0,
                                "quiet_gap_probability_ceiling": gap_ceiling,
                                "quiet_gap_window_seconds": gap_window,
                                "quiet_gap_min_seconds": gap_minimum,
                            }
                        )
                        candidate_adapter = SegmentationAdapter(
                            **{**asdict(best), "start_quality": candidate_model}
                        )
                        candidate_report = evaluate(candidate_adapter, 1000 + len(candidates))
                        candidates.append((candidate_adapter, candidate_report))
                        if (
                            candidate_report["active_time_recall"] + 1e-9
                            >= best_report["active_time_recall"]
                            and candidate_report[
                                "truth_rallies_with_at_least_98_percent_coverage"
                            ]
                            >= best_report["truth_rallies_with_at_least_98_percent_coverage"]
                            and candidate_report["editing_quality"]["premature_cut_seconds"]
                            <= best_report["editing_quality"]["premature_cut_seconds"] + 1e-9
                            and _adapter_rank(candidate_report) < _adapter_rank(best_report)
                        ):
                            best = candidate_adapter
                            best_report = candidate_report

    if hashlib.sha256(truth_csv.read_bytes()).hexdigest() != truth_hash_before:
        raise RuntimeError(f"Truth file changed while fitting adapter: {truth_csv}")

    metadata = {
        "truth": str(truth_csv),
        "truth_sha256": truth_hash_before,
        "evaluated_candidates": len(candidates) + len(buffer_candidates),
        "metrics": {
            "active_time_precision": best_report["active_time_precision"],
            "active_time_recall": best_report["active_time_recall"],
            "active_time_f1": best_report["active_time_f1"],
            "editing_quality": best_report["editing_quality"],
        },
    }
    best.write(output_json, metadata)
    return {"adapter": asdict(best), **metadata}
