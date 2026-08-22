from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .io import load_rallies


def _interval_length(ranges: list[tuple[float, float]]) -> float:
    if not ranges:
        return 0.0
    ordered = sorted(ranges)
    total = 0.0
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
            continue
        total += max(0.0, end - start)
        start, end = next_start, next_end
    return total + max(0.0, end - start)


def _overlap_length(first: tuple[float, float], second: tuple[float, float]) -> float:
    return max(0.0, min(first[1], second[1]) - max(first[0], second[0]))


def editing_quality_metrics(
    predicted: list[tuple[float, float]], truth: list[tuple[float, float]]
) -> dict[str, float | int]:
    """Measure errors by their impact on a finished rally edit."""
    missed_rallies = 0
    incomplete_rallies = 0
    uncovered_truth_seconds = 0.0
    premature_cut_seconds = 0.0
    for target in truth:
        overlaps = [candidate for candidate in predicted if _overlap_length(candidate, target) > 0]
        covered = _interval_length(
            [(max(candidate[0], target[0]), min(candidate[1], target[1])) for candidate in overlaps]
        )
        duration = max(1e-6, target[1] - target[0])
        uncovered_truth_seconds += max(0.0, duration - covered)
        if covered / duration < 0.50:
            missed_rallies += 1
        if covered / duration < 0.98:
            incomplete_rallies += 1
        latest_covered = max((min(candidate[1], target[1]) for candidate in overlaps), default=target[0])
        premature_cut_seconds += max(0.0, target[1] - latest_covered)

    predicted_time = _interval_length(predicted)
    true_positive_time = _interval_length(
        [
            (max(candidate[0], target[0]), min(candidate[1], target[1]))
            for candidate in predicted
            for target in truth
            if _overlap_length(candidate, target) > 0
        ]
    )
    false_positive_seconds = max(0.0, predicted_time - true_positive_time)
    adjacent_overlap_seconds = sum(
        max(0.0, first[1] - second[0])
        for first, second in zip(sorted(predicted), sorted(predicted)[1:])
    )
    meaningful_overlaps = []
    for candidate in predicted:
        covered_truth = [
            target
            for target in truth
            if _overlap_length(candidate, target)
            >= min(0.50, max(0.10, (target[1] - target[0]) * 0.10))
        ]
        meaningful_overlaps.append(len(covered_truth))
    merged_predicted_intervals = sum(count > 1 for count in meaningful_overlaps)
    merged_truth_rallies = sum(max(0, count - 1) for count in meaningful_overlaps)

    fragmented_truth_rallies = 0
    for target in truth:
        covering_predictions = sum(
            _overlap_length(candidate, target)
            >= min(0.50, max(0.10, (target[1] - target[0]) * 0.10))
            for candidate in predicted
        )
        fragmented_truth_rallies += int(covering_predictions > 1)
    rally_count_error = abs(len(predicted) - len(truth))
    loss = (
        premature_cut_seconds * 80.0
        + missed_rallies * 120.0
        + incomplete_rallies * 60.0
        + merged_truth_rallies * 20.0
        + fragmented_truth_rallies * 20.0
        + false_positive_seconds
        + adjacent_overlap_seconds * 12.0
    )
    return {
        "premature_cut_seconds": premature_cut_seconds,
        "missed_rallies": missed_rallies,
        "incomplete_rallies": incomplete_rallies,
        "uncovered_truth_seconds": uncovered_truth_seconds,
        "false_positive_seconds": false_positive_seconds,
        "adjacent_overlap_seconds": adjacent_overlap_seconds,
        "merged_predicted_intervals": merged_predicted_intervals,
        "merged_truth_rallies": merged_truth_rallies,
        "fragmented_truth_rallies": fragmented_truth_rallies,
        "rally_count_error": rally_count_error,
        "editing_quality_loss": loss,
    }


def evaluate_rallies(
    predicted_csv: Path, truth_csv: Path, output_json: Path | None = None, sample_fps: float = 20.0
) -> dict:
    predicted = load_rallies(predicted_csv)
    truth = load_rallies(truth_csv)
    duration = max([end for _start, end in predicted + truth], default=0.0)
    times = np.arange(0.0, duration + 1.0 / sample_fps, 1.0 / sample_fps)

    def mask(ranges):
        result = np.zeros(len(times), dtype=bool)
        for start, end in ranges:
            result |= (times >= start) & (times <= end)
        return result

    predicted_mask, truth_mask = mask(predicted), mask(truth)
    true_positive = int(np.sum(predicted_mask & truth_mask))
    precision = true_positive / max(1, int(np.sum(predicted_mask)))
    recall = true_positive / max(1, int(np.sum(truth_mask)))
    per_rally = []
    for number, (start, end) in enumerate(truth, 1):
        target = (times >= start) & (times <= end)
        coverage = float(np.sum(predicted_mask & target) / max(1, np.sum(target)))
        predicted_starts = [
            candidate[0] for candidate in predicted if candidate[1] >= start - 4 and candidate[0] <= end + 4
        ]
        predicted_ends = [
            candidate[1] for candidate in predicted if candidate[1] >= start - 4 and candidate[0] <= end + 4
        ]
        per_rally.append(
            {
                "rally": number,
                "truth_start": start,
                "truth_end": end,
                "coverage": coverage,
                "nearest_start_error_seconds": min((abs(value - start) for value in predicted_starts), default=None),
                "nearest_end_error_seconds": min((abs(value - end) for value in predicted_ends), default=None),
                "pass_no_lost_play": coverage >= 0.98,
            }
        )
    report = {
        "predicted_rallies": len(predicted),
        "truth_rallies": len(truth),
        "active_time_precision": precision,
        "active_time_recall": recall,
        "active_time_f1": 2 * precision * recall / max(precision + recall, 1e-9),
        "truth_rallies_with_at_least_98_percent_coverage": sum(row["pass_no_lost_play"] for row in per_rally),
        "all_truth_rallies_preserved": all(row["pass_no_lost_play"] for row in per_rally),
        "per_rally": per_rally,
        "editing_quality": editing_quality_metrics(predicted, truth),
    }
    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
