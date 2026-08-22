from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .io import load_rallies


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
    }
    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
