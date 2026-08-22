from pathlib import Path

import numpy as np
import pandas as pd

from smart_badminton.adaptation import SegmentationAdapter, _adapter_rank, fit_segmentation_adapter
from smart_badminton.interval_quality import interval_quality_probabilities


def test_adapter_rank_prioritizes_complete_rallies_over_precision() -> None:
    complete = {
        "active_time_recall": 0.99,
        "active_time_precision": 0.55,
        "editing_quality": {
            "missed_rallies": 0,
            "incomplete_rallies": 0,
            "premature_cut_seconds": 0.0,
            "uncovered_truth_seconds": 0.0,
            "editing_quality_loss": 500.0,
        },
    }
    precise_but_cut = {
        "active_time_recall": 0.95,
        "active_time_precision": 0.9,
        "editing_quality": {
            "missed_rallies": 0,
            "incomplete_rallies": 1,
            "premature_cut_seconds": 0.1,
            "uncovered_truth_seconds": 0.1,
            "editing_quality_loss": 100.0,
        },
    }

    assert _adapter_rank(complete) < _adapter_rank(precise_but_cut)


def test_local_adapter_is_fitted_without_modifying_truth(tmp_path: Path) -> None:
    times = np.round(np.arange(0.0, 8.0, 0.1), 3)
    active = ((times >= 1.0) & (times <= 2.5)) | ((times >= 5.0) & (times <= 6.5))
    features = tmp_path / "features.csv"
    probabilities = tmp_path / "probabilities.csv"
    truth = tmp_path / "truth.csv"
    output = tmp_path / "adapter.json"
    pd.DataFrame(
        {
            "time_seconds": times,
            "frame_index": np.arange(len(times)),
            "near_swing_score": active.astype(float) * 0.7,
            "far_swing_score": 0.0,
            "near_lunge_score": 0.0,
            "far_lunge_score": 0.0,
            "near_flow_mean": 0.0,
            "far_flow_mean": 0.0,
            "near_motion_fraction": 0.0,
            "far_motion_fraction": 0.0,
            "audio_hit_score": 0.0,
            "shuttle_visible": 0.0,
            "shuttle_speed_normalized": 0.0,
        }
    ).to_csv(features, index=False)
    pd.DataFrame(
        {"time_seconds": times, "frame_index": np.arange(len(times)), "rally_probability": np.where(active, 0.95, 0.01)}
    ).to_csv(probabilities, index=False)
    truth.write_text("rally,start_seconds,end_seconds\n1,1,2.5\n2,5,6.5\n", encoding="utf-8")
    original_truth = truth.read_bytes()

    report = fit_segmentation_adapter(features, probabilities, truth, output)

    assert output.exists()
    assert SegmentationAdapter.from_json(output).soft_gap_seconds > 0
    assert report["evaluated_candidates"] == 174
    assert truth.read_bytes() == original_truth


def test_interval_quality_model_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "adapter.json"
    model = {
        "feature_names": ["duration"],
        "means": [3.0],
        "scales": [1.0],
        "weights": [2.0],
        "intercept": -1.0,
        "threshold": 0.4,
    }

    SegmentationAdapter(interval_quality=model).write(path)

    assert SegmentationAdapter.from_json(path).interval_quality == model


def test_decision_tree_interval_quality_model_is_json_runnable() -> None:
    model = {
        "type": "decision_tree",
        "feature_names": ["duration"],
        "children_left": [1, -1, -1],
        "children_right": [2, -1, -1],
        "split_features": [0, -2, -2],
        "thresholds": [2.0, -2.0, -2.0],
        "positive_probability": [0.5, 0.1, 0.9],
    }

    scores = interval_quality_probabilities(pd.DataFrame({"duration": [1.0, 3.0]}), model)

    assert scores.tolist() == [0.1, 0.9]
