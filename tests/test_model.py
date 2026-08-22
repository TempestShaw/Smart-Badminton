import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from smart_badminton.model import benchmark_multi_models, build_feature_matrix, predict_model, train_multi_model


def _write_source(root: Path, source_id: str, offset: float) -> tuple[Path, Path]:
    times = np.arange(0, 12, 0.1)
    active = ((times >= 2 + offset) & (times <= 5 + offset)) | ((times >= 7) & (times <= 9))
    features = root / f"{source_id}-features.csv"
    rallies = root / f"{source_id}-rallies.csv"
    pd.DataFrame(
        {
            "time_seconds": times,
            "frame_index": np.arange(len(times)),
            "court_motion_fraction": active.astype(float),
            "near_motion_fraction": active.astype(float) * 0.8,
            "far_motion_fraction": active.astype(float) * 0.7,
            "near_flow_mean": active.astype(float) * 3.0,
            "far_flow_mean": active.astype(float) * 2.5,
            "near_swing_score": active.astype(float) * 0.6,
            "far_swing_score": active.astype(float) * 0.5,
            "near_lunge_score": np.zeros(len(times)),
            "far_lunge_score": np.zeros(len(times)),
            "audio_hit_score": active.astype(float) * 0.3,
            "shuttle_visible": np.zeros(len(times)),
            "shuttle_speed_normalized": np.zeros(len(times)),
        }
    ).to_csv(features, index=False)
    rallies.write_text(
        f"rally,start_seconds,end_seconds\n1,{2 + offset},{5 + offset}\n2,7,9\n",
        encoding="utf-8",
    )
    return features, rallies


def test_feature_matrix_couples_owned_shuttle_with_player_contact() -> None:
    frame = pd.DataFrame(
        {
            "time_seconds": [0.0, 0.1, 0.2],
            "shuttle_visible": [0.0, 1.0, 0.0],
            "shuttle_x_normalized": [0.0, 0.52, 0.0],
            "shuttle_y_normalized": [0.0, 0.48, 0.0],
            "shuttle_speed_normalized": [0.0, 2.0, 0.0],
            "near_active_wrist_visible": [0.0, 1.0, 0.0],
            "near_active_wrist_x_normalized": [0.0, 0.5, 0.0],
            "near_active_wrist_y_normalized": [0.0, 0.5, 0.0],
            "near_swing_score": [0.0, 0.8, 0.0],
            "audio_hit_score": [0.0, 0.7, 0.0],
        }
    )

    matrix = build_feature_matrix(frame, sample_fps=10.0)

    assert matrix.loc[1, "shuttle_near_contact_support"] > 0.7
    assert matrix.loc[1, "shuttle_motion_support"] == 2.0
    assert matrix.loc[1, "shuttle_audio_support"] == 0.7


def test_multi_video_training_holds_out_entire_sources(tmp_path: Path) -> None:
    source_a = _write_source(tmp_path, "a", 0.0)
    source_b = _write_source(tmp_path, "b", 0.2)
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps(
            {
                "sources": [
                    {"id": "a", "features": source_a[0].name, "rallies": source_a[1].name},
                    {"id": "b", "features": source_b[0].name, "rallies": source_b[1].name},
                ]
            }
        ),
        encoding="utf-8",
    )
    model = tmp_path / "multi.joblib"
    report_path = tmp_path / "report.json"

    report = train_multi_model(dataset, model, report_path)

    assert [row["held_out"] for row in report["leave_one_video_out"]] == ["a", "b"]
    assert report["training_samples"] == 240
    bundle = joblib.load(model)
    assert bundle["training_sources"] == ["a", "b"]
    assert bundle["rally_gap_profile"]["sample_count"] == 2
    assert bundle["rally_gap_profile"]["soft_min_seconds"] >= bundle["rally_gap_profile"]["hard_min_seconds"]
    assert json.loads(report_path.read_text(encoding="utf-8"))["leave_one_video_out_mean"]["recall"] >= 0

    probabilities = tmp_path / "probabilities.csv"
    predict_model(source_a[0], model, probabilities)
    prediction_frame = pd.read_csv(probabilities)
    assert prediction_frame["gap_hard_min_seconds"].iloc[0] == bundle["rally_gap_profile"]["hard_min_seconds"]
    assert prediction_frame["gap_soft_min_seconds"].iloc[0] == bundle["rally_gap_profile"]["soft_min_seconds"]


def test_model_benchmark_runs_real_segmentation_and_selects_family(tmp_path: Path) -> None:
    source_a = _write_source(tmp_path, "a", 0.0)
    source_b = _write_source(tmp_path, "b", 0.2)
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps(
            {
                "sources": [
                    {"id": "a", "features": source_a[0].name, "rallies": source_a[1].name},
                    {"id": "b", "features": source_b[0].name, "rallies": source_b[1].name},
                ]
            }
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "benchmark.json"

    report = benchmark_multi_models(
        dataset,
        report_path,
        families=["hist_gradient_boosting", "logistic_regression"],
    )

    assert report["selected_family"] in {"hist_gradient_boosting", "logistic_regression"}
    assert all("segmentation_mean" in result for result in report["families"])
    assert json.loads(report_path.read_text(encoding="utf-8"))["selection_score"] >= 0
