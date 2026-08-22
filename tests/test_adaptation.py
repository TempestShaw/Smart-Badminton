from pathlib import Path

import numpy as np
import pandas as pd

from smart_badminton.adaptation import SegmentationAdapter, fit_segmentation_adapter


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
    assert report["evaluated_candidates"] == 153
    assert truth.read_bytes() == original_truth
