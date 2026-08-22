from pathlib import Path

import pandas as pd

from smart_badminton.phase_examples import capture_phase_examples


def test_phase_examples_capture_corrected_boundaries_and_evidence(tmp_path: Path) -> None:
    proposed = tmp_path / "automatic.csv"
    proposed.write_text("rally,start_seconds,end_seconds\n1,1.0,3.0\n", encoding="utf-8")
    features = tmp_path / "features.csv"
    probabilities = tmp_path / "probabilities.csv"
    times = [0.9, 1.0, 1.1, 2.9, 3.0, 3.1]
    pd.DataFrame(
        {
            "time_seconds": times,
            "near_swing_score": 0.0,
            "far_swing_score": 0.0,
            "near_lunge_score": 0.0,
            "far_lunge_score": 0.0,
            "near_flow_mean": 0.0,
            "far_flow_mean": 0.0,
            "near_motion_fraction": 0.0,
            "far_motion_fraction": 0.0,
            "audio_hit_score": 0.0,
            "near_person_visible": 1.0,
            "far_person_visible": 1.0,
            "near_foot_speed_normalized": 0.0,
            "far_foot_speed_normalized": 0.0,
            "near_stance_width_normalized": 0.05,
            "far_stance_width_normalized": 0.03,
        }
    ).to_csv(features, index=False)
    pd.DataFrame({"time_seconds": times, "rally_probability": [0.1, 0.6, 0.8, 0.7, 0.3, 0.1]}).to_csv(
        probabilities, index=False
    )
    output = tmp_path / "phase-examples.csv"

    summary = capture_phase_examples(
        "match-a",
        [{"start_seconds": 1.1, "end_seconds": 3.1}],
        proposed,
        output,
        features,
        probabilities,
    )

    rows = pd.read_csv(output)
    assert summary["changed_boundaries"] == 2
    assert rows.delta_seconds.round(1).tolist() == [0.1, 0.1]
    assert rows.rally_probability.notna().all()
