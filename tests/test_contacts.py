from pathlib import Path

import numpy as np
import pandas as pd

from smart_badminton.contacts import analyze_contacts, infer_contact_events


def _contact_features() -> pd.DataFrame:
    times = np.asarray([0.9, 1.0, 1.1])
    return pd.DataFrame(
        {
            "time_seconds": times,
            "shuttle_visible": 1.0,
            "shuttle_x_normalized": [0.45, 0.50, 0.44],
            "shuttle_y_normalized": [0.52, 0.50, 0.48],
            "near_active_wrist_x_normalized": [0.48, 0.50, 0.49],
            "near_active_wrist_y_normalized": [0.52, 0.50, 0.49],
            "near_active_wrist_visible": 1.0,
            "near_stance_width_normalized": 0.04,
            "near_swing_score": [0.2, 0.8, 0.3],
            "far_active_wrist_x_normalized": 0.8,
            "far_active_wrist_y_normalized": 0.5,
            "far_active_wrist_visible": 1.0,
            "far_stance_width_normalized": 0.02,
            "far_swing_score": 0.0,
            "audio_hit_score": [0.0, 0.7, 0.0],
        }
    )


def test_contact_requires_wrist_proximity_and_supporting_evidence() -> None:
    events = infer_contact_events(_contact_features())

    assert len(events) == 1
    assert events[0]["player"] == "near"
    assert events[0]["confidence"] > 0.7


def test_contact_analysis_assigns_event_to_rally(tmp_path: Path) -> None:
    features = tmp_path / "features.csv"
    rallies = tmp_path / "rallies.csv"
    output = tmp_path / "contacts.csv"
    summary = tmp_path / "summary.json"
    _contact_features().to_csv(features, index=False)
    rallies.write_text("rally,start_seconds,end_seconds\n1,0.5,1.5\n", encoding="utf-8")

    result = analyze_contacts(features, rallies, output, summary)

    assert result["available"] is True
    assert result["events"] == 1
    assert pd.read_csv(output).iloc[0].rally == 1
