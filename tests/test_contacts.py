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


def _rally_frame(events: list[tuple[float, str, float, float]], seconds: float = 6.0) -> pd.DataFrame:
    """A 10 fps feature timeline with one contact-like sample per (time, side, swing, direction) event."""
    times = np.round(np.arange(0.0, seconds, 0.1), 2)
    frame = pd.DataFrame(
        {
            "time_seconds": times,
            "shuttle_visible": 1.0,
            "shuttle_x_normalized": 0.5,
            "shuttle_y_normalized": 0.1,
            "near_active_wrist_visible": 1.0,
            "near_active_wrist_x_normalized": 0.9,
            "near_active_wrist_y_normalized": 0.9,
            "near_stance_width_normalized": 0.04,
            "near_swing_score": 0.05,
            "far_active_wrist_visible": 1.0,
            "far_active_wrist_x_normalized": 0.1,
            "far_active_wrist_y_normalized": 0.4,
            "far_stance_width_normalized": 0.02,
            "far_swing_score": 0.02,
            "audio_hit_score": 0.0,
        }
    )
    for time_seconds, side, swing, _direction in events:
        index = int(np.argmin(np.abs(times - time_seconds)))
        frame.loc[index, f"{side}_active_wrist_x_normalized"] = 0.5
        frame.loc[index, f"{side}_active_wrist_y_normalized"] = 0.1
        frame.loc[index, f"{side}_swing_score"] = swing
    return frame


def test_one_swing_seen_twice_counts_once_and_sides_never_repeat_within_an_exchange() -> None:
    frame = _rally_frame([(1.0, "near", 0.8, 0.0), (1.3, "near", 0.6, 0.0), (2.0, "far", 0.5, 0.0), (2.4, "far", 0.5, 0.0)])
    events = infer_contact_events(frame)

    assert [(event["time_seconds"], event["player"]) for event in events] == [(1.0, "near"), (2.0, "far")]


def test_a_wrist_beside_a_resting_shuttle_is_not_a_hit() -> None:
    # After the rally the shuttle lies by a player who walks over to pick it up: proximity without any impact.
    frame = _rally_frame([(time, "near", 0.06, 0.0) for time in np.arange(1.0, 3.0, 0.1)])

    assert infer_contact_events(frame) == []


def test_far_side_swings_are_judged_against_the_far_players_own_scale() -> None:
    # The far player looks half as tall, so their fast swings score half; they must still count.
    events = [(1.0, "near", 0.8, 0.0), (1.8, "far", 0.2, 0.0), (2.7, "near", 0.75, 0.0), (3.5, "far", 0.21, 0.0)]
    frame = _rally_frame(events)

    assert [event["player"] for event in infer_contact_events(frame)] == ["near", "far", "near", "far"]


def test_contact_evaluation_reports_precision_recall_and_impossible_repeats(tmp_path: Path) -> None:
    predicted = tmp_path / "predicted.csv"
    truth = tmp_path / "truth.csv"
    pd.DataFrame({"time_seconds": [1.0, 1.3, 2.05, 4.0], "player": ["near", "near", "far", "far"]}).to_csv(predicted, index=False)
    pd.DataFrame({"time_seconds": [1.02, 2.0, 3.0], "player": ["near", "far", "near"]}).to_csv(truth, index=False)
    from smart_badminton.contacts import evaluate_contacts

    report = evaluate_contacts(predicted, truth)

    assert (report["matched"], report["precision"], report["recall"]) == (2, 0.5, 0.667)
    assert report["impossible_same_side_repeats"] == 1
