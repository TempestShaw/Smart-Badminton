from pathlib import Path

import pandas as pd

from smart_badminton.analytics import analyze_rally_actions


def test_action_analytics_builds_entertainment_metrics(tmp_path: Path) -> None:
    times = [index / 10 for index in range(61)]
    frame = pd.DataFrame(
        {
            "time_seconds": times,
            "audio_hit_score": [1.0 if index in (12, 20, 28, 36, 44) else 0.0 for index in range(61)],
            "near_swing_score": [0.9 if index in (12, 28, 44) else 0.0 for index in range(61)],
            "far_swing_score": [0.8 if index in (20, 36) else 0.0 for index in range(61)],
            "near_lunge_score": [0.8 if index in (25, 40) else 0.0 for index in range(61)],
            "far_lunge_score": [0.7 if index == 32 else 0.0 for index in range(61)],
            "near_flow_mean": [2.5] * 61,
            "far_flow_mean": [2.0] * 61,
            "near_motion_fraction": [0.09] * 61,
            "far_motion_fraction": [0.08] * 61,
        }
    )
    features = tmp_path / "features.csv"
    rallies = tmp_path / "rallies.csv"
    output = tmp_path / "actions.csv"
    summary_path = tmp_path / "summary.json"
    frame.to_csv(features, index=False)
    rallies.write_text("rally,start_seconds,end_seconds\n1,1.0,5.0\n", encoding="utf-8")

    summary = analyze_rally_actions(features, rallies, output, summary_path)

    action = summary["rallies"][0]
    assert summary["available"] is True
    assert summary["schema_version"] == 3
    assert action["estimated_hits"] == 5
    assert action["smash_candidates"] == 5
    assert action["near_lunges"] == 2
    assert action["far_lunges"] == 1
    assert "火力全开" in action["tags"]
    assert summary["match"]["top_highlights"] == [1]
    assert output.exists()
    assert summary_path.exists()


def test_raw_audio_anchors_ignore_noisy_pose_and_add_one_quiet_strong_swing(tmp_path: Path) -> None:
    times = [index / 10 for index in range(121)]
    noisy_near = {27: 0.57, 34: 0.47, 40: 0.57, 45: 0.38, 61: 0.37, 69: 0.86, 84: 0.43, 89: 0.43}
    frame = pd.DataFrame(
        {
            "time_seconds": times,
            "audio_hit_score": [0.9 if index in (45, 57, 60, 69) else 0.0 for index in range(121)],
            "near_swing_score": [noisy_near.get(index, 0.0) for index in range(121)],
            "far_swing_score": [1.0 if index == 92 else 0.0 for index in range(121)],
            "near_lunge_score": [0.0] * 121,
            "far_lunge_score": [0.0] * 121,
            "near_flow_mean": [1.0] * 121,
            "far_flow_mean": [1.0] * 121,
            "near_motion_fraction": [0.04] * 121,
            "far_motion_fraction": [0.04] * 121,
        }
    )
    features = tmp_path / "features.csv"
    rallies = tmp_path / "rallies.csv"
    audio_events = tmp_path / "audio-events.csv"
    contacts = tmp_path / "contacts.csv"
    frame.to_csv(features, index=False)
    rallies.write_text("rally,start_seconds,end_seconds\n1,0.0,11.0\n", encoding="utf-8")
    audio_events.write_text(
        "time_seconds,score\n4.48,26.9\n5.68,26.6\n6.01,14.1\n6.86,24.8\n",
        encoding="utf-8",
    )
    contacts.write_text(
        "rally,time_seconds,player,confidence\n1,4.50,near,0.8\n",
        encoding="utf-8",
    )

    action = analyze_rally_actions(features, rallies, audio_events_csv=audio_events, contacts_csv=contacts)["rallies"][0]

    assert action["audio_hit_candidates"] == 4
    assert action["pose_only_hit_candidates"] == 1
    assert action["estimated_hits"] == 5
    assert action["contact_hit_candidates"] == 1
    assert action["hit_estimation_source"] == "contact+audio-pose-fusion"
