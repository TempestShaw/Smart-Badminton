from pathlib import Path

from smart_badminton.analytics import analyze_rally_actions
from smart_badminton.geometry import CourtGeometry
from smart_badminton.shot_analysis import classify_shots


def _geometry() -> CourtGeometry:
    court = [(0.1, 0.9), (0.9, 0.9), (0.7, 0.2), (0.3, 0.2)]
    return CourtGeometry(
        reference_width=1000,
        reference_height=1000,
        court_ground_polygon=court,
        near_player_zone=court,
        far_player_zone=court,
        net_band=[(0.2, 0.45), (0.8, 0.45), (0.8, 0.55), (0.2, 0.55)],
        shuttle_airspace_polygon=[(0, 0), (1, 0), (1, 1), (0, 1)],
        static_false_positive_polygons=[],
        background_court_polygons=[],
        active_court_polygon=court,
    )


def test_shot_classifier_labels_slow_contact_near_net_as_candidate(tmp_path: Path) -> None:
    trajectory = tmp_path / "trajectory.csv"
    trajectory.write_text(
        "time_seconds,center_x,center_y,source_width,source_height,status\n"
        "0.98,500,500,1000,1000,tracked\n"
        "1.18,540,520,1000,1000,tracked\n",
        encoding="utf-8",
    )

    shots = classify_shots([{"time_seconds": "1.0", "player": "near"}], trajectory, _geometry())

    assert shots[0]["shot_type"] == "net_candidate"
    assert 0 < shots[0]["confidence"] < 1


def test_action_summary_exposes_heatmap_and_terminal_side(tmp_path: Path) -> None:
    features = tmp_path / "features.csv"
    rallies = tmp_path / "rallies.csv"
    contacts = tmp_path / "contacts.csv"
    events = tmp_path / "events.csv"
    trajectory = tmp_path / "trajectory.csv"
    features.write_text(
        "time_seconds,audio_hit_score,near_swing_score,far_swing_score,near_lunge_score,far_lunge_score,"
        "near_flow_mean,far_flow_mean,near_motion_fraction,far_motion_fraction\n"
        "0.9,0,0,0,0,0,1,1,0.1,0.1\n1.0,0,0,0,0,0,1,1,0.1,0.1\n"
        "1.2,0,0,0,0,0,1,1,0.1,0.1\n",
        encoding="utf-8",
    )
    rallies.write_text("rally,start_seconds,end_seconds\n1,0.8,1.3\n", encoding="utf-8")
    contacts.write_text("rally,time_seconds,player,confidence\n1,1.0,near,0.8\n", encoding="utf-8")
    events.write_text(
        "rally,event,confidence,last_hitter,court_x_meters,court_y_meters\n"
        "1,landing_in_candidate,0.6,near,2.5,8.0\n",
        encoding="utf-8",
    )
    trajectory.write_text(
        "time_seconds,center_x,center_y,source_width,source_height,status\n"
        "0.98,500,500,1000,1000,tracked\n1.18,540,700,1000,1000,tracked\n",
        encoding="utf-8",
    )

    result = analyze_rally_actions(
        features,
        rallies,
        contacts_csv=contacts,
        events_csv=events,
        trajectory_csv=trajectory,
    )

    assert result["match"]["court_heatmap"][0]["x_meters"] == 2.5
    assert result["rallies"][0]["terminal_landing_side"] == "unknown"
