from pathlib import Path

from smart_badminton.analytics import analyze_rally_actions


def _features(path: Path) -> None:
    path.write_text("time_seconds\n0.0\n1.0\n2.0\n3.0\n", encoding="utf-8")


def test_action_analytics_uses_owned_trajectory(tmp_path: Path) -> None:
    features = tmp_path / "features.csv"
    rallies = tmp_path / "rallies.csv"
    trajectory = tmp_path / "trajectory.csv"
    events = tmp_path / "events.csv"
    output = tmp_path / "actions.csv"
    summary_path = tmp_path / "summary.json"
    _features(features)
    rallies.write_text("rally,start_seconds,end_seconds\n1,0.0,3.0\n", encoding="utf-8")
    trajectory.write_text(
        "time_seconds,center_x,center_y,source_width,source_height,status,flight_id,detection_status,"
        "ownership_confidence,ownership_evidence\n"
        "0.5,100,300,1000,1000,tracked,1,detected,0.8,player_contact\n"
        "0.6,250,450,1000,1000,tracked,1,detected,0.8,court_continuity\n"
        "0.7,400,550,1000,1000,tracked,1,detected,0.8,net_crossing\n"
        "0.8,600,700,1000,1000,tracked,1,detected,0.8,court_continuity\n",
        encoding="utf-8",
    )
    events.write_text(
        "rally,event,confidence,last_hitter,landing_side,court_x_meters,court_y_meters\n"
        "1,landing_in_candidate,0.82,far,near,2.5,10.0\n",
        encoding="utf-8",
    )

    summary = analyze_rally_actions(
        features,
        rallies,
        output,
        summary_path,
        events_csv=events,
        trajectory_csv=trajectory,
    )

    action = summary["rallies"][0]
    assert summary["schema_version"] == 4
    assert action["trajectory_available"] is True
    assert action["trajectory_points"] == 4
    assert action["trajectory_visible_seconds"] == 0.3
    assert action["terminal_landing_side"] == "near"
    assert action["terminal_event"] == "landing_in_candidate"
    assert summary["match"]["court_heatmap"][0]["x_meters"] == 2.5
    assert output.exists()
    assert summary_path.exists()


def test_action_analytics_ignores_unowned_neighbor_trajectory(tmp_path: Path) -> None:
    features = tmp_path / "features.csv"
    rallies = tmp_path / "rallies.csv"
    trajectory = tmp_path / "trajectory.csv"
    _features(features)
    rallies.write_text("rally,start_seconds,end_seconds\n1,0.0,2.0\n", encoding="utf-8")
    trajectory.write_text(
        "time_seconds,center_x,center_y,source_width,source_height,status,flight_id,detection_status,"
        "ownership_confidence,ownership_evidence\n"
        "0.5,800,300,1000,1000,tracked,9,detected,0.2,unknown\n"
        "0.6,700,400,1000,1000,tracked,9,detected,0.2,unknown\n",
        encoding="utf-8",
    )

    action = analyze_rally_actions(features, rallies, trajectory_csv=trajectory)["rallies"][0]

    assert action["trajectory_available"] is False
    assert action["highlight_score"] == 0.0
