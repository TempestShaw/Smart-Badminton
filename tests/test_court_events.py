import pandas as pd

from smart_badminton.court_events import _infer_pickup_landing, _infer_terminal_event
from smart_badminton.geometry import CourtGeometry


def _geometry() -> CourtGeometry:
    court = [(0.2, 0.5), (0.8, 0.5), (0.9, 1.0), (0.1, 1.0)]
    return CourtGeometry(
        reference_width=1000,
        reference_height=1000,
        court_ground_polygon=court,
        near_player_zone=court,
        far_player_zone=court,
        net_band=[(0.1, 0.45), (0.9, 0.45), (0.9, 0.55), (0.1, 0.55)],
        shuttle_airspace_polygon=[(0.05, 0.05), (0.95, 0.05), (1.0, 1.0), (0.0, 1.0)],
        static_false_positive_polygons=[],
        background_court_polygons=[],
        active_court_polygon=court,
    )


def test_descending_terminal_inside_ground_is_candidate_but_not_score_usable() -> None:
    points = [
        {"time": 4.7, "x": 500.0, "y": 650.0, "flight_id": "1"},
        {"time": 4.8, "x": 510.0, "y": 700.0, "flight_id": "1"},
        {"time": 4.9, "x": 520.0, "y": 760.0, "flight_id": "1"},
    ]

    event = _infer_terminal_event(points, 5.0, _geometry(), 1000, 1000, "near")

    assert event["event"] == "landing_in_candidate"
    assert event["last_hitter"] == "near"
    assert event["score_usable"] == "no"
    assert event["height_ambiguous"] == "yes"


def test_apparent_net_crossing_is_reported_but_height_remains_ambiguous() -> None:
    geometry = _geometry()
    geometry.court_corners = [(0.1, 0.9), (0.9, 0.9), (0.7, 0.2), (0.3, 0.2)]
    points = [
        {"time": 1.0, "x": 500.0, "y": 500.0, "source_width": 1280.0, "source_height": 720.0},
        {"time": 1.2, "x": 700.0, "y": 300.0, "source_width": 1280.0, "source_height": 720.0},
    ]

    event = _infer_terminal_event(points, 1.3, geometry, 1280, 720, "near")

    assert event["height_ambiguous"] == "yes"
    assert event["net_crossing_confidence"] <= 0.55


def test_missing_trajectory_stays_unknown() -> None:
    event = _infer_terminal_event([], 5.0, _geometry(), 1000, 1000, "unknown")

    assert event["event"] == "unknown"
    assert event["confidence"] == 0.0


def test_confirmed_landing_in_near_court_is_score_usable_without_homography_or_last_hitter() -> None:
    geometry = _geometry()
    geometry.near_player_zone = [(0.1, 0.7), (0.9, 0.7), (0.9, 1.0), (0.1, 1.0)]
    geometry.far_player_zone = [(0.2, 0.5), (0.8, 0.5), (0.85, 0.7), (0.15, 0.7)]
    points = [
        {"time": 4.5, "x": 500.0, "y": 520.0, "flight_id": "1", "status": "tracked"},
        {"time": 4.7, "x": 510.0, "y": 610.0, "flight_id": "1", "status": "tracked"},
        {"time": 4.8, "x": 520.0, "y": 680.0, "flight_id": "1", "status": "tracked"},
        {"time": 4.95, "x": 530.0, "y": 900.0, "flight_id": "1", "status": "manual"},
    ]

    event = _infer_terminal_event(points, 5.0, geometry, 1000, 1000, "unknown", 0.1)

    assert event["event"] == "landing_in_candidate"
    assert event["landing_side"] == "near"
    assert event["score_usable"] == "yes"


def test_continued_rally_probability_prevents_a_premature_landing_score() -> None:
    geometry = _geometry()
    geometry.near_player_zone = [(0.1, 0.7), (0.9, 0.7), (0.9, 1.0), (0.1, 1.0)]
    geometry.far_player_zone = [(0.2, 0.5), (0.8, 0.5), (0.85, 0.7), (0.15, 0.7)]
    points = [
        {"time": 4.7, "x": 510.0, "y": 610.0, "flight_id": "1", "status": "tracked"},
        {"time": 4.8, "x": 520.0, "y": 680.0, "flight_id": "1", "status": "tracked"},
        {"time": 4.95, "x": 530.0, "y": 900.0, "flight_id": "1", "status": "manual"},
    ]

    event = _infer_terminal_event(points, 5.0, geometry, 1000, 1000, "unknown", 0.95)

    assert event["event"] == "landing_in_candidate"
    assert event["score_usable"] == "no"


def test_player_pickup_inside_active_court_recovers_missed_landing_side() -> None:
    rows = []
    for time, near_wrist_y in [(4.5, 0.68), (4.9, 0.69), (5.4, 0.86)]:
        rows.append(
            {
                "time_seconds": time,
                "near_person_visible": 1.0,
                "near_active_wrist_visible": 1.0,
                "near_active_wrist_x_normalized": 0.62,
                "near_active_wrist_y_normalized": near_wrist_y,
                "near_foot_x_normalized": 0.60,
                "near_foot_y_normalized": 0.90,
                "far_person_visible": 1.0,
                "far_active_wrist_visible": 1.0,
                "far_active_wrist_x_normalized": 0.50,
                "far_active_wrist_y_normalized": 0.55,
                "far_foot_x_normalized": 0.50,
                "far_foot_y_normalized": 0.62,
            }
        )

    pickup = _infer_pickup_landing(pd.DataFrame(rows), 5.0, _geometry())

    assert pickup is not None
    assert pickup["side"] == "near"


def test_low_wrist_without_post_rally_descent_is_not_a_pickup() -> None:
    rows = []
    for time in (4.5, 4.9, 5.4):
        rows.append(
            {
                "time_seconds": time,
                "near_person_visible": 1.0,
                "near_active_wrist_visible": 1.0,
                "near_active_wrist_x_normalized": 0.62,
                "near_active_wrist_y_normalized": 0.84,
                "near_foot_x_normalized": 0.60,
                "near_foot_y_normalized": 0.90,
                "far_person_visible": 1.0,
                "far_active_wrist_visible": 1.0,
                "far_active_wrist_x_normalized": 0.50,
                "far_active_wrist_y_normalized": 0.55,
                "far_foot_x_normalized": 0.50,
                "far_foot_y_normalized": 0.62,
            }
        )

    assert _infer_pickup_landing(pd.DataFrame(rows), 5.0, _geometry()) is None
