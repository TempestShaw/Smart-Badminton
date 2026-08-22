import csv
import json
from pathlib import Path

import cv2
import numpy as np

from smart_badminton.geometry import CourtGeometry
from smart_badminton.trajectory import (
    _drop_weak_partial_tracks,
    _link_dynamic_rows,
    _manual_annotation_rows,
    _predict_track_point,
    _select_primary_track_indices,
    _stitch_primary_flights,
    analyze_shuttle_trajectory,
)


def test_trajectory_suppresses_static_false_positive_and_keeps_flight(tmp_path: Path) -> None:
    detections = tmp_path / "detections.csv"
    output = tmp_path / "trajectory.csv"
    fields = ["time_seconds", "frame", "confidence", "center_x", "center_y", "width", "height"]
    with detections.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for index in range(40):
            writer.writerow(
                {
                    "time_seconds": index / 30,
                    "frame": index,
                    "confidence": 0.7,
                    "center_x": 100.0,
                    "center_y": 200.0,
                    "width": 8.0,
                    "height": 8.0,
                }
            )
        for index, (x, y) in enumerate(((300, 300), (340, 280), (385, 270), (430, 275)), start=10):
            writer.writerow(
                {
                    "time_seconds": index / 30,
                    "frame": index,
                    "confidence": 0.4,
                    "center_x": x,
                    "center_y": y,
                    "width": 8.0,
                    "height": 8.0,
                }
            )

    result = analyze_shuttle_trajectory(detections, output)

    assert result["stationary_candidates"] == 40
    assert result["tracked_points"] == 4
    assert result["tracks"] == 1
    assert result["flights"] == 1


def test_trajectory_suppresses_persistent_jittering_feather_fragment(tmp_path: Path) -> None:
    detections = tmp_path / "detections.csv"
    output = tmp_path / "trajectory.csv"
    fields = [
        "time_seconds",
        "frame",
        "source_width",
        "source_height",
        "confidence",
        "center_x",
        "center_y",
        "width",
        "height",
    ]
    with detections.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for index in range(120):
            writer.writerow(
                {
                    "time_seconds": index / 15,
                    "frame": index * 2,
                    "source_width": 1920,
                    "source_height": 1080,
                    "confidence": 0.72,
                    "center_x": 920 + (index % 7 - 3) * 2.2,
                    "center_y": 646 + (index % 5 - 2) * 2.0,
                    "width": 13,
                    "height": 14,
                }
            )
        for offset, (x, y) in enumerate(((500, 780), (560, 730), (630, 690), (710, 670)), start=30):
            writer.writerow(
                {
                    "time_seconds": offset / 15,
                    "frame": offset * 2,
                    "source_width": 1920,
                    "source_height": 1080,
                    "confidence": 0.24,
                    "center_x": x,
                    "center_y": y,
                    "width": 12,
                    "height": 13,
                }
            )

    result = analyze_shuttle_trajectory(detections, output)

    assert result["stationary_candidates"] == 120
    assert result["tracked_points"] == 4
    assert result["tracks"] == 1


def test_trajectory_preserves_detection_coordinate_dimensions(tmp_path: Path) -> None:
    detections = tmp_path / "detections.csv"
    output = tmp_path / "trajectory.csv"
    fields = [
        "time_seconds",
        "frame",
        "source_width",
        "source_height",
        "confidence",
        "center_x",
        "center_y",
        "width",
        "height",
    ]
    with detections.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for index, x in enumerate((200, 240, 290, 350)):
            writer.writerow(
                {
                    "time_seconds": index / 15,
                    "frame": index * 2,
                    "source_width": 1280,
                    "source_height": 720,
                    "confidence": 0.5,
                    "center_x": x,
                    "center_y": 300 - index * 15,
                    "width": 8,
                    "height": 8,
                }
            )

    analyze_shuttle_trajectory(detections, output)

    with output.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    assert rows[0]["source_width"] == "1280"
    assert rows[0]["source_height"] == "720"


def test_single_shuttle_competition_rejects_short_overlapping_neighbor_track() -> None:
    rows = []
    proposed = {}
    for track_id, points in (
        (1, [(1.00, 500, 100, 0.55), (1.10, 560, 130, 0.60), (1.20, 620, 170, 0.62)]),
        (2, [(1.05, 1100, 300, 0.18), (1.15, 1140, 320, 0.20), (1.25, 1180, 340, 0.19)]),
        (3, [(1.45, 700, 260, 0.50), (1.55, 740, 300, 0.55), (1.65, 780, 350, 0.58)]),
    ):
        for time_seconds, x, y, confidence in points:
            index = len(rows)
            rows.append(
                {
                    "time_seconds": time_seconds,
                    "frame": round(time_seconds * 30),
                    "confidence": confidence,
                    "center_x": x,
                    "center_y": y,
                    "width": 8.0,
                    "height": 8.0,
                }
            )
            proposed[index] = track_id

    selected = {proposed[index] for index in _select_primary_track_indices(rows, proposed)}

    assert selected == {1, 3}


def test_racket_contact_hint_can_select_the_lower_confidence_competing_track() -> None:
    rows = []
    proposed = {}
    for track_id, confidence, points in (
        (1, 0.72, [(1.0, 1050, 250), (1.1, 1090, 270), (1.2, 1130, 300)]),
        (2, 0.30, [(1.0, 610, 365), (1.1, 650, 350), (1.2, 700, 345)]),
    ):
        for time_seconds, x, y in points:
            index = len(rows)
            rows.append(
                {
                    "time_seconds": time_seconds,
                    "frame": round(time_seconds * 30),
                    "source_width": 1280,
                    "source_height": 720,
                    "confidence": confidence,
                    "center_x": x,
                    "center_y": y,
                    "width": 8.0,
                    "height": 8.0,
                }
            )
            proposed[index] = track_id
    contact_hints = [{"time": 1.1, "x": 0.50, "y": 0.50, "support": 0.9, "side": "near"}]

    selected = {proposed[index] for index in _select_primary_track_indices(rows, proposed, contact_hints)}

    assert selected == {2}


def test_tight_player_possession_hint_can_select_held_shuttle_track() -> None:
    rows = []
    proposed = {}
    for track_id, confidence, points in (
        (1, 0.50, [(1.0, 1040, 620), (1.1, 1050, 620), (1.2, 1060, 620)]),
        (2, 0.32, [(1.0, 610, 365), (1.1, 620, 360), (1.2, 635, 355)]),
    ):
        for time_seconds, x, y in points:
            index = len(rows)
            rows.append(
                {
                    "time_seconds": time_seconds,
                    "frame": round(time_seconds * 30),
                    "source_width": 1280,
                    "source_height": 720,
                    "confidence": confidence,
                    "center_x": x,
                    "center_y": y,
                    "width": 8.0,
                    "height": 8.0,
                }
            )
            proposed[index] = track_id
    possession_hints = [
        {"time": 1.1, "x": 0.485, "y": 0.50, "support": 0.32, "radius": 0.045, "side": "near"}
    ]

    selected = {proposed[index] for index in _select_primary_track_indices(rows, proposed, possession_hints)}

    assert selected == {2}


def test_tracklets_are_stitched_into_one_flight_across_short_occlusion() -> None:
    rows = [
        {"time_seconds": 1.0, "center_x": 100.0, "center_y": 100.0},
        {"time_seconds": 1.1, "center_x": 150.0, "center_y": 120.0},
        {"time_seconds": 1.4, "center_x": 220.0, "center_y": 150.0},
        {"time_seconds": 1.5, "center_x": 260.0, "center_y": 180.0},
        {"time_seconds": 6.0, "center_x": 300.0, "center_y": 200.0},
        {"time_seconds": 6.1, "center_x": 340.0, "center_y": 220.0},
    ]
    accepted = {0: 1, 1: 1, 2: 2, 3: 2, 4: 3, 5: 3}

    flights = _stitch_primary_flights(rows, accepted)

    assert flights[1] == flights[2]
    assert flights[3] != flights[2]


def test_track_prediction_uses_recent_velocity_and_bounded_acceleration() -> None:
    rows = [
        {"time_seconds": 0.0, "center_x": 10.0, "center_y": 30.0},
        {"time_seconds": 0.1, "center_x": 20.0, "center_y": 26.0},
        {"time_seconds": 0.2, "center_x": 30.0, "center_y": 24.0},
    ]

    x, y = _predict_track_point(rows, [0, 1, 2], 0.3)

    assert 39.0 <= x <= 41.0
    assert 23.0 <= y <= 25.0


def test_endpoint_validated_optical_flow_recovers_short_detection_gap(tmp_path: Path) -> None:
    video = tmp_path / "moving-dot.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (96, 64))
    assert writer.isOpened()
    for frame_index in range(10):
        frame = np.zeros((64, 96, 3), dtype=np.uint8)
        cv2.circle(frame, (15 + frame_index * 3, 32), 4, (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    detections = tmp_path / "detections.csv"
    output = tmp_path / "trajectory.csv"
    fields = [
        "time_seconds",
        "frame",
        "source_width",
        "source_height",
        "confidence",
        "center_x",
        "center_y",
        "width",
        "height",
    ]
    with detections.open("w", newline="", encoding="utf-8") as destination:
        csv_writer = csv.DictWriter(destination, fieldnames=fields)
        csv_writer.writeheader()
        for frame_index in (0, 3, 9):
            csv_writer.writerow(
                {
                    "time_seconds": frame_index / 30,
                    "frame": frame_index,
                    "source_width": 96,
                    "source_height": 64,
                    "confidence": 0.8,
                    "center_x": 15 + frame_index * 3,
                    "center_y": 32,
                    "width": 8,
                    "height": 8,
                }
            )

    result = analyze_shuttle_trajectory(detections, output, video=video)

    assert result["recovered_points"] > 0
    with output.open(newline="", encoding="utf-8") as source:
        statuses = [row["status"] for row in csv.DictReader(source)]
    assert "recovered" in statuses


def test_user_annotations_remove_false_detection_and_add_manual_point(tmp_path: Path) -> None:
    detections = tmp_path / "detections.csv"
    annotations = tmp_path / "annotations.csv"
    output = tmp_path / "trajectory.csv"
    fields = [
        "time_seconds",
        "frame",
        "source_width",
        "source_height",
        "confidence",
        "center_x",
        "center_y",
        "width",
        "height",
    ]
    with detections.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for frame, x in ((0, 100), (3, 130), (6, 160), (9, 190)):
            writer.writerow(
                {
                    "time_seconds": frame / 30,
                    "frame": frame,
                    "source_width": 640,
                    "source_height": 360,
                    "confidence": 0.7,
                    "center_x": x,
                    "center_y": 180,
                    "width": 8,
                    "height": 8,
                }
            )
    annotation_fields = [
        "id",
        "time_seconds",
        "frame",
        "source_width",
        "source_height",
        "x_normalized",
        "y_normalized",
        "action",
        "note",
    ]
    with annotations.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=annotation_fields)
        writer.writeheader()
        writer.writerow(
            {
                "id": "reject-1",
                "time_seconds": 0.1,
                "frame": 3,
                "source_width": 640,
                "source_height": 360,
                "x_normalized": 130 / 640,
                "y_normalized": 0.5,
                "action": "reject",
                "note": "neighbor court",
            }
        )
        writer.writerow(
            {
                "id": "add-1",
                "time_seconds": 0.15,
                "frame": 4,
                "source_width": 640,
                "source_height": 360,
                "x_normalized": 0.25,
                "y_normalized": 0.4,
                "action": "add",
                "note": "missed shuttle",
            }
        )

    result = analyze_shuttle_trajectory(detections, output, annotations_csv=annotations)

    assert result["user_rejected_points"] == 1
    assert result["manual_points"] == 1
    with output.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    assert sum(row["status"] == "manual" for row in rows) == 1
    assert all(not (row["frame"] == "3" and row["status"] != "manual") for row in rows)


def test_manual_annotations_form_local_flights_and_attach_to_nearby_model_flight() -> None:
    annotations = [
        {
            "id": "manual-1",
            "time_seconds": 1.2,
            "frame": 36,
            "source_width": 640,
            "source_height": 360,
            "x_normalized": 0.25,
            "y_normalized": 0.5,
            "action": "add",
            "note": "studio",
        },
        {
            "id": "manual-2",
            "time_seconds": 1.5,
            "frame": 45,
            "source_width": 640,
            "source_height": 360,
            "x_normalized": 0.31,
            "y_normalized": 0.48,
            "action": "add",
            "note": "studio",
        },
        {
            "id": "manual-3",
            "time_seconds": 4.0,
            "frame": 120,
            "source_width": 640,
            "source_height": 360,
            "x_normalized": 0.75,
            "y_normalized": 0.3,
            "action": "add",
            "note": "studio",
        },
    ]
    references = [
        {
            "time_seconds": 1.0,
            "frame": 30,
            "source_width": 640,
            "source_height": 360,
            "center_x": 130.0,
            "center_y": 180.0,
            "status": "tracked",
            "flight_id": "flight-7",
        }
    ]

    rows = _manual_annotation_rows(annotations, references)

    assert [row["flight_id"] for row in rows] == ["flight-7", "flight-7", "manual-2"]
    assert [row["track_length"] for row in rows] == [2, 2, 1]
    assert all(row["status"] == "manual" for row in rows)


def test_manual_anchor_beats_a_competing_false_contact_track() -> None:
    rows = []
    proposed = {}
    for frame, time_seconds in enumerate((0.0, 0.1, 0.2)):
        for track_id, x, confidence in ((1, 200.0, 0.45), (2, 500.0, 0.92)):
            index = len(rows)
            rows.append(
                {
                    "time_seconds": time_seconds,
                    "frame": frame,
                    "source_width": 640,
                    "source_height": 360,
                    "confidence": confidence,
                    "center_x": x + frame * 12.0,
                    "center_y": 180.0 - frame * 4.0,
                    "width": 8.0,
                    "height": 8.0,
                    "detection_status": "detected",
                    "evidence_weight": 1.0,
                }
            )
            proposed[index] = track_id
    hints = [
        {"time": 0.1, "x": 212 / 640, "y": 176 / 360, "support": 1.0, "radius": 0.04, "kind": "manual_anchor"},
        {"time": 0.1, "x": 512 / 640, "y": 176 / 360, "support": 1.0, "radius": 0.04, "kind": "contact"},
    ]

    accepted = _select_primary_track_indices(rows, proposed, hints)

    assert 2 in accepted
    assert 3 not in accepted


def test_weak_two_point_prefix_of_rejected_neighbor_track_is_dropped() -> None:
    rows = [
        {
            "time_seconds": index * 0.02,
            "ownership_evidence": "court_continuity" if index < 2 else "unknown",
        }
        for index in range(12)
    ]
    proposed = {index: 1 for index in range(12)}

    assert _drop_weak_partial_tracks(rows, proposed, {0, 1}) == set()


def test_short_complete_track_is_not_dropped() -> None:
    rows = [
        {"time_seconds": 0.0, "ownership_evidence": "court_continuity"},
        {"time_seconds": 0.05, "ownership_evidence": "court_continuity"},
    ]

    assert _drop_weak_partial_tracks(rows, {0: 1, 1: 1}, {0, 1}) == {0, 1}


def test_inpainted_candidates_cannot_create_a_track() -> None:
    rows = [
        {
            "time_seconds": 1.0,
            "frame": 30,
            "center_x": 400.0,
            "center_y": 300.0,
            "confidence": 0.28,
            "detection_status": "inpainted",
            "evidence_weight": 0.0,
        },
        {
            "time_seconds": 1.1,
            "frame": 33,
            "center_x": 460.0,
            "center_y": 280.0,
            "confidence": 0.28,
            "detection_status": "inpainted",
            "evidence_weight": 0.0,
        },
        {
            "time_seconds": 1.2,
            "frame": 36,
            "center_x": 520.0,
            "center_y": 260.0,
            "confidence": 0.28,
            "detection_status": "inpainted",
            "evidence_weight": 0.0,
        },
    ]

    assert _link_dynamic_rows(rows, list(range(len(rows)))) == {}


def test_active_court_contact_beats_long_neighbour_track(tmp_path: Path) -> None:
    config = tmp_path / "court.json"
    config.write_text(
        json.dumps(
            {
                "reference_frame": {"width": 1920, "height": 1080},
                "calibration": {"court_corners_normalized": None, "shuttle_perspective_axis_normalized": []},
                "masks_normalized": {
                    "active_court_polygon": [[0.20, 0.58], [0.82, 0.58], [0.88, 1.0], [0.25, 1.0]],
                    "court_ground_polygon": [[0.20, 0.58], [0.82, 0.58], [0.88, 1.0], [0.25, 1.0]],
                    "near_player_zone": [[0.25, 0.70], [0.75, 0.70], [0.82, 1.0], [0.28, 1.0]],
                    "far_player_zone": [[0.25, 0.55], [0.75, 0.55], [0.70, 0.70], [0.30, 0.70]],
                    "net_band": [[0.25, 0.60], [0.75, 0.60], [0.75, 0.66], [0.25, 0.66]],
                    "shuttle_airspace_polygon": [[0.10, 0.05], [0.90, 0.05], [0.95, 1.0], [0.05, 1.0]],
                    "static_false_positive_polygons": [],
                    "background_court_polygons": [[[0.73, 0.35], [0.95, 0.35], [0.95, 0.70], [0.73, 0.70]]],
                },
            }
        ),
        encoding="utf-8",
    )
    rows = []
    proposed = {}
    for track_id, points in (
        (1, [(1.00, 760, 460), (1.10, 775, 430), (1.20, 790, 410), (1.30, 805, 430)]),
        (2, [(0.95, 1430, 470), (1.05, 1440, 480), (1.15, 1450, 490), (1.25, 1460, 500), (1.35, 1450, 490), (1.45, 1430, 480), (1.55, 1400, 470), (1.65, 1370, 460)]),
    ):
        for time_seconds, x, y in points:
            index = len(rows)
            rows.append(
                {
                    "time_seconds": time_seconds,
                    "frame": round(time_seconds * 30),
                    "source_width": 1920,
                    "source_height": 1080,
                    "center_x": x,
                    "center_y": y,
                    "confidence": 0.30,
                    "evidence_weight": 0.65,
                    "width": 9.0,
                    "height": 10.0,
                    "detection_status": "detected",
                }
            )
            proposed[index] = track_id

    from smart_badminton.geometry import CourtGeometry

    geometry = CourtGeometry.from_json(config)
    accepted = _select_primary_track_indices(
        rows,
        proposed,
        [{"time": 1.12, "x": 0.40, "y": 0.42, "support": 0.90, "radius": 0.10, "side": "near"}],
        geometry,
    )
    selected = {proposed[index] for index in accepted}

    assert selected == {1}


def test_high_confidence_background_track_needs_ownership_credential(tmp_path: Path) -> None:
    geometry = CourtGeometry(
        reference_width=1920,
        reference_height=1080,
        court_ground_polygon=[(0.2, 0.58), (0.82, 0.58), (0.88, 1.0), (0.25, 1.0)],
        near_player_zone=[(0.25, 0.70), (0.75, 0.70), (0.82, 1.0), (0.28, 1.0)],
        far_player_zone=[(0.25, 0.55), (0.75, 0.55), (0.70, 0.70), (0.30, 0.70)],
        net_band=[(0.25, 0.58), (0.70, 0.58), (0.70, 0.66), (0.25, 0.66)],
        shuttle_airspace_polygon=[(0.1, 0.05), (0.9, 0.05), (0.95, 1.0), (0.05, 1.0)],
        static_false_positive_polygons=[],
        background_court_polygons=[[(0.73, 0.30), (1.0, 0.30), (1.0, 0.75), (0.73, 0.62)]],
        active_court_polygon=[(0.2, 0.58), (0.82, 0.58), (0.88, 1.0), (0.25, 1.0)],
        shuttle_perspective_axis=[(0.46, 0.05), (0.50, 0.90)],
    )
    rows = []
    proposed = {}
    for time_seconds, x, y in [(1.0, 1500, 450), (1.1, 1520, 470), (1.2, 1540, 500), (1.3, 1560, 530)]:
        index = len(rows)
        rows.append(
            {
                "time_seconds": time_seconds,
                "frame": round(time_seconds * 60),
                "source_width": 1920,
                "source_height": 1080,
                "center_x": x,
                "center_y": y,
                "confidence": 0.95,
                "evidence_weight": 1.0,
                "detection_status": "detected",
            }
        )
        proposed[index] = 1

    assert _select_primary_track_indices(rows, proposed, geometry=geometry) == set()


def test_contact_credential_keeps_high_clear_through_background_region(tmp_path: Path) -> None:
    geometry = CourtGeometry(
        reference_width=1920,
        reference_height=1080,
        court_ground_polygon=[(0.2, 0.58), (0.82, 0.58), (0.88, 1.0), (0.25, 1.0)],
        near_player_zone=[(0.25, 0.70), (0.75, 0.70), (0.82, 1.0), (0.28, 1.0)],
        far_player_zone=[(0.25, 0.55), (0.75, 0.55), (0.70, 0.70), (0.30, 0.70)],
        net_band=[(0.25, 0.58), (0.70, 0.58), (0.70, 0.66), (0.25, 0.66)],
        shuttle_airspace_polygon=[(0.1, 0.05), (0.9, 0.05), (0.95, 1.0), (0.05, 1.0)],
        static_false_positive_polygons=[],
        background_court_polygons=[[(0.73, 0.30), (1.0, 0.30), (1.0, 0.75), (0.73, 0.62)]],
        active_court_polygon=[(0.2, 0.58), (0.82, 0.58), (0.88, 1.0), (0.25, 1.0)],
        shuttle_perspective_axis=[(0.46, 0.05), (0.50, 0.90)],
    )
    rows = []
    proposed = {}
    points = [(1.0, 960, 760), (1.1, 1120, 650), (1.2, 1320, 560), (1.3, 1480, 480)]
    for time_seconds, x, y in points:
        index = len(rows)
        rows.append(
            {
                "time_seconds": time_seconds,
                "frame": round(time_seconds * 60),
                "source_width": 1920,
                "source_height": 1080,
                "center_x": x,
                "center_y": y,
                "confidence": 0.65,
                "evidence_weight": 1.0,
                "detection_status": "detected",
            }
        )
        proposed[index] = 1
    hints = [{"time": 1.0, "x": 0.50, "y": 0.70, "support": 0.9, "radius": 0.12, "kind": "contact"}]

    accepted = _select_primary_track_indices(rows, proposed, hints, geometry)

    assert accepted == set(range(len(rows)))


def test_court_axis_is_a_soft_tiebreaker_for_simultaneous_tracks() -> None:
    geometry = CourtGeometry(
        reference_width=1920,
        reference_height=1080,
        court_ground_polygon=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        near_player_zone=[],
        far_player_zone=[],
        net_band=[],
        shuttle_airspace_polygon=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        static_false_positive_polygons=[],
        background_court_polygons=[],
        active_court_polygon=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        shuttle_perspective_axis=[(0.50, 0.0), (0.50, 1.0)],
    )
    rows = []
    proposed = {}
    for track_id, x in ((1, 960), (2, 1320)):
        for time_seconds, y in ((1.0, 420), (1.1, 460), (1.2, 510)):
            index = len(rows)
            rows.append(
                {
                    "time_seconds": time_seconds,
                    "frame": round(time_seconds * 60),
                    "source_width": 1920,
                    "source_height": 1080,
                    "center_x": x,
                    "center_y": y,
                    "confidence": 0.70,
                    "evidence_weight": 1.0,
                    "detection_status": "detected",
                }
            )
            proposed[index] = track_id

    accepted = _select_primary_track_indices(rows, proposed, geometry=geometry)

    assert {proposed[index] for index in accepted} == {1}


def test_target_net_crossing_credentials_background_extension() -> None:
    geometry = CourtGeometry(
        reference_width=1920,
        reference_height=1080,
        court_ground_polygon=[(0.2, 0.58), (0.82, 0.58), (0.88, 1.0), (0.25, 1.0)],
        near_player_zone=[],
        far_player_zone=[],
        net_band=[(0.25, 0.54), (0.70, 0.54), (0.70, 0.66), (0.25, 0.66)],
        shuttle_airspace_polygon=[(0.1, 0.05), (0.9, 0.05), (0.95, 1.0), (0.05, 1.0)],
        static_false_positive_polygons=[],
        background_court_polygons=[[(0.73, 0.30), (1.0, 0.30), (1.0, 0.75), (0.73, 0.62)]],
        active_court_polygon=[(0.2, 0.58), (0.82, 0.58), (0.88, 1.0), (0.25, 1.0)],
        shuttle_perspective_axis=[(0.46, 0.05), (0.50, 0.90)],
    )
    rows = []
    proposed = {}
    for time_seconds, x, y in [(1.0, 950, 760), (1.1, 1050, 650), (1.2, 1160, 590), (1.3, 1480, 500)]:
        index = len(rows)
        rows.append(
            {
                "time_seconds": time_seconds,
                "frame": round(time_seconds * 60),
                "source_width": 1920,
                "source_height": 1080,
                "center_x": x,
                "center_y": y,
                "confidence": 0.65,
                "evidence_weight": 1.0,
                "detection_status": "detected",
            }
        )
        proposed[index] = 1

    accepted = _select_primary_track_indices(rows, proposed, geometry=geometry)

    assert accepted == set(range(len(rows)))


def test_interval_selection_keeps_non_overlapping_tracklets_after_overlap_chain() -> None:
    rows = []
    proposed = {}
    for track_id, points in (
        (1, [(0.0, 100, 100), (0.5, 140, 120), (1.0, 180, 140)]),
        (2, [(0.9, 500, 300), (1.4, 540, 320), (2.0, 580, 340)]),
        (3, [(1.9, 200, 150), (2.4, 240, 170), (3.0, 280, 190)]),
    ):
        for time_seconds, x, y in points:
            index = len(rows)
            rows.append(
                {
                    "time_seconds": time_seconds,
                    "frame": round(time_seconds * 30),
                    "center_x": x,
                    "center_y": y,
                    "confidence": 0.70 if track_id != 2 else 0.20,
                    "evidence_weight": 1.0,
                }
            )
            proposed[index] = track_id

    accepted = _select_primary_track_indices(rows, proposed)

    assert {proposed[index] for index in accepted} == {1, 2, 3}


def test_local_ownership_trims_landed_tail_when_new_flight_starts() -> None:
    rows = []
    proposed = {}
    for track_id, confidence, points in (
        (1, 0.72, [(1.00, 700, 500), (1.10, 740, 460), (1.20, 780, 420), (1.30, 820, 390), (1.40, 860, 380)]),
        (2, 0.22, [(1.20, 1430, 480), (1.30, 1450, 490), (1.40, 1470, 500)]),
        (3, 0.82, [(1.40, 860, 380), (1.50, 820, 420), (1.60, 780, 460), (1.70, 740, 500)]),
    ):
        for time_seconds, x, y in points:
            index = len(rows)
            rows.append(
                {
                    "time_seconds": time_seconds,
                    "frame": round(time_seconds * 60),
                    "source_width": 1920,
                    "source_height": 1080,
                    "center_x": x,
                    "center_y": y,
                    "confidence": confidence,
                    "evidence_weight": 0.9,
                    "width": 8.0,
                    "height": 8.0,
                    "detection_status": "detected",
                }
            )
            proposed[index] = track_id

    accepted = _select_primary_track_indices(rows, proposed)

    assert {proposed[index] for index in accepted} == {1, 3}
    assert any(proposed[index] == 1 and rows[index]["time_seconds"] < 1.4 for index in accepted)
    assert any(proposed[index] == 3 and rows[index]["time_seconds"] >= 1.4 for index in accepted)
    assert all(proposed[index] != 2 for index in accepted)


def test_match5_active_point_wins_local_neighbour_competition() -> None:
    rows = []
    proposed = {}
    for track_id, points in (
        (
            15,
            [
                (22.183333, 758.0, 471.0, 0.756),
                (22.200000, 764.642, 458.888, 0.738),
                (22.216667, 774.0, 443.0, 0.706),
                (22.300000, 800.0, 398.0, 0.718),
            ],
        ),
        (
            16,
            [
                (22.200000, 1433.425, 479.110, 0.146),
                (22.216667, 1443.0, 491.0, 0.303),
                (22.266667, 1463.0, 519.0, 0.280),
            ],
        ),
    ):
        for time_seconds, x, y, confidence in points:
            index = len(rows)
            rows.append(
                {
                    "time_seconds": time_seconds,
                    "frame": round(time_seconds * 60),
                    "source_width": 1920,
                    "source_height": 1080,
                    "confidence": confidence,
                    "center_x": x,
                    "center_y": y,
                    "width": 8.0,
                    "height": 8.0,
                    "detection_status": "detected",
                    "evidence_weight": 1.0,
                }
            )
            proposed[index] = track_id

    accepted = _select_primary_track_indices(rows, proposed)

    assert 1 in accepted
    assert proposed[1] == 15
    assert proposed[4] == 16
    assert 4 not in accepted
