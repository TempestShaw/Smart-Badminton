import csv
from pathlib import Path

from smart_badminton.features import ShuttleDetectionIndex
from smart_badminton.geometry import CourtGeometry


def test_shuttle_index_uses_detection_coordinate_dimensions(tmp_path: Path) -> None:
    detections = tmp_path / "trajectory.csv"
    fields = [
        "time_seconds",
        "frame",
        "source_width",
        "source_height",
        "confidence",
        "center_x",
        "center_y",
        "status",
    ]
    with detections.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "time_seconds": 1.0,
                "frame": 30,
                "source_width": 1280,
                "source_height": 720,
                "confidence": 0.8,
                "center_x": 640,
                "center_y": 360,
                "status": "tracked",
            }
        )
    geometry = CourtGeometry(
        reference_width=1280,
        reference_height=720,
        court_ground_polygon=[(0, 0), (1, 0), (1, 1), (0, 1)],
        near_player_zone=[(0, 0), (1, 0), (1, 1), (0, 1)],
        far_player_zone=[(0, 0), (1, 0), (1, 1), (0, 1)],
        net_band=[(0, 0), (1, 0), (1, 1), (0, 1)],
        shuttle_airspace_polygon=[(0, 0), (1, 0), (1, 1), (0, 1)],
        static_false_positive_polygons=[],
        background_court_polygons=[],
    )

    index = ShuttleDetectionIndex(detections, geometry, width=1920, height=1080)
    values, _point = index.sample(1.0, 0.05, None, None)

    assert values["shuttle_visible"] == 1.0
    assert values["shuttle_x_normalized"] == 0.5
    assert values["shuttle_y_normalized"] == 0.5
