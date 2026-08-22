from smart_badminton.geometry import CourtGeometry
from smart_badminton.trajectory_metrics import TrajectoryPoint, summarize_trajectory


def test_trajectory_metrics_count_stable_net_zone_transit() -> None:
    geometry = CourtGeometry(
        reference_width=1000,
        reference_height=1000,
        court_ground_polygon=[(0.1, 0.9), (0.9, 0.9), (0.7, 0.2), (0.3, 0.2)],
        near_player_zone=[],
        far_player_zone=[],
        net_band=[(0.2, 0.55), (0.2, 0.45), (0.8, 0.45), (0.8, 0.55)],
        shuttle_airspace_polygon=[(0, 0), (1, 0), (1, 1), (0, 1)],
        static_false_positive_polygons=[],
        background_court_polygons=[],
    )
    points = [
        TrajectoryPoint(0.00, 0.40, 0.42, "1", 1.0),
        TrajectoryPoint(0.05, 0.45, 0.46, "1", 1.0),
        TrajectoryPoint(0.10, 0.50, 0.50, "1", 1.0),
        TrajectoryPoint(0.15, 0.55, 0.54, "1", 1.0),
        TrajectoryPoint(0.20, 0.60, 0.58, "1", 1.0),
    ]

    metrics = summarize_trajectory(points, 0.0, 1.0, geometry)

    assert metrics["trajectory_available"] is True
    assert metrics["trajectory_coverage_percent"] == 20.0
    assert metrics["net_zone_transits"] == 1
