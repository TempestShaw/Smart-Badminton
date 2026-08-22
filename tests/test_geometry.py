import pytest

from smart_badminton.geometry import CourtGeometry


def _calibrated_geometry() -> CourtGeometry:
    """Return deterministic geometry; live Studio calibration is user-owned data."""
    return CourtGeometry(
        reference_width=1280,
        reference_height=720,
        court_ground_polygon=[
            (0.1921, 0.5968), (0.4162, 0.5655), (0.8996, 0.7467), (0.8352, 0.9982), (0.2758, 1.0)
        ],
        active_court_polygon=[
            (0.1921, 0.5968), (0.4162, 0.5655), (0.8996, 0.7467), (0.8352, 0.9982), (0.2758, 1.0)
        ],
        near_player_zone=[(0.2768, 0.9982), (0.2144, 0.6853), (0.5440, 0.6284), (0.8992, 0.7649), (0.8464, 0.9982)],
        far_player_zone=[(0.2080, 0.6881), (0.1952, 0.5942), (0.4240, 0.5857), (0.5344, 0.6284), (0.3120, 0.6682)],
        net_band=[(0.20, 0.58), (0.90, 0.72), (0.90, 0.77), (0.20, 0.63)],
        shuttle_airspace_polygon=[(0.18, 0.16), (0.80, 0.16), (0.91, 0.78), (0.85, 1.0), (0.27, 1.0)],
        static_false_positive_polygons=[],
        background_court_polygons=[],
        shuttle_perspective_axis=[(0.4176, 0.0196), (0.4096, 0.6540)],
    )


def test_user_calibrated_active_court_keeps_player_and_rejects_spectator() -> None:
    geometry = _calibrated_geometry()

    far_player_foot = (380.76 / 1280.0, 427.35 / 720.0)
    seated_spectator_foot = (835.85 / 1280.0, 453.10 / 720.0)
    near_player_foot = (700.0 / 1280.0, 600.0 / 720.0)

    assert geometry.contains_active_court(*far_player_foot, margin=0.015)
    assert geometry.contains_active_court(*near_player_foot, margin=0.015)
    assert not geometry.contains_active_court(*seated_spectator_foot, margin=0.015)


def test_player_inference_crop_includes_upper_body_above_foot_zone() -> None:
    geometry = _calibrated_geometry()

    near_x1, near_y1, near_x2, near_y2 = geometry.player_inference_bounds("near", 1280, 720)
    far_x1, far_y1, far_x2, far_y2 = geometry.player_inference_bounds("far", 1280, 720)
    near_zone_y = geometry.polygon("near_player_zone", 1280, 720)[:, 1].min()
    far_zone_y = geometry.polygon("far_player_zone", 1280, 720)[:, 1].min()

    assert near_y1 < near_zone_y - 200
    assert far_y1 < far_zone_y
    assert near_x1 < near_x2 and near_y1 < near_y2
    assert far_x1 < far_x2 and far_y1 < far_y2


def test_player_inference_crop_rejects_unknown_side() -> None:
    geometry = _calibrated_geometry()

    with pytest.raises(ValueError, match="near or far"):
        geometry.player_inference_bounds("middle", 1280, 720)


def test_perspective_shuttle_volume_keeps_high_clear_and_rejects_neighbor_court() -> None:
    geometry = _calibrated_geometry()

    assert not geometry.contains_normalized("shuttle_airspace_polygon", 0.50, 0.05)
    assert geometry.contains_shuttle_volume(0.50, 0.05)
    assert not geometry.contains_shuttle_volume(0.05, 0.20)
    assert not geometry.contains_shuttle_volume(0.95, 0.20)
    assert geometry.shuttle_vanishing_point() is not None


def test_homography_reports_click_sensitivity_instead_of_zero_reprojection_error() -> None:
    geometry = CourtGeometry(
        reference_width=1280,
        reference_height=720,
        court_ground_polygon=[(0.1, 0.95), (0.9, 0.95), (0.65, 0.35), (0.35, 0.35)],
        near_player_zone=[],
        far_player_zone=[],
        net_band=[],
        shuttle_airspace_polygon=[],
        static_false_positive_polygons=[],
        background_court_polygons=[],
        active_court_polygon=[(0.1, 0.95), (0.9, 0.95), (0.65, 0.35), (0.35, 0.35)],
        court_corners=[(0.1, 0.95), (0.9, 0.95), (0.65, 0.35), (0.35, 0.35)],
    )

    quality = geometry.homography_quality(1280, 720)

    assert quality["available"] is True
    assert 0 < quality["median_uncertainty_meters"] < quality["p95_uncertainty_meters"]
    assert 0 <= quality["confidence"] <= 1
