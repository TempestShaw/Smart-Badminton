import pandas as pd

from smart_badminton.serve_inference import infer_serve_observations


def _features(near_x: float, far_x: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_seconds": [0.0, 0.1, 0.2],
            "near_active_wrist_visible": [1.0, 1.0, 1.0],
            "near_active_wrist_x_normalized": [near_x] * 3,
            "near_active_wrist_y_normalized": [0.5] * 3,
            "far_active_wrist_visible": [1.0, 1.0, 1.0],
            "far_active_wrist_x_normalized": [far_x] * 3,
            "far_active_wrist_y_normalized": [0.5] * 3,
        }
    )


def _trajectory(x: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_seconds": [0.1, 0.2],
            "flight_id": [1, 1],
            "source_width": [1000, 1000],
            "source_height": [500, 500],
            "center_x": [x * 1000, (x + 0.01) * 1000],
            "center_y": [250, 245],
            "ownership_evidence": ["player_contact", "player_contact"],
            "ownership_confidence": [0.9, 0.9],
        }
    )


def test_owned_trajectory_near_far_wrist_identifies_server() -> None:
    observations = infer_serve_observations(
        [(0.0, 1.0)],
        _features(0.8, 0.2),
        _trajectory(0.2),
    )

    assert observations == [
        {
            "rally": 1,
            "time": 0.1,
            "server": "far",
            "confidence": 0.93,
            "source": "owned-trajectory-wrist",
        }
    ]


def test_ambiguous_wrist_proximity_stays_unknown() -> None:
    observations = infer_serve_observations(
        [(0.0, 1.0)],
        _features(0.48, 0.52),
        _trajectory(0.5),
    )

    assert observations == []
