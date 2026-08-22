from pathlib import Path

import numpy as np
import pandas as pd

from smart_badminton.rally_evidence import build_rally_evidence
from smart_badminton.segmenter import segment_rallies


def test_padding_overlap_is_deduplicated(tmp_path: Path) -> None:
    times = np.round(np.arange(0.0, 7.0, 0.1), 3)
    frame = pd.DataFrame(
        {
            "time_seconds": times,
            "frame_index": np.arange(len(times)),
            "near_swing_score": 0.0,
            "far_swing_score": 0.0,
            "near_lunge_score": 0.0,
            "far_lunge_score": 0.0,
            "near_flow_mean": 0.0,
            "far_flow_mean": 0.0,
            "near_motion_fraction": 0.0,
            "far_motion_fraction": 0.0,
            "audio_hit_score": 0.0,
            "shuttle_visible": 0.0,
            "shuttle_speed_normalized": 0.0,
        }
    )
    probability = np.where(((times >= 1.0) & (times <= 2.3)) | ((times >= 3.5) & (times <= 4.8)), 0.95, 0.01)
    features_path = tmp_path / "features.csv"
    probabilities_path = tmp_path / "probabilities.csv"
    output_path = tmp_path / "rallies.csv"
    frame.to_csv(features_path, index=False)
    pd.DataFrame({"time_seconds": times, "frame_index": frame.frame_index, "rally_probability": probability}).to_csv(
        probabilities_path, index=False
    )
    result = segment_rallies(features_path, probabilities_path, output_path)
    assert len(result) == 2
    assert result[0].end <= result[1].start


def test_learned_gap_prior_blocks_an_immediate_unsupported_restart(tmp_path: Path) -> None:
    times = np.round(np.arange(0.0, 8.0, 0.1), 3)
    active = ((times >= 1.0) & (times <= 2.5)) | ((times >= 3.4) & (times <= 4.9))
    frame = pd.DataFrame(
        {
            "time_seconds": times,
            "frame_index": np.arange(len(times)),
            "near_swing_score": 0.0,
            "far_swing_score": 0.0,
            "near_lunge_score": 0.0,
            "far_lunge_score": 0.0,
            "near_flow_mean": 0.0,
            "far_flow_mean": 0.0,
            "near_motion_fraction": 0.0,
            "far_motion_fraction": 0.0,
            "audio_hit_score": 0.0,
            "shuttle_visible": 0.0,
            "shuttle_speed_normalized": 0.0,
        }
    )
    features_path = tmp_path / "features.csv"
    probabilities_path = tmp_path / "probabilities.csv"
    output_path = tmp_path / "rallies.csv"
    frame.to_csv(features_path, index=False)
    pd.DataFrame(
        {
            "time_seconds": times,
            "frame_index": frame.frame_index,
            "rally_probability": np.where(active, 0.95, 0.01),
            "gap_hard_min_seconds": 2.0,
            "gap_soft_min_seconds": 4.0,
        }
    ).to_csv(probabilities_path, index=False)

    result = segment_rallies(features_path, probabilities_path, output_path)

    assert len(result) == 1


def test_precision_mode_suppresses_weak_handoff_between_real_exchanges(tmp_path: Path) -> None:
    times = np.round(np.arange(0.0, 14.0, 0.1), 3)
    true_play = ((times >= 1.0) & (times <= 3.0)) | ((times >= 7.0) & (times <= 10.0))
    handoff = (times >= 5.0) & (times <= 6.0)
    frame = pd.DataFrame(
        {
            "time_seconds": times,
            "frame_index": np.arange(len(times)),
            "near_swing_score": np.where(true_play, 0.8, np.where(handoff, 0.3, 0.0)),
            "far_swing_score": 0.0,
            "near_lunge_score": 0.0,
            "far_lunge_score": 0.0,
            "near_flow_mean": 0.0,
            "far_flow_mean": 0.0,
            "near_motion_fraction": 0.0,
            "far_motion_fraction": 0.0,
            "audio_hit_score": np.where(true_play & ((times * 10).astype(int) % 10 == 0), 0.8, 0.0),
            "shuttle_visible": 0.0,
            "shuttle_speed_normalized": 0.0,
        }
    )
    probability = np.where(true_play, 0.98, np.where(handoff, 0.70, 0.01))
    features_path = tmp_path / "features.csv"
    probabilities_path = tmp_path / "probabilities.csv"
    output_path = tmp_path / "rallies.csv"
    frame.to_csv(features_path, index=False)
    pd.DataFrame({"time_seconds": times, "frame_index": frame.frame_index, "rally_probability": probability}).to_csv(
        probabilities_path, index=False
    )

    result = segment_rallies(features_path, probabilities_path, output_path)

    assert len(result) == 2
    assert result[0].end < 5.0
    assert result[1].start > 6.0


def test_descending_shuttle_protects_rally_end_until_landing(tmp_path: Path) -> None:
    times = np.round(np.arange(0.0, 5.0, 0.1), 3)
    active = (times >= 1.0) & (times <= 2.0)
    frame = pd.DataFrame(
        {
            "time_seconds": times,
            "frame_index": np.arange(len(times)),
            "near_swing_score": np.where((times >= 1.7) & (times <= 1.9), 0.8, 0.0),
            "far_swing_score": 0.0,
            "near_lunge_score": 0.0,
            "far_lunge_score": 0.0,
            "near_flow_mean": np.where(active, 2.0, 0.0),
            "far_flow_mean": np.where(active, 1.0, 0.0),
            "near_motion_fraction": 0.0,
            "far_motion_fraction": 0.0,
            "audio_hit_score": 0.0,
            "shuttle_visible": 0.0,
            "shuttle_speed_normalized": 0.0,
        }
    )
    probability = np.where(active, 0.95, 0.01)
    features_path = tmp_path / "features.csv"
    probabilities_path = tmp_path / "probabilities.csv"
    trajectory_path = tmp_path / "trajectory.csv"
    output_path = tmp_path / "rallies.csv"
    frame.to_csv(features_path, index=False)
    pd.DataFrame({"time_seconds": times, "frame_index": frame.frame_index, "rally_probability": probability}).to_csv(
        probabilities_path, index=False
    )
    pd.DataFrame(
        {
            "time_seconds": [1.8, 2.1, 2.4, 2.7],
            "center_y": [100.0, 120.0, 155.0, 205.0],
            "status": ["tracked"] * 4,
            "track_id": [1] * 4,
        }
    ).to_csv(trajectory_path, index=False)

    result = segment_rallies(
        features_path,
        probabilities_path,
        output_path,
        shuttle_trajectory_csv=trajectory_path,
    )

    assert len(result) == 1
    assert result[0].end >= 3.0
    assert "trajectory" in result[0].reason


def test_handoff_opening_is_trimmed_to_ready_serve(tmp_path: Path) -> None:
    times = np.round(np.arange(0.0, 10.0, 0.1), 3)
    handoff = (times >= 1.0) & (times <= 1.1)
    serve = (times >= 4.0) & (times <= 4.1)
    foot_speed = np.where(times < 1.5, 0.2, 0.0)
    frame = pd.DataFrame(
        {
            "time_seconds": times,
            "frame_index": np.arange(len(times)),
            "near_swing_score": np.where(handoff, 0.3, np.where(serve, 0.6, 0.0)),
            "far_swing_score": 0.0,
            "near_lunge_score": 0.0,
            "far_lunge_score": 0.0,
            "near_flow_mean": np.where(handoff | serve, 1.5, 0.0),
            "far_flow_mean": 0.0,
            "near_motion_fraction": 0.0,
            "far_motion_fraction": 0.0,
            "audio_hit_score": np.where(serve, 0.7, 0.0),
            "shuttle_visible": 0.0,
            "shuttle_speed_normalized": 0.0,
            "near_person_visible": 1.0,
            "far_person_visible": 1.0,
            "near_foot_speed_normalized": foot_speed,
            "far_foot_speed_normalized": foot_speed,
            "near_stance_width_normalized": 0.04,
            "far_stance_width_normalized": 0.02,
        }
    )
    probability = np.where((times >= 0.5) & (times <= 8.5), 0.92, 0.01)
    features_path = tmp_path / "features.csv"
    probabilities_path = tmp_path / "probabilities.csv"
    output_path = tmp_path / "rallies.csv"
    frame.to_csv(features_path, index=False)
    pd.DataFrame({"time_seconds": times, "frame_index": frame.frame_index, "rally_probability": probability}).to_csv(
        probabilities_path, index=False
    )

    result = segment_rallies(features_path, probabilities_path, output_path)

    assert len(result) == 1
    assert result[0].start >= 3.5
    assert "handoff opening trimmed" in result[0].reason


def test_formal_serve_requires_the_inferred_receiver_to_be_ready() -> None:
    times = np.round(np.arange(0.0, 6.0, 0.1), 3)
    early_serve = (times >= 2.0) & (times <= 2.1)
    formal_serve = (times >= 4.0) & (times <= 4.1)
    far_moving = times < 2.7
    frame = pd.DataFrame(
        {
            "time_seconds": times,
            "rally_probability": 0.1,
            "near_swing_score": np.where(early_serve | formal_serve, 0.65, 0.0),
            "far_swing_score": 0.0,
            "near_lunge_score": 0.0,
            "far_lunge_score": 0.0,
            "near_flow_mean": np.where(early_serve | formal_serve, 1.0, 0.0),
            "far_flow_mean": np.where(far_moving, 2.0, 0.0),
            "near_motion_fraction": 0.0,
            "far_motion_fraction": 0.0,
            "audio_hit_score": np.where(early_serve | formal_serve, 0.7, 0.0),
            "near_person_visible": 1.0,
            "far_person_visible": 1.0,
            "near_foot_speed_normalized": 0.0,
            "far_foot_speed_normalized": np.where(far_moving, 0.25, 0.0),
            "near_stance_width_normalized": 0.04,
            "far_stance_width_normalized": 0.02,
        }
    )

    evidence = build_rally_evidence(frame)
    serve_times = times[evidence.formal_serve]

    assert not np.any((serve_times >= 1.9) & (serve_times <= 2.2))
    assert np.any((serve_times >= 3.9) & (serve_times <= 4.2))
    assert np.any(evidence.near_serving)
    assert not np.any(evidence.far_serving)


def test_qualified_flight_after_ready_stance_recovers_missed_serve(tmp_path: Path) -> None:
    times = np.round(np.arange(0.0, 5.0, 0.1), 3)
    serve = (times >= 2.0) & (times <= 2.2)
    frame = pd.DataFrame(
        {
            "time_seconds": times,
            "frame_index": np.arange(len(times)),
            "near_swing_score": np.where(serve, 0.16, 0.0),
            "far_swing_score": 0.0,
            "near_lunge_score": 0.0,
            "far_lunge_score": 0.0,
            "near_flow_mean": 0.0,
            "far_flow_mean": 0.0,
            "near_motion_fraction": 0.0,
            "far_motion_fraction": 0.0,
            "audio_hit_score": 0.0,
            "shuttle_visible": 0.0,
            "shuttle_speed_normalized": 0.0,
            "near_person_visible": 1.0,
            "far_person_visible": 1.0,
            "near_foot_speed_normalized": 0.0,
            "far_foot_speed_normalized": 0.0,
            "near_stance_width_normalized": 0.04,
            "far_stance_width_normalized": 0.02,
        }
    )
    features = tmp_path / "features.csv"
    probabilities = tmp_path / "probabilities.csv"
    trajectory = tmp_path / "trajectory.csv"
    output = tmp_path / "rallies.csv"
    frame.to_csv(features, index=False)
    pd.DataFrame({"time_seconds": times, "rally_probability": 0.05}).to_csv(probabilities, index=False)
    pd.DataFrame(
        {
            "time_seconds": np.arange(2.1, 3.01, 0.1),
            "center_x": np.linspace(400, 900, 10),
            "center_y": np.linspace(550, 250, 10),
            "source_width": 1920,
            "source_height": 1080,
            "status": "tracked",
            "track_id": 1,
            "flight_id": 1,
        }
    ).to_csv(trajectory, index=False)

    result = segment_rallies(features, probabilities, output, shuttle_trajectory_csv=trajectory)

    assert len(result) == 1
    assert result[0].start <= 2.1
    assert result[0].end >= 3.3


def test_trajectory_serve_keeps_a_short_rally_through_landing(tmp_path: Path) -> None:
    times = np.round(np.arange(0.0, 3.0, 0.1), 3)
    serve = (times >= 1.0) & (times <= 1.1)
    frame = pd.DataFrame(
        {
            "time_seconds": times,
            "frame_index": np.arange(len(times)),
            "near_swing_score": np.where(serve, 0.18, 0.0),
            "far_swing_score": 0.0,
            "near_lunge_score": 0.0,
            "far_lunge_score": 0.0,
            "near_flow_mean": 0.0,
            "far_flow_mean": 0.0,
            "near_motion_fraction": 0.0,
            "far_motion_fraction": 0.0,
            "audio_hit_score": 0.0,
            "shuttle_visible": 0.0,
            "shuttle_speed_normalized": 0.0,
            "near_person_visible": 1.0,
            "far_person_visible": 1.0,
            "near_foot_speed_normalized": 0.0,
            "far_foot_speed_normalized": 0.0,
            "near_stance_width_normalized": 0.04,
            "far_stance_width_normalized": 0.02,
        }
    )
    features = tmp_path / "features.csv"
    probabilities = tmp_path / "probabilities.csv"
    trajectory = tmp_path / "trajectory.csv"
    output = tmp_path / "rallies.csv"
    frame.to_csv(features, index=False)
    pd.DataFrame({"time_seconds": times, "rally_probability": 0.05}).to_csv(probabilities, index=False)
    pd.DataFrame(
        {
            "time_seconds": [1.0, 1.1, 1.2, 1.3],
            "center_x": [500, 570, 650, 730],
            "center_y": [300, 360, 470, 650],
            "source_width": 1920,
            "source_height": 1080,
            "status": "tracked",
            "track_id": 1,
            "flight_id": 1,
        }
    ).to_csv(trajectory, index=False)

    result = segment_rallies(features, probabilities, output, shuttle_trajectory_csv=trajectory)

    assert len(result) == 1
    assert result[0].start <= 1.0
    assert result[0].end >= 1.7
    assert result[0].end - result[0].start < 1.5


def test_short_track_gap_during_active_play_is_not_a_landing(tmp_path: Path) -> None:
    times = np.round(np.arange(0.0, 5.0, 0.1), 3)
    active = ((times >= 1.0) & (times <= 1.9)) | ((times >= 2.6) & (times <= 3.3))
    frame = pd.DataFrame(
        {
            "time_seconds": times,
            "frame_index": np.arange(len(times)),
            "near_swing_score": np.where((times >= 1.0) & (times <= 1.1), 0.5, 0.0),
            "far_swing_score": 0.0,
            "near_lunge_score": 0.0,
            "far_lunge_score": 0.0,
            "near_flow_mean": np.where(active, 1.3, 0.0),
            "far_flow_mean": np.where(active, 0.8, 0.0),
            "near_motion_fraction": 0.0,
            "far_motion_fraction": 0.0,
            "audio_hit_score": 0.0,
            "shuttle_visible": 0.0,
            "shuttle_speed_normalized": 0.0,
        }
    )
    features = tmp_path / "features.csv"
    probabilities = tmp_path / "probabilities.csv"
    trajectory = tmp_path / "trajectory.csv"
    output = tmp_path / "rallies.csv"
    frame.to_csv(features, index=False)
    probability = np.where((times >= 2.6) & (times <= 3.3), 0.35, 0.01)
    probability[(times >= 1.0) & (times <= 1.1)] = 0.9
    pd.DataFrame({"time_seconds": times, "rally_probability": probability}).to_csv(probabilities, index=False)
    pd.DataFrame(
        {
            "time_seconds": [1.0, 1.3, 1.6, 1.9, 3.0, 3.1, 3.2, 3.3],
            "center_x": [500, 560, 620, 680, 720, 780, 840, 900],
            "center_y": [280, 350, 470, 650, 520, 430, 360, 310],
            "source_width": 1920,
            "source_height": 1080,
            "status": "tracked",
            "track_id": 1,
            "flight_id": 1,
        }
    ).to_csv(trajectory, index=False)

    result = segment_rallies(features, probabilities, output, shuttle_trajectory_csv=trajectory)

    assert len(result) == 1
    assert result[0].start <= 1.0
    assert result[0].end >= 3.6


def test_contact_backed_flight_can_start_when_pose_misses_ready_stance(tmp_path: Path) -> None:
    times = np.round(np.arange(0.0, 5.0, 0.1), 3)
    first_contact = (times >= 2.0) & (times <= 2.1)
    frame = pd.DataFrame(
        {
            "time_seconds": times,
            "frame_index": np.arange(len(times)),
            "near_swing_score": 0.0,
            "far_swing_score": np.where(first_contact, 0.24, 0.0),
            "near_lunge_score": 0.0,
            "far_lunge_score": 0.0,
            "near_flow_mean": 0.0,
            "far_flow_mean": np.where((times >= 2.0) & (times <= 3.0), 1.2, 0.0),
            "near_motion_fraction": 0.0,
            "far_motion_fraction": 0.0,
            "audio_hit_score": 0.0,
            "shuttle_visible": 0.0,
            "shuttle_speed_normalized": 0.0,
            "near_person_visible": 1.0,
            "far_person_visible": 1.0,
            "near_foot_speed_normalized": 0.2,
            "far_foot_speed_normalized": 0.2,
            "near_stance_width_normalized": 0.04,
            "far_stance_width_normalized": 0.02,
        }
    )
    features = tmp_path / "features.csv"
    probabilities = tmp_path / "probabilities.csv"
    trajectory = tmp_path / "trajectory.csv"
    output = tmp_path / "rallies.csv"
    frame.to_csv(features, index=False)
    probability = np.where((times >= 2.0) & (times <= 3.0), 0.12, 0.01)
    pd.DataFrame({"time_seconds": times, "rally_probability": probability}).to_csv(probabilities, index=False)
    pd.DataFrame(
        {
            "time_seconds": np.arange(2.0, 3.01, 0.1),
            "center_x": np.linspace(620, 1100, 11),
            "center_y": np.r_[np.linspace(520, 260, 6), np.linspace(300, 650, 5)],
            "source_width": 1920,
            "source_height": 1080,
            "status": "tracked",
            "track_id": 1,
            "flight_id": 1,
        }
    ).to_csv(trajectory, index=False)

    result = segment_rallies(features, probabilities, output, shuttle_trajectory_csv=trajectory)

    assert len(result) == 1
    assert result[0].start <= 2.0
    assert result[0].end >= 3.3


def test_handoff_flight_without_ready_stance_does_not_start_rally(tmp_path: Path) -> None:
    times = np.round(np.arange(0.0, 5.0, 0.1), 3)
    frame = pd.DataFrame(
        {
            "time_seconds": times,
            "near_swing_score": 0.0,
            "far_swing_score": 0.0,
            "near_lunge_score": 0.0,
            "far_lunge_score": 0.0,
            "near_flow_mean": 0.0,
            "far_flow_mean": 0.0,
            "near_motion_fraction": 0.0,
            "far_motion_fraction": 0.0,
            "audio_hit_score": 0.0,
            "shuttle_visible": 0.0,
            "shuttle_speed_normalized": 0.0,
            "near_person_visible": 1.0,
            "far_person_visible": 1.0,
            "near_foot_speed_normalized": 0.2,
            "far_foot_speed_normalized": 0.2,
            "near_stance_width_normalized": 0.04,
            "far_stance_width_normalized": 0.02,
        }
    )
    features = tmp_path / "features.csv"
    probabilities = tmp_path / "probabilities.csv"
    trajectory = tmp_path / "trajectory.csv"
    output = tmp_path / "rallies.csv"
    frame.to_csv(features, index=False)
    pd.DataFrame({"time_seconds": times, "rally_probability": 0.05}).to_csv(probabilities, index=False)
    pd.DataFrame(
        {
            "time_seconds": np.arange(2.0, 2.81, 0.1),
            "center_x": np.linspace(500, 900, 9),
            "center_y": np.linspace(600, 300, 9),
            "source_width": 1920,
            "source_height": 1080,
            "status": "tracked",
            "track_id": 1,
            "flight_id": 1,
        }
    ).to_csv(trajectory, index=False)

    result = segment_rallies(features, probabilities, output, shuttle_trajectory_csv=trajectory)

    assert result == []


def test_confirmed_landing_prevents_immediate_handoff_from_extending_clip(tmp_path: Path) -> None:
    times = np.round(np.arange(0.0, 5.0, 0.1), 3)
    active = (times >= 1.0) & (times <= 1.8)
    handoff = (times >= 2.4) & (times <= 3.0)
    frame = pd.DataFrame(
        {
            "time_seconds": times,
            "near_swing_score": np.where(active, 0.55, 0.0),
            "far_swing_score": 0.0,
            "near_lunge_score": 0.0,
            "far_lunge_score": 0.0,
            "near_flow_mean": np.where(active, 2.0, 0.0),
            "far_flow_mean": 0.0,
            "near_motion_fraction": 0.0,
            "far_motion_fraction": 0.0,
            "audio_hit_score": 0.0,
            "shuttle_visible": 0.0,
            "shuttle_speed_normalized": 0.0,
        }
    )
    features = tmp_path / "features.csv"
    probabilities = tmp_path / "probabilities.csv"
    trajectory = tmp_path / "trajectory.csv"
    output = tmp_path / "rallies.csv"
    frame.to_csv(features, index=False)
    probability = np.where(active, 0.95, np.where(handoff, 0.70, 0.01))
    pd.DataFrame({"time_seconds": times, "rally_probability": probability}).to_csv(probabilities, index=False)
    pd.DataFrame(
        {
            "time_seconds": [1.2, 1.4, 1.6, 1.8, 2.0, 2.4, 2.6, 2.8, 3.0],
            "center_x": [500, 560, 620, 680, 740, 760, 700, 640, 580],
            "center_y": [250, 300, 380, 500, 680, 650, 540, 430, 350],
            "source_width": 1920,
            "source_height": 1080,
            "status": "tracked",
            "track_id": [1, 1, 1, 1, 1, 2, 2, 2, 2],
            "flight_id": [1, 1, 1, 1, 1, 2, 2, 2, 2],
        }
    ).to_csv(trajectory, index=False)

    result = segment_rallies(features, probabilities, output, shuttle_trajectory_csv=trajectory)

    assert len(result) == 1
    assert result[0].end < 3.0


def test_sparse_trajectory_points_do_not_create_flight_evidence(tmp_path: Path) -> None:
    times = np.round(np.arange(0.0, 3.0, 0.1), 3)
    frame = pd.DataFrame({"time_seconds": times, "rally_probability": 0.05})
    trajectory = tmp_path / "trajectory.csv"
    pd.DataFrame(
        {
            "time_seconds": [1.0, 1.2],
            "center_x": [400, 800],
            "center_y": [500, 300],
            "status": "tracked",
            "track_id": 1,
            "flight_id": 1,
        }
    ).to_csv(trajectory, index=False)

    evidence = build_rally_evidence(frame, trajectory)

    assert not np.any(evidence.trajectory_visible)
    assert not np.any(evidence.trajectory_flight_start)


def test_landing_and_next_serve_split_without_probability_valley(tmp_path: Path) -> None:
    times = np.round(np.arange(0.0, 9.0, 0.1), 3)
    first_play = (times >= 1.0) & (times <= 2.0)
    next_serve = (times >= 4.0) & (times <= 4.1)
    foot_speed = np.where(times < 2.6, 0.18, 0.0)
    frame = pd.DataFrame(
        {
            "time_seconds": times,
            "near_swing_score": np.where(first_play, 0.55, np.where(next_serve, 0.65, 0.0)),
            "far_swing_score": 0.0,
            "near_lunge_score": 0.0,
            "far_lunge_score": 0.0,
            "near_flow_mean": np.where(first_play | next_serve, 1.5, 0.0),
            "far_flow_mean": 0.0,
            "near_motion_fraction": 0.0,
            "far_motion_fraction": 0.0,
            "audio_hit_score": np.where(next_serve, 0.75, 0.0),
            "shuttle_visible": 0.0,
            "shuttle_speed_normalized": 0.0,
            "near_person_visible": 1.0,
            "far_person_visible": 1.0,
            "near_foot_speed_normalized": foot_speed,
            "far_foot_speed_normalized": foot_speed,
            "near_stance_width_normalized": 0.04,
            "far_stance_width_normalized": 0.02,
        }
    )
    features = tmp_path / "features.csv"
    probabilities = tmp_path / "probabilities.csv"
    trajectory = tmp_path / "trajectory.csv"
    output = tmp_path / "rallies.csv"
    frame.to_csv(features, index=False)
    probability = np.where((times >= 0.7) & (times <= 8.0), 0.92, 0.01)
    pd.DataFrame({"time_seconds": times, "rally_probability": probability}).to_csv(probabilities, index=False)
    pd.DataFrame(
        {
            "time_seconds": [1.2, 1.4, 1.6, 1.8, 2.0, 4.0, 4.2, 4.4, 4.6],
            "center_x": [450, 520, 590, 660, 730, 500, 570, 650, 740],
            "center_y": [250, 300, 390, 520, 690, 620, 520, 400, 280],
            "source_width": 1920,
            "source_height": 1080,
            "status": "tracked",
            "track_id": [1, 1, 1, 1, 1, 2, 2, 2, 2],
            "flight_id": [1, 1, 1, 1, 1, 2, 2, 2, 2],
        }
    ).to_csv(trajectory, index=False)

    result = segment_rallies(features, probabilities, output, shuttle_trajectory_csv=trajectory)

    assert len(result) == 2
    assert result[0].end < result[1].start


def test_complete_handoff_flight_stays_between_points_until_formal_serve(tmp_path: Path) -> None:
    times = np.round(np.arange(0.0, 8.1, 0.1), 3)
    first_play = (times >= 1.0) & (times <= 2.0)
    handoff = (times >= 3.0) & (times <= 4.0)
    serve = (times >= 6.0) & (times <= 7.0)
    swing = np.isclose(times, 1.0) | np.isclose(times, 3.0) | np.isclose(times, 6.0)
    ready_window = (times >= 4.6) & (times <= 6.0)
    frame = pd.DataFrame(
        {
            "time_seconds": times,
            "near_swing_score": np.where(swing, 0.65, 0.0),
            "far_swing_score": 0.0,
            "near_lunge_score": 0.0,
            "far_lunge_score": 0.0,
            "near_flow_mean": np.where(first_play | handoff | serve, 1.5, 0.0),
            "far_flow_mean": 0.0,
            "near_motion_fraction": 0.0,
            "far_motion_fraction": 0.0,
            "audio_hit_score": np.where(np.isclose(times, 6.0), 0.75, 0.0),
            "shuttle_visible": 0.0,
            "shuttle_speed_normalized": 0.0,
            "near_person_visible": 1.0,
            "far_person_visible": 1.0,
            "near_foot_speed_normalized": np.where(ready_window, 0.0, 0.2),
            "far_foot_speed_normalized": np.where(ready_window, 0.0, 0.2),
            "near_stance_width_normalized": 0.04,
            "far_stance_width_normalized": 0.02,
        }
    )
    features = tmp_path / "features.csv"
    probabilities = tmp_path / "probabilities.csv"
    trajectory = tmp_path / "trajectory.csv"
    output = tmp_path / "rallies.csv"
    frame.to_csv(features, index=False)
    probability = np.where(first_play | handoff | serve, 0.95, 0.01)
    pd.DataFrame({"time_seconds": times, "rally_probability": probability}).to_csv(probabilities, index=False)

    track_rows = []
    for flight_id, start in enumerate((1.0, 3.0, 6.0), 1):
        for offset, (x, y) in enumerate(zip((500, 580, 660, 740, 820, 900), (260, 300, 370, 450, 560, 680))):
            track_rows.append(
                {
                    "time_seconds": start + offset * 0.2,
                    "center_x": x,
                    "center_y": y,
                    "source_width": 1920,
                    "source_height": 1080,
                    "status": "tracked",
                    "track_id": flight_id,
                    "flight_id": flight_id,
                }
            )
    pd.DataFrame(track_rows).to_csv(trajectory, index=False)

    evidence_frame = frame.assign(rally_probability=probability)
    evidence = build_rally_evidence(evidence_frame, trajectory)
    handoff_index = int(np.flatnonzero(np.isclose(times, 3.0))[0])
    serve_index = int(np.flatnonzero(np.isclose(times, 6.0))[0])
    assert evidence.between_points[handoff_index]
    assert evidence.handoff_candidate[handoff_index]
    assert not evidence.trajectory_contact_start[handoff_index]
    assert not evidence.between_points[serve_index]

    result = segment_rallies(features, probabilities, output, shuttle_trajectory_csv=trajectory)

    assert len(result) == 2
    assert result[0].end < 3.0
    assert result[1].start >= 5.5
    assert result[0].end < 4.0
