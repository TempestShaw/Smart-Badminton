from pathlib import Path

import numpy as np
import pandas as pd

from smart_badminton.adaptation import SegmentationAdapter
from smart_badminton.local_boundaries import split_at_local_starts
from smart_badminton.rally_evidence import RallyEvidence, build_rally_evidence
from smart_badminton.segmenter import Interval, segment_rallies


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


def test_learned_gap_prior_softly_penalizes_an_immediate_unsupported_restart(tmp_path: Path) -> None:
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
            "rally_probability": np.where((times >= 1.0) & (times <= 2.5), 0.95, np.where(active, 0.65, 0.01)),
            "gap_hard_min_seconds": 2.0,
            "gap_soft_min_seconds": 4.0,
        }
    ).to_csv(probabilities_path, index=False)

    result = segment_rallies(features_path, probabilities_path, output_path)

    assert len(result) == 1


def test_short_gap_cannot_block_a_strong_formal_rally_candidate(tmp_path: Path) -> None:
    times = np.round(np.arange(0.0, 8.0, 0.1), 3)
    active = ((times >= 1.0) & (times <= 2.5)) | ((times >= 3.4) & (times <= 4.9))
    frame = pd.DataFrame(
        {
            "time_seconds": times,
            "frame_index": np.arange(len(times)),
            "near_swing_score": np.where(active, 0.7, 0.0),
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

    assert len(result) == 2


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

    result = segment_rallies(
        features,
        probabilities,
        output,
        shuttle_trajectory_csv=trajectory,
        adapter=SegmentationAdapter(split_handoff_before_serve=True),
    )

    assert len(result) == 2
    assert result[0].end < result[1].start


def test_trajectory_tracklet_restart_does_not_split_active_rally(tmp_path: Path) -> None:
    times = np.round(np.arange(0.0, 9.0, 0.1), 3)
    first_play = (times >= 1.0) & (times <= 2.0)
    ready_window = (times >= 2.6) & (times <= 4.0)
    frame = pd.DataFrame(
        {
            "time_seconds": times,
            "near_swing_score": np.where(first_play, 0.55, 0.0),
            "far_swing_score": 0.0,
            "near_lunge_score": 0.0,
            "far_lunge_score": 0.0,
            "near_flow_mean": np.where(first_play, 1.5, 0.0),
            "far_flow_mean": 0.0,
            "near_motion_fraction": 0.0,
            "far_motion_fraction": 0.0,
            "audio_hit_score": 0.0,
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
    pd.DataFrame(
        {"time_seconds": times, "rally_probability": np.where((times >= 0.7) & (times <= 8.0), 0.92, 0.01)}
    ).to_csv(probabilities, index=False)
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

    assert len(result) == 1


def test_uncertain_landing_cannot_block_a_later_model_backed_rally(tmp_path: Path) -> None:
    times = np.round(np.arange(0.0, 8.0, 0.1), 3)
    first = (times >= 1.0) & (times <= 1.2)
    second = (times >= 5.0) & (times <= 6.0)
    frame = pd.DataFrame(
        {
            "time_seconds": times,
            "frame_index": np.arange(len(times)),
            "near_swing_score": np.where(first | second, 0.7, 0.0),
            "far_swing_score": 0.0,
            "near_lunge_score": 0.0,
            "far_lunge_score": 0.0,
            "near_flow_mean": np.where(first | second, 2.0, 0.0),
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
    pd.DataFrame(
        {
            "time_seconds": times,
            "frame_index": frame.frame_index,
            "rally_probability": np.where(first | second, 0.95, 0.01),
        }
    ).to_csv(probabilities, index=False)
    pd.DataFrame(
        {
            "time_seconds": [1.0, 1.1, 1.2, 1.3],
            "center_x": [500, 560, 620, 680],
            "center_y": [280, 360, 470, 650],
            "source_width": 1920,
            "source_height": 1080,
            "status": "tracked",
            "track_id": 1,
            "flight_id": 1,
        }
    ).to_csv(trajectory, index=False)

    result = segment_rallies(features, probabilities, output, shuttle_trajectory_csv=trajectory)

    assert len(result) == 2
    assert result[1].start <= 5.0


def test_protect_only_trajectory_extends_end_without_changing_clip_count(tmp_path: Path) -> None:
    times = np.round(np.arange(0.0, 7.0, 0.1), 3)
    active = ((times >= 1.0) & (times <= 2.0)) | ((times >= 4.0) & (times <= 5.0))
    frame = pd.DataFrame(
        {
            "time_seconds": times,
            "frame_index": np.arange(len(times)),
            "near_swing_score": np.where(
                active,
                0.7,
                np.where(np.isclose(times, 2.3), 0.17, 0.0),
            ),
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
    pd.DataFrame(
        {
            "time_seconds": times,
            "frame_index": frame.frame_index,
            "rally_probability": np.where(
                active,
                0.95,
                np.where(np.isclose(times, 2.3), 0.10, 0.01),
            ),
        }
    ).to_csv(probabilities, index=False)
    pd.DataFrame(
        {
            "time_seconds": [1.8, 2.0, 2.4, 2.5, 2.6],
            "center_x": [500, 560, 620, 680, 740],
            "center_y": [260, 320, 400, 510, 650],
            "source_width": 1920,
            "source_height": 1080,
            "status": "tracked",
            "track_id": [1, 1, 2, 2, 2],
            "flight_id": [1, 1, 2, 2, 2],
        }
    ).to_csv(trajectory, index=False)

    result = segment_rallies(
        features,
        probabilities,
        output,
        shuttle_trajectory_csv=trajectory,
        trajectory_policy="protect_only",
    )

    assert len(result) == 2
    assert result[0].end >= 3.0
    assert result[0].end < result[1].start

    continuous = pd.read_csv(trajectory)
    continuous[["track_id", "flight_id"]] = 1
    continuous.to_csv(trajectory, index=False)
    unchanged = segment_rallies(
        features,
        probabilities,
        output,
        shuttle_trajectory_csv=trajectory,
        trajectory_policy="protect_only",
    )

    assert unchanged[0].end == 2.55


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


def test_unknown_neighbor_trajectory_is_not_editing_evidence(tmp_path: Path) -> None:
    times = np.round(np.arange(0.0, 3.0, 0.1), 3)
    frame = pd.DataFrame({"time_seconds": times, "rally_probability": 0.05})
    trajectory = tmp_path / "trajectory.csv"
    pd.DataFrame(
        {
            "time_seconds": [1.0, 1.1, 1.2, 1.3],
            "center_x": [1400, 1450, 1500, 1550],
            "center_y": [300, 350, 420, 500],
            "source_width": 1920,
            "source_height": 1080,
            "status": "tracked",
            "track_id": 1,
            "flight_id": 1,
            "ownership_confidence": 0.25,
            "ownership_evidence": "unknown",
        }
    ).to_csv(trajectory, index=False)

    evidence = build_rally_evidence(frame, trajectory)

    assert not np.any(evidence.trajectory_visible)
    assert not np.any(evidence.protected_flight)


def test_stationary_gap_splits_reused_flight_id(tmp_path: Path) -> None:
    times = np.round(np.arange(0.0, 4.0, 0.1), 3)
    frame = pd.DataFrame(
        {
            "time_seconds": times,
            "rally_probability": 0.8,
            "near_swing_score": 0.3,
        }
    )
    trajectory = tmp_path / "trajectory.csv"
    rows = []
    for time_seconds in np.round(np.arange(1.0, 3.1, 0.1), 3):
        if time_seconds <= 1.5:
            center_x = 500 + (time_seconds - 1.0) * 300
        elif time_seconds < 2.5:
            center_x = 650
        else:
            center_x = 650 + (time_seconds - 2.5) * 300
        rows.append(
            {
                "time_seconds": time_seconds,
                "center_x": center_x,
                "center_y": 400,
                "source_width": 1920,
                "source_height": 1080,
                "status": "tracked",
                "track_id": 1,
                "flight_id": 1,
                "ownership_confidence": 0.9,
                "ownership_evidence": "net_crossing",
            }
        )
    pd.DataFrame(rows).to_csv(trajectory, index=False)

    evidence = build_rally_evidence(frame, trajectory)

    assert evidence.trajectory_visible[np.flatnonzero(np.isclose(times, 1.3))[0]]
    assert not evidence.trajectory_visible[np.flatnonzero(np.isclose(times, 2.0))[0]]
    assert evidence.trajectory_visible[np.flatnonzero(np.isclose(times, 2.8))[0]]


def test_local_start_adapter_splits_a_merged_interval(tmp_path: Path) -> None:
    times = np.round(np.arange(0.0, 10.0, 0.1), 3)
    contacts = np.isclose(times, 1.0) | np.isclose(times, 6.0)
    frame = pd.DataFrame(
        {
            "time_seconds": times,
            "near_swing_score": np.where(contacts, 0.55, 0.0),
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
    features = tmp_path / "features.csv"
    probabilities = tmp_path / "probabilities.csv"
    trajectory = tmp_path / "trajectory.csv"
    output = tmp_path / "rallies.csv"
    frame.to_csv(features, index=False)
    pd.DataFrame({"time_seconds": times, "rally_probability": 0.92}).to_csv(
        probabilities, index=False
    )
    rows = []
    for flight_id, start in enumerate((1.0, 6.0), 1):
        for offset in range(5):
            rows.append(
                {
                    "time_seconds": start + offset * 0.2,
                    "center_x": 500 + offset * 80,
                    "center_y": 600 - offset * 50,
                    "source_width": 1920,
                    "source_height": 1080,
                    "status": "tracked",
                    "track_id": flight_id,
                    "flight_id": flight_id,
                }
            )
    pd.DataFrame(rows).to_csv(trajectory, index=False)
    start_model = {
        "type": "decision_tree",
        "feature_names": ["trajectory_flight_start"],
        "children_left": [-1],
        "children_right": [-1],
        "split_features": [-2],
        "thresholds": [-2.0],
        "positive_probability": [1.0],
        "threshold": 0.9,
        "lead_seconds": 0.3,
        "dedupe_seconds": 1.0,
        "quiet_probability_ceiling": 0.2,
        "quiet_min_seconds": 0.6,
        "quiet_search_seconds": 6.0,
    }

    result = segment_rallies(
        features,
        probabilities,
        output,
        shuttle_trajectory_csv=trajectory,
        adapter=SegmentationAdapter(start_quality=start_model),
    )

    assert len(result) == 2
    assert result[0].end <= result[1].start


def test_local_adapter_splits_a_sustained_quiet_gap_without_a_serve_candidate() -> None:
    times = np.round(np.arange(0.0, 10.1, 0.1), 3)
    probability = np.where((times >= 4.0) & (times <= 6.0), 0.05, 0.92)
    zeros = np.zeros(len(times), dtype=bool)
    flight_start = zeros.copy()
    flight_start[10] = True
    evidence = RallyEvidence(
        activity=np.full(len(times), 0.8),
        player_engagement=np.full(len(times), 0.5),
        trajectory_visible=zeros.copy(),
        trajectory_descending=zeros.copy(),
        trajectory_occluded=zeros.copy(),
        trajectory_flight_start=flight_start,
        trajectory_flight_end=zeros.copy(),
        trajectory_serve_start=zeros.copy(),
        trajectory_contact_start=zeros.copy(),
        landing_candidate=zeros.copy(),
        protected_flight=zeros.copy(),
        near_ready=zeros.copy(),
        far_ready=zeros.copy(),
        players_ready=zeros.copy(),
        between_points=zeros.copy(),
        handoff_candidate=zeros.copy(),
        formal_serve=zeros.copy(),
        near_serving=zeros.copy(),
        far_serving=zeros.copy(),
        serve_confidence=np.zeros(len(times)),
    )
    data = pd.DataFrame(
        {
            "time_seconds": times,
            "rally_probability": probability,
            "audio_hit_score": 0.0,
            "near_swing_score": 0.0,
            "far_swing_score": 0.0,
        }
    )
    model = {
        "feature_names": ["trajectory_flight_start"],
        "children_left": [-1],
        "children_right": [-1],
        "split_features": [-2],
        "thresholds": [-2.0],
        "positive_probability": [1.0],
        "threshold": 0.9,
        "lead_seconds": 1.0,
        "dedupe_seconds": 1.0,
        "quiet_gap_probability_ceiling": 0.2,
        "quiet_gap_window_seconds": 1.0,
        "quiet_gap_min_seconds": 0.6,
    }

    result = split_at_local_starts(
        [Interval(0.0, 10.0, 0.9, "test")],
        data,
        times,
        probability,
        evidence,
        model,
    )

    assert len(result) == 2
    assert result[0].end < result[1].start
