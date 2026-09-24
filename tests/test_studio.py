import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from smart_badminton import studio
from smart_badminton.project_layout import ProjectLayout
from smart_badminton.studio import StudioState, app, calibration, insights, jobs, media, pipeline, project, shuttle
from smart_badminton.studio import state as studio_state
from smart_badminton.studio.insights import evidence_payload, serve_observations
from smart_badminton.studio.pipeline import POSE_ASSOCIATION_FIELDS, analyze_video, trajectory_needs_refinement
from smart_badminton.studio.project import discover_videos, ensure_working_timeline, save_segments, timeline_rows
from smart_badminton.studio.shuttle import run_shuttle_detection

STUDIO_MODULES = (app, calibration, insights, jobs, media, pipeline, project, shuttle, studio_state)


@pytest.fixture
def patch_studio(monkeypatch):
    """Replace a name in every Studio module that imported it."""

    def patch(name: str, value) -> None:
        targets = [module for module in STUDIO_MODULES if hasattr(module, name)]
        assert targets, name
        for module in targets:
            monkeypatch.setattr(module, name, value)

    return patch


def client_for(state: StudioState) -> TestClient:
    return TestClient(studio.create_studio_app(state), base_url="http://127.0.0.1")


def test_studio_validates_non_overlapping_segments() -> None:
    rows = timeline_rows(
        [{"start": 1.0, "end": 2.0}, {"start": 2.0, "end": 3.5}],
        duration=5.0,
    )
    assert rows[0]["end_seconds"] == rows[1]["start_seconds"]


def test_trajectory_refines_when_features_or_user_annotations_are_newer(tmp_path: Path) -> None:
    features = tmp_path / "features.csv"
    trajectory = tmp_path / "trajectory.csv"
    annotations = tmp_path / "annotations.csv"
    features.write_text("features", encoding="utf-8")
    annotations.write_text("annotations", encoding="utf-8")
    trajectory.write_text("trajectory", encoding="utf-8")
    os.utime(features, (100, 100))
    os.utime(annotations, (100, 100))
    os.utime(trajectory, (200, 200))

    assert trajectory_needs_refinement(features, trajectory, annotations) is False
    os.utime(annotations, (300, 300))
    assert trajectory_needs_refinement(features, trajectory, annotations) is True


def test_studio_save_creates_backup(tmp_path: Path) -> None:
    path = tmp_path / "rallies.csv"
    path.write_text("rally,start_seconds,end_seconds\n1,0,1\n", encoding="utf-8")
    rows = timeline_rows([{"start": 0.5, "end": 1.5}], duration=2.0)
    backup = save_segments(path, rows)
    assert backup.exists()
    assert "0.500" in path.read_text(encoding="utf-8")
    assert "0,1" in backup.read_text(encoding="utf-8")


def test_studio_api_saves_validated_timeline(tmp_path: Path, patch_studio) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"placeholder")
    timeline = tmp_path / "rallies.csv"
    timeline.write_text("rally,start_seconds,end_seconds\n1,1,2\n", encoding="utf-8")
    metadata = {"name": video.name, "duration": 10.0, "fps": 30.0, "frame_count": 300, "width": 1280, "height": 720}
    patch_studio("video_metadata", lambda _path: metadata)
    client = client_for(StudioState(video=video, rallies=timeline, output=tmp_path / "edited.mp4"))
    shell = client.get("/")
    assert shell.status_code == 200
    assert "Smart Badminton Studio" in shell.text
    assert "/static/_next/" in shell.text
    assert "studio.js" not in shell.text
    tutorial = client.get("/api/tutorial")
    assert tutorial.status_code == 200
    assert "校准保存后怎么重新预测" in tutorial.json()["markdown"]
    project = client.get("/api/project").json()
    assert project["segments"][0]["start"] == 1.0
    assert project["automatic_analysis"]["settings"]["preroll"] == 0.35
    assert project["automatic_analysis"]["settings"]["postroll"] == 0.55
    assert project["automatic_analysis"]["settings"]["suppress_handoffs"] is True
    assert project["evidence"]["available"] is False
    response = client.put(
        "/api/timeline",
        json={"project_id": "source.mp4", "segments": [{"start": 1.5, "end": 3.0}]},
    )
    assert response.status_code == 200
    assert timeline.with_suffix(".csv.bak").exists()
    assert "1.500" in timeline.read_text(encoding="utf-8")
    assert response.json()["phase_examples"]["examples"] == 2

    stale = client.put(
        "/api/timeline",
        json={"project_id": "other.mp4", "segments": [{"start": 2.0, "end": 4.0}]},
    )
    assert stale.status_code == 409
    assert "1.500" in timeline.read_text(encoding="utf-8")


def test_studio_directory_browser_and_output_selection_are_real(tmp_path: Path, patch_studio) -> None:
    source_folder = tmp_path / "source"
    output_folder = tmp_path / "exports"
    child_folder = source_folder / "nested"
    child_folder.mkdir(parents=True)
    output_folder.mkdir()
    video = source_folder / "source.mp4"
    video.write_bytes(b"placeholder")
    timeline = source_folder / "rallies.csv"
    timeline.write_text("rally,start_seconds,end_seconds\n1,1,2\n", encoding="utf-8")
    metadata = {"name": video.name, "duration": 10.0, "fps": 30.0, "frame_count": 300, "width": 1280, "height": 720}
    patch_studio("video_metadata", lambda _path: metadata)
    client = client_for(StudioState(video=video, rallies=timeline, output=source_folder / "edited.mp4", library=source_folder))

    browser = client.get("/api/filesystem/directories", params={"path": str(source_folder)})
    assert browser.status_code == 200
    assert browser.json()["path"] == str(source_folder.resolve())
    assert {row["name"] for row in browser.json()["directories"]} == {"nested"}

    changed = client.put(
        "/api/output",
        json={"project_id": "source.mp4", "directory": str(output_folder), "filename": "match-final.mp4"},
    )
    assert changed.status_code == 200
    assert changed.json()["path"] == str(output_folder.resolve() / "match-final.mp4")
    assert changed.json()["directory_exists"] is True
    assert changed.json()["file_exists"] is False
    project = client.get("/api/project").json()
    assert project["api_schema_version"] == 8
    assert project["output"]["filename"] == "match-final.mp4"

    stale = client.put(
        "/api/output",
        json={"project_id": "other.mp4", "directory": str(output_folder), "filename": "bad.mp4"},
    )
    assert stale.status_code == 409


def test_studio_reports_missing_ffmpeg(tmp_path: Path, patch_studio) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"placeholder")
    timeline = tmp_path / "rallies.csv"
    timeline.write_text("rally,start_seconds,end_seconds\n1,1,2\n", encoding="utf-8")
    config = tmp_path / "court.json"
    model = tmp_path / "rallies.joblib"
    pose_model = tmp_path / "pose.pt"
    config.write_text("{}", encoding="utf-8")
    model.write_bytes(b"model")
    pose_model.write_bytes(b"pose")
    metadata = {
        "name": video.name,
        "duration": 10.0,
        "fps": 30.0,
        "frame_count": 300,
        "width": 1280,
        "height": 720,
    }
    patch_studio("video_metadata", lambda _path: metadata)
    patch_studio("calibration_ready", lambda _state: True)

    def missing_ffmpeg(_path=None):
        raise RuntimeError("FFmpeg unavailable in test")

    patch_studio("resolve_ffmpeg", missing_ffmpeg)
    state = StudioState(
        video=video,
        rallies=timeline,
        output=tmp_path / "edited.mp4",
        config=config,
        model=model,
        pose_model=pose_model,
    )
    client = client_for(state)

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["api_schema_version"] == 8
    assert health.json()["runtime"]["ffmpeg"]["available"] is False
    project = client.get("/api/project").json()
    assert project["runtime"]["ffmpeg"]["reason"] == "FFmpeg unavailable in test"
    assert project["automatic_analysis"]["configured"] is False
    assert project["automatic_analysis"]["pose_overlay_configured"] is False


def test_studio_reports_missing_h264_encoder(tmp_path: Path, patch_studio) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"placeholder")
    timeline = tmp_path / "rallies.csv"
    timeline.write_text("rally,start_seconds,end_seconds\n1,1,2\n", encoding="utf-8")
    metadata = {
        "name": video.name,
        "duration": 10.0,
        "fps": 30.0,
        "frame_count": 300,
        "width": 1280,
        "height": 720,
    }
    patch_studio("video_metadata", lambda _path: metadata)
    patch_studio("resolve_ffmpeg", lambda _path=None: Path("ffmpeg"))

    def missing_encoder(_requested, _ffmpeg):
        raise RuntimeError("FFmpeg does not provide a supported H.264 encoder")

    patch_studio("choose_working_video_encoder", missing_encoder)
    state = StudioState(video=video, rallies=timeline, output=tmp_path / "edited.mp4")
    client = client_for(state)

    project = client.get("/api/project").json()
    assert project["runtime"]["ffmpeg"]["available"] is False
    assert project["runtime"]["ffmpeg"]["selected_encoder"] is None
    assert "H.264 encoder" in project["runtime"]["ffmpeg"]["reason"]


def test_studio_render_can_include_trajectory(tmp_path: Path, patch_studio) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"placeholder")
    timeline = tmp_path / "rallies.csv"
    timeline.write_text("rally,start_seconds,end_seconds\n1,1,2\n", encoding="utf-8")
    analysis = tmp_path / "Analysis" / "Auto" / video.stem
    analysis.mkdir(parents=True)
    trajectory = analysis / "shuttle-track.csv"
    trajectory.write_text("trajectory", encoding="utf-8")
    captured: dict[str, object] = {}

    patch_studio("runtime_payload", lambda _state: {"ffmpeg": {"available": True, "selected_encoder": "libx264", "reason": None}},
    )

    def fake_render(*args, **kwargs):
        captured.update(kwargs)
        return "libx264"

    patch_studio("render_rallies", fake_render)
    state = StudioState(
        video=video,
        rallies=timeline,
        output=tmp_path / "edited.mp4",
        library=tmp_path,
    )
    client = client_for(state)

    response = client.post("/api/render", json={"include_trajectory": True})
    assert response.status_code == 200
    for _ in range(100):
        if client.get("/api/render").json()["state"] != "running":
            break
        time.sleep(0.01)

    status = client.get("/api/render").json()
    assert status["state"] == "complete"
    assert status["include_trajectory"] is True
    assert captured["trajectory_csv"] == trajectory


def test_studio_render_can_filter_winner_and_show_score(tmp_path: Path, patch_studio) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"placeholder")
    timeline = tmp_path / "rallies.csv"
    timeline.write_text("rally,start_seconds,end_seconds\n1,1,2\n", encoding="utf-8")
    analysis = tmp_path / "Analysis" / "Auto" / video.stem
    analysis.mkdir(parents=True)
    score_path = analysis / "score-state.csv"
    score_path.write_text(
        "rally,winner,near_score,far_score,game_finished\n1,near,1,0,\n",
        encoding="utf-8",
    )
    (analysis / "score-summary.json").write_text(
        json.dumps({"available": True, "rallies": []}),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}
    patch_studio("runtime_payload", lambda _state: {"ffmpeg": {"available": True, "selected_encoder": "libx264", "reason": None}},
    )

    def fake_render(*args, **kwargs):
        captured.update(kwargs)
        return "libx264"

    patch_studio("render_rallies", fake_render)
    client = client_for(StudioState(video=video, rallies=timeline, output=tmp_path / "edited.mp4", library=tmp_path))

    response = client.post(
        "/api/render",
        json={"winner_filter": "near", "include_score": True},
    )
    assert response.status_code == 200
    for _ in range(100):
        if client.get("/api/render").json()["state"] != "running":
            break
        time.sleep(0.01)

    status = client.get("/api/render").json()
    assert status["state"] == "complete"
    assert status["winner_filter"] == "near"
    assert status["include_score"] is True
    assert captured["score_csv"] == score_path
    assert captured["winner_filter"] == "near"
    assert captured["include_score"] is True


def test_studio_score_correction_does_not_mutate_timeline(tmp_path: Path, patch_studio) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"placeholder")
    timeline = tmp_path / "rallies.csv"
    timeline.write_text("rally,start_seconds,end_seconds\n1,1,2\n", encoding="utf-8")
    original = timeline.read_text(encoding="utf-8")
    metadata = {"name": video.name, "duration": 10.0, "fps": 30.0, "frame_count": 300, "width": 1280, "height": 720}
    patch_studio("video_metadata", lambda _path: metadata)
    client = client_for(StudioState(video=video, rallies=timeline, output=tmp_path / "edited.mp4", library=tmp_path))

    response = client.put(
        "/api/score",
        json={
            "project_id": "source.mp4",
            "corrections": [
                {
                    "rally": 1,
                    "winner": "near",
                    "server_override": "near",
                    "server": "far",
                    "last_hitter": "near",
                    "terminal_event": "landing_in",
                    "landing_side": "far",
                    "post_rally_event": "handoff",
                    "note": "reviewed",
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["rallies"][0]["near_score"] == 1
    assert response.json()["rallies"][0]["server_next"] == "near"
    assert response.json()["corrections"][0]["terminal_event"] == "landing_in"
    assert response.json()["corrections"][0]["post_rally_event"] == "handoff"
    assert timeline.read_text(encoding="utf-8") == original
    stale = client.put("/api/score", json={"project_id": "other.mp4", "corrections": []})
    assert stale.status_code == 409


def test_studio_calculates_score_on_demand_and_invalidates_it_after_timeline_edit(
    tmp_path: Path, patch_studio
) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"placeholder")
    timeline = tmp_path / "rallies.csv"
    timeline.write_text("rally,start_seconds,end_seconds\n1,1,2\n2,3,4\n", encoding="utf-8")
    metadata = {
        "name": video.name,
        "duration": 10.0,
        "fps": 30.0,
        "frame_count": 300,
        "width": 1280,
        "height": 720,
    }
    patch_studio("video_metadata", lambda _path: metadata)
    client = client_for(StudioState(video=video, rallies=timeline, output=tmp_path / "edited.mp4", library=tmp_path))

    assert client.get("/api/score").json()["generated"] is False
    calculated = client.post("/api/score/analyze", json={"project_id": "source.mp4"})
    assert calculated.status_code == 200
    assert calculated.json()["available"] is True
    assert len(calculated.json()["rallies"]) == 2

    saved = client.put(
        "/api/timeline",
        json={
            "project_id": "source.mp4",
            "segments": [
                {"start": 1.0, "end": 2.2},
                {"start": 3.0, "end": 4.0},
            ],
        },
    )
    assert saved.status_code == 200
    assert saved.json()["score"]["available"] is False
    assert saved.json()["score"]["stale"] is True


def test_studio_accepts_a_machine_score_suggestion_as_reviewed_truth(tmp_path: Path, patch_studio) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"placeholder")
    timeline = tmp_path / "rallies.csv"
    timeline.write_text("rally,start_seconds,end_seconds\n1,1,2\n", encoding="utf-8")
    original = timeline.read_bytes()
    metadata = {"name": video.name, "duration": 10.0, "fps": 30.0, "frame_count": 300, "width": 1280, "height": 720}
    patch_studio("video_metadata", lambda _path: metadata)
    labels = tmp_path / "Analysis" / "Score_Labeling" / "machine-score-labels.json"
    labels.parent.mkdir(parents=True)
    labels.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "labels": [
                    {
                        "rally": 1,
                        "status": "consensus",
                        "model": "vision",
                        "suggestion": {
                            "winner": "far",
                            "terminal_event": "net",
                            "last_hitter": "near",
                            "landing_side": "near",
                            "post_rally_event": "none",
                            "confidence": 0.91,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    client = client_for(StudioState(video=video, rallies=timeline, output=tmp_path / "edited.mp4", library=tmp_path))

    response = client.put(
        "/api/score-labels/review",
        json={"project_id": "source.mp4", "rally": 1, "decision": "accepted"},
    )

    assert response.status_code == 200
    assert response.json()["score"]["rallies"][0]["winner"] == "far"
    assert response.json()["score_labeling"]["accepted"] == 1
    corrections = (tmp_path / "Metadata" / "source-score-corrections.csv").read_text(encoding="utf-8")
    assert "far" in corrections and "net" in corrections
    assert timeline.read_bytes() == original


def test_pose_only_flight_cannot_decide_server(tmp_path: Path, patch_studio) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"placeholder")
    timeline = tmp_path / "rallies.csv"
    timeline.write_text("rally,start_seconds,end_seconds\n1,1,2\n2,3,5\n", encoding="utf-8")
    analysis = tmp_path / "Analysis" / "Auto" / "source"
    analysis.mkdir(parents=True)
    pd.DataFrame(
        {
            "time_seconds": [1.0, 2.0, 3.0, 3.2, 4.0],
            "near_swing_score": [0.0, 0.0, 0.05, 0.1, 0.0],
            "far_swing_score": [0.0, 0.0, 0.1, 0.7, 0.0],
            "audio_hit_score": [0.0, 0.0, 0.0, 0.6, 0.0],
        }
    ).to_csv(analysis / "smart-features.csv", index=False)
    pd.DataFrame(
        {
            "time_seconds": [1.0, 2.0, 3.0, 3.2, 4.0],
            "rally_probability": [0.1, 0.1, 0.3, 0.9, 0.8],
        }
    ).to_csv(analysis / "rally-probabilities.csv", index=False)
    (analysis / "shuttle-track.csv").write_text("time_seconds,x,y\n3.2,0.5,0.5\n", encoding="utf-8")
    patch_studio("build_rally_evidence", lambda *_args, **_kwargs: SimpleNamespace(
            trajectory_flight_start=[False, False, False, True, False],
            near_ready=[False, False, False, True, False],
            far_ready=[False, False, True, True, False],
        ),
    )
    state = StudioState(video=video, rallies=timeline, output=tmp_path / "edited.mp4", library=tmp_path)

    observations = serve_observations(state, {"available": True, "serves": [], "contacts": []})

    assert observations == []


def test_studio_evidence_payload_exports_compact_signal_spans(tmp_path: Path) -> None:
    video = tmp_path / "source.mp4"
    timeline = tmp_path / "rallies.csv"
    timeline.write_text("rally,start_seconds,end_seconds\n1,0,2\n", encoding="utf-8")
    analysis = tmp_path / "Analysis" / "Auto" / "source"
    analysis.mkdir(parents=True)
    times = [index / 10 for index in range(21)]
    pd.DataFrame(
        {
            "time_seconds": times,
            "near_swing_score": [0.7 if index == 15 else 0.0 for index in range(21)],
            "far_swing_score": 0.0,
            "near_lunge_score": 0.0,
            "far_lunge_score": 0.0,
            "near_flow_mean": 0.0,
            "far_flow_mean": 0.0,
            "near_motion_fraction": 0.0,
            "far_motion_fraction": 0.0,
            "audio_hit_score": [0.8 if index == 15 else 0.0 for index in range(21)],
            "near_person_visible": 1.0,
            "far_person_visible": 1.0,
            "near_foot_speed_normalized": 0.0,
            "far_foot_speed_normalized": 0.0,
            "near_stance_width_normalized": 0.04,
            "far_stance_width_normalized": 0.02,
        }
    ).to_csv(analysis / "smart-features.csv", index=False)
    pd.DataFrame(
        {"time_seconds": times, "rally_probability": [0.8 if 5 <= index <= 17 else 0.05 for index in range(21)]}
    ).to_csv(analysis / "rally-probabilities.csv", index=False)
    state = StudioState(video=video, rallies=timeline, output=tmp_path / "edited.mp4", library=tmp_path)

    payload = evidence_payload(state)

    assert payload["available"] is True
    assert payload["signals"]["model_active"]
    assert payload["signals"]["near_ready"]
    assert payload["signals"]["far_ready"]
    assert payload["serves"][0]["server"] == "near"


def test_studio_payload_cache_reuses_results_until_an_input_changes(tmp_path: Path) -> None:
    video = tmp_path / "source.mp4"
    timeline = tmp_path / "rallies.csv"
    source = tmp_path / "evidence.csv"
    video.write_bytes(b"video")
    timeline.write_text("rally,start_seconds,end_seconds\n1,0,2\n", encoding="utf-8")
    source.write_text("first", encoding="utf-8")
    state = StudioState(video=video, rallies=timeline, output=tmp_path / "edited.mp4")
    calls = 0

    def build() -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"value": source.read_text(encoding="utf-8")}

    assert studio_state.cached_payload(state, "test", (source,), build)["value"] == "first"
    assert studio_state.cached_payload(state, "test", (source,), build)["value"] == "first"
    assert calls == 1

    source.write_text("second value", encoding="utf-8")
    assert studio_state.cached_payload(state, "test", (source,), build)["value"] == "second value"
    assert calls == 2


def test_studio_api_creates_and_backs_up_interactive_calibration(tmp_path: Path, patch_studio) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"placeholder")
    timeline = tmp_path / "rallies.csv"
    timeline.write_text("rally,start_seconds,end_seconds\n1,1,2\n", encoding="utf-8")
    metadata = {
        "name": video.name,
        "duration": 10.0,
        "fps": 30.0,
        "frame_count": 300,
        "width": 1280,
        "height": 720,
    }
    patch_studio("video_metadata", lambda _path: metadata)
    state = StudioState(video=video, rallies=timeline, output=tmp_path / "edited.mp4")
    client = client_for(state)

    calibration = client.get("/api/calibration").json()
    assert calibration["exists"] is False
    assert calibration["required_total"] == 5
    for index, region in enumerate(calibration["regions"]):
        if region["required"]:
            offset = index * 0.01
            region["points"] = [[0.1 + offset, 0.1], [0.8, 0.1 + offset], [0.7, 0.8], [0.2, 0.8]]
        elif region["id"] == "shuttle_perspective_axis":
            region["points"] = [[0.5, 0.1], [0.5, 0.75]]

    response = client.put(
        "/api/calibration",
        json={"project_id": "source.mp4", "source_time_seconds": 3.5, "regions": calibration["regions"]},
    )
    assert response.status_code == 200
    assert response.json()["ready"] is True
    assert state.config == tmp_path / "Calibration" / "court-config.json"
    assert state.config.exists()
    assert '"active_court_polygon"' in state.config.read_text(encoding="utf-8")
    assert '"shuttle_perspective_axis_normalized"' in state.config.read_text(encoding="utf-8")

    second = client.put(
        "/api/calibration",
        json={"project_id": "source.mp4", "source_time_seconds": 4.0, "regions": response.json()["regions"]},
    )
    assert second.status_code == 200
    assert state.config.with_suffix(".json.bak").exists()


def test_studio_api_saves_user_shuttle_annotations(tmp_path: Path, patch_studio) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"placeholder")
    timeline = tmp_path / "rallies.csv"
    timeline.write_text("rally,start_seconds,end_seconds\n1,1,2\n", encoding="utf-8")
    metadata = {
        "name": video.name,
        "duration": 10.0,
        "fps": 30.0,
        "frame_count": 300,
        "width": 1280,
        "height": 720,
    }
    patch_studio("video_metadata", lambda _path: metadata)
    state = StudioState(
        video=video,
        rallies=timeline,
        output=tmp_path / "edited.mp4",
        library=tmp_path,
        config=tmp_path / "court.json",
        shuttle_model=tmp_path / "shuttle.pt",
    )
    state.config.write_text("{}", encoding="utf-8")
    state.shuttle_model.write_bytes(b"model")
    client = client_for(state)

    initial = client.get("/api/shuttle-annotations")
    assert initial.status_code == 200
    assert initial.json()["available"] is False
    assert initial.json()["configured"] is True
    assert initial.json()["generated"] is False
    response = client.put(
        "/api/shuttle-annotations",
        json={
            "project_id": "source.mp4",
            "annotations": [
                {
                    "id": "user-1",
                    "time_seconds": 1.5,
                    "x_normalized": 0.4,
                    "y_normalized": 0.3,
                    "action": "add",
                    "note": "studio",
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["annotations"][0]["action"] == "add"
    annotation_path = tmp_path / "Analysis" / "Auto" / "source" / "Shuttle_Annotations" / "user-shuttle-points.csv"
    assert annotation_path.exists()
    trajectory_path = annotation_path.parents[1] / "shuttle-track.csv"
    trajectory_path.write_text(
        "time_seconds,frame,center_x,center_y,confidence,source_width,source_height,status,flight_id\n"
        "1.5,45,512,216,1.0,1280,720,manual,manual-1\n",
        encoding="utf-8",
    )
    visible = client.get("/api/shuttle-annotations").json()
    assert visible["point_count"] == 1
    assert visible["flight_count"] == 1
    assert visible["detections"] == [
        {
            "time": 1.5,
            "frame": 45,
            "x": 0.4,
            "y": 0.3,
            "confidence": 1.0,
            "status": "manual",
            "source": "unknown",
            "detection_status": "detected",
            "evidence_weight": 1.0,
            "flight_id": "manual-1",
        }
    ]
    stale = client.put(
        "/api/shuttle-annotations",
        json={"project_id": "other.mp4", "annotations": []},
    )
    assert stale.status_code == 409


def test_studio_runs_shuttle_analysis_without_changing_timeline(tmp_path: Path, patch_studio) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"placeholder")
    timeline = tmp_path / "rallies.csv"
    timeline.write_text("rally,start_seconds,end_seconds\n1,1,2\n", encoding="utf-8")
    config = tmp_path / "court.json"
    shuttle_model = tmp_path / "shuttle.pt"
    config.write_text("{}", encoding="utf-8")
    shuttle_model.write_bytes(b"model")
    metadata = {
        "name": video.name,
        "duration": 10.0,
        "fps": 30.0,
        "frame_count": 300,
        "width": 1280,
        "height": 720,
    }
    patch_studio("video_metadata", lambda _path: metadata)
    patch_studio("calibration_ready", lambda _state: True)

    def fake_detect(_video, _config, _model, _packages, output, **_kwargs):
        output.write_text(
            "time_seconds,frame,center_x,center_y,confidence,source_width,source_height\n"
            "1.0,30,640,360,0.9,1280,720\n",
            encoding="utf-8",
        )
        return {"sampled_frames": 150, "detections_in_roi": 1, "output": str(output)}

    def fake_track(_raw, output, _features, _video, _annotations, _config):
        output.write_text(
            "time_seconds,frame,center_x,center_y,confidence,source_width,source_height,status,flight_id\n"
            "1.0,30,640,360,0.9,1280,720,tracked,flight-1\n",
            encoding="utf-8",
        )

    patch_studio("detect_shuttle", fake_detect)
    patch_studio("analyze_shuttle_trajectory", fake_track)
    state = StudioState(
        video=video,
        rallies=timeline,
        output=tmp_path / "edited.mp4",
        library=tmp_path,
        config=config,
        shuttle_model=shuttle_model,
    )
    client = client_for(state)
    original_timeline = timeline.read_bytes()
    before = client.get("/api/project").json()
    assert before["shuttle_analysis"]["generated"] is False
    assert before["automatic_analysis"]["trajectory_boundary_protection"] is False

    stale = client.post("/api/analyze/shuttle", json={"project_id": "other.mp4"})
    assert stale.status_code == 409
    started = client.post("/api/analyze/shuttle", json={"project_id": "source.mp4", "force": False})
    assert started.status_code == 200
    deadline = time.monotonic() + 2.0
    status = started.json()
    while status["state"] == "running" and time.monotonic() < deadline:
        time.sleep(0.01)
        status = client.get("/api/analyze").json()

    assert status["state"] == "complete"
    assert status["mode"] == "shuttle"
    assert status["background"] is True
    assert status["project_id"] == "source.mp4"
    assert len(status["job_id"]) == 12
    persisted_status = tmp_path / ".smart-badminton" / "jobs" / "analysis-status.json"
    assert json.loads(persisted_status.read_text(encoding="utf-8"))["state"] == "complete"
    assert timeline.read_bytes() == original_timeline
    after = client.get("/api/project").json()
    assert after["shuttle_analysis"]["generated"] is True
    assert after["shuttle_analysis"]["current"] is True
    assert after["shuttle_analysis"]["stale"] is False
    assert after["shuttle_analysis"]["point_count"] == 1
    assert after["shuttle_analysis"]["flight_count"] == 1
    assert after["automatic_analysis"]["trajectory_boundary_protection"] is True
    annotations = client.get("/api/shuttle-annotations").json()
    assert annotations["available"] is True
    assert annotations["detections"][0]["flight_id"] == "flight-1"
    assert annotations["detections"][0]["confidence"] == 0.9

    trajectory = tmp_path / "Analysis" / "Auto" / "source" / "shuttle-track.csv"
    os.utime(trajectory, (100, 100))
    os.utime(config, (200, 200))
    outdated = client.get("/api/project").json()
    assert outdated["shuttle_analysis"]["stale"] is True
    assert outdated["shuttle_analysis"]["stale_reason"] == "球场校准已更改；原始检测已更新"
    assert outdated["automatic_analysis"]["trajectory_boundary_protection"] is False

    restarted = client.post("/api/analyze/shuttle", json={"project_id": "source.mp4", "force": False})
    assert restarted.status_code == 200
    status = restarted.json()
    deadline = time.monotonic() + 2.0
    while status["state"] == "running" and time.monotonic() < deadline:
        time.sleep(0.01)
        status = client.get("/api/analyze").json()
    assert status["state"] == "complete"
    refreshed = client.get("/api/project").json()
    assert refreshed["shuttle_analysis"]["current"] is True
    assert refreshed["shuttle_analysis"]["stale"] is False
    assert timeline.read_bytes() == original_timeline


def test_studio_restores_interrupted_background_job_as_actionable_error(tmp_path: Path) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    timeline = tmp_path / "rallies.csv"
    timeline.write_text("rally,start_seconds,end_seconds\n1,1,2\n", encoding="utf-8")
    status_path = tmp_path / ".smart-badminton" / "jobs" / "analysis-status.json"
    status_path.parent.mkdir(parents=True)
    status_path.write_text(
        json.dumps(
            {
                "job_id": "job-before-restart",
                "state": "running",
                "mode": "visual",
                "stage": "pose",
                "label": "正在分析整段人物姿态",
                "progress": 0.42,
                "started_at": 100.0,
            }
        ),
        encoding="utf-8",
    )

    state = StudioState(
        video=video,
        rallies=timeline,
        output=tmp_path / "edited.mp4",
        library=tmp_path,
    )
    client = client_for(state)

    restored = client.get("/api/analyze").json()
    assert restored["state"] == "error"
    assert restored["job_id"] == "job-before-restart"
    assert "服务停止" in restored["label"]
    assert "缓存" in restored["message"]


def test_studio_precomputes_full_video_pose_and_shuttle_overlays(tmp_path: Path, patch_studio) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    timeline = tmp_path / "rallies.csv"
    timeline.write_text("rally,start_seconds,end_seconds\n1,1,2\n", encoding="utf-8")
    config = tmp_path / "court.json"
    pose_model = tmp_path / "pose.pt"
    shuttle_model = tmp_path / "shuttle.pt"
    config.write_text("{}", encoding="utf-8")
    pose_model.write_bytes(b"pose")
    shuttle_model.write_bytes(b"shuttle")
    analysis = tmp_path / "Analysis" / "Auto" / "source"
    analysis.mkdir(parents=True)
    state_features = analysis / "smart-features.csv"
    state_features.write_text("time_seconds,legacy_activity\n0.0,0.2\n", encoding="utf-8")
    original_state_features = state_features.read_bytes()
    metadata = {
        "name": video.name,
        "duration": 10.0,
        "fps": 30.0,
        "frame_count": 300,
        "width": 1280,
        "height": 720,
    }
    patch_studio("video_metadata", lambda _path: metadata)
    patch_studio("calibration_ready", lambda _state: True)
    patch_studio("runtime_payload", lambda _state: {
            "ffmpeg": {
                "available": True,
                "path": "ffmpeg",
                "reason": None,
                "requested_encoder": "auto",
                "selected_encoder": "libx264",
                "warning": None,
            }
        },
    )

    def fake_pose(_video, start, end, output, *_args, progress_callback=None, **_kwargs):
        assert start == 0.0
        assert end == 10.0
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"pose-overlay")
        if progress_callback:
            progress_callback(1.0)
        return output

    def fake_detect(_video, _config, _model, _packages, output, **_kwargs):
        output.write_text(
            "time_seconds,frame,center_x,center_y,confidence,source_width,source_height\n"
            "1.0,30,640,360,0.9,1280,720\n",
            encoding="utf-8",
        )
        return {"sampled_frames": 150, "detections_in_roi": 1}

    def fake_track(_raw, output, _features, _video, _annotations, _config):
        output.write_text(
            "time_seconds,frame,center_x,center_y,confidence,source_width,source_height,status,flight_id\n"
            "1.0,30,640,360,0.9,1280,720,tracked,flight-1\n",
            encoding="utf-8",
        )

    def fake_audio(_video, output, **_kwargs):
        output.write_text("time_seconds,score\n", encoding="utf-8")

    def fake_features(_video, _config, output, *_args, progress_callback=None, **_kwargs):
        output.write_text(",".join(sorted(POSE_ASSOCIATION_FIELDS)) + "\n", encoding="utf-8")
        if progress_callback:
            progress_callback(1.0)

    patch_studio("audio_available", lambda _state, _video=None: True)
    patch_studio("render_pose_overlay", fake_pose)
    patch_studio("detect_shuttle", fake_detect)
    patch_studio("analyze_shuttle_trajectory", fake_track)
    patch_studio("analyze_audio", fake_audio)
    patch_studio("extract_features", fake_features)
    state = StudioState(
        video=video,
        rallies=timeline,
        output=tmp_path / "edited.mp4",
        library=tmp_path,
        config=config,
        pose_model=pose_model,
        shuttle_model=shuttle_model,
    )
    client = client_for(state)
    original_timeline = timeline.read_bytes()

    started = client.post("/api/analyze/visual", json={"project_id": "source.mp4", "force": False})
    assert started.status_code == 200
    status = started.json()
    deadline = time.monotonic() + 2.0
    while status["state"] == "running" and time.monotonic() < deadline:
        time.sleep(0.01)
        status = client.get("/api/analyze").json()

    assert status["state"] == "complete"
    assert status["mode"] == "visual"
    assert timeline.read_bytes() == original_timeline
    assert state_features.read_bytes() == original_state_features
    assert (analysis / "vision-features.csv").exists()
    project = client.get("/api/project").json()
    assert project["pose_analysis"]["current"] is True
    assert project["pose_analysis"]["url"].startswith("/media/pose-overlay/full")
    assert project["shuttle_analysis"]["current"] is True
    assert client.get("/media/pose-overlay/full").content == b"pose-overlay"

def test_studio_discovers_videos_and_seeds_working_copy(tmp_path: Path) -> None:
    video = tmp_path / "DJI_TEST_0007_D.MP4"
    video.write_bytes(b"video")
    video.with_suffix(".LRF").write_bytes(b"proxy")
    truth = tmp_path / "Standard_Answers" / "0007_rallies_ground_truth.csv"
    truth.parent.mkdir()
    truth.write_text("rally,start_seconds,end_seconds\n1,1,2\n", encoding="utf-8")

    assert discover_videos(tmp_path) == [video]
    working = ensure_working_timeline(tmp_path, video)
    assert working == tmp_path / "Analysis" / "Studio" / "DJI_TEST_0007_D.csv"
    assert "1,1,2" in working.read_text(encoding="utf-8")


def test_studio_prefers_latest_auto_cut_for_new_working_copy(tmp_path: Path) -> None:
    video = tmp_path / "Match1" / "Match1_clip1.mp4"
    video.parent.mkdir()
    video.write_bytes(b"video")
    latest = tmp_path / "Latest_Auto_Cut" / "Match1-rallies.csv"
    latest.parent.mkdir()
    latest.write_text("rally,start_seconds,end_seconds\n1,3,7\n", encoding="utf-8")
    truth = video.parent / "Metadata" / "rallies-ground-truth.csv"
    truth.parent.mkdir()
    truth.write_text("rally,start_seconds,end_seconds\n1,1,2\n", encoding="utf-8")

    working = ensure_working_timeline(tmp_path, video)

    assert working == video.parent / "Metadata" / "rallies-studio-review.csv"
    assert "1,3,7" in working.read_text(encoding="utf-8")


def test_studio_api_switches_projects_from_library(tmp_path: Path, patch_studio) -> None:
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    timeline = tmp_path / "first.csv"
    timeline.write_text("rally,start_seconds,end_seconds\n1,1,2\n", encoding="utf-8")

    def metadata(path: Path) -> dict[str, float | int | str]:
        return {"name": path.name, "duration": 10.0, "fps": 30.0, "frame_count": 300, "width": 1280, "height": 720}

    patch_studio("video_metadata", metadata)
    state = StudioState(video=first, rallies=timeline, output=tmp_path / "edited.mp4", library=tmp_path)
    client = client_for(state)
    library = client.get("/api/library").json()
    assert [item["name"] for item in library["videos"]] == ["first.mp4", "second.mp4"]

    response = client.post("/api/project/open", json={"id": "second"})
    assert response.status_code == 200
    assert response.json()["video"]["name"] == "second.mp4"
    assert state.rallies == tmp_path / "Analysis" / "Studio" / "second.csv"


def test_studio_recursively_discovers_match_sources_but_not_derivatives(tmp_path: Path) -> None:
    match = tmp_path / "Match2"
    source = match / "Match2_clip1.mp4"
    source.parent.mkdir()
    source.write_bytes(b"source")
    proxy = match / "Proxy" / "Match2_clip1_proxy_720p.mp4"
    proxy.parent.mkdir()
    proxy.write_bytes(b"proxy")
    edited = match / "Edited" / "Match2_final_1080p60.mp4"
    edited.parent.mkdir()
    edited.write_bytes(b"edited")

    assert discover_videos(tmp_path) == [source]
    layout = ProjectLayout.for_video(tmp_path, source)
    assert layout.root == match
    assert layout.working_timeline == match / "Metadata" / "rallies-studio-review.csv"
    assert layout.default_output == match / "Edited" / "Match2_final_1080p60.mp4"


def test_studio_opens_recursive_video_by_stable_relative_id(tmp_path: Path, patch_studio) -> None:
    first = tmp_path / "Match2" / "clip.mp4"
    second = tmp_path / "Match3" / "clip.mp4"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    timeline = first.parent / "Metadata" / "rallies-studio-review.csv"
    timeline.parent.mkdir()
    timeline.write_text("rally,start_seconds,end_seconds\n1,1,2\n", encoding="utf-8")
    metadata = {"name": "clip.mp4", "duration": 10.0, "fps": 30.0, "frame_count": 300, "width": 1280, "height": 720}
    patch_studio("video_metadata", lambda path: {**metadata, "name": path.name})

    state = StudioState(video=first, rallies=timeline, output=tmp_path / "edited.mp4", library=tmp_path)
    client = client_for(state)
    library = client.get("/api/library").json()
    assert [item["id"] for item in library["videos"]] == ["Match2/clip.mp4", "Match3/clip.mp4"]

    response = client.post("/api/project/open", json={"id": "Match3/clip.mp4"})
    assert response.status_code == 200
    assert state.video == second
    assert state.rallies == second.parent / "Metadata" / "rallies-studio-review.csv"


def test_studio_analysis_pipeline_publishes_editable_timeline(tmp_path: Path, patch_studio) -> None:
    video = tmp_path / "Match2" / "clip.mp4"
    video.parent.mkdir()
    video.write_bytes(b"video")
    config = tmp_path / "camera.json"
    model = tmp_path / "model.joblib"
    config.write_text("{}", encoding="utf-8")
    model.write_bytes(b"model")
    proxy = video.parent / "Proxy" / "clip_proxy_720p.mp4"
    proxy.parent.mkdir()
    proxy.write_bytes(b"proxy")

    def fake_audio(_video, output, **_kwargs):
        output.write_text("time_seconds,score\n", encoding="utf-8")

    def fake_features(_video, _config, output, *_args):
        output.write_text("time_seconds\n0\n", encoding="utf-8")

    def fake_predict(_features, _model, output):
        output.write_text("time_seconds,rally_probability\n0,1\n", encoding="utf-8")

    segment_options = {}

    def fake_segment(_features, _probabilities, output, **kwargs):
        segment_options.update(kwargs)
        output.write_text("rally,start_seconds,end_seconds\n1,1.0,4.0\n", encoding="utf-8")
        return [object()]

    patch_studio("make_proxy", lambda *_args: proxy)
    patch_studio("analyze_audio", fake_audio)
    patch_studio("extract_features", fake_features)
    patch_studio("predict_model", fake_predict)
    patch_studio("segment_rallies", fake_segment)
    patch_studio("video_metadata", lambda path: {"name": path.name, "duration": 10.0, "fps": 30.0, "frame_count": 300, "width": 1280, "height": 720},
    )
    timeline = video.parent / "Metadata" / "rallies-studio-review.csv"
    state = StudioState(video=video, rallies=timeline, output=tmp_path / "out.mp4", library=tmp_path, config=config, model=model)

    result = analyze_video(state, video, 1, 1)

    assert result["rallies"] == 1
    assert _public_start(timeline) == 1.0
    assert state.proxy == proxy
    assert segment_options["preroll"] == 0.35
    assert segment_options["postroll"] == 0.55
    assert segment_options["suppress_handoffs"] is True


def test_hybrid_routes_yolo_and_tracknet_package_paths_separately(tmp_path: Path, patch_studio) -> None:
    video = tmp_path / "video.mp4"
    config = tmp_path / "camera.json"
    yolo_model = tmp_path / "shuttle.pt"
    tracknet_model = tmp_path / "tracknet.pt"
    inpaint_model = tmp_path / "inpaint.pt"
    analysis_root = tmp_path / "analysis"
    for path in (video, config, yolo_model, tracknet_model, inpaint_model):
        path.write_bytes(b"test")
    analysis_root.mkdir()
    yolo_packages = tmp_path / "yolo-packages"
    tracknet_packages = tmp_path / "cuda-packages"
    captured = {}

    def fake_yolo(*args, **_kwargs):
        captured["yolo_packages"] = args[3]
        args[4].write_text("time_seconds,center_x,center_y\n", encoding="utf-8")
        return {"points": 0}

    def fake_tracknet(*args, **_kwargs):
        captured["tracknet_packages"] = args[5]
        args[3].write_text("time_seconds,center_x,center_y\n", encoding="utf-8")
        return {"points": 0}

    def fake_fuse(*args, **_kwargs):
        args[3].write_text("time_seconds,center_x,center_y\n", encoding="utf-8")
        return {"points": 0}

    patch_studio("detect_shuttle", fake_yolo)
    patch_studio("detect_tracknet", fake_tracknet)
    patch_studio("fuse_shuttle_detections", fake_fuse)
    state = StudioState(
        video=video,
        rallies=tmp_path / "rallies.csv",
        output=tmp_path / "out.mp4",
        config=config,
        shuttle_model=yolo_model,
        tracknet_model=tracknet_model,
        inpaint_model=inpaint_model,
        shuttle_mode="hybrid",
        packages=yolo_packages,
        tracknet_packages=tracknet_packages,
    )

    run_shuttle_detection(state, video, analysis_root, force=True)

    assert captured == {
        "yolo_packages": yolo_packages,
        "tracknet_packages": tracknet_packages,
    }


def _public_start(path: Path) -> float:
    return float(path.read_text(encoding="utf-8").splitlines()[1].split(",")[1])


def test_studio_tutorial_supports_english_and_chinese(tmp_path: Path) -> None:
    state = StudioState(video=tmp_path / "video.mp4", rallies=tmp_path / "rallies.csv", output=tmp_path / "edited.mp4")
    client = client_for(state)
    english = client.get("/api/tutorial?language=en")
    chinese = client.get("/api/tutorial?language=zh")
    assert english.status_code == chinese.status_code == 200
    assert "# Smart Badminton Studio manual" in english.json()["markdown"]
    assert "完整使用教程" in chinese.json()["markdown"]
    assert client.get("/api/tutorial").json() == chinese.json()


def test_studio_rejects_requests_for_a_foreign_host(tmp_path: Path) -> None:
    state = StudioState(video=tmp_path / "video.mp4", rallies=tmp_path / "rallies.csv", output=tmp_path / "edited.mp4")
    rebinding = TestClient(studio.create_studio_app(state), base_url="http://attacker.example")
    assert rebinding.get("/api/health").status_code == 400
    assert client_for(state).get("/api/health").status_code == 200


def test_studio_refuses_to_analyze_a_video_without_audio(tmp_path: Path, patch_studio) -> None:
    first, second = tmp_path / "first.mp4", tmp_path / "second.mp4"
    for video in (first, second):
        video.write_bytes(b"video")
    timeline = tmp_path / "rallies.csv"
    timeline.write_text("rally,start_seconds,end_seconds\n1,1,2\n", encoding="utf-8")
    config, model = tmp_path / "court.json", tmp_path / "model.joblib"
    config.write_text("{}", encoding="utf-8")
    model.write_bytes(b"model")
    metadata = {"name": "first.mp4", "duration": 10.0, "fps": 30.0, "frame_count": 300, "width": 1280, "height": 720}
    patch_studio("video_metadata", lambda _path: metadata)
    patch_studio("calibration_ready", lambda _state: True)
    patch_studio("ffmpeg_available", lambda _state: True)
    patch_studio("audio_available", lambda _state, video=None: False)
    state = StudioState(
        video=first, rallies=timeline, output=tmp_path / "edited.mp4", library=tmp_path, config=config, model=model
    )
    client = client_for(state)

    automatic = client.get("/api/project").json()["automatic_analysis"]
    assert automatic["configured"] is False
    assert automatic["audio_available"] is False
    assert automatic["configuration_issue"] == "此视频没有音轨，自动分析需要击球声，无法分析"
    for path in ("/api/analyze", "/api/analyze/visual"):
        response = client.post(path, json={"project_id": "first.mp4"})
        assert response.status_code == 400
        assert response.json()["detail"] == "此视频没有音轨，自动分析需要击球声，无法分析"
    batch = client.post("/api/analyze/batch", json={})
    assert batch.status_code == 400
    assert batch.json()["detail"] == "这些视频没有音轨，无法自动分析：first.mp4、second.mp4"
    assert state.analysis_status["state"] == "idle"


def test_studio_releases_the_analysis_lock_when_options_are_invalid(tmp_path: Path, patch_studio) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    timeline = tmp_path / "rallies.csv"
    timeline.write_text("rally,start_seconds,end_seconds\n1,1,2\n", encoding="utf-8")
    patch_studio("audio_available", lambda _state, _video=None: True)
    state = StudioState(video=video, rallies=timeline, output=tmp_path / "edited.mp4", library=tmp_path)
    client = client_for(state)

    rejected = client.post("/api/analyze", json={"preroll": "not-a-number"})

    assert rejected.status_code == 400
    assert state.analysis_lock.locked() is False


def test_studio_starts_when_the_saved_job_status_is_unreadable(tmp_path: Path) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    status_path = tmp_path / ".smart-badminton" / "jobs" / "analysis-status.json"
    status_path.parent.mkdir(parents=True)
    state = StudioState(video=video, rallies=tmp_path / "rallies.csv", output=tmp_path / "edited.mp4", library=tmp_path)
    for content in ('{"state": "runn', "[1, 2]"):
        status_path.write_text(content, encoding="utf-8")
        assert client_for(state).get("/api/analyze").json() == {"state": "idle"}


def test_studio_releases_the_analysis_lock_when_job_status_cannot_be_saved(tmp_path: Path, patch_studio) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    timeline = tmp_path / "rallies.csv"
    timeline.write_text("rally,start_seconds,end_seconds\n1,1,2\n", encoding="utf-8")
    (tmp_path / ".smart-badminton").write_text("not a directory", encoding="utf-8")
    patch_studio("audio_available", lambda _state, _video=None: True)
    state = StudioState(video=video, rallies=timeline, output=tmp_path / "edited.mp4", library=tmp_path)

    rejected = client_for(state).post("/api/analyze", json={})

    assert rejected.status_code == 400
    assert state.analysis_lock.locked() is False


def test_studio_output_filename_cannot_escape_the_chosen_folder(tmp_path: Path) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    exports = tmp_path / "exports"
    exports.mkdir()
    state = StudioState(video=video, rallies=tmp_path / "rallies.csv", output=tmp_path / "edited.mp4", library=tmp_path)
    client = client_for(state)

    for filename in ("../../escaped.mp4", str(tmp_path / "elsewhere" / "absolute.mp4")):
        response = client.put("/api/output", json={"directory": str(exports), "filename": filename})
        assert response.status_code == 200
        assert state.output.parent == exports.resolve()


def test_studio_rejects_invalid_input_before_changing_anything(tmp_path: Path, patch_studio) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    timeline = tmp_path / "rallies.csv"
    timeline.write_text("rally,start_seconds,end_seconds\n1,1,2\n", encoding="utf-8")
    original_timeline = timeline.read_bytes()
    metadata = {"name": video.name, "duration": 10.0, "fps": 30.0, "frame_count": 300, "width": 1280, "height": 720}
    patch_studio("video_metadata", lambda _path: metadata)
    patch_studio("audio_available", lambda _state, _video=None: True)
    state = StudioState(video=video, rallies=timeline, output=tmp_path / "edited.mp4", library=tmp_path)
    client = client_for(state)

    for segments in (
        [{"start": -1, "end": 2}],
        [{"start": 1, "end": 1.01}],
        [{"start": 3, "end": 2}],
        [{"start": 1, "end": 11}],
        [{"start": "nan", "end": 2}],
        [{"start": 1, "end": 3}, {"start": 2, "end": 4}],
    ):
        response = client.put("/api/timeline", json={"project_id": "source.mp4", "segments": segments})
        assert response.status_code == 400, segments
    assert timeline.read_bytes() == original_timeline

    regions = client.get("/api/calibration").json()["regions"]
    square = [[0.1, 0.1], [0.8, 0.1], [0.8, 0.8], [0.1, 0.8]]
    for region in regions:
        if region["required"]:
            region["points"] = square
    for bad in ([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]], [[0.1, 0.1], [1.4, 0.1], [0.5, 0.8]], [[0.1, 0.1], [0.8, 0.1]]):
        regions[0]["points"] = bad
        response = client.put("/api/calibration", json={"project_id": "source.mp4", "regions": regions})
        assert response.status_code == 400, bad
    regions[0]["points"] = square
    corners = next(region for region in regions if region["id"] == "court_corners")
    corners["points"] = [[0.1, 0.1], [0.8, 0.8], [0.8, 0.1], [0.1, 0.8]]  # self-intersecting order
    assert client.put("/api/calibration", json={"project_id": "source.mp4", "regions": regions}).status_code == 400
    assert state.config is None

    empty = tmp_path / "empty"
    empty.mkdir()
    for path in (empty, tmp_path / "missing"):
        assert client.post("/api/library", json={"path": str(path)}).status_code == 400
    assert state.library == tmp_path

    for options in ({"preroll": -10}, {"postroll": 100}, {"preroll": "nan"}):
        assert client.post("/api/analyze", json=options).status_code == 400, options
    assert state.analysis_lock.locked() is False
    assert state.analysis_options["preroll"] == 0.35
