from pathlib import Path

from smart_badminton.model_registry import ModelRegistry
from smart_badminton.project_layout import LibraryLayout, ProjectLayout


def test_match_project_layout_keeps_user_and_generated_files_separate(tmp_path: Path) -> None:
    video = tmp_path / "Match5" / "Match5_clip1.mp4"
    layout = ProjectLayout.for_video(tmp_path, video)

    assert layout.root == tmp_path / "Match5"
    assert layout.working_timeline == tmp_path / "Match5" / "Metadata" / "rallies-studio-review.csv"
    assert layout.latest_auto_cut == tmp_path / "Latest_Auto_Cut" / "Match5-rallies.csv"
    assert layout.analysis.features == tmp_path / "Match5" / "Analysis" / "smart-features.csv"
    assert layout.analysis.vision_features == tmp_path / "Match5" / "Analysis" / "vision-features.csv"
    assert layout.analysis.shuttle_track == tmp_path / "Match5" / "Analysis" / "shuttle-track.csv"
    assert layout.analysis.score_labeling == tmp_path / "Match5" / "Analysis" / "Score_Labeling"
    assert layout.default_output == tmp_path / "Match5" / "Edited" / "Match5_final_1080p60.mp4"


def test_loose_video_gets_an_isolated_analysis_directory(tmp_path: Path) -> None:
    video = tmp_path / "recording.mp4"
    layout = ProjectLayout.for_video(tmp_path, video)

    assert layout.root == tmp_path
    assert layout.analysis.root == tmp_path / "Analysis" / "Auto" / "recording"
    assert layout.analysis.score_labeling == tmp_path / "Analysis" / "Auto" / "recording" / "Score_Labeling"
    assert layout.score_corrections == tmp_path / "Metadata" / "recording-score-corrections.csv"


def test_library_runtime_files_are_isolated_from_match_data(tmp_path: Path) -> None:
    layout = LibraryLayout(tmp_path)

    assert layout.models == tmp_path / ".smart-badminton" / "models"
    assert layout.analysis_status == tmp_path / ".smart-badminton" / "jobs" / "analysis-status.json"


def test_model_registry_reports_roles_without_credentials(tmp_path: Path) -> None:
    rally = tmp_path / "rally.joblib"
    tracknet = tmp_path / "tracknet.pt"
    rally.write_bytes(b"model")
    tracknet.write_bytes(b"model")
    registry = ModelRegistry(
        rally_state=rally,
        tracknet=tracknet,
        vision_models=("provider/model-a", "provider/model-b"),
    )

    payload = registry.public_payload()
    assert payload["configured_count"] == 4
    assert payload["local_count"] == 2
    assert payload["remote_count"] == 2
    assert {entry["role"] for entry in payload["entries"]} >= {
        "rally_segmentation",
        "shuttle_tracking",
        "score_labeling",
    }
    assert "api_key" not in str(payload)
