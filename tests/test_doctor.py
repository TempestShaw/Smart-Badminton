import json
from pathlib import Path

from smart_badminton.doctor import (
    choose_video_encoder,
    initialize_project,
    inspect_model_license,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_choose_video_encoder_falls_back_to_cpu(monkeypatch) -> None:
    monkeypatch.setattr("smart_badminton.encoding.available_ffmpeg_encoders", lambda _path=None: {"libx264"})

    selected, warning = choose_video_encoder("h264_nvenc")

    assert selected == "libx264"
    assert "falling back" in str(warning)


def test_model_license_requires_matching_checksum(tmp_path: Path) -> None:
    model = tmp_path / "shuttle.pt"
    model.write_bytes(b"checkpoint")
    unverified = inspect_model_license(model)
    assert unverified["redistributable"] is False

    sidecar = model.with_suffix(".pt.license.json")
    sidecar.write_text(
        json.dumps(
            {
                "name": "synthetic",
                "source_url": "https://example.test/source",
                "license": "Apache-2.0",
                "redistributable": True,
                "sha256": unverified["sha256"],
            }
        ),
        encoding="utf-8",
    )

    verified = inspect_model_license(model)
    assert verified["status"] == "verified-redistributable"
    assert verified["redistributable"] is True


def test_published_rally_model_has_verified_license() -> None:
    model = REPOSITORY_ROOT / "models" / "rally-state-final-v4-frozen.joblib"

    verified = inspect_model_license(model)

    assert verified["status"] == "verified-redistributable"
    assert verified["manifest"]["license"] == "Apache-2.0"


def test_initialize_project_creates_first_run_layout(tmp_path: Path) -> None:
    result = initialize_project(tmp_path / "project")

    assert Path(result["calibration"]).exists()
    assert Path(result["license_template"]).exists()
    assert (tmp_path / "project" / "Analysis").is_dir()
