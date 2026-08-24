import hashlib
import json
from pathlib import Path

from smart_badminton import model_assets
from smart_badminton.model_assets import (
    DOWNLOADABLE_MODELS,
    MODEL_FILENAMES,
    DownloadableModel,
    find_model,
    install_models,
    resolve_model,
)


def test_download_manifest_has_unique_roles_and_checksum_bound_assets() -> None:
    assert {asset.role for asset in DOWNLOADABLE_MODELS} == {"pose", "shuttle", "tracknet", "inpaint"}
    assert len({asset.filename for asset in DOWNLOADABLE_MODELS}) == len(DOWNLOADABLE_MODELS)
    assert all(len(asset.sha256) == 64 for asset in DOWNLOADABLE_MODELS)
    assert all(asset.license in {"AGPL-3.0-only", "MIT"} for asset in DOWNLOADABLE_MODELS)


def test_find_model_prefers_configured_directory(tmp_path: Path, monkeypatch) -> None:
    model_directory = tmp_path / "models"
    model_directory.mkdir()
    model = model_directory / MODEL_FILENAMES["pose"]
    model.write_bytes(b"pose")
    monkeypatch.setenv("SMART_BADMINTON_MODELS_DIR", str(model_directory))

    assert find_model("pose") == model.resolve()
    assert resolve_model(None, "pose") == model.resolve()


def test_local_download_manifest_matches_published_files() -> None:
    root = Path(__file__).resolve().parents[1] / "models"
    for asset in DOWNLOADABLE_MODELS:
        model = root / asset.filename
        if not model.exists():
            continue
        assert hashlib.sha256(model.read_bytes()).hexdigest() == asset.sha256


def test_install_models_downloads_and_writes_license_sidecar(tmp_path: Path, monkeypatch) -> None:
    content = b"model"
    asset = DownloadableModel(
        role="pose",
        name="Synthetic pose",
        filename="pose.pt",
        sha256=hashlib.sha256(content).hexdigest(),
        source_url="https://example.test/source",
        license="MIT",
        download_url="https://example.test/pose.pt",
    )
    monkeypatch.setattr(model_assets, "DOWNLOADABLE_MODELS", (asset,))
    monkeypatch.setattr(model_assets, "_download", lambda _url, output: output.write_bytes(content))

    result = install_models(tmp_path / "models")

    model = tmp_path / "models" / "pose.pt"
    assert model.read_bytes() == content
    assert json.loads(model.with_suffix(".pt.license.json").read_text(encoding="utf-8"))["license"] == "MIT"
    assert result["installed"] == ["pose.pt"]
