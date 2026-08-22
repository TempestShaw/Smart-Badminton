import json
from pathlib import Path

import pytest

from smart_badminton.shuttle_training import audit_shuttle_dataset, prepare_shuttle_dataset


def _dataset(tmp_path: Path, redistributable: bool = True) -> Path:
    sources = []
    for camera in ("phone-low", "tripod-high"):
        images = tmp_path / camera / "images"
        labels = tmp_path / camera / "labels"
        images.mkdir(parents=True)
        labels.mkdir()
        for index in range(5):
            (images / f"frame-{index}.jpg").write_bytes(b"synthetic")
            (labels / f"frame-{index}.txt").write_text("0 0.5 0.4 0.02 0.03\n", encoding="utf-8")
        sources.append(
            {
                "id": camera,
                "camera_id": camera,
                "image_directory": str(images),
                "label_directory": str(labels),
                "source_url": f"https://example.test/{camera}",
                "license": "CC-BY-4.0",
                "redistributable": redistributable,
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"sources": sources}), encoding="utf-8")
    return manifest


def test_dataset_audit_requires_rights_and_multiple_cameras(tmp_path: Path) -> None:
    manifest = _dataset(tmp_path)

    audit = audit_shuttle_dataset(manifest)

    assert audit["ready"] is True
    assert audit["images"] == 10
    assert set(audit["cameras"]) == {"phone-low", "tripod-high"}


def test_prepare_uses_camera_disjoint_validation(tmp_path: Path) -> None:
    report = prepare_shuttle_dataset(_dataset(tmp_path), tmp_path / "prepared", "phone-low")

    assert report["split_strategy"] == "camera-disjoint"
    assert report["training_images"] == 5
    assert report["validation_images"] == 5
    assert Path(report["dataset_yaml"]).exists()


def test_dataset_audit_blocks_unconfirmed_redistribution(tmp_path: Path) -> None:
    manifest = _dataset(tmp_path, redistributable=False)

    audit = audit_shuttle_dataset(manifest)

    assert audit["ready"] is False
    assert any("redistribution permission" in error for error in audit["errors"])
    with pytest.raises(ValueError, match="Dataset audit failed"):
        prepare_shuttle_dataset(manifest, tmp_path / "prepared")
