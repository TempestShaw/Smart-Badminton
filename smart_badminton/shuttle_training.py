from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from .doctor import inspect_model_license

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Training manifest must be a JSON object")
    return payload


def _source_paths(manifest_path: Path, source: dict[str, Any]) -> tuple[Path, Path]:
    root = manifest_path.parent

    def resolve(value: Any) -> Path:
        candidate = Path(str(value))
        return candidate if candidate.is_absolute() else (root / candidate).resolve()

    return resolve(source.get("image_directory", "")), resolve(source.get("label_directory", ""))


def _validate_label(path: Path) -> list[str]:
    errors: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{path}:{line_number}: expected YOLO class/x/y/width/height")
            continue
        try:
            class_id = int(parts[0])
            values = [float(value) for value in parts[1:]]
        except ValueError:
            errors.append(f"{path}:{line_number}: non-numeric YOLO label")
            continue
        if class_id != 0:
            errors.append(f"{path}:{line_number}: shuttle must use class 0")
        if any(value < 0 or value > 1 for value in values) or values[2] <= 0 or values[3] <= 0:
            errors.append(f"{path}:{line_number}: box coordinates must be normalized and non-empty")
    return errors


def audit_shuttle_dataset(manifest_path: Path) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        payload = _manifest(manifest_path)
    except (OSError, json.JSONDecodeError, TypeError) as error:
        return {"ready": False, "errors": [str(error)], "warnings": [], "cameras": {}, "images": 0}
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        return {
            "ready": False,
            "errors": ["Manifest needs a non-empty sources array"],
            "warnings": [],
            "cameras": {},
            "images": 0,
        }
    cameras: Counter[str] = Counter()
    seen_ids: set[str] = set()
    total_images = 0
    resolved_sources: list[dict[str, Any]] = []
    for index, raw_source in enumerate(sources, 1):
        if not isinstance(raw_source, dict):
            errors.append(f"source {index}: expected an object")
            continue
        source_id = str(raw_source.get("id", "")).strip()
        camera_id = str(raw_source.get("camera_id", "")).strip()
        if not source_id or source_id in seen_ids:
            errors.append(f"source {index}: id is missing or duplicated")
        seen_ids.add(source_id)
        if not camera_id:
            errors.append(f"source {source_id or index}: camera_id is required")
        required_rights = ("source_url", "license", "redistributable")
        missing_rights = [name for name in required_rights if raw_source.get(name) in {None, ""}]
        if missing_rights:
            errors.append(f"source {source_id or index}: missing rights fields {', '.join(missing_rights)}")
        if raw_source.get("redistributable") is not True:
            errors.append(f"source {source_id or index}: redistribution permission is not confirmed")
        image_directory, label_directory = _source_paths(manifest_path, raw_source)
        if not image_directory.is_dir() or not label_directory.is_dir():
            errors.append(f"source {source_id or index}: image or label directory does not exist")
            continue
        if image_directory.name.lower() != "images" or label_directory.name.lower() != "labels":
            errors.append(
                f"source {source_id or index}: directories must be named images and labels for YOLO path mapping"
            )
        images = sorted(path for path in image_directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            errors.append(f"source {source_id or index}: no supported images")
            continue
        paired = []
        for image in images:
            label = label_directory / f"{image.stem}.txt"
            if not label.exists():
                errors.append(f"source {source_id or index}: missing label for {image.name}")
                continue
            errors.extend(_validate_label(label))
            paired.append((image, label))
        cameras[camera_id] += len(paired)
        total_images += len(paired)
        resolved_sources.append(
            {
                **raw_source,
                "image_directory": str(image_directory),
                "label_directory": str(label_directory),
                "images": [str(image) for image, _label in paired],
            }
        )
    if len(cameras) < 2:
        errors.append("At least two distinct camera_id values are required for a multi-angle model")
    if total_images < 20:
        warnings.append("Fewer than 20 labelled frames; this is only enough to test the pipeline")
    if any(count < 5 for count in cameras.values()):
        warnings.append("Each camera should contribute at least five labelled frames")
    return {
        "ready": not errors,
        "errors": errors,
        "warnings": warnings,
        "cameras": dict(sorted(cameras.items())),
        "images": total_images,
        "sources": resolved_sources,
        "manifest": str(manifest_path),
    }


def prepare_shuttle_dataset(
    manifest_path: Path,
    output_directory: Path,
    validation_camera: str | None = None,
) -> dict[str, Any]:
    audit = audit_shuttle_dataset(manifest_path)
    if not audit["ready"]:
        raise ValueError("Dataset audit failed: " + "; ".join(audit["errors"]))
    cameras = sorted(audit["cameras"])
    validation_camera = validation_camera or cameras[-1]
    if validation_camera not in cameras:
        raise ValueError(f"Unknown validation camera: {validation_camera}")
    train_images: list[str] = []
    validation_images: list[str] = []
    for source in audit["sources"]:
        destination = validation_images if source["camera_id"] == validation_camera else train_images
        destination.extend(source["images"])
    if not train_images or not validation_images:
        raise ValueError("Camera-disjoint training and validation splits must both be non-empty")
    output_directory.mkdir(parents=True, exist_ok=True)
    train_file = output_directory / "train.txt"
    validation_file = output_directory / "validation.txt"
    train_file.write_text("\n".join(Path(item).as_posix() for item in train_images) + "\n", encoding="utf-8")
    validation_file.write_text("\n".join(Path(item).as_posix() for item in validation_images) + "\n", encoding="utf-8")
    dataset_yaml = output_directory / "shuttle-dataset.yaml"
    dataset_yaml.write_text(
        f"train: '{train_file.as_posix()}'\nval: '{validation_file.as_posix()}'\nnc: 1\nnames: ['shuttle']\n",
        encoding="utf-8",
    )
    report = {
        **audit,
        "split_strategy": "camera-disjoint",
        "validation_camera": validation_camera,
        "training_images": len(train_images),
        "validation_images": len(validation_images),
        "dataset_yaml": str(dataset_yaml),
    }
    (output_directory / "dataset-audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def train_shuttle_model(
    manifest_path: Path,
    output_model: Path,
    base_model: Path,
    work_directory: Path,
    validation_camera: str | None = None,
    epochs: int = 80,
    image_size: int = 1280,
    device: str = "",
    checkpoint_license: str = "AGPL-3.0",
) -> dict[str, Any]:
    if not checkpoint_license.strip() or checkpoint_license.lower() in {"unknown", "proprietary"}:
        raise ValueError("An explicit redistributable checkpoint license is required")
    base_license = inspect_model_license(base_model)
    if base_license.get("redistributable") is not True:
        raise ValueError("Base model needs a checksum-matched redistributable .license.json sidecar")
    base_license_name = str(base_license.get("manifest", {}).get("license", ""))
    if "agpl" in base_license_name.lower() and "agpl" not in checkpoint_license.lower():
        raise ValueError("An AGPL base checkpoint cannot be relabelled under a non-AGPL checkpoint license")
    report = prepare_shuttle_dataset(manifest_path, work_directory / "dataset", validation_camera)
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError('Install the vision extra with: pip install -e ".[vision]"') from error
    model = YOLO(str(base_model))
    result = model.train(
        data=report["dataset_yaml"],
        epochs=epochs,
        imgsz=image_size,
        device=device or None,
        project=str(work_directory / "runs"),
        name="shuttle-multi-angle",
    )
    best = Path(str(result.save_dir)) / "weights" / "best.pt"
    if not best.exists():
        raise RuntimeError(f"Training finished without a best checkpoint: {best}")
    output_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, output_model)
    checksum = hashlib.sha256(output_model.read_bytes()).hexdigest().upper()
    payload = _manifest(manifest_path)
    license_manifest = {
        "name": output_model.stem,
        "source_url": payload.get("project_url", "local-training-manifest"),
        "license": checkpoint_license,
        "redistributable": True,
        "sha256": checksum,
        "base_model": {
            "path": str(base_model),
            "sha256": base_license["sha256"],
            "source_url": base_license["manifest"]["source_url"],
            "license": base_license_name,
        },
        "training_sources": [
            {
                "id": source["id"],
                "camera_id": source["camera_id"],
                "source_url": source["source_url"],
                "license": source["license"],
            }
            for source in payload["sources"]
        ],
        "validation_camera": report["validation_camera"],
        "notes": "Camera-disjoint validation; classification remains candidate-only in monocular footage.",
    }
    sidecar = output_model.with_suffix(output_model.suffix + ".license.json")
    sidecar.write_text(json.dumps(license_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {**report, "model": str(output_model), "license_manifest": str(sidecar), "sha256": checksum}
