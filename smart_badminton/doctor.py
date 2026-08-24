from __future__ import annotations

import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import platform
import sys
from importlib.resources import files
from pathlib import Path
from typing import Any

from .encoding import available_ffmpeg_encoders, choose_video_encoder, choose_working_video_encoder
from .io import resolve_ffmpeg

__all__ = ["available_ffmpeg_encoders", "choose_video_encoder", "doctor_report", "initialize_project", "inspect_model_license"]


def model_license_path(model: Path) -> Path:
    return model.with_suffix(model.suffix + ".license.json")


def inspect_model_license(model: Path | None) -> dict[str, Any]:
    if model is None:
        return {"configured": False, "status": "not-configured"}
    result: dict[str, Any] = {"configured": True, "path": str(model), "exists": model.exists()}
    if not model.exists():
        return {**result, "status": "missing"}
    result["sha256"] = hashlib.sha256(model.read_bytes()).hexdigest().upper()
    sidecar = model_license_path(model)
    result["license_manifest"] = str(sidecar)
    if not sidecar.exists():
        return {**result, "status": "local-only-unverified", "redistributable": False}
    try:
        manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {**result, "status": "invalid-license-manifest", "error": str(error), "redistributable": False}
    required = {"name", "source_url", "license", "redistributable", "sha256"}
    missing = sorted(required - manifest.keys())
    checksum_matches = str(manifest.get("sha256", "")).upper() == result["sha256"]
    redistributable = bool(manifest.get("redistributable")) and checksum_matches and not missing
    return {
        **result,
        "status": "verified-redistributable" if redistributable else "verified-local-only",
        "redistributable": redistributable,
        "checksum_matches": checksum_matches,
        "missing_fields": missing,
        "manifest": manifest,
    }


def doctor_report(
    config: Path | None = None,
    rally_model: Path | None = None,
    pose_model: Path | None = None,
    shuttle_model: Path | None = None,
    ffmpeg: Path | None = None,
    encoder: str = "auto",
    packages: Path | None = None,
    tracknet_model: Path | None = None,
    inpaint_model: Path | None = None,
) -> dict[str, Any]:
    def package_available(name: str) -> bool:
        if importlib.util.find_spec(name) is not None:
            return True
        return bool(packages and packages.is_dir() and importlib.machinery.PathFinder.find_spec(name, [str(packages)]))

    checks: dict[str, Any] = {
        "python": {"version": platform.python_version(), "supported": sys.version_info >= (3, 10)},
        "config": {"configured": config is not None, "exists": bool(config and config.exists())},
        "packages": {
            name: package_available(name)
            for name in ("cv2", "numpy", "pandas", "sklearn", "fastapi", "ultralytics", "torch")
        },
        "package_search_path": str(packages) if packages is not None else None,
        "models": {
            "rally": inspect_model_license(rally_model),
            "pose": inspect_model_license(pose_model),
            "shuttle": inspect_model_license(shuttle_model),
            "tracknet": inspect_model_license(tracknet_model),
            "inpaint": inspect_model_license(inpaint_model),
        },
    }
    torch_runtime: dict[str, Any] = {"available": False}
    if checks["packages"]["torch"]:
        inserted = False
        try:
            if packages is not None and str(packages) not in sys.path:
                sys.path.append(str(packages))
                inserted = True
            torch = importlib.import_module("torch")
            cuda_available = bool(torch.cuda.is_available())
            torch_runtime = {
                "available": True,
                "version": str(torch.__version__),
                "cuda_available": cuda_available,
                "cuda_version": str(torch.version.cuda) if torch.version.cuda else None,
                "device": torch.cuda.get_device_name(0) if cuda_available else "cpu",
            }
        except Exception as error:  # noqa: BLE001 - doctor reports import failures instead of raising
            torch_runtime = {"available": False, "error": str(error)}
        finally:
            if inserted:
                sys.path.remove(str(packages))
    checks["torch"] = torch_runtime
    try:
        executable = resolve_ffmpeg(ffmpeg)
        selected, warning, probe_failures = choose_working_video_encoder(encoder, executable)
        checks["ffmpeg"] = {
            "available": True,
            "path": str(executable),
            "requested_encoder": encoder,
            "selected_encoder": selected,
            "warning": warning,
            "probe_failures": probe_failures,
        }
    except (FileNotFoundError, RuntimeError, OSError) as error:
        checks["ffmpeg"] = {"available": False, "error": str(error)}
    checks["ready_for_basic_editing"] = bool(
        checks["python"]["supported"]
        and checks["ffmpeg"].get("available")
        and checks["packages"]["cv2"]
    )
    checks["ready_for_automatic_analysis"] = bool(
        checks["ready_for_basic_editing"]
        and checks["config"]["exists"]
        and rally_model is not None
        and rally_model.exists()
    )
    checks["ready_for_tracknet"] = bool(
        checks["ready_for_basic_editing"]
        and checks["config"]["exists"]
        and tracknet_model is not None
        and tracknet_model.exists()
        and torch_runtime.get("available")
    )
    checks["release_safe"] = all(
        not model["configured"] or bool(model.get("redistributable"))
        for model in checks["models"].values()
    )
    return checks


def initialize_project(directory: Path) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    calibration = directory / "Calibration" / "court-config.json"
    calibration.parent.mkdir(parents=True, exist_ok=True)
    if not calibration.exists():
        source = Path(str(files("smart_badminton").joinpath("configs/example_fixed_camera.json")))
        calibration.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    for name in ("Analysis", "Metadata", "Edited", "Proxy", "Models"):
        (directory / name).mkdir(exist_ok=True)
    template = directory / "Models" / "model.pt.license.json.example"
    if not template.exists():
        template.write_text(
            json.dumps(
                {
                    "name": "replace-me",
                    "source_url": "https://example.invalid/model-source",
                    "license": "replace-me",
                    "redistributable": False,
                    "sha256": "replace-with-model-sha256",
                    "notes": "Do not mark redistributable until both checkpoint and training-data rights are verified.",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return {"project": str(directory), "calibration": str(calibration), "license_template": str(template)}
