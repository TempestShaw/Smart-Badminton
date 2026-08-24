from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

MODEL_FILENAMES = {
    "rally": "rally-state-final-v4-frozen.joblib",
    "pose": "yolo11n-pose.pt",
    "shuttle": "yolo11s-ball.pt",
    "tracknet": "TrackNet_best.pt",
    "inpaint": "InpaintNet_best.pt",
}


@dataclass(frozen=True)
class DownloadableModel:
    role: str
    name: str
    filename: str
    sha256: str
    source_url: str
    license: str
    download_url: str
    archive_member: str | None = None
    notes: str | None = None

    def license_manifest(self) -> dict[str, object]:
        manifest: dict[str, object] = {
            "name": self.name,
            "source_url": self.source_url,
            "license": self.license,
            "redistributable": True,
            "sha256": self.sha256,
        }
        if self.notes:
            manifest["notes"] = self.notes
        return manifest


TRACKNET_ARCHIVE_URL = (
    "https://drive.usercontent.google.com/download?"
    "id=1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA&export=download&confirm=t"
)

DOWNLOADABLE_MODELS = (
    DownloadableModel(
        role="pose",
        name="Ultralytics YOLO11n Pose",
        filename=MODEL_FILENAMES["pose"],
        sha256="869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0",
        source_url="https://github.com/ultralytics/assets/releases/tag/v8.3.0",
        license="AGPL-3.0-only",
        download_url="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt",
    ),
    DownloadableModel(
        role="shuttle",
        name="Good-Badminton YOLO11s Shuttlecock",
        filename=MODEL_FILENAMES["shuttle"],
        sha256="c21113960fadce7f96b9f70f86802142482775405f60a9761328491868edf31b",
        source_url="https://github.com/yo-WASSUP/Good-Badminton/releases/tag/v0.1.0",
        license="AGPL-3.0-only",
        download_url="https://github.com/yo-WASSUP/Good-Badminton/releases/download/v0.1.0/yolo11s-ball.pt",
        notes=(
            "Good-Badminton declares this checkpoint Apache-2.0; its embedded Ultralytics metadata declares "
            "AGPL-3.0, so the bundle applies AGPL-3.0."
        ),
    ),
    DownloadableModel(
        role="tracknet",
        name="TrackNetV3 TrackNet",
        filename=MODEL_FILENAMES["tracknet"],
        sha256="df867641a02712b021f04548ff4b1208ddfdb47f629ab2094ceb978667e83b1a",
        source_url="https://github.com/qaz812345/TrackNetV3",
        license="MIT",
        download_url=TRACKNET_ARCHIVE_URL,
        archive_member="ckpts/TrackNet_best.pt",
    ),
    DownloadableModel(
        role="inpaint",
        name="TrackNetV3 InpaintNet",
        filename=MODEL_FILENAMES["inpaint"],
        sha256="5749b66b8002f3ad9e0af841604004706fc796df30599e6bf01952696009688c",
        source_url="https://github.com/qaz812345/TrackNetV3",
        license="MIT",
        download_url=TRACKNET_ARCHIVE_URL,
        archive_member="ckpts/InpaintNet_best.pt",
    ),
)


def model_directories(library: Path | None = None) -> tuple[Path, ...]:
    configured = os.environ.get("SMART_BADMINTON_MODELS_DIR")
    candidates = [Path(configured).expanduser()] if configured else []
    if library is not None:
        candidates.append(library / ".smart-badminton" / "models")
    candidates.extend(
        (
            Path.cwd() / "models",
            Path(__file__).resolve().parents[1] / "models",
            Path(os.sys.prefix) / "share" / "smart-badminton" / "models",
        )
    )
    unique: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in unique:
            unique.append(resolved)
    return tuple(unique)


def find_model(role: str, library: Path | None = None) -> Path | None:
    filename = MODEL_FILENAMES[role]
    return next((directory / filename for directory in model_directories(library) if (directory / filename).is_file()), None)


def resolve_model(explicit: Path | None, role: str, library: Path | None = None) -> Path | None:
    return explicit.expanduser().resolve() if explicit is not None else find_model(role, library)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, output: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "smart-badminton/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response, output.open("wb") as stream:
        shutil.copyfileobj(response, stream, length=1024 * 1024)


def install_models(destination: Path, force: bool = False) -> dict[str, object]:
    destination = destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    installed: list[str] = []
    reused: list[str] = []
    archives: dict[str, Path] = {}
    with tempfile.TemporaryDirectory(prefix=".smart-badminton-models-", dir=destination) as temporary:
        temporary_root = Path(temporary)
        for asset in DOWNLOADABLE_MODELS:
            target = destination / asset.filename
            if target.exists() and _sha256(target) == asset.sha256 and not force:
                reused.append(asset.filename)
            else:
                if asset.archive_member:
                    archive = archives.get(asset.download_url)
                    if archive is None:
                        archive = temporary_root / "models.zip"
                        _download(asset.download_url, archive)
                        archives[asset.download_url] = archive
                    with zipfile.ZipFile(archive) as bundle, bundle.open(asset.archive_member) as source:
                        partial = temporary_root / asset.filename
                        with partial.open("wb") as stream:
                            shutil.copyfileobj(source, stream, length=1024 * 1024)
                else:
                    partial = temporary_root / asset.filename
                    _download(asset.download_url, partial)
                actual = _sha256(partial)
                if actual != asset.sha256:
                    raise RuntimeError(f"Checksum mismatch for {asset.filename}: expected {asset.sha256}, got {actual}")
                partial.replace(target)
                installed.append(asset.filename)
            sidecar = target.with_suffix(target.suffix + ".license.json")
            sidecar.write_text(json.dumps(asset.license_manifest(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"directory": str(destination), "installed": installed, "reused": reused}
